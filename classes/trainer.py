import os.path
import torch
from timeit import default_timer as timer
from datetime import timedelta
from .utils import Cache
from .statistics import Stats, Monitor
import math
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
import random
import gc

class Trainer(object):
    class TrainerDecorators(object):
        @classmethod
        def training_wrapper(cls, training_function):
            def wrapper(self, *args, **kw):
                # Set model into training mode
                self.model.train()

                # Define monitor to track time and avg. loss over time
                mon = Monitor(self.epochs * self.n_batches, 1000, agr=Monitor.Aggregate.AVG, title=self.title, json_dir=self.json)

                # Train model if no cache file exists or the train flag is set, otherwise load cached model
                chache_file_name = self.cache.cache_dir + self.cache.key_prefix + '_' + self.cache_filename + '.sdc'
                if self.cache.disabled or not os.path.isfile(chache_file_name):

                    # Train the model
                    print('Training model...')
                    for epoch in range(self.epochs):
                        losses_epoch = []
                        for batch_data in self.training_data:
                            if self._scaler is None:
                                # --------- Decorated function -----------
                                loss = training_function(self, batch_data)
                                # ----------------------------------------

                                # Reset optimizer loss
                                self.optimizer.zero_grad()

                                # Unscaled backward propagation
                                loss.backward()
                                
                                # Clip gradient to prevent exploding gradient
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                                
                                # Optimizer step in direction of gradient
                                self.optimizer.step()
                            else:
                                with torch.cuda.amp.autocast():
                                    # --------- Decorated function -----------
                                    loss = training_function(self, batch_data)
                                    # ----------------------------------------

                                # Reset optimizer loss
                                self.optimizer.zero_grad()

                                # Scaled backward propagation
                                self._scaler.scale(loss).backward()

                                # Unscales the gradients of optimizer's assigned params in-place
                                self._scaler.unscale_(self.optimizer)

                                # Clip gradient to prevent exploding gradient
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

                                # Scaled optimizer update step
                                self._scaler.step(self.optimizer)
                                self._scaler.update()

                            # Plot to tensorboard
                            self.writer.add_scalar(self.title + ' loss', loss, global_step=mon.iter)

                            # Append loss for avg epoch loss calculation to be used in learning rate scheduler
                            losses_epoch.append(loss.item())

                            # Calculate time left and save avg. loss of last interval
                            if mon(loss.item()):
                                time_left_h, time_left_m = mon.time_left
                                print (f'{self.title}, Epoch [{epoch+1}/{self.epochs}], Step [{mon.iter}/{self.epochs*self.n_batches}], Moving avg. Loss: {mon.measurements[-1]:.4f}, Time left: {time_left_h}h {time_left_m}m')
                        
                        # Free up unused variables
                        gc.collect()

                        # Update scheduler
                        mean_loss_epoch = sum(losses_epoch) / len(losses_epoch)
                        self.scheduler.step(mean_loss_epoch)

                        # Calculate validation loss after each epoch
                        if self.validation and epoch == self.epochs-1:
                        #if self.validation and (epoch+1) % 20 == 0:
                            accuracy, loss = self.validate()
                            self.model.train()
                            self.writer.add_scalar("Validation accuracy", accuracy, global_step=epoch)
                            self.writer.add_scalar("Validation mean loss", loss, global_step=epoch)


                    # Get stats
                    self.stats.add_monitor(mon)
                    self.stats.losses = mon.measurements

                    # Store trained model
                    print(f'Storing model to cache {chache_file_name}...',end='')
                    torch.save(self.model.state_dict(), chache_file_name)
                    print('done')

                    # Store statistics object
                    self.cache.save('stats', self.stats, msg='Storing statistics to cache')
                else:
                    # Load cached model
                    print(f'(Cache) Loading trained model {chache_file_name}...',end='')
                    self.model.load_state_dict(torch.load(chache_file_name))
                    print('done')

                    if self.cache.exists('stats'):
                        # Load statistics object
                        self.stats = self.cache.load('stats', msg='Loading statistics object')
            return wrapper

        @classmethod       
        def validation_wrapper(cls, validation_function):
            def wrapper(self, *args, **kw):
                # Put model into evaluation mode
                self.model.eval()

                # Validate model
                print('Validating model...')
                with torch.no_grad():
                    n_correct = n_samples = n_false_positive = n_false_negative = 0
                    validation_losses = []
                    for batch_data in self.validation_data:

                        # -------------------------- Decorated function --------------------------
                        loss, predicted, target, categories = validation_function(self, batch_data)
                        # ------------------------------------------------------------------------

                        # Evaluate results
                        n_samples += target.size()[0]
                        n_correct += (predicted == target).sum().item()
                        n_false_negative += (predicted < target).sum().item()
                        n_false_positive += (predicted > target).sum().item()
                        assert n_correct == n_samples - n_false_negative - n_false_positive

                        # Append loss
                        validation_losses.append(loss.item())

                        # Add to class stats
                        self.stats.class_stats.add((predicted == target), categories)

                    # Save and cache validation results
                    self.stats.n_samples = n_samples
                    self.stats.n_false_negative = n_false_negative
                    self.stats.n_false_positive = n_false_positive
                    mean_loss = sum(validation_losses)/len(validation_losses)
                
                print(f'Validation size {self.stats.val_percent}%: Accuracy {(self.stats.accuracy * 100.0):.3f}%, False p.: {self.stats.false_positive:.3f}%, False n.: {self.stats.false_negative:.3f}%, Mean loss: {mean_loss:.3f}')
                return self.stats.accuracy, mean_loss
            return wrapper
    
    def __init__(self, model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer, mixed_precision=False):
        # Strings to be used for file and console outputs
        self.title = "Training"
        self.cache_filename = "trained_model"
        self.validation = False
        self.model = model
        assert isinstance(model, nn.Module)
        self.training_data = training_data
        self.validation_data = validation_data
        assert isinstance(training_data, DataLoader)
        assert isinstance(validation_data, DataLoader)
        self.criterion = criterion
        self.optimizer = optimizer
        assert isinstance(optimizer, Optimizer)
        self.epochs = epochs
        assert epochs > 0
        self.stats = stats
        assert isinstance(stats, Stats)
        self.cache = cache
        assert isinstance(cache, Cache)
        self.device = device
        self.n_batches = len(self.training_data)
        if mixed_precision:
            self._scaler = torch.cuda.amp.GradScaler()
        else:
            self._scaler = None
        self.json = json
        self.writer = writer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=10, verbose=True
        )

    def evaluate(self):
        self.stats.save_stats()
        self.stats.save_losses()
        self.stats.plot_losses()

    def parallel_forward(self, input, seq_lens, in_batch_first=False, out_batch_first=False):
        # Get batch_size and number of available GPUs 
        batch_size = input.size()[0] if in_batch_first else input.size()[1]
        n_gpu = torch.cuda.device_count()
        assert batch_size % n_gpu == 0

        # Replicate model for each GPU
        device_ids = range(n_gpu)
        output_device = device_ids[0]
        replicas = nn.parallel.replicate(self.model, device_ids)

        # Split inputs along batch dimension into equal chunks
        inputs = nn.parallel.scatter(input, device_ids, dim=(0 if in_batch_first else 1))
        chunk_size = batch_size // n_gpu
        inputs = list(inputs)
        for i in range(n_gpu):
            inputs[i] = (nn.utils.rnn.pack_padded_sequence(inputs[i], seq_lens[chunk_size*i:chunk_size*(i+1)], enforce_sorted=False, batch_first=in_batch_first),)
        inputs = tuple(inputs)
                
        # Apply chunks to model replicas and gather outputs on GPU with ID 0
        replicas = replicas[:len(inputs)]
        outputs = nn.parallel.parallel_apply(replicas, inputs)

        # If the chunks have different maximum sequence lengths, pad the shorter ones so all are the same length
        if len(outputs[0].size()) > 1:
            seq_dim = 1 if out_batch_first else 0
            max_seq_lens = [out.size()[seq_dim] for out in outputs]

            # If the max sequence lengths of all the chunks are the same, we can skip this
            if not max_seq_lens.count(max_seq_lens[0]) == len(max_seq_lens):
                max_seq_len = max(max_seq_lens)
                for i, out in enumerate(outputs):
                    out = out.to(output_device)
                    current_max_seq_len = out.size()[seq_dim]
                    if current_max_seq_len < max_seq_len:
                        padding = torch.zeros((out.size()[0], max_seq_len - current_max_seq_len, out.size()[2])) if out_batch_first else torch.zeros((max_seq_len - current_max_seq_len, out.size()[1], out.size()[2]))
                        padding = padding.to(output_device)
                        outputs[i] = torch.cat((out, padding), seq_dim)

        return nn.parallel.gather(outputs, output_device, dim=(0 if out_batch_first else 1))

    @TrainerDecorators.validation_wrapper
    def validate(self, batch_data):
        # exptected to return tuple of (loss, predicted, target, category)
        return 0,0,0,0

    @TrainerDecorators.training_wrapper
    def train(self, batch_data):
        # returns nothing
        pass

class Transformer():
    class Supervised(Trainer):
        def __init__(self, model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer):
            super().__init__(model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer, mixed_precision=False)
            # Strings to be used for file and console outputs
            self.title = "Supervised"
            self.cache_filename = "trained_model"
            self.validation = True

        @Trainer.TrainerDecorators.training_wrapper
        def train(self, batch_data):

            # Unpack batch data
            (data, seq_lens), labels, _  = batch_data

            # Get input and targets and get to cuda
            data = data.to(self.device)
            labels = labels[0,:,0].to(self.device)
            
            # Forward prop
            out = self.parallel_forward(data, seq_lens=seq_lens, out_batch_first=True)

            # Calculate loss
            loss = self.criterion(out, labels)

            return loss

        @Trainer.TrainerDecorators.validation_wrapper
        def validate(self, batch_data):
            # Unpack batch data
            (data, seq_lens), labels, categories = batch_data

            # Move data to selected device 
            data = data.to(self.device)
            labels = labels.to(self.device)
            categories = categories.to(self.device)

            # Masked forward pass
            logits = self.parallel_forward(data, seq_lens=seq_lens, out_batch_first=True)

            # Apply sigmoid function and round
            sigmoided_output = torch.sigmoid(logits)
            predicted = torch.round(sigmoided_output)

            # Extract single categories and label vector out of seq (they are all the same)
            targets = labels[0, :, 0].squeeze()
            categories = categories[0, :, 0].squeeze()  

            # Calculate loss
            loss = self.criterion(logits, targets)

            return loss, predicted, targets, categories

    class Interpolation(Trainer):
        def __init__(self, model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer):
            super().__init__(model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer)
            # Strings to be used for file and console outputs
            self.title = "Interpolation"
            self.cache_filename = "pretrained_model"
            self.validate = False

        @Trainer.TrainerDecorators.training_wrapper
        def train(self, batch_data):
            # Unpack data and move to device
            (_, data), _, _ = batch_data

            data = data.to(self.device)
            data_unpacked, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(data)

            # Select every even idx of data as src and every odd idx as target
            max_seq_len = data_unpacked.size()[0]
            trg_idx = torch.arange(1, max_seq_len, step=2)
            src_idx = trg_idx - 1
            trg_data = torch.nn.utils.rnn.pack_padded_sequence(data_unpacked[trg_idx,:,:], seq_lens // 2, enforce_sorted=False)
            src_data = torch.nn.utils.rnn.pack_padded_sequence(data_unpacked[src_idx,:,:], seq_lens // 2, enforce_sorted=False)

            #TODO: Figure out what to do with sequences of length 1 (results in length 0 after interpolation split)

            # Forward pass
            out = self.model(src_data, trg_data)

            # Create mask for non-padded items only
            loss = self.criterion(out, trg_data)

            return loss

    class Autoencode(Trainer):
        def __init__(self, model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer):
            super().__init__(model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer)
            # Strings to be used for file and console outputs
            self.title = "Autoencoder"
            self.cache_filename = "pretrained_model"

        @Trainer.TrainerDecorators.training_wrapper
        def train(self, batch_data):

            # Unpack data and move to device
            (data, seq_lens), _, _ = batch_data
            data = data.to(self.device)

            # Forward pass
            out = self.parallel_forward(data, seq_lens=seq_lens)

            # Create mask for non-padded items only
            loss = self.criterion(out, data)

            return loss

    class ObscureFeature(Trainer):
        def __init__(self, model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer):
            super().__init__(model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer)
            # Strings to be used for file and console outputs
            self.title = "ObscureFeature"
            self.cache_filename = "pretrained_model"

        def obscure(self, data, i_start, i_end):
            assert i_end >= i_start
            assert i_end < data.size()[2]
            masked_data = data
            data_size = data.size()
            masked_data[:, :, i_start:i_end] = -torch.ones(data_size[0], data_size[1], i_end-i_start)
            return masked_data

        def obscure_random(self, data, n_features):
            masked_data = data
            max_seq_length, batch_size, input_size = data.shape
            for _ in range(n_features):
                idx = random.randint(0, input_size-1)
                masked_data[:, :, idx] = torch.zeros(max_seq_length, batch_size)
            return masked_data

        def mask(self, op_size, seq_lens):
            mask = torch.zeros(op_size, dtype=torch.bool)
            for index, length in enumerate(seq_lens):
                mask[:length, index,:] = True
            return mask

        @Trainer.TrainerDecorators.training_wrapper
        def train(self, batch_data):
            # Unpack batch data
            (data, seq_lens), _, _ = batch_data

            data = data.to(self.device)

            # Obscure features
            masked_data = self.obscure_random(data, 1)

            # Pack data
            masked_data = torch.nn.utils.rnn.pack_padded_sequence(data, seq_lens, enforce_sorted=False)

            # Move data to selected device 
            masked_data = masked_data
            
            # Forwards pass
            outputs = self.parallel_forward(data, seq_lens=seq_lens)

            op_mask = self.mask(outputs.size(), seq_lens)
            loss = self.criterion(outputs[op_mask], data[op_mask])

            return loss

    class MaskPacket(Trainer):
        def __init__(self, model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer):
            super().__init__(model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer)
            # Strings to be used for file and console outputs
            self.title = "MaskPacket"
            self.cache_filename = "pretrained_model"

        def mask_packets(self, data, seq_lens, n_features):
            masked_data = data
            mask = torch.zeros(data.size(), dtype=torch.bool)
            _, _, input_size = data.shape
            for _ in range(n_features):
                for batch_idx, length in enumerate(seq_lens):
                    seq_idx = random.randint(0, length-1)
                    masked_data[seq_idx, batch_idx, :] = torch.zeros(input_size)
                    mask[seq_idx, batch_idx, :] = True
            return masked_data, mask

        @Trainer.TrainerDecorators.training_wrapper
        def train(self, batch_data):
            # Unpack batch data
            (data, seq_lens), _, _ = batch_data

            data = data.to(self.device)

            # Obscure features
            masked_data, mask = self.mask_packets(data, seq_lens, 1)

            # Pack data
            masked_data = torch.nn.utils.rnn.pack_padded_sequence(masked_data, seq_lens, enforce_sorted=False)

            # Forwards pass
            outputs = self.parallel_forward(data, seq_lens=seq_lens)
            loss = self.criterion(outputs[mask], data[mask])

            return loss
            
class LSTM():
    class Supervised(Trainer):
        def __init__(self, model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer):
            super().__init__(model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer, mixed_precision=True)
            # Strings to be used for file and console outputs
            self.title = "Supervised"
            self.cache_filename = "supervised_trained_model"
            self.validation = True

        def mask(self, op_size, seq_lens):
            mask = torch.zeros(op_size, dtype=torch.bool)
            for index, length in enumerate(seq_lens):
                mask[index, :length,:] = True
            return mask
            
        def logit_mask(self, op_size, seq_lens):
            logit_mask = torch.zeros(op_size, dtype=torch.bool)
            for index, length in enumerate(seq_lens):
                logit_mask[index, length-1,:] = True
            return logit_mask

        @Trainer.TrainerDecorators.training_wrapper
        def train(self, batch_data):
            # Unpack batch data
            (data, seq_lens), labels, _ = batch_data

            # Move data to selected device 
            data = data.to(self.device)
            labels = labels.to(self.device)
                
            # Forwards pass
            outputs = self.parallel_forward(data, seq_lens, in_batch_first=True, out_batch_first=True)
            mask = self.mask(outputs.size(), seq_lens)
            op_view = outputs[mask].view(-1)
            lab_view = labels[mask].view(-1)
            loss = self.criterion(op_view, lab_view)

            return loss

        @Trainer.TrainerDecorators.validation_wrapper
        def validate(self, batch_data):
            # Unpack batch data
            (data, seq_lens), labels, categories = batch_data

            # Move data to selected device 
            data = data.to(self.device)
            labels = labels.to(self.device)
            categories = categories.to(self.device)

            # Forward pass
            outputs = self.parallel_forward(data, seq_lens=seq_lens, in_batch_first=True, out_batch_first=True)
            logit_mask = self.logit_mask(outputs.size(), seq_lens)

            # Max returns (value ,index)
            sigmoided_output = torch.sigmoid(outputs.data[logit_mask].detach())
            predicted = torch.round(sigmoided_output)
            targets = labels[:, 0, 0].squeeze()
            categories = categories[:, 0, 0].squeeze()  

            # Calculate loss
            mask = self.mask(outputs.size(), seq_lens)
            op_view = outputs[mask].view(-1)
            lab_view = labels[mask].view(-1)
            loss = self.criterion(op_view, lab_view)

            return loss, predicted, targets, categories

    class PredictPacket(Trainer):
        def __init__(self, model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer):
            super().__init__(model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer, mixed_precision=True)
            # Strings to be used for file and console outputs
            self.title = "PredictPacket"
            self.cache_filename = "pretrained_model"

        def masks(self, op_size, seq_lens):
            src_mask = torch.zeros(op_size, dtype=torch.bool)
            trg_mask = torch.zeros(op_size, dtype=torch.bool)
            for index, length in enumerate(seq_lens):
                src_mask[index, :length-1,:] = True
                trg_mask[index, 1:length,:] = True
                #src_mask[index, length-2,:] = True
                #trg_mask[index, length-1,:] = True
            return src_mask, trg_mask

        @Trainer.TrainerDecorators.training_wrapper
        def train(self, batch_data):
            # Unpack batch data
            (data, seq_lens), _, _ = batch_data

            # Move data to selected device 
            data = data.to(self.device)

            # Forwards pass
            outputs = self.parallel_forward(data, seq_lens, in_batch_first=True, out_batch_first=True)
            src_mask, trg_mask = self.masks(outputs.size(), seq_lens)
            loss = self.criterion(outputs[src_mask], data[trg_mask])

            return loss

    class ObscureFeature(Trainer):
        def __init__(self, model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer):
            super().__init__(model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer)
            # Strings to be used for file and console outputs
            self.title = "ObscureFeature"
            self.cache_filename = "pretrained_model"

        def obscure(self, data, i_start, i_end):
            batch_size, max_seq_length, input_size = data.size()
            assert i_end >= i_start
            assert i_end < input_size
            mask = torch.zeros(data.size(), dtype=torch.bool)
            masked_data = data
            masked_data[:, :, i_start:i_end] = -torch.ones(batch_size, max_seq_length, i_end-i_start)
            mask[:, :, i_start:i_end] = True
            return masked_data, mask

        def obscure_random(self, data, i_start, i_end, obscuration_rate):
            batch_size, max_seq_length, input_size = data.shape
            assert i_end < input_size
            assert i_end >= i_start
            masked_data = data
            mask = torch.zeros(data.size(), dtype=torch.bool)
            n_packets = math.ceil(max_seq_length * obscuration_rate)
            for _ in range(n_packets):
                idx = random.randint(0, max_seq_length-1)
                masked_data[:, idx, i_start:i_end] = -torch.ones(batch_size, i_end-i_start)
                mask[:, idx, i_start:i_end] = True
            return masked_data, mask

        @Trainer.TrainerDecorators.training_wrapper
        def train(self, batch_data):
            # Unpack batch data
            (data, seq_lens), _, _ = batch_data

            # Obscure features
            masked_data, mask = self.obscure_random(data, 6, 9, 0.3)

            # Move data to selected device 
            masked_data = data.to(self.device)
            data = data.to(self.device)
            mask = mask.to(self.device)

            # Forwards pass
            outputs = self.parallel_forward(masked_data, seq_lens, in_batch_first=True, out_batch_first=True)
            loss = self.criterion(outputs[mask], data[mask])

            return loss

    class MaskPacket(Trainer):
        def __init__(self, model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer):
            super().__init__(model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer, mixed_precision=True)
            # Strings to be used for file and console outputs
            self.title = "MaskPacket"
            self.cache_filename = "pretrained_model"

        def mask_packets(self, data, seq_lens, n_packets):
            masked_data = data
            mask = torch.zeros(data.size(), dtype=torch.bool)
            _, _, input_size = data.shape
            for _ in range(n_packets):
                for batch_idx, length in enumerate(seq_lens):
                    seq_idx = random.randint(0, length-1)
                    masked_data[batch_idx, seq_idx, :] = -torch.ones(input_size)
                    mask[batch_idx, seq_idx, :] = True
            return masked_data, mask

        @Trainer.TrainerDecorators.training_wrapper
        def train(self, batch_data):
            # Unpack batch data
            (data, seq_lens), _, _ = batch_data

            # Obscure features
            masked_data, mask = self.mask_packets(data, seq_lens, 1)

            # Move data to selected device 
            masked_data = masked_data.to(self.device)
            data = data.to(self.device)
            mask = mask.to(self.device)

            # Forwards pass
            outputs = self.parallel_forward(masked_data, seq_lens, in_batch_first=True, out_batch_first=True)
            loss = self.criterion(outputs[mask], data[mask])

            return loss

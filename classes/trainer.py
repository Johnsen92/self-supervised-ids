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

                            # Scaled forward propagation
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

                            # Since the gradients of optimizer's assigned params are unscaled, clip as usual
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
                        
                        # Update scheduler
                        mean_loss_epoch = sum(losses_epoch) / len(losses_epoch)
                        self.scheduler.step(mean_loss_epoch)

                        # Calculate validation loss after each epoch
                        if self.validation:
                            accuracy, loss = self.validate()
                            self.writer.add_scalar("Validation accuracy", accuracy, global_step=epoch)
                            self.writer.add_scalar("Validation mean loss", loss, global_step=epoch)


                    # Get stats
                    self.stats.add_monitor(mon)
                    self.stats.losses = mon.measurements

                    # Store trained model
                    print('Storing model to cache...',end='')
                    torch.save(self.model.state_dict(), chache_file_name)
                    print('done')

                    # Store statistics object
                    self.cache.save('stats', self.stats, msg='Storing statistics to cache')
                else:
                    # Load cached model
                    print('(Cache) Loading trained model...',end='')
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
    
    def __init__(self, model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer):
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
        self._scaler = torch.cuda.amp.GradScaler()
        self.json = json
        self.writer = writer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=10, verbose=True
        )

    def evaluate(self):
        self.stats.save_stats()
        self.stats.save_losses()
        self.stats.plot_losses()

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
            super().__init__(model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer)
            # Strings to be used for file and console outputs
            self.title = "Supervised"
            self.cache_filename = "trained_model"
            self.validation = True

        @Trainer.TrainerDecorators.training_wrapper
        def train(self, batch_data):

            # Unpack batch data
            (_, data), labels, _ = batch_data

            # Get input and targets and get to cuda
            data = data.to(self.device)
            labels = labels[0,:,0].to(self.device)
            
            # Forward prop
            out = self.model(data)

            # Calculate loss
            loss = self.criterion(out, labels)

            return loss

        @Trainer.TrainerDecorators.validation_wrapper
        def validate(self, batch_data):
            # Unpack batch data
            (_, data), labels, categories = batch_data

            # Move data to selected device 
            data = data.to(self.device)
            labels = labels.to(self.device)
            categories = categories.to(self.device)

            # Masked forward pass
            logits = self.model(data)

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
            (data_unpacked, data), _, _ = batch_data
            data_unpacked = data_unpacked.to(self.device)
            data = data.to(self.device)

            # Forward pass
            out = self.model(data, data)

            # Create mask for non-padded items only
            loss = self.criterion(out, data_unpacked)

            return loss

class LSTM():
    class Supervised(Trainer):
        def __init__(self, model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer):
            super().__init__(model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer)
            # Strings to be used for file and console outputs
            self.title = "Supervised"
            self.cache_filename = "trained_model"
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
            (_, data), labels, _ = batch_data

            # Move data to selected device 
            data = data.to(self.device)
            labels = labels.to(self.device)
                
            # Forwards pass
            outputs, seq_lens = self.model(data)
            mask = self.mask(outputs.size(), seq_lens)
            op_view = outputs[mask].view(-1)
            lab_view = labels[mask].view(-1)
            loss = self.criterion(op_view, lab_view)

            return loss

        @Trainer.TrainerDecorators.validation_wrapper
        def validate(self, batch_data):
            # Unpack batch data
            (_, data), labels, categories = batch_data

            # Move data to selected device 
            data = data.to(self.device)
            labels = labels.to(self.device)
            categories = categories.to(self.device)

            # Forward pass
            outputs, seq_lens = self.model(data)
            logit_mask = self.logit_mask(outputs.size(), seq_lens)

            # Max returns (value ,index)
            sigmoided_output = torch.sigmoid(outputs.data[logit_mask].detach())
            predicted = torch.round(sigmoided_output)
            targets = labels[:, 0, :].squeeze()
            categories = categories[:, 0, :].squeeze()  

            # Calculate loss
            mask = self.mask(outputs.size(), seq_lens)
            op_view = outputs[mask].view(-1)
            lab_view = labels[mask].view(-1)
            loss = self.criterion(op_view, lab_view)

            return loss, predicted, targets, categories

    class PredictPacket(Trainer):
        def __init__(self, model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer):
            super().__init__(model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer)
            # Strings to be used for file and console outputs
            self.title = "PredictPacket"
            self.cache_filename = "pretrained_model"

        def masks(self, op_size, seq_lens):
            logit_mask = torch.zeros(op_size, dtype=torch.bool)
            target_mask = torch.zeros(op_size, dtype=torch.bool)
            for index, length in enumerate(seq_lens):
                target_mask[index, :length-1,:] = True
                logit_mask[index, 1:length,:] = True
            return logit_mask, target_mask

        @Trainer.TrainerDecorators.training_wrapper
        def train(self, batch_data):
            # Unpack batch data
            (_, data), _, _ = batch_data

            # Move data to selected device 
            data = data.to(self.device)
            data_unpacked, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(data, batch_first=True)

            # Forwards pass
            outputs, seq_lens = self.model(data)
            logit_mask, target_mask = self.masks(outputs.size(), seq_lens)
            loss = self.criterion(outputs[logit_mask], data_unpacked[target_mask])

            return loss

    class ObscureFeature(Trainer):
        def __init__(self, model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer):
            super().__init__(model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json, writer)
            # Strings to be used for file and console outputs
            self.title = "ObscureFeature"
            self.cache_filename = "pretrained_model"

        def obscure(self, data, i_start, i_end):
            assert i_end >= i_start
            assert i_end < data.size()[1]
            masked_data = data
            data_size = data.size()
            masked_data[:, i_start:i_end, :] = torch.zeros(data_size[0], i_end-i_start, data_size[2])
            return masked_data

        def mask(self, op_size, seq_lens):
            mask = torch.zeros(op_size, dtype=torch.bool)
            for index, length in enumerate(seq_lens):
                mask[index, :length,:] = True
            return mask

        @Trainer.TrainerDecorators.training_wrapper
        def train(self, batch_data):
            # Unpack batch data
            (_, data), _, _ = batch_data

            # Unpack data
            data_unpacked, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(data, batch_first=True)

            # Obscure features
            masked_data = self.obscure(data_unpacked, 6, 9)

            # Pack data
            masked_data = torch.nn.utils.rnn.pack_padded_sequence(masked_data, seq_lens, batch_first=True, enforce_sorted=False)

            # Move data to selected device 
            masked_data = masked_data.to(self.device)
            data_unpacked = data_unpacked.to(self.device)

            # Forwards pass
            outputs, _ = self.model(masked_data)
            op_mask = self.mask(outputs.size(), seq_lens)
            loss = self.criterion(outputs[op_mask], data_unpacked[op_mask])

            return loss
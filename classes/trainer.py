import os.path
import torch
from timeit import default_timer as timer
from datetime import timedelta
from classes import utils, datasets
from .statistics import Stats, Monitor, PDData, NeuronData
from .datasets import Flows, FlowsSubset
from .utils import Cache
import math
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
import random
import gc
from pympler import muppy, summary
from pandas import DataFrame
from enum import Enum
from torch.utils.data import Dataset, Subset
import numpy as np
import pickle
import json
from collections import Counter

def memory_dump():
    # Add to leaky code within python_script_being_profiled.py
    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)# Prints out a summary of the large objects
    summary.print_(sum1)# Get references to certain types of objects such as dataframe
    dataframes = [ao for ao in all_objects if isinstance(ao, DataFrame)]
    for d in dataframes:
        print(d.columns.values)
        print(len(d))

def get_nth_split(dataset, n_fold, index, maxSize):
	dataset_size = len(dataset)
	indices = list(range(dataset_size))
	bottom, top = int(math.floor(float(dataset_size)*index/n_fold)), int(math.floor(float(dataset_size)*(index+1)/n_fold))
	train_indices, test_indices = indices[0:bottom]+indices[top:], indices[bottom:top]
	return train_indices[:maxSize], test_indices[:maxSize]

class Trainer(object):
    class TrainerDecorators(object):
        @classmethod
        def training_wrapper(cls, training_function):
            def wrapper(self, *args, **kw):
                # Set model into training mode
                self.model.train()

                # Define monitor to track time and avg. loss over time
                mon = Monitor(self.epochs * self.n_batches, 1000, agr=Monitor.Aggregate.AVG, title=self.title, json_dir=self.json)

                # Train the model
                chache_file_name = self.cache_filename + str(self.training_data.dataset)
                if self.cache.disabled or not self.cache.exists(chache_file_name):
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

                            # Append loss for avg epoch loss calculation to be used in learning rate scheduler
                            losses_epoch.append(loss.item())

                            # Calculate time left and save avg. loss of last interval
                            if mon(loss.item()):
                                time_left_h, time_left_m = mon.time_left
                                print (f'{self.title}, Epoch [{epoch+1}/{self.epochs}], Step [{mon.iter}/{self.epochs*self.n_batches}], Moving avg. Loss: {mon.measurements[-1]:.4f}, Time left: {time_left_h}h {time_left_m}m')
                        
                        # Free up unused variables
                        gc.collect()

                        # Update scheduler
                        mean_loss_epoch = sum(losses_epoch) / max(len(losses_epoch),1)
                        self.scheduler.step(mean_loss_epoch)

                        # Plot to tensorboard
                        self.writer.add_scalar(self.title + ' loss', mean_loss_epoch, global_step=epoch)

                        # Validation is performed if enabled and after the last epoch or periodically if val_epochs is set not set to 0
                        validate_periodically = (epoch + 1) % self.val_epochs == 0 if self.val_epochs != 0 else False
                        if self.validation and (epoch == self.epochs-1 or validate_periodically):
                            self.stats.new_epoch(
                                epoch = epoch, 
                                training_time = mon.time_passed, 
                                training_loss = mean_loss_epoch)
                            accuracy, loss = self.validate()
                            self.model.train()
                            if self.stats.last_epoch.epoch == self.stats.best_epoch.epoch:
                                self.cache.save_model(chache_file_name, self.model, tmp=True)
                                self.cache.save('best_epoch', self.stats.last_epoch)
                                print(f"New best epoch: {self.stats.last_epoch.epoch}, Time: {self.stats.training_time_to_best_epoch[0]}h {self.stats.training_time_to_best_epoch[1]}m")
                            self.writer.add_scalar('Validation accuracy', accuracy, global_step=epoch)
                            self.writer.add_scalar('Validation mean loss', loss, global_step=epoch)

                    # Store statistics object and load and save best epoch
                    if self.validation:
                        self.cache.load_model(chache_file_name, self.model, tmp=True)
                        self.cache.save('stats', self.stats, msg='Storing statistics to cache')

                    # Safe model
                    self.cache.save_model(chache_file_name, self.model)
                else:
                    # Load cached model
                    self.cache.load_model(chache_file_name, self.model)

                    if self.cache.exists('stats'):
                        # Load statistics object
                        stats_dir = self.stats.stats_dir
                        self.stats = self.cache.load('stats', msg='Loading statistics object')
                        self.stats.set_stats_dir(stats_dir)
                        self.stats.set_category_mapping(self.training_data.dataset.mapping)
                        self.stats.make_stats_dir()
                        print(f"Best epoch: {self.stats.best_epoch.epoch}, Time: {self.stats.training_time_to_best_epoch[0]}h {self.stats.training_time_to_best_epoch[1]}m")
                    elif self.cache.exists('best_epoch'):
                        best_epoch = self.cache.load('best_epoch', msg='Loading best epoch')
                        self.stats.add_epoch(best_epoch)
                        print(f"Best epoch: {self.stats.best_epoch.epoch}, Time: {self.stats.training_time_to_best_epoch[0]}h {self.stats.training_time_to_best_epoch[1]}m")
                    elif self.validation:
                        self.stats.new_epoch(self.epochs)
                        self.validate()
                        # Store statistics object
                        self.cache.save('stats', self.stats, msg='Storing statistics to cache')

            return wrapper

        @classmethod       
        def validation_wrapper(cls, validation_function):
            @torch.no_grad()
            def wrapper(self, *args, **kw):
                # Put model into evaluation mode
                self.model.eval()

                # Validate model
                print('Validating model...')
                epoch = self.stats.last_epoch
                with torch.no_grad():
                    validation_losses = []
                    # Reset metric counters in stats and class stats
                    for batch_data in self.validation_data:
                        
                        # -------------------------- Decorated function --------------------------
                        loss, _, predicted, target, categories = validation_function(self, batch_data)
                        # ------------------------------------------------------------------------

                        # Evaluate results
                        epoch.add_batch(predicted, target, categories)

                        # Append loss
                        validation_losses.append(loss.item())
                        
                    # Save and cache validation results
                    mean_loss = sum(validation_losses)/len(validation_losses)
                
                epoch.validation_loss = mean_loss
                print(f'Validation size {self.stats.val_percent}%: Accuracy {(epoch.accuracy * 100.0):.3f}%, FAR: {(epoch.false_alarm_rate * 100.0):.3f}%, Precision: {(epoch.precision * 100.0):.3f}%, Mean loss: {mean_loss:.4f}')
                return epoch.accuracy, mean_loss
            return wrapper

    def __init__(self, model, training_data, validation_data, test_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title, mixed_precision=False):
        # Strings to be used for file and console outputs
        self.title = title
        self.cache_filename = 'trained_model'
        self.validation = False
        self.model = model
        assert isinstance(model, nn.Module)
        self.training_data = training_data
        self.validation_data = validation_data
        self.test_data = test_data
        assert isinstance(training_data, DataLoader)
        assert isinstance(validation_data, DataLoader) or validation_data is None
        assert isinstance(test_data, DataLoader) or validation_data is None
        self.criterion = criterion
        self.optimizer = optimizer
        assert isinstance(optimizer, Optimizer)
        self.epochs = epochs
        self.val_epochs = val_epochs
        assert epochs > 0
        self.stats = stats
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
            optimizer, factor=0.1, patience=self.epochs // 10, verbose=True
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
        outputs, neurons, states = zip(*nn.parallel.parallel_apply(replicas, inputs))
        outputs = list(outputs)
        neurons = list(neurons)
        hidden_states, cell_states = zip(*states)

        assert len(outputs) == len(neurons), 'Length of outputs and neurons do not match'

        # If the chunks have different maximum sequence lengths, pad the shorter ones so all are the same length
        seq_dim = 1 if out_batch_first else 0
        
        # Padding for output sequences
        max_seq_lens = [out.size()[seq_dim] for out in outputs]
        max_seq_len = max(max_seq_lens)
        for i, out in enumerate(outputs):
            out = out.to(output_device)
            seq_len = out.size()[seq_dim]
            if seq_len < max_seq_len:
                padding_out = torch.zeros((out.size()[0], max_seq_len - seq_len, out.size()[2])) if out_batch_first else torch.zeros((max_seq_len - seq_len, out.size()[1], out.size()[2]))
                padding_out = padding_out.to(output_device)
                outputs[i] = torch.cat((out, padding_out), seq_dim)

        # Padding for neuron sequences
        max_seq_lens = [neuron.size()[seq_dim] for neuron in neurons]
        max_seq_len = max(max_seq_lens)
        for i, neuron in enumerate(neurons):
            neuron = neuron.to(output_device)
            seq_len = neuron.size()[seq_dim]
            if seq_len < max_seq_len:
                padding_neuron = torch.zeros((neuron.size()[0], max_seq_len - seq_len, neuron.size()[2])) if out_batch_first else torch.zeros((max_seq_len - seq_len, neuron.size()[1], neuron.size()[2]))
                padding_neuron = padding_neuron.to(output_device)
                neurons[i] = torch.cat((neuron, padding_neuron), seq_dim)

        merged_output = nn.parallel.gather(outputs, output_device, dim=(0 if out_batch_first else 1))
        merged_neurons = nn.parallel.gather(neurons, output_device, dim=(0 if out_batch_first else 1))
        merged_hidden_state = nn.parallel.gather(hidden_states, output_device, dim=1)
        merged_cell_state = nn.parallel.gather(cell_states, output_device, dim=1)
        return merged_output, merged_neurons, (merged_hidden_state, merged_cell_state)

    @TrainerDecorators.validation_wrapper
    def validate(self, batch_data):
        # exptected to return tuple of (loss, predicted, target, category)
        return 0,0,0,0

    @TrainerDecorators.training_wrapper
    def train(self, batch_data):
        # returns nothing
        pass

    @torch.no_grad()
    def pdp(self, id, config_file, batch_first=True):
        with open(config_file, 'r') as f:
            config = json.load(f)

        self.model.eval()
        pdp_base_dir = os.path.dirname(self.test_data.dataset.data_pickle) + '/pdp/'
        pdp_file = pdp_base_dir + id + '.pickle'
        if os.path.exists(pdp_file):
            print(f'PDP data for {id} already exists. Returning...')
            return

        minmax = self.test_data.dataset.minmax
        stds = self.test_data.dataset.stds
        means = self.test_data.dataset.means
        mapping = self.stats.mapping
        reverse_mapping = {v:k for k,v in mapping.items()}
        

        # PDP data generation parameters
        max_batch_size = 512
        max_samples = 1024

        pd_data = PDData(id, config)
        for feature_key, feature_name in config['features'].items():
            feature_ind = int(feature_key)
            for category in config['categories']:
                # Get at most max_samples flows of attack_number
                good_subset = FlowsSubset(self.test_data.dataset, mapping, dist={category: max_samples}, ditch=[-1, category])

                pd_data.features[(category, feature_ind)] = np.array([(item[0][0,feature_ind]*stds[feature_ind] + means[feature_ind]).item() for item in good_subset])

                # Calculate optimal batch size but at most max_batch_size
                batch_size = len(good_subset)
                div = 2
                while batch_size > max_batch_size:
                    batch_size = len(good_subset) // div
                    div += 1
                # Assert even batch size (only needed if you have dual GPU)
                batch_size = (batch_size // 2) * 2

                # If too few samples, continue
                if len(good_subset) < 128:
                    print(f'Did not find enough samples ({len(good_subset)}) for attack category {category}. Continuing...')
                    continue

                print(f'Generating PDP data for flow category {reverse_mapping[category]} ({category}) and feature {feature_name} ({feature_ind})...',end='')
                feat_min, feat_max = minmax[feature_ind]
                values = np.linspace(feat_min, feat_max, 100)
                pdp = np.zeros([values.size])   

                for i in range(values.size):
                    for index, sample in enumerate(good_subset):
                        for j in range(sample[0].shape[0]):
                            sample[0][j,feature_ind] = values[i]

                    loader = DataLoader(dataset=good_subset, batch_size=batch_size, shuffle=True, num_workers=12, collate_fn=datasets.collate_flows_batch_first if batch_first else datasets.collate_flows, drop_last=True)
                    outputs = []
                    for (input_data, seq_lens), _, _ in loader:
                        output, _, _ = self.parallel_forward(input_data, seq_lens=seq_lens, in_batch_first=batch_first, out_batch_first=True)

                        # Data is (Sequence Index, Batch Index, Feature Index)
                        for batch_index in range(output.shape[0]):
                            flow_length = seq_lens[batch_index]
                            if len(output.size()) == 3:
                                flow_output = output[batch_index,:flow_length,:].detach().cpu().numpy()
                            elif len(output.size()) == 1:
                                flow_output = output[batch_index].detach().cpu().repeat(flow_length, 1).numpy()
                            else:
                                print(f'No case for output with {len(output.size())} dimensions. Error...')
                            outputs.append(flow_output)

                    pdp[i] = np.mean(np.array([utils.numpy_sigmoid(output[-1]) for output in outputs] ))

                rescaled = values * stds[feature_ind] + means[feature_ind]
                pd_data.results[(category, feature_ind)] = np.vstack((rescaled,pdp))
                print(f'done')

        # Save PDP Data
        with open(pdp_file, 'wb') as f:
            pickle.dump(pd_data, f)

    @torch.no_grad()
    def neuron_activation(self, id, config_file, title=None, postfix=None, batch_first=True, out_batch_first=False):
        with open(config_file, 'r') as f:
            config = json.load(f)

        self.model.eval()
        mapping = self.stats.mapping
        na_base_dir = os.path.dirname(self.test_data.dataset.data_pickle) + '/neurons/'
        na_file = na_base_dir + id + ('_' + postfix if not postfix is None else '') + '.pickle'
        if os.path.exists(na_file):
            print(f'Neuron data for {id} already exists. Returning...')
            return

        # PDP data generation parameters
        max_batch_size = 1024
        max_samples = 1024

        if len(config['categories']) == 0:
            categories = [v for _, v in mapping.items()]
        else:
            categories = config['categories']

        neuron_data = NeuronData(id, config, title)

        for category in categories:
            # Get at most max_samples flows of attack_number
            good_subset = FlowsSubset(self.test_data.dataset, mapping, dist={category: max_samples}, ditch=[-1, category])

            # Calculate optimal batch size but at most max_batch_size
            batch_size = len(good_subset)
            div = 2
            while batch_size > max_batch_size:
                batch_size = len(good_subset) // div
                div += 1
            # Assert even batch size (only needed if you have dual GPU)
            batch_size = (batch_size // 2) * 2

            # If too few samples, continue
            if len(good_subset) < 128:
                print(f'Did not find enough samples ({len(good_subset)}) for attack category {category}. Continuing...')
                continue

            loader = DataLoader(dataset=good_subset, batch_size=batch_size, shuffle=True, num_workers=12, collate_fn=datasets.collate_flows_batch_first if batch_first else datasets.collate_flows, drop_last=True)
            neurons_latest = []
            neurons_means = []
            for (input_data, seq_lens), _, _ in loader:
                _, neurons, _ = self.parallel_forward(input_data, seq_lens=seq_lens, in_batch_first=batch_first, out_batch_first=out_batch_first)
                neurons_adjusted = (neurons if batch_first else neurons.permute(1,0,2)).detach().cpu()

                # neurons is (Batch Index, Sequence Index, Feature Index)
                for batch_index, seq_len in enumerate(seq_lens):
                    neurons_latest.append(neurons_adjusted[batch_index, seq_len-1, :].numpy())
                    neurons_means.append(torch.mean(neurons_adjusted[batch_index, :seq_len, :], 0).numpy())

            neuron_data.latest[category] = np.mean(neurons_latest, axis=0)
            neuron_data.means[category] = np.mean(neurons_means, axis=0)
            print(f'done')

        # Save Neuron Data
        with open(na_file, 'wb') as f:
            pickle.dump(neuron_data, f)

class Transformer():
    class Supervised(Trainer):
        def __init__(self, model, training_data, validation_data, test_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title):
            super().__init__(model, training_data, validation_data, test_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title, mixed_precision=False)
            # Strings to be used for file and console outputs
            self.cache_filename = 'trained_model'
            self.validation = True

        @Trainer.TrainerDecorators.training_wrapper
        def train(self, batch_data):

            # Unpack batch data
            (data, seq_lens), labels, _  = batch_data

            # Get input and targets and get to cuda
            data = data.to(self.device)
            labels = labels[0,:,0].unsqueeze(1).unsqueeze(0).to(self.device)
            
            # Forward prop
            out, _, _ = self.parallel_forward(data, seq_lens=seq_lens)

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
            out, _, _ = self.parallel_forward(data, seq_lens=seq_lens)

            out = out[0, :, 0]

            # Apply sigmoid function and round
            sigmoided_output = torch.sigmoid(out)
            predicted = torch.round(sigmoided_output)

            # Extract single categories and label vector out of seq (they are all the same)
            targets = labels[0, :, 0]
            categories = categories[0, :, 0]

            # Calculate loss
            loss = self.criterion(out, targets)

            return loss, sigmoided_output, predicted, targets, categories

    class Interpolation(Trainer):
        def __init__(self, model, training_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title, test_data=None):
            super().__init__(model, training_data, None, test_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title, mixed_precision=True)
            # Strings to be used for file and console outputs
            self.cache_filename = 'pretrained_model'

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
            out, _, _ = self.model(src_data, trg_data)

            # Create mask for non-padded items only
            loss = self.criterion(out, trg_data)

            return loss

    class AutoEncode(Trainer):
        def __init__(self, model, training_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title, test_data=None):
            super().__init__(model, training_data, None, test_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title, mixed_precision=False)
            # Strings to be used for file and console outputs
            self.cache_filename = 'pretrained_model'

        def mask(self, op_size, seq_lens):
            mask = torch.zeros(op_size, dtype=torch.bool)
            for index, length in enumerate(seq_lens):
                mask[:length, index,:] = True
            return mask

        @Trainer.TrainerDecorators.training_wrapper
        def train(self, batch_data):

            # Unpack data and move to device
            (data, seq_lens), _, _ = batch_data
            data = data.to(self.device)

            # Forward pass
            out, _, _ = self.parallel_forward(data, seq_lens=seq_lens)

            # Create mask for non-padded items only
            mask = self.mask(data.size(), seq_lens)

            # Calculate loss
            loss = self.criterion(out[mask], data[mask])

            return loss

    class ObscureFeature(Trainer):
        def __init__(self, model, training_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title, test_data=None):
            super().__init__(model, training_data, None, test_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title, mixed_precision=False)
            # Strings to be used for file and console outputs
            self.cache_filename = 'pretrained_model'

        def obscure(self, data, i_start, i_end):
            assert i_end >= i_start
            mask = torch.zeros(data.size(), dtype=torch.bool)
            assert i_end < data.size()[2]
            masked_data = data
            data_size = data.size()
            masked_data[:, :, i_start:i_end] = -torch.ones(data_size[0], data_size[1], i_end-i_start)
            mask[:, :, i_start:i_end] = True
            return masked_data, mask

        def obscure_random(self, data, n_features):
            masked_data = data
            max_seq_length, batch_size, input_size = data.shape
            mask = torch.zeros(data.size(), dtype=torch.bool)
            for _ in range(n_features):
                idx = random.randint(0, input_size-1)
                masked_data[:, :, idx] = torch.zeros(max_seq_length, batch_size)
                mask[:, :, idx] = True
            return masked_data, mask

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
            masked_data, mask = self.obscure(data, 6, 7)

            # Pack data
            masked_data = torch.nn.utils.rnn.pack_padded_sequence(data, seq_lens, enforce_sorted=False)

            # Move data to selected device 
            masked_data = masked_data
            
            # Forwards pass
            out, _, _ = self.parallel_forward(data, seq_lens=seq_lens)

            # mask = self.mask(out.size(), seq_lens)
            loss = self.criterion(out[mask], data[mask])

            return loss

    class MaskPacket(Trainer):
        def __init__(self, model, training_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title, test_data=None):
            super().__init__(model, training_data, None, test_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title, mixed_precision=True)
            # Strings to be used for file and console outputs
            self.cache_filename = 'pretrained_model'

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
            out, _, _ = self.parallel_forward(data, seq_lens=seq_lens)
            loss = self.criterion(out[mask], data[mask])

            return loss
            
class LSTM():
    class Supervised(Trainer):
        def __init__(self, model, training_data, validation_data, test_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title):
            super().__init__(model, training_data, validation_data, test_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title, mixed_precision=False)
            # Strings to be used for file and console outputs
            self.cache_filename = 'supervised_trained_model'
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
            outputs, _, _ = self.parallel_forward(data, seq_lens, in_batch_first=True, out_batch_first=True)
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
            outputs, _, _ = self.parallel_forward(data, seq_lens=seq_lens, in_batch_first=True, out_batch_first=True)
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

            return loss, sigmoided_output, predicted, targets, categories

    class PredictPacket(Trainer):
        def __init__(self, model, training_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title, test_data=None):
            super().__init__(model, training_data, None, test_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title, mixed_precision=True)
            # Strings to be used for file and console outputs
            self.cache_filename = 'pretrained_model'

        def masks(self, op_size, seq_lens):
            src_mask = torch.zeros(op_size, dtype=torch.bool)
            trg_mask = torch.zeros(op_size, dtype=torch.bool)
            for index, length in enumerate(seq_lens):
                src_mask[index, :length-1,:] = True
                trg_mask[index, 1:length,:] = True
            return src_mask, trg_mask

        @Trainer.TrainerDecorators.training_wrapper
        def train(self, batch_data):
            # Unpack batch data
            (data, seq_lens), _, _ = batch_data

            # Move data to selected device 
            data = data.to(self.device)

            # Forwards pass
            outputs, _, _ = self.parallel_forward(data, seq_lens, in_batch_first=True, out_batch_first=True)
            src_mask, trg_mask = self.masks(outputs.size(), seq_lens)

            # Calculate loss
            loss = self.criterion(outputs[src_mask], data[trg_mask])

            return loss

    class AutoEncoder(Trainer):
        def __init__(self, model, training_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title, test_data=None):
            super().__init__(model, training_data, None, test_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title, mixed_precision=False)
            # Strings to be used for file and console outputs
            self.cache_filename = 'pretrained_model'

        def masks(self, op_size, seq_lens):
            mask = torch.zeros(op_size, dtype=torch.bool)
            for index, length in enumerate(seq_lens):
                mask[index, :length,:] = True
            return mask

        def reverse_sequences(self, seqs, seq_lens):
            seqs_reversed = torch.zeros(seqs.size(), dtype=torch.float32)
            for i, seq_len in enumerate(seq_lens):
                rev_idx = [i for i in range(seq_len-1, -1, -1)]
                rev_idx = torch.LongTensor(rev_idx).to(seqs.get_device())
                seqs_reversed[i, :seq_len, :] = torch.index_select(seqs[i, :seq_len, :],0,rev_idx)
            return seqs_reversed

        @Trainer.TrainerDecorators.training_wrapper
        def train(self, batch_data):
            # Unpack batch data
            (data, seq_lens), _, _ = batch_data

            # Move data to selected device 
            data = data.to(self.device)
            data_reversed = self.reverse_sequences(data, seq_lens).to(self.device)

            # Forwards pass
            outputs, _, _ = self.parallel_forward(data, seq_lens, in_batch_first=True, out_batch_first=True)
            mask = self.masks(outputs.size(), seq_lens)
            loss = self.criterion(outputs[mask], data_reversed[mask])

            return loss

    class Composite(Trainer):
        def __init__(self, model, training_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title, test_data=None):
            super().__init__(model, training_data, None, test_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title, mixed_precision=False)
            # Strings to be used for file and console outputs
            self.title = 'Composite'
            self.cache_filename = 'pretrained_model'

        def masks(self, op_size, seq_lens):
            mask = torch.zeros(op_size, dtype=torch.bool)
            for index, length in enumerate(seq_lens):
                mask[index, :length, :] = True
            return mask

        @Trainer.TrainerDecorators.training_wrapper
        def train(self, batch_data):
            # Unpack batch data
            (data, seq_lens), _, _ = batch_data

            # Move data to selected device 
            data = data.to(self.device)

            # Forwards pass
            if max(seq_lens) <= 1:
                mask = self.masks(data.size(), seq_lens)
                loss = self.criterion(data[mask], data[mask])
            else:
                outputs, _, _ = self.parallel_forward(data, seq_lens, in_batch_first=True, out_batch_first=True)
                mask = self.masks(outputs.size(), seq_lens)
                loss = self.criterion(outputs[mask], data[mask])

            return loss

    class Identity(Trainer):
        def __init__(self, model, training_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title, test_data=None):
            super().__init__(model, training_data, None, test_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title, mixed_precision=True)
            # Strings to be used for file and console outputs
            self.cache_filename = 'pretrained_model'

        def masks(self, op_size, seq_lens):
            mask = torch.zeros(op_size, dtype=torch.bool)
            for index, length in enumerate(seq_lens):
                mask[index, :length,:] = True
            return mask

        @Trainer.TrainerDecorators.training_wrapper
        def train(self, batch_data):
            # Unpack batch data
            (data, seq_lens), _, _ = batch_data

            # Move data to selected device 
            data = data.to(self.device)

            # Forwards pass
            outputs, _, _ = self.parallel_forward(data, seq_lens, in_batch_first=True, out_batch_first=True)
            mask = self.masks(outputs.size(), seq_lens)
            loss = self.criterion(outputs[mask], data[mask])

            return loss

    class ObscureFeature(Trainer):
        def __init__(self, model, training_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title, test_data=None):
            super().__init__(model, training_data, None, test_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title)
            # Strings to be used for file and console outputs
            self.cache_filename = 'pretrained_model'

        def obscure(self, data, i_start, i_end):
            batch_size, max_seq_length, input_size = data.size()
            assert i_end >= i_start
            assert i_end < input_size
            mask = torch.zeros(data.size(), dtype=torch.bool)
            masked_data = data
            #masked_data[:, :, i_start:i_end] = -torch.ones(batch_size, max_seq_length, i_end-i_start)
            masked_data[:, :, i_start:i_end] = torch.zeros(batch_size, max_seq_length, i_end-i_start)
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
            #masked_data, mask = self.obscure_random(data, 6, 9, 0.3)
            masked_data, mask = self.obscure(data, 6, 7)

            # Move data to selected device 
            masked_data = data.to(self.device)
            data = data.to(self.device)
            mask = mask.to(self.device)

            # Forwards pass
            outputs, _, _ = self.parallel_forward(masked_data, seq_lens, in_batch_first=True, out_batch_first=True)
            loss = self.criterion(outputs[mask], data[mask])

            return loss

    class MaskPacket(Trainer):
        def __init__(self, model, training_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title, test_data=None):
            super().__init__(model, training_data, None, test_data, device, criterion, optimizer, epochs, val_epochs, stats, cache, json, writer, title, mixed_precision=True)
            # Strings to be used for file and console outputs
            self.title = 'MaskPacket'
            self.cache_filename = 'pretrained_model'

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
            outputs, _, _ = self.parallel_forward(masked_data, seq_lens, in_batch_first=True, out_batch_first=True)
            loss = self.criterion(outputs[mask], data[mask])

            return loss

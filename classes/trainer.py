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

class Trainer():
    def __init__(self, model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json):
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
        self._scaler = torch.cuda.amp.GradScaler() if stats.gpu else None
        self.json = json

    def evaluate(self):
        pass

    def train(self):
        pass

    def validate(self):
        pass

class Supervised(Trainer):
    def __init__(self, model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json):
        super().__init__(model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json)

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

    def train(self):
        # Set model into training mode
        self.model.train()

        # Define monitor to track time and avg. loss over time
        mon = Monitor(self.epochs * self.n_batches, 1000, agr=Monitor.Aggregate.AVG, title='Training', json_dir=self.json)

        # Train model if no cache file exists or the train flag is set, otherwise load cached model
        chache_file_name = self.cache.cache_dir + self.cache.key_prefix + '_trained_model.sdc'
        if self.cache.disabled or not os.path.isfile(chache_file_name):

            # Train the model
            print('Training model...')
            for epoch in range(self.epochs):
                for (_, data), labels, _ in self.training_data: 

                    # Move data to selected device 
                    data = data.to(self.device)
                    labels = labels.to(self.device)
                         
                    # Clear gradients
                    self.optimizer.zero_grad()

                    # If GPU training and GPU > Nvidia 2000, fp16 should be enabled
                    if not self._scaler == None:
                        with torch.cuda.amp.autocast():
                            # Forwards pass
                            outputs, seq_lens = self.model(data)
                            mask = self.mask(outputs.size(), seq_lens)
                            op_view = outputs[mask].view(-1)
                            lab_view = labels[mask].view(-1)
                            loss = self.criterion(op_view, lab_view)

                        # Backward and optimize
                        self._scaler.scale(loss).backward()
                        self._scaler.step(self.optimizer)
                        self._scaler.update()
                    else:
                        # Forwards pass
                        outputs, seq_lens = self.model(data)
                        mask = self.mask(outputs.size(), seq_lens)
                        op_view = outputs[mask].view(-1)
                        lab_view = labels[mask].view(-1)
                        loss = self.criterion(op_view, lab_view)

                        # Backward and optimize
                        loss.backward()
                        self.optimizer.step()
                    
                    # Calculate time left and save avg. loss of last interval
                    if mon(loss.item()):
                        time_left_h, time_left_m = mon.time_left
                        print (f'Supervised, Epoch [{epoch+1}/{self.epochs}], Step [{mon.iter}/{self.epochs*self.n_batches}], Moving avg. Loss: {mon.measurements[-1]:.4f}, Time left: {time_left_h}h {time_left_m}m')

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
            self.model.eval()
            print('done')

            if self.cache.exists('stats'):
                # Load statistics object
                self.stats = self.cache.load('stats', msg='Loading statistics object')

    def validate(self):

        # Define monitor to track time and avg. loss over time
        n_val_samples = len(self.validation_data)
        mon = Monitor(n_val_samples, 1000, agr=Monitor.Aggregate.NONE, title='Validation', json_dir=self.json)

        if self.cache.exists('stats_completed'):
            self.stats = self.cache.load('stats_completed', msg='Loading cached validation results')
        else:
            # Validate model
            print('Validating model...')
            with torch.no_grad():
                n_correct = n_samples = n_false_positive = n_false_negative = 0
                for (_, data), labels, categories in self.validation_data:

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
                    target = labels[:, 0, :].squeeze()
                    categories = categories[:, 0, :].squeeze()  
                    n_samples += labels.size(0)
                    n_correct += (predicted == target).sum().item()
                    n_false_negative += (predicted < target).sum().item()
                    n_false_positive += (predicted > target).sum().item()
                    self.stats.class_stats.add((predicted == target), categories)
                    assert n_correct == n_samples - n_false_negative - n_false_positive

                    if mon(0):
                        time_left_h, time_left_m = mon.time_left
                        print (f'Validation [{mon.iter}/{n_val_samples}], Time left: {time_left_h}h {time_left_m}m')

                # Save and cache validation results
                self.stats.n_samples = n_samples
                self.stats.n_false_negative = n_false_negative
                self.stats.n_false_positive = n_false_positive
                self.cache.save('stats_completed', self.stats, msg='Storing validation results')
        
        print(f'Accuracy with validation size {self.stats.val_percent}% of data samples: Accuracy {(self.stats.accuracy * 100.0):.3f}%, False p.: {self.stats.false_positive:.3f}%, False n.: {self.stats.false_negative:.3f}%')

    def evaluate(self):
        self.stats.save_stats()
        self.stats.save_losses()
        self.stats.plot_losses()

class PredictPacket(Trainer):
    def __init__(self, model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json):
        super().__init__(model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json)

    def masks(self, op_size, seq_lens):
        logit_mask = torch.zeros(op_size, dtype=torch.bool)
        target_mask = torch.zeros(op_size, dtype=torch.bool)
        for index, length in enumerate(seq_lens):
            target_mask[index, :length-1,:] = True
            logit_mask[index, 1:length,:] = True
        return logit_mask, target_mask

    def train(self):
        # Set model into training mode
        self.model.train()

        # Define monitor to track time and avg. loss over time
        mon = Monitor(self.epochs * self.n_batches, 1000, agr=Monitor.Aggregate.AVG, title='Pretraining', json_dir=self.json)

        # Train model if no cache file exists or the train flag is set, otherwise load cached model
        chache_file_name = self.cache.cache_dir + self.cache.key_prefix + '_pretrained_model.sdc'
        if self.cache.disabled or not os.path.isfile(chache_file_name):

            # Train the model
            print('Pretraining model (PacketPrediction)...')
            for epoch in range(self.epochs):
                for (_, data), _, _ in self.training_data: 

                    # Move data to selected device 
                    data_unpacked, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(data, batch_first=True)
                    data_unpacked = data_unpacked.to(self.device)
                    data = data.to(self.device)
                    seq_lens = seq_lens.to(self.device)

                    # Clear gradients
                    self.optimizer.zero_grad()
                    
                    # If GPU training and GPU > Nvidia 2000, fp16 should be enabled
                    if not self._scaler == None:
                        with torch.cuda.amp.autocast():
                            # Forwards pass
                            outputs, seq_lens = self.model(data)
                            logit_mask, target_mask = self.masks(outputs.size(), seq_lens)
                            loss = self.criterion(outputs[logit_mask], data_unpacked[target_mask])

                        # Backward and optimize
                        self._scaler.scale(loss).backward()
                        self._scaler.step(self.optimizer)
                        self._scaler.update()
                    else:
                        # Forwards pass
                        outputs, seq_lens = self.model(data)
                        logit_mask, target_mask = self.masks(outputs.size(), seq_lens)
                        loss = self.criterion(outputs[logit_mask], data_unpacked[target_mask])

                        # Backward and optimize
                        loss.backward()
                        self.optimizer.step()

                    # Calculate time left and save avg. loss of last interval
                    if mon(loss.item()):
                        time_left_h, time_left_m = mon.time_left
                        print (f'Pretraining, Epoch [{epoch+1}/{self.epochs}], Step [{mon.iter}/{self.epochs*self.n_batches}], Moving avg. Loss: {mon.measurements[-1]:.4f}, Time left: {time_left_h}h {time_left_m}m')

            # Get stats
            self.stats.add_monitor(mon)
            self.stats.losses = mon.measurements

            # Store trained model
            print('(Cache) Storing pretrained model to cache', end='')
            torch.save(self.model.state_dict(), chache_file_name)
            print('...done')

            # Store statistics object
            self.cache.save('pretraining_stats', self.stats, msg='Storing pretraining statistics to cache')
        else:
            # Load cached model
            print('(Cache) Loading pretraining model...',end='')
            self.model.load_state_dict(torch.load(chache_file_name))
            self.model.eval()
            print('done')

            # Load statistics object
            self.stats = self.cache.load('pretraining_stats', msg='Loading pretraining statistics object')

    def validate(self):
        print('Though shalt not validate an only pretrained model')

class ObscureFeature(Trainer):
    def __init__(self, model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json):
        super().__init__(model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json)

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

    def train(self):
        # Set model into training mode
        self.model.train()

        # Define monitor to track time and avg. loss over time
        mon = Monitor(self.epochs * self.n_batches, 1000, agr=Monitor.Aggregate.AVG, title='Pretraining', json_dir=self.json)

        # Train model if no cache file exists or the train flag is set, otherwise load cached model
        chache_file_name = self.cache.cache_dir + self.cache.key_prefix + '_pretrained_model.sdc'
        if self.cache.disabled or not os.path.isfile(chache_file_name):

            # Train the model
            print('Pretraining model (ObscureFeature)...')
            for epoch in range(self.epochs):
                for (_, data), _, _ in self.training_data: 

                    # Unpack data
                    data_unpacked, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(data, batch_first=True)

                    # Obscure features
                    masked_data = self.obscure(data_unpacked, 6, 9)

                    # Pack data
                    masked_data = torch.nn.utils.rnn.pack_padded_sequence(masked_data, seq_lens, batch_first=True, enforce_sorted=False)

                    # Move data to selected device 
                    masked_data = masked_data.to(self.device)
                    data_unpacked = data_unpacked.to(self.device)

                    # Clear gradients
                    self.optimizer.zero_grad()
                    
                    # If GPU training and GPU > Nvidia 2000, fp16 should be enabled
                    if not self._scaler == None:
                        with torch.cuda.amp.autocast():
                            # Forwards pass
                            outputs, _ = self.model(masked_data)
                            op_mask = self.mask(outputs.size(), seq_lens)
                            loss = self.criterion(outputs[op_mask], data_unpacked[op_mask])

                        # Backward and optimize
                        self._scaler.scale(loss).backward()
                        self._scaler.step(self.optimizer)
                        self._scaler.update()
                    else:
                        # Forwards pass
                        outputs, _ = self.model(masked_data)
                        op_mask = self.mask(outputs.size(), seq_lens)
                        loss = self.criterion(outputs[op_mask], data_unpacked[op_mask])

                        # Backward and optimize
                        loss.backward()
                        self.optimizer.step()

                    # Calculate time left and save avg. loss of last interval
                    if mon(loss.item()):
                        time_left_h, time_left_m = mon.time_left
                        print (f'Pretraining, Epoch [{epoch+1}/{self.epochs}], Step [{mon.iter}/{self.epochs*self.n_batches}], Moving avg. Loss: {mon.measurements[-1]:.4f}, Time left: {time_left_h}h {time_left_m}m')

            # Get stats
            self.stats.add_monitor(mon)
            self.stats.losses = mon.measurements

            # Store trained model
            print('(Cache) Storing pretrained model to cache', end='')
            torch.save(self.model.state_dict(), chache_file_name)
            print('...done')

            # Store statistics object
            self.cache.save('pretraining_stats', self.stats, msg='Storing pretraining statistics to cache')
        else:
            # Load cached model
            print('(Cache) Loading pretraining model...',end='')
            self.model.load_state_dict(torch.load(chache_file_name))
            self.model.eval()
            print('done')

            # Load statistics object
            self.stats = self.cache.load('pretraining_stats', msg='Loading pretraining statistics object')

    def validate(self):
        print('Though shalt not validate an only pretrained model')
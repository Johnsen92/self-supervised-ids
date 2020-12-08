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
    def __init__(self, model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache):
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

    def train(self):
        pass

    def validate(self):
        pass

class Supvervised(Trainer):
    def __init__(self, model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache):
        super().__init__(model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache)
        
    def train(self):
        # Set model into training mode
        self.model.train()

        # Define monitor to track time and avg. loss over time
        mon = Monitor(self.epochs * self.n_batches, 1000, agr=Monitor.Aggregate.NONE)

        # Train model if no cache file exists or the train flag is set, otherwise load cached model
        chache_file_name = self.cache.cache_dir + self.cache.key_prefix + '_trained_model.sdc'
        if self.cache.disabled or not os.path.isfile(chache_file_name):

            # Train the model
            print('Training model...')
            for epoch in range(self.epochs):
                for data, labels, categories in self.training_data: 

                    # Move data to selected device 
                    data = data.to(self.device)
                    labels = labels.to(self.device)
                    categories = categories.to(self.device)

                    # Forward pass
                    outputs = self.model(data, pretraining=False)
                    op_view = outputs.view(-1, 2)
                    lab_view = labels.view(-1)
                    loss = self.criterion(op_view, lab_view)
                    
                    # Backward and optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    # Calculate time left and save avg. loss of last interval
                    if mon(loss.item()):
                        time_left_h, time_left_m = mon.time_left
                        print (f'Epoch [{epoch+1}/{self.epochs}], Step [{mon.iter}/{self.epochs*self.n_batches}], Moving avg. Loss: {mon.measurements[-1]:.4f}, Time left: {time_left_h}h {time_left_m}m')

            # Get stats
            self.stats.training_time = mon.duration_s
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

            # Load statistics object
            self.stats = self.cache.load('stats', msg='Loading statistics object')

    def validate(self):
        # Validate model
        print('Validating model...')
        with torch.no_grad():
            n_correct = n_samples = n_false_positive = n_false_negative = 0
            for data, labels, categories in self.validation_data:

                # Move data to selected device 
                data = data.to(self.device)
                labels = labels.to(self.device)
                categories = categories.to(self.device)

                # Forward pass
                outputs = self.model(data)

                # Max returns (value ,index)
                _, predicted = torch.max(outputs.data[:,-1,:], 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels[:, 0]).sum().item()
                n_false_negative += (predicted < labels[:, 0]).sum().item()
                n_false_positive += (predicted > labels[:, 0]).sum().item()
                assert n_correct == n_samples - n_false_negative - n_false_positive

            # Calculate statistics
            false_p = 100.0 * n_false_positive/(n_samples - n_correct)
            false_n = 100.0 * n_false_negative/(n_samples - n_correct)

            # Save and cache statistics
            self.stats.n_false_negative = n_false_negative
            self.stats.n_false_positive = n_false_positive
            print(f'Accuracy with validation size {(100 - self.stats.train_percent)}% of data samples: {(stats.accuracy*100):.3f}%, False p.: {false_p:.3f}%, False n.: {false_n:.3f}%')
            self.stats.save_stats()
            self.stats.save_losses()
            self.stats.plot_losses()

class PredictPacket(Trainer):
    def __init__(self, model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache):
        super().__init__(model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache)

    def train(self):
        # Set model into training mode
        self.model.train()

        # Define monitor to track time and avg. loss over time
        mon = Monitor(self.epochs * self.n_batches, 1000, agr=Monitor.Aggregate.NONE)

        # Train model if no cache file exists or the train flag is set, otherwise load cached model
        chache_file_name = self.cache.cache_dir + self.cache.key_prefix + '_trained_model.sdc'
        if self.cache.disabled or not os.path.isfile(chache_file_name):

            # Train the model
            print('Training model...')
            for epoch in range(self.epochs):
                for data, _, _ in self.training_data: 

                    # Move data to selected device 
                    data = data.to(self.device)

                    # Forward pass
                    outputs = self.model(data, pretraining=True)
                    loss = self.criterion(outputs[:, :-1, :], data[:, 1:, :])
                    
                    # Backward and optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    # Calculate time left and save avg. loss of last interval
                    if mon(loss.item()):
                        time_left_h, time_left_m = mon.time_left
                        print (f'Epoch [{epoch+1}/{self.epochs}], Step [{mon.iter}/{self.epochs*self.n_batches}], Moving avg. Loss: {mon.measurements[-1]:.4f}, Time left: {time_left_h}h {time_left_m}m')

            # Get stats
            self.stats.training_time = mon.duration_s
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
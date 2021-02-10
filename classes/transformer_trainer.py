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

class Interpolation(Trainer):
    def __init__(self, model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json):
        super().__init__(model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json)

    def train(self):

        # Tensorboard to get nice loss plot
        writer = SummaryWriter("runs/loss_plot")
        step = 0

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
                for (_, data), _, _ in self.training_data: 

                    data_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(data)

                    # Get input and targets and get to cuda
                    data = data_unpacked.to(self.device)
                    labels = labels.to(self.device)

                    # Select every even idx of data as src and every odd idx as target
                    seq_len = data_unpacked.size()[0]
                    trg_idx = torch.arange(1, seq_len, step=2)
                    src_idx = trg_idx - 1
                    src_data = data[src_idx,:,:]
                    trg_data = data[trg_idx,:,:]
                    
                    # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
                    # doesn't take input in that form. For example if we have MNIST we want to have
                    # output to be: (N, 10) and targets just (N). Here we can view it in a similar
                    # way that we have output_words * batch_size that we want to send in into
                    # our cost function, so we need to do some reshapin.
                    # Let's also remove the start token while we're at it
                    with torch.cuda.amp.autocast():
                        # Forward prop
                        out = self.model(src_data, trg_data)

                        # Create mask for non-padded items only
                        #mask = self.mask(out.size(), seq_lens).to(self.device)
                        out = out.view(-1)
                        trg = trg_data.view(-1)
                        #labels = labels[mask].view(-1)
                        self.optimizer.zero_grad()
                        loss = self.criterion(out, trg)

                    # Backward and optimize
                    self._scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                    self._scaler.step(self.optimizer)
                    self._scaler.update()

                    # Back prop
                    #loss.backward()
                    # Clip to avoid exploding gradient issues, makes sure grads are
                    # within a healthy range
                    #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

                    # Gradient descent step
                    #self.optimizer.step()

                    # plot to tensorboard
                    writer.add_scalar("Training loss", loss, global_step=step)
                    step += 1

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

class Supervised(Trainer):
    def __init__(self, model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json):
        super().__init__(model, training_data, validation_data, device, criterion, optimizer, epochs, stats, cache, json)

    def src_mask(self, size, seq_lens):
        mask = torch.ones(size[:2], dtype=torch.bool).transpose(0, 1)
        for index, length in enumerate(seq_lens):
            mask[index, :length] = False
        return mask

    def trg_mask(self, size, seq_lens):
        mask = torch.zeros(size, dtype=torch.bool)
        for index, length in enumerate(seq_lens):
            mask[:length-1, index, :] = True
        return mask

    def logit_mask(self, size, seq_lens):
        mask = torch.zeros(size, dtype=torch.bool)
        for index, length in enumerate(seq_lens):
            mask[:length-1, index, :] = True
        return mask

    def logits(self, output, seq_lens):
        logits = torch.zeros(output.size()[1], dtype=torch.float)
        for index, length in enumerate(seq_lens):
            logits[index] = torch.sum(output[:length-1, index, :])/length
        return logits

    def train(self):

        # Tensorboard to get nice loss plot
        writer = SummaryWriter("runs/loss_plot")
        step = 0

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

                    data_unpacked, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(data)

                    # Get input and targets and get to cuda
                    data_unpacked = data_unpacked.to(self.device)
                    labels = labels.to(self.device)

                    # Create mask for non-padded items only
                    src_mask = self.src_mask(data_unpacked.size(), seq_lens).to(self.device)
                    trg_mask = self.trg_mask(labels.size(), seq_lens).to(self.device)
                    
                    # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
                    # doesn't take input in that form. For example if we have MNIST we want to have
                    # output to be: (N, 10) and targets just (N). Here we can view it in a similar
                    # way that we have output_words * batch_size that we want to send in into
                    # our cost function, so we need to do some reshapin.
                    # Let's also remove the start token while we're at it
                    with torch.cuda.amp.autocast():

                        # Forward prop
                        out = self.model(data_unpacked, src_mask)
                        out = out[trg_mask].view(-1)
                        labels = labels[trg_mask].view(-1)
                        self.optimizer.zero_grad()
                        loss = self.criterion(out, labels)

                    # Backward and optimize
                    self._scaler.scale(loss).backward()
                    #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                    self._scaler.step(self.optimizer)
                    self._scaler.update()

                    # Back prop
                    #loss.backward()
                    # Clip to avoid exploding gradient issues, makes sure grads are
                    # within a healthy range
                    #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

                    # Gradient descent step
                    #self.optimizer.step()

                    # plot to tensorboard
                    writer.add_scalar("Training loss", loss, global_step=step)
                    step += 1

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

        self.model.eval()

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
                    data_unpacked, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(data)
                    data_unpacked = data_unpacked.to(self.device)
                    labels = labels.to(self.device)
                    categories = categories.to(self.device)

                    # Forward pass
                    src_mask = self.src_mask(data_unpacked.size(), seq_lens).to(self.device)
                    outputs = self.model(data_unpacked, src_mask)
                    #logit_mask = self.logit_mask(outputs.size(), seq_lens)
                    logits = self.logits(outputs, seq_lens).to(self.device)

                    # Max returns (value ,index)
                    #sigmoided_output = torch.sigmoid(outputs.data[logit_mask].detach())
                    sigmoided_output = torch.sigmoid(logits)
                    predicted = torch.round(sigmoided_output)
                    target = labels[0, :, :].squeeze()
                    categories = categories[0, :, :].squeeze()  
                    n_samples += labels.size(1)
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

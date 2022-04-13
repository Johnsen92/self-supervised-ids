import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import math
from enum import Enum
from datetime import datetime
from timeit import default_timer as timer
from datetime import timedelta
import json
import os
import errno
from sklearn.inspection import plot_partial_dependence
from collections import Counter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import re
from operator import itemgetter
import seaborn as sns
import pandas as pd
import torch


def formatTime(time_s):
    time_h = time_s // 3600
    time_m = math.floor((float(time_s) / 3600.0 - time_h) * 60.0)
    return int(time_h), int(time_m)

class Monitor():
    class Aggregate(Enum):
        NONE = 0,
        SUM = 1,
        AVG = 2

    index = 0

    def __init__(self, iterations, n_measurements=1000, agr=Aggregate.NONE, json_dir=None, title=None):
        self.iterations = iterations
        self.n_samples = n_measurements
        self.agr = agr
        self._percent_interval = iterations // 100 if iterations // 100 > 0 else 1
        self._interval = iterations // n_measurements if iterations // n_measurements > 0 else 1
        self._seq = []
        self._agr_seq = [] 
        self._i = 0
        self._start_time = None
        self._end_time = None  
        self._progress = 0
        self._time_left_s = -1
        if not json_dir == None:
            self._json_dir = json_dir if json_dir[-1] == '/' else json_dir+'/'

        if title == None:
            self.title = "Monitor #" + str(Monitor.index)
        else:
            self.title = title
        Monitor.index += 1

        if json_dir == None:
            self._json = False
        else:
            self._json = True
            self._export_progress()

    def _export_progress(self):
        assert self._json   
        prog_dict = {}
        prog_dict['title'] = self.title
        prog_dict['progress'] = self._progress
        prog_dict['time_left_h'], prog_dict['time_left_m'] = self.time_left
        with open(self._json_dir + 'progress.json', 'w') as f:
            f.write(json.dumps(prog_dict))

    def __call__(self, val):
        # Check if first or last iteration and set start-/endtime
        if self._i == 0:
            self._start_time = timer()
        elif self._i == self.iterations-1:
            self._end_time = timer()

        # Update iteration counter
        self._i += 1

        # Update progress in percent for every percent of training done
        if (self._i-1) % self._percent_interval == 0:
            self._progress += 1
            # If json is true, export progress
            if self._json:
                self._export_progress()

        self._seq.append(val)
        if (self._i-1) % self._interval == 0:

            # Calculate expected time left
            time_passed = timer() - self._start_time
            time_per_iteration = float(time_passed) / float(self._i)
            self._time_left_s = int(time_per_iteration * float(self.iterations - self._i))
            
            # Calculate aggregate
            if self.agr == self.Aggregate.NONE:
                self._agr_seq.append(self._seq[-1])
            elif self.agr == self.Aggregate.SUM:
                self._agr_seq.append(sum(self._seq))
            elif self.agr == self.Aggregate.AVG:
                self._agr_seq.append(sum(self._seq)/len(self._seq))
            else:
                print(f'Invalid enum value: {self.agr}')
            self._seq = []
            return True
        else:
            return False

    @property
    def time_left(self):
        if self._time_left_s >= 0:
            return formatTime(self._time_left_s)
        # If called before first call of __call__ function, return 0
        else:
            return formatTime(0)

    @property
    def time_passed(self):
        if self._start_time == None:
            print('Start time not set, setting it to now')
            self._start_time = timer()
        return formatTime(timer() - self._start_time)
        
    @property
    def measurements(self):
        return self._agr_seq

    @property
    def iter(self):
        return self._i

    @property
    def duration_s(self):
        if self._start_time == None:
            print('Start time not set, setting it to now')
            self._start_time = timer()
        if self._end_time == None:
            print('End time not set, setting it to now')
            self._end_time = timer()
        return self._end_time - self._start_time

class ClassStats():
    def __init__(self, mapping, stats_dir='./', benign=10):
        self.stats_dir = stats_dir
        self.make_stats_dir()
        self.benign = benign
        self.mapping = mapping
        self.reverse_mapping = { val: key for key, val in mapping.items() }
        self.number = { c : 0 for c in mapping.values() }
        self.right = { c : 0 for c in mapping.values() }
        
    def reset(self):
        self.number = { c : 0 for c in self.mapping.values() }
        self.right = { c : 0 for c in self.mapping.values() }

    @property
    def labels(self):
        return self.mapping.keys()

    @property
    def classes(self):
        return self.mapping.values()

    def add_batch(self, results, classes):
        for index, c in enumerate(classes):
            c_val = c.item()
            self.number[c_val] += 1
            if results[index]:
                self.right[c_val] += 1

    def save_stats(self, proxy_task, file_name=''):
        now = datetime.now().strftime('%d%m%Y_%H-%M-%S')
        stats_file_name = file_name if file_name else f'class_stats_{now}.csv'
        with open(os.path.join(self.stats_dir, stats_file_name), 'w') as f:
            f.write(f'Class, #, Occurance, Right, {proxy_task}\n')
            for k,v in self.mapping.items():
                f.write(f'{k}, {v}, {self.number[v]}, {self.right[v]}, {self.accuracy_per_category[v]*100.0:.3f}%\n')
            f.write('\n')
            f.write(f'Benign, {self.benign}, {self.number[self.benign]}, {self.right[self.benign]}, {self.accuracy_benign*100.0:.3f}%\n')
            f.write(f'Attack, !{self.benign}, {self.n_attack}, {self.n_right_attack}, {self.accuracy_attack*100.0:.3f}%\n')
            f.write(f'Overall, ALL, {self.n_samples}, {self.n_right}, {self.accuracy*100.0:.3f}%\n')

    @property
    def accuracy_per_category(self):
        return {v: (float(self.right[v]) / float(self.number[v]) if self.number[v] != 0 else 1.0) for _,v in self.mapping.items()}

    @property
    def n_attack(self):
        return self.n_samples - self.number[self.benign]

    @property
    def n_right_attack(self):
        return self.n_right - self.right[self.benign]

    @property
    def accuracy_benign(self):
        return float(self.right[self.benign]) / float(self.number[self.benign]) if not self.number[self.benign] == 0 else 1.0

    @property
    def accuracy_attack(self):
        return float(self.n_right_attack) / float(self.n_attack) if not self.n_attack == 0 else 1.0

    @property
    def benign_rate(self):
        return float(self.number[self.benign]) / float(self.n_samples)

    @property
    def attack_rate(self):
        return float(self.n_attack) / float(self.n_samples)

    @property
    def n_samples(self):
        return sum(self.number.values())

    @property
    def n_false(self):
        return self.n_samples - self.n_right
    
    @property
    def n_right(self):
        return sum(self.right.values())

    @property
    def accuracy(self):
        return float(self.n_right) / float(self.n_samples)

    def make_stats_dir(self):
        if not os.path.exists(os.path.dirname(self.stats_dir)):
            try:
                os.makedirs(os.path.dirname(self.stats_dir))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    def set_stats_dir(self, stats_dir):
        self.stats_dir = stats_dir if stats_dir[-1] == '/' else stats_dir + '/'
        self.make_stats_dir()

class Epoch():
    def __init__(self, epoch, class_stats, training_time, training_loss = 0, validation_loss = 0):
        self.epoch = epoch
        self.n_true_positive = 0
        self.n_samples_counted = 0
        self.n_true_positive = 0
        self.n_true_negative = 0
        self.n_false_positive = 0
        self.n_false_negative = 0
        self.training_loss = training_loss
        self.validation_loss = validation_loss
        self.training_time = training_time
        self.class_stats = class_stats

    def add_batch(self, predicted, target, categories):
        self.n_samples_counted += target.size()[0]
        self.n_true_negative += torch.logical_and((predicted == target),(target == 0)).sum().item()
        self.n_true_positive += torch.logical_and((predicted == target),(target == 1)).sum().item()
        assert torch.logical_and((predicted == target),(target == 0)).sum().item() + torch.logical_and((predicted == target),(target == 1)).sum().item() == (predicted == target).sum().item()
        self.n_false_negative += (predicted < target).sum().item()
        self.n_false_positive += (predicted > target).sum().item()
        self.class_stats.add_batch((predicted == target), categories)

    @property
    def accuracy(self):
        if self.n_samples == 0:
            return 1.0
        else:
            return float(self.n_true) / float(self.n_samples)

    @property
    def error_rate(self):
        if self.n_samples == 0:
            return 1.0
        else:
            return float(self.n_false) / float(self.n_samples)

    @property
    def detection_rate(self):
        if self.n_true_positive + self.n_false_negative == 0:
            return 1.0
        else:
            return float(self.n_true_positive) / float(self.n_true_positive + self.n_false_negative)

    @property
    def precision(self):
        if self.n_true_positive + self.n_false_positive == 0:
            return 1.0
        else:
            return float(self.n_true_positive) / (float(self.n_true_positive + self.n_false_positive))

    @property
    def recall(self):
        return self.detection_rate

    @property
    def f1_measure(self):
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def false_alarm_rate(self):
        return self.false_positive_rate

    @property
    def missed_alarm_rate(self):
        return self.false_negative_rate

    @property
    def specificity(self):
        if self.n_true_negative + self.n_false_positive == 0:
            return 1.0
        else:
            return float(self.n_true_negative) / (float(self.n_true_negative + self.n_false_positive))

    @property
    def false_positive_rate(self):
        if self.n_false_positive + self.n_true_positive == 0:
            return 0.0
        else:
            return float(self.n_false_positive) / (float(self.n_false_positive + self.n_true_positive))

    @property
    def false_negative_rate(self):
        if self.n_false_negative + self.n_true_positive == 0:
            return 0.0
        else:
            return float(self.n_false_negative) / (float(self.n_false_negative + self.n_true_positive))

    @property
    def n_true(self):
        return self.n_true_negative + self.n_true_positive

    @property
    def n_false(self):
        return self.n_false_negative + self.n_false_positive

    @property
    def n_samples(self):
        assert self.n_true + self.n_false == self.n_samples_counted
        return self.n_true + self.n_false

class Stats():
    index = 0

    def __init__(
        self,
        train_percent, 
        val_percent, 
        n_epochs,
        batch_size, 
        learning_rate, 
        category_mapping,
        benign,
        model_parameters={},
        n_epochs_pretraining=0, 
        pretrain_percent=0, 
        proxy_task=None, 
        title=None, 
        random_seed=None, 
        subset=False,
        stats_dir='./'
    ):
        self.stats_dir = stats_dir if stats_dir[-1] == '/' else stats_dir + '/'
        self.make_stats_dir()
        self.train_percent = train_percent
        self.pretrain_percent = pretrain_percent
        self.proxy_task = proxy_task
        self.val_percent = val_percent
        self.n_epochs = n_epochs
        self.n_epochs_pretraining = n_epochs_pretraining
        self.batch_size = batch_size
        self.learning_rate = learning_rate  
        self.losses = []
        self.mapping = category_mapping
        self.benign = benign
        self.model_parameters = model_parameters
        self.random_seed = random_seed
        self.epochs = []
        self.subset = subset

        if title == None:
            self.title = "Statistics #" + str(Stats.index)
        else:
            self.title = title
        Stats.index += 1

    def __str__(self): 
        return f'bs{self.batch_size}_ep{self.n_epochs}_tp{self.train_percent}_lr{str(self.learning_rate).replace(".", "")}'

    def make_stats_dir(self):
        if not os.path.exists(os.path.dirname(self.stats_dir)):
            try:
                os.makedirs(os.path.dirname(self.stats_dir))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    def set_stats_dir(self, stats_dir):
        self.stats_dir = stats_dir if stats_dir[-1] == '/' else stats_dir + '/'
        for e in self.epochs:
            e.class_stats.set_stats_dir(stats_dir)
        self.make_stats_dir()

    def set_category_mapping(self, mapping):
        self.mapping = mapping
        self.class_stats.mapping = mapping
        self.class_stats.reverse_mapping = { val: key for key, val in mapping.items() }

    def plot_stats(self):
        pass

    def save_losses(self):
        assert not self.losses == None
        now = datetime.now().strftime('%d%m%Y_%H-%M-%S')
        print('Save loss progression...', end='')
        with open(self.stats_dir + 'training_losses_' + now + '.csv', 'w') as f:
            for epoch, loss in self.training_losses:
                f.write(f'{epoch}, {loss:.6f}\n')

        with open(self.stats_dir + 'validation_losses_' + now + '.csv', 'w') as f:
            for epoch, loss in self.validation_losses:
                f.write(f'{epoch}, {loss:.6f}\n')
        print('done.')

    def save_stats(self, file_name=''):
        now = datetime.now().strftime('%d%m%Y_%H-%M-%S')
        stats_file_name = file_name if file_name else f'stats_{now}.csv'
        print('Save statistics...', end='')
        with open(os.path.join(self.stats_dir, stats_file_name), 'w') as f:
            f.write(f'Proxy task, {self.proxy_task if self.pretrain_percent > 0 else "NONE"}\n')
            f.write(f'Epochs Supervised, {self.n_epochs}\n')
            f.write(f'Training percentage, {(self.train_percent / 10.0):.2f} %\n')
            f.write(f'Specialized subset, {self.subset}\n')
            f.write(f'\nTraining metrics,\n')     
            f.write(f'Best epoch, {self.best_epoch.epoch}\n')
            f.write(f'Time to best epoch, {self.best_epoch.training_time[0]}h {self.best_epoch.training_time[1]}m\n')
            f.write(f'\nPerformance metrics,\n')
            f.write(f'Accuracy, {self.accuracy*100.0:.3f} %\n')
            f.write(f'Detection rate, {self.detection_rate*100.0:.3f} %\n')
            f.write(f'Precision, {self.precision*100.0:.3f} %\n')
            f.write(f'Specificity, {self.specificity*100.0:.3f} %\n')
            f.write(f'F1-Measure, {self.f1_measure*100.0:.3f} %\n')
            f.write(f'False alarm rate, {self.false_alarm_rate*100.0:.3f} %\n')
            f.write(f'Missed alarm rate, {self.missed_alarm_rate*100.0:.3f} %\n')
        print('done.')
        if not self.class_stats == None:
            self.class_stats.save_stats(self.proxy_task, f'{file_name[:-4]}_class.csv')

    def save_stats_complete(self):
        now = datetime.now().strftime('%d%m%Y_%H-%M-%S')
        print('Save statistics...', end='')
        with open(self.stats_dir + 'stats_' + now + '.csv', 'w') as f:
            f.write(f'Hyperparameters,\n')
            f.write(f'Epochs Supervised, {self.n_epochs}\n')
            f.write(f'Epochs Pretraining, {0 if self.proxy_task is None else self.n_epochs_pretraining}\n')
            f.write(f'Batch size, {self.batch_size}\n')
            f.write(f'Proxy task, {self.proxy_task if self.pretrain_percent > 0 else "NONE"}\n')
            f.write(f'Pretraining percentage, {(self.pretrain_percent / 10.0):.2f} %\n')
            f.write(f'Training percentage, {(self.train_percent / 10.0):.2f} %\n')
            f.write(f'Validation percentage, {(self.val_percent / 10.0):.2f} %\n')
            f.write(f'Specialized subset, {self.subset}\n')
            f.write(f'Learning rate, {self.learning_rate}\n')
            f.write(f'Random Seed, {self.random_seed}\n')
            f.write(f'\nModelparameters,\n')
            if not self.model_parameters is None:
                for key, val in self.model_parameters.items():
                    f.write(f'{key}, {val}\n')
            f.write(f'\nTraining metrics,\n')     
            f.write(f'Best epoch, {self.best_epoch.epoch}\n')
            f.write(f'Time to best epoch, {self.best_epoch.training_time[0]}h {self.best_epoch.training_time[1]}m\n')
            f.write(f'\nPerformance metrics,\n')
            #f.write(f'Accuracy, {self.best_epoch[1]:.3f} %\n')
            f.write(f'Accuracy, {self.accuracy*100.0:.3f} %\n')
            f.write(f'False alarm rate, {self.false_alarm_rate*100.0:.3f} %\n')
            f.write(f'Missed alarm rate, {self.missed_alarm_rate*100.0:.3f} %\n')
            f.write(f'Detection rate, {self.detection_rate*100.0:.3f} %\n')
            f.write(f'Precision, {self.precision*100.0:.3f} %\n')
            f.write(f'Specificity, {self.specificity*100.0:.3f} %\n')
            f.write(f'Recall, {self.recall*100.0:.3f} %\n')
            f.write(f'F1-Measure, {self.f1_measure*100.0:.3f} %\n')
        print('done.')
        if not self.class_stats == None:
            self.class_stats.save_stats()

    def plot_losses(self):
        pass

    def add_epoch(self, epoch):
        self.epochs.append(epoch)

    def new_epoch(self, epoch, training_time, training_loss):
        class_stats = ClassStats(
            stats_dir = self.stats_dir,
            mapping = self.mapping,
            benign = self.benign
        )  
        new_epoch = Epoch(
            epoch = epoch, 
            class_stats = class_stats, 
            training_time = training_time, 
            training_loss = training_loss
        )
        self.epochs.append(new_epoch)
        return new_epoch

    @property
    def validation_losses(self):
        return [(e.epoch, e.validation_loss) for e in self.epochs]

    @property
    def training_losses(self):
        return [(e.epoch, e.training_loss) for e in self.epochs]

    @property
    def last_epoch(self):
        return self.epochs[-1]

    def plot_pdp(self, X, Y, features=[0], category=0):
        pdp_plot = PDPlot(X, Y, self.mapping)
        pdp_plot.plot(features, category)

    @property
    def best_epoch(self):
        return max(self.epochs, key=lambda e: e.accuracy)

    @property
    def class_stats(self):
        return self.best_epoch.class_stats

    @property
    def n_true_positive(self):
        return self.best_epoch.n_true_positive

    @property
    def n_true_negative(self):
        return self.best_epoch.n_true_negative

    @property
    def n_false_negative(self):
        return self.best_epoch.n_false_negative

    @property
    def n_false_positive(self):
        return self.best_epoch.n_false_positive

    @property
    def n_samples_counted(self):
        return self.best_epoch.n_samples_counted

    @property
    def accuracy(self):
        return self.best_epoch.accuracy

    @property
    def error_rate(self):
        return self.best_epoch.error_rate

    @property
    def detection_rate(self):
        return self.best_epoch.detection_rate

    @property
    def precision(self):
        return self.best_epoch.precision

    @property
    def recall(self):
        return self.best_epoch.recall

    @property
    def f1_measure(self):
        return self.best_epoch.f1_measure
    
    @property
    def false_alarm_rate(self):
        return self.best_epoch.false_alarm_rate

    @property
    def missed_alarm_rate(self):
        return self.best_epoch.missed_alarm_rate

    @property
    def specificity(self):
        return self.best_epoch.specificity

    @property
    def false_positive_rate(self):
        return self.best_epoch.false_positive_rate

    @property
    def false_negative_rate(self):
        return self.best_epoch.false_negative_rate

    @property
    def n_true(self):
        return self.n_true_negative + self.n_true_positive

    @property
    def n_false(self):
        return self.n_false_negative + self.n_false_positive

    @property
    def n_samples(self):
        assert self.n_true + self.n_false == self.n_samples_counted
        return self.n_true + self.n_false

    @property
    def training_time_to_best_epoch(self):
        return self.best_epoch.training_time

class NeuronData():
    def __init__(self, id, config, title=None):
        self.id = id
        self.title = title
        self.config = config
        self.latest = {}
        self.means = {}

    @property
    def means_means(self):
        return {k:np.mean(v, axis=0) for k,v in self.means.items()}

    @property
    def latest_means(self):
        return {k:np.mean(v, axis=0) for k,v in self.latest.items()}

    @property
    def label(self):
        return re.search(r'\_xy(\w+)', self.id).group(1)

    def compare(self, nd):
        latest_class_diff = {k:0 for k in self.latest.keys()}
        means_class_diff = {k:0 for k in self.means.keys()}
        for k in self.latest.keys():
            latest_mask = np.ones(self.latest[k].shape, dtype=np.int) - np.round(abs(self.latest[k])).astype(np.int)
            means_mask = np.ones(self.means[k].shape, dtype=np.int) - np.round(abs(self.means[k])).astype(np.int)
            x_means = ma.array(self.means[k], mask=means_mask)
            y_means = ma.array(nd.means[k], mask=means_mask)
            x_latest = ma.array(self.latest[k], mask=latest_mask)
            y_latest = ma.array(nd.latest[k], mask=latest_mask)
            # Calculate mean difference. The x2 in the denominator is for normalizing the final value between 0 and 1. The value range is -1 - +1 so the max. difference is 2.
            latest_class_diff[k] = np.sum(abs(x_latest - y_latest))/(sum(np.round(abs(self.latest[k])).astype(np.int))*2)
            means_class_diff[k] = np.sum(abs(x_means - y_means))/(sum(np.round(abs(self.latest[k])).astype(np.int))*2)
            #print(latest_class_diff)
            #print(means_class_diff)

        avg_latest_diff = sum([v for v in latest_class_diff.values()])/len(latest_class_diff.values())
        avg_means_diff = sum([v for v in means_class_diff.values()])/len(means_class_diff.values())
        return avg_latest_diff, avg_means_diff, latest_class_diff, means_class_diff

class NeuronPlot():
    def __init__(self, config, mapping, neuron_data, plot_dir='plots/neurons/', use_titles=False, compare=False, base_name=None):
        if base_name == None:
            self.base_name = ''
        else:
            self.base_name = f'{base_name}_'
        self.mapping = mapping
        self.compare = compare
        self.reverse_mapping = {v: k for k, v in mapping.items()}
        self.neuron_data = neuron_data
        self.plot_dir = f'{plot_dir}{self.base_name}{self.label}/'
        self.plot_dir_latest = self.plot_dir + 'latest/'
        self.plot_dir_means = self.plot_dir + 'means/'
        self.use_titles = use_titles
        # If compare is set, two NeuronData rows are expected for the plot which shall be compared. The comparison is not kommutative
        if compare:
            assert len(self.neuron_data) == 2
            self.avg_latest_diff, self.avg_means_diff, self.latest_class_diff, self.means_class_diff = self.neuron_data[1].compare(self.neuron_data[0])
        os.makedirs(self.plot_dir_latest, exist_ok=True)
        os.makedirs(self.plot_dir_means, exist_ok=True)
        self.config = config
        self._cmap = sns.diverging_palette(240, 10, n=9)
    
    @property
    def label(self):
        assert len(self.neuron_data) > 0
        id_string = ''
        if self.compare:
            id_string += self.id(self.neuron_data[0]) + '_'
        else:
            for nd in self.neuron_data:
                id_string += self.id(nd) + '_'
        return id_string[:-1]

    def id(self, data):
        return re.search(r'\_xy(\w+)', data.id).group(1)

    # print comparison on plot

    def plot_means(self, category):
        fig, ax = plt.subplots(figsize=(25,len(self.neuron_data)))
        means_means = []
        labels = []
        for nd in self.neuron_data:
            if self.use_titles:
                labels.append(nd.title)
            else:
                labels.append(nd.label)
            means_means.append(nd.means[category])
            #latest_means.append(np.mean(nd.means[category], 0))

        means_data = np.vstack(means_means)
        data_latest = pd.DataFrame(data=means_data, index=labels)
        ax = sns.heatmap(data_latest, vmin=-1, vmax=1, cmap=self._cmap, linewidth=0.0001)
        label = f'Neurons - means - {self.reverse_mapping[category]}' + (f' - L1 difference {self.means_class_diff[category]:.2f}' if self.compare else '')
        ax.set_xlabel(label)
        file_name = self.plot_dir_means + f'{category}_{self.reverse_mapping[category].replace("/", "-").replace(":", "-")}'
        plt.savefig(file_name, bbox_inches = 'tight', pad_inches = 0.1)
        plt.clf()

    def plot_latest(self, category):
        fig, ax = plt.subplots(figsize=(25,len(self.neuron_data)))
        latest_means = []
        labels = []
        for nd in self.neuron_data:
            if self.use_titles:
                labels.append(nd.title)
            else:
                labels.append(nd.label)
            latest_means.append(nd.latest[category])
            #latest_means.append(np.mean(nd.latest[category], 0))

        means_data = np.vstack(latest_means)
        data_latest = pd.DataFrame(data=means_data, index=labels)
        ax = sns.heatmap(data_latest, vmin=-1, vmax=1, cmap=self._cmap, linewidth=0.0001)
        label = f'Neurons -  latest - {self.reverse_mapping[category]}' + f' - L1 difference {self.latest_class_diff[category]:.2f}' if self.compare else ''
        ax.set_xlabel(label)
        file_name = self.plot_dir_latest + f'{category}_{self.reverse_mapping[category].replace("/", "-").replace(":", "-")}'
        plt.savefig(file_name, bbox_inches = 'tight', pad_inches = 0.1)
        plt.clf()

    def plot_all(self):
        for category in self.config['categories']:
            self.plot_means(category)
            self.plot_latest(category)

class PDData():
    def __init__(self, id, config, title=None):
        self.id = id
        self.config = config
        self.title = title
        self.results = {}
        self.features = {}

    @property
    def label(self):
        return self.title if not self.title is None else re.search(r'\_xy(\w+)', self.id).group(1)

class PDPlot():
    def __init__(self, config, mapping, pd_data, plot_dir='plots/pdp/', base_name=None):
        if base_name == None:
            self.base_name = ''
        else:
            self.base_name = f'{base_name}_'
        self.mapping = mapping
        self.reverse_mapping = {v: k for k, v in mapping.items()}
        self.pd_data = pd_data
        self.plot_dir = f'{plot_dir}{self.base_name}{self.label}/'
        os.makedirs(self.plot_dir, exist_ok=True)
        self.config = config
        self._colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self._bar_color = '#000000'
    
    @property
    def label(self):
        assert len(self.pd_data) > 0
        id_string = ''
        for pd in self.pd_data:
            id_string += self.pd_id(pd) + '_'
        return id_string[:-1]

    def pd_id(self, pd_data):
        return re.search(r'\_xy(\w+)', pd_data.id).group(1)

    def plot(self, category, feature_index, feature_name):
        #fig, ax = plt.subplots(figsize=(5,2.4))
        fig, ax = plt.subplots(figsize=(15,7.2))
        plt.rcParams["font.family"] = "serif"
        ax_bar = ax.twinx()
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position("left")
        ax_bar.set_ylabel("Flow number")
        ax_bar.yaxis.tick_right()
        ax_bar.yaxis.set_label_position("right")
        ax.set_ylabel('Prediction')
        all_legends = []
        all_labels = []

        # Barplot for occurances
        as_ints = list(self.pd_data[0].features[(category, feature_index)].astype(np.int32))
        counted = Counter(as_ints)
        keys = counted.keys()
        values = counted.values()

        range = max(self.pd_data[0].results[(category, feature_index)][0,:]) - min(self.pd_data[0].results[(category, feature_index)][0,:])
        width = range / 500
        ax_bar.bar(keys, values, width=width, color=self._bar_color, alpha=0.2, label=f'{feature_name} occurrence')

        for index, pdp in enumerate(self.pd_data):
            print(f'({category},{feature_index}) ({self.reverse_mapping[category]}, {feature_name})')
            if not (category, feature_index) in pdp.results or pdp.results[(category, feature_index)] is None:
                print(f'Invalid key pair ({category},{feature_index}) in {pdp.label} or values None. Continuing...')
                return
            ax.plot(pdp.results[(category, feature_index)][0,:], pdp.results[(category, feature_index)][1,:], color=self._colors[index], label=f'{feature_name} confidence')
            all_legends.append(Rectangle((0,0), 1, 1, color=self._colors[index]))
            all_labels.append(self.pd_id(pdp))

        ax.legend(all_legends[::-1], all_labels[::-1], loc='upper left', bbox_to_anchor=(0.00,1))
        ax.set_xlabel(f'{feature_name} - {self.reverse_mapping[category]}')
        ax.set_ylabel('Partial dependence')    
        plt.tight_layout()
        file_name = (self.plot_dir + f'{feature_name}_{category}_{self.reverse_mapping[category].replace("/", "-").replace(":", "-")}_{feature_index}').replace(" ","_")
        plt.savefig(file_name, bbox_inches = 'tight', pad_inches = 0.1)
        plt.clf()

    def plot_all(self):
        for category in self.config['categories']:
            for feature_key, feature_name in self.config['features'].items():
                feature_ind = int(feature_key)
                self.plot(category, feature_ind, feature_name)
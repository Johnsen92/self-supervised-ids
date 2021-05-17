import matplotlib.pyplot as plt
import numpy as np
import math
from enum import Enum
from datetime import datetime
from timeit import default_timer as timer
from datetime import timedelta
import json
import os
import errno

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
        self._prev_timer = timer()
        self._timer = timer()
        self._seq = []
        self._agr_seq = [] 
        self._i = 0
        self._start_time = None
        self._end_time = None  
        self._progress = 0
        self._time_left_s = -1
        self._time_left_s_last = -1
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
            self._prev_timer = self._timer
            self._timer = timer()
            interval_time = self._timer - self._prev_timer
            self._time_left_s_last = self._time_left_s
            self._time_left_s = int(float(interval_time) * float(self.iterations - self._i) / float(self._interval))
            
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
            # If last value available, return avg.
            if self._time_left_s_last >= 0:
                return formatTime((self._time_left_s + self._time_left_s_last) // 2)
            else:
                return formatTime(self._time_left_s)
        # If called before first call of __call__ function, return 0
        else:
            return formatTime(0)
        
    @property
    def measurements(self):
        return self._agr_seq

    @property
    def iter(self):
        return self._i

    @property
    def duration_s(self):
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

    def add(self, results, classes):
        for index, c in enumerate(classes):
            c_val = c.item()
            self.number[c_val] += 1
            if results[index]:
                self.right[c_val] += 1

    def save_stats(self):
        now = datetime.now().strftime('%d%m%Y_%H-%M-%S')
        with open(self.stats_dir + 'class_stats_' + now + '.csv', 'w') as f:
            f.write('Class, Alias, Occurance, Right, Accuracy\n')
            for key, val in self.mapping.items():
                accuracy = self.right[val] / self.number[val] * 100.0 if not self.number[val] == 0 else 100.0
                f.write(f'{key}, {val}, {self.number[val]}, {self.right[val]}, {accuracy:.3f}\n')

            n_attack = self.n_samples - self.number[self.benign]
            n_right_attack = self.n_right - self.right[self.benign]
            accuracy_benign = (self.right[self.benign] * 100.0) / self.number[self.benign] if not self.number[self.benign] == 0 else 100.0
            accuracy_attack = (n_right_attack * 100.0) / n_attack if not n_attack == 0 else 100.0
            accuracy = (float(self.n_right) * 100.0) / float(self.n_samples)
            f.write(f'Benign, {self.benign}, {self.number[self.benign]}, {self.right[self.benign]}, {accuracy_benign:.3f}%\n')
            f.write(f'Attack, !{self.benign}, {n_attack}, {n_right_attack}, {accuracy_attack:.3f}%\n')
            f.write(f'Overall, ALL, {self.n_samples}, {self.n_right}, {accuracy:.3f}%\n')
            f.write('\n')
            benign_rate = float(self.number[self.benign]) / float(self.n_samples) * 100.0
            attack_rate = float(n_attack) / float(self.n_samples) * 100.0
            
            f.write(f'Samples, {self.n_samples}\n')
            f.write(f'Benign, {benign_rate:.2f}%\n')
            f.write(f'Attack, {attack_rate:.2f}%\n')

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
            
class Stats():

    index = 0

    def __init__(self, stats_dir='./', n_samples=None, train_percent=None, pretrain_percent=None, proxy_task=None, val_percent=None, n_epochs=None, n_epochs_pretraining=None, model_parameters=None, batch_size=None, learning_rate=None, losses=None, class_stats=None, n_false_positive=None, n_false_negative=None, title=None, random_seed=None, subset=False):
        self.stats_dir = stats_dir if stats_dir[-1] == '/' else stats_dir + '/'
        self.make_stats_dir()
        self.n_samples = n_samples
        self.n_false_positive = n_false_positive
        self.n_false_negative = n_false_negative
        self.train_percent = train_percent
        self.pretrain_percent = pretrain_percent
        self.proxy_task = proxy_task
        self.val_percent = val_percent
        self.n_epochs = n_epochs
        self.n_epochs_pretraining = n_epochs_pretraining
        self.batch_size = batch_size
        self.learning_rate = learning_rate  
        self.losses = losses
        self.class_stats = class_stats
        self.model_parameters = model_parameters
        self.random_seed = random_seed
        self.monitors = []
        self.accuracies = []
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

    def plot_stats(self):
        pass

    def save_losses(self):
        assert not self.losses == None
        now = datetime.now().strftime('%d%m%Y_%H-%M-%S')
        print('Save loss progression...', end='')
        with open(self.stats_dir + 'losses_' + now + '.csv', 'w') as f:
            for item in self.losses:
                f.write(f'{item:.6f}\n')
        print('done.')

    def save_stats(self):
        time_h, time_m = formatTime(self.training_time_s)
        n_wrong = self.n_false_negative + self.n_false_positive
        n_right = self.n_samples - n_wrong
        p_acc = float(n_right)/float(self.n_samples)*100
        now = datetime.now().strftime('%d%m%Y_%H-%M-%S')
        print('Save statistics...', end='')
        with open(self.stats_dir + 'stats_' + now + '.csv', 'w') as f:
            f.write(f'Hyperparameters,\n')
            f.write(f'Epochs Supervised, {self.n_epochs}\n')
            f.write(f'Epochs Pretraining, {0 if self.proxy_task == 0 else self.n_epochs_pretraining}\n')
            f.write(f'Batch size, {self.batch_size}\n')
            f.write(f'Proxy task, {self.proxy_task if self.pretrain_percent > 0 else "NONE"}\n')
            f.write(f'Pretraining percentage, {(self.pretrain_percent / 10.0):.2f} %\n')
            f.write(f'Training percentage, {(self.train_percent / 10.0):.2f} %\n')
            f.write(f'Validation percentage, {(self.val_percent / 10.0):.2f} %\n')
            f.write(f'Specialized subset, {self.subset}\n')
            f.write(f'Training time, {time_h} h {time_m} m\n')
            f.write(f'Learning rate, {self.learning_rate}\n')
            f.write(f'Random Seed, {self.random_seed}\n')
            f.write(f'\nModelparameters,\n')
            if not self.model_parameters is None:
                for key, val in self.model_parameters.items():
                    f.write(f'{key}, {val}\n')
            f.write(f'\nResults,\n')
            f.write(f'Final accuracy, {p_acc:.3f} %\n')
            f.write(f'Highest observed acc., {self.highest_observed_accuracy:.3f} %\n')
            f.write(f'# false positves, {self.n_false_positive}\n')
            f.write(f'# false negatives, {self.n_false_negative}\n')
            f.write(f'% false positves, {(self.false_positive * 100):.3f} %\n')
            f.write(f'% false negatives, {(self.false_negative * 100):.3f} %\n')
        print('done.')
        if not self.class_stats == None:
            self.class_stats.save_stats()

    def plot_losses(self):
        assert not self.losses == None
        x = np.array(range(len(self.losses)), dtype=float)
        x = np.round(x/len(self.losses)*100,3)
        y = np.array(self.losses, dtype=float)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='% of Training', ylabel='Loss', title='Loss progression')
        ax.grid()
        now = datetime.now().strftime('%d%m%Y_%H-%M-%S')
        fig.savefig(self.stats_dir + 'loss_' + now + '.png')
        #plt.show()

    @property
    def accuracy(self):
        assert not self.n_false_positive == None and not self.n_false_negative == None and not self.n_samples == None
        n_right = self.n_samples - self.n_false_negative - self.n_false_positive
        return float(n_right) / float(self.n_samples)

    @property
    def false_positive(self):
        assert not self.n_false_positive == None and not self.n_false_negative == None and not self.n_samples == None
        # Avoid division by 0 by adding minor float value
        return float(self.n_false_positive) / (float(self.n_false_positive + self.n_false_negative) + 0.00001)

    @property
    def false_negative(self):
        assert not self.n_false_positive == None and not self.n_false_negative == None and not self.n_samples == None
        # Avoid division by 0 by adding minor float value
        return float(self.n_false_negative) / (float(self.n_false_positive + self.n_false_negative) + 0.00001)

    @property
    def false_alarm_rate(self):
        pass

    def add_monitor(self, monitor):
        self.monitors.append(monitor)

    @property
    def training_time_s(self):
        time_s = 0
        for mon in self.monitors:
            time_s += mon.duration_s
        return time_s

    @property
    def highest_observed_accuracy(self):
        if len(self.accuracies) == 0:
            return 0.0
        else:
            return max([acc for _, acc in self.accuracies]) * 100.0
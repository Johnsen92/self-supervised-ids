import matplotlib.pyplot as plt
import numpy as np
import math
from enum import Enum
from datetime import datetime
from timeit import default_timer as timer
from datetime import timedelta
import json

def formatTime(time_s):
    time_h = time_s // 3600
    time_m = math.floor((float(time_s) / 3600.0 - time_h) * 60.0)
    return time_h, time_m

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
        self._time_left_s = 0
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
        return formatTime(self._time_left_s)
        
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

class Stats():

    index = 0

    def __init__(self, stats_dir='./', training_time_s=None, n_samples=None, train_percent=None, val_percent=None, n_epochs=None, batch_size=None, learning_rate=None, losses=None, n_false_positive=None, n_false_negative=None, gpu=True, title=None):
        self.stats_dir = stats_dir if stats_dir[-1] == '/' else stats_dir+'/'
        self.n_samples = n_samples
        self.n_false_positive = n_false_positive
        self.n_false_negative = n_false_negative
        self.train_percent = train_percent
        self.val_percent = val_percent
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gpu = gpu
        self.learning_rate = learning_rate  
        self.losses = losses
        self.training_time_s = training_time_s
        if title == None:
            self.title = "Statistics #" + str(Stats.index)
        else:
            self.title = title
        Stats.index += 1

    def plot_stats(self):
        pass

    def save_losses(self):
        assert not self.losses == None
        now = datetime.now().strftime('%d%m%Y_%H-%M-%S')
        with open(self.stats_dir + now + '_losses.csv', 'w') as f:
            for item in self.losses:
                f.write(f'{item:.6f}\n')

    def save_stats(self):
        time_h, time_m = formatTime(self.training_time_s)
        n_wrong = self.n_false_negative + self.n_false_positive
        n_right = self.n_samples - n_wrong
        p_acc = float(n_right)/float(self.n_samples)*100
        now = datetime.now().strftime('%d%m%Y_%H-%M-%S')
        with open(self.stats_dir + 'stats_' + now + '.csv', 'w') as f:
            f.write(f'Trained on, {"GPU" if self.gpu else "CPU"} %\n')
            f.write(f'Epochs, {self.n_epochs}\n')
            f.write(f'Batch size, {self.batch_size}\n')
            f.write(f'Training percentage, {self.train_percent:.2f}\n')
            f.write(f'Validation percentage, {self.val_percent:.2f}\n')
            f.write(f'Training time, {time_h} h {time_m} m\n')
            f.write(f'Learning rate, {self.learning_rate}\n')
            f.write(f'Accuracy, {p_acc:.2f} %\n')
            f.write(f'# false positves, {self.n_false_positive}\n')
            f.write(f'# false negatives, {self.n_false_negative}\n')
            f.write(f'% false positves, {(self.false_positive * 100):.2f} %\n')
            f.write(f'% false negatives, {(self.false_negative * 100):.2f} %\n')

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
        plt.show()

    @property
    def accuracy(self):
        assert not self.n_false_positive == None and not self.n_false_negative == None and not self.n_samples == None
        n_right = self.n_samples - self.n_false_negative - self.n_false_positive
        return float(n_right)/float(self.n_samples)

    @property
    def false_positive(self):
        assert not self.n_false_positive == None and not self.n_false_negative == None and not self.n_samples == None
        # Avoid division by 0 by adding minor float value
        return float(self.n_false_positive)/(float(self.n_false_positive + self.n_false_negative) + 0.00001)

    @property
    def false_negative(self):
        assert not self.n_false_positive == None and not self.n_false_negative == None and not self.n_samples == None
        # Avoid division by 0 by adding minor float value
        return float(self.n_false_negative)/(float(self.n_false_positive + self.n_false_negative) + 0.00001)

    @property
    def false_alarm_rate(self):
        pass
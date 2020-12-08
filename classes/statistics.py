import matplotlib.pyplot as plt
import numpy as np
import math
from enum import Enum
from datetime import datetime
from timeit import default_timer as timer
from datetime import timedelta

def formatTime(time_s):
    time_h = float(time_s) // 3600.0
    time_m = math.floor((float(time_s) / 3600.0 - time_h) * 60.0)
    return time_h, time_m

class Aggregate(Enum):
    NONE = 0,
    SUM = 1,
    AVG = 2

class Monitor():
    aggregates = Aggregate()

    def __init__(self, iterations, n_measurements=1000, agr=Aggregate.NONE, time_tracking=False):
        self.iterations = iterations
        self.n_samples = n_measurements
        self.agr = agr
        self.time_tracking = time_tracking
        self._interval = iterations // n_measurements
        self._prev_timer = timer()
        self._timer = timer()
        self._first = True
        self._seq = []
        self._agr_seq = [] 
        self._i = 0

    def __call__(self, val):
        self._i += 1
        self._seq.append(val)
        if self._i-1 % self._interval == 0:
            if self.agr == Aggregate.NONE:
                self._agr_seq = self._seq[-1]
            elif self.agr == Aggregate.SUM:
                self._agr_seq = sum(self._seq)
            elif self.agr == Aggregate.AVG:
                self._agr_seq = sum(self._seq)/len(self._seq)
            else:
                print(f'Invalid enum value: {self.agr}')
            self._seq = []
            if self.time_tracking:
                self._time_left()
            return True
        else:
            return False

    def _time_left(self):
        self._prev_timer = self._timer
        self._timer = timer()
        interval_time = self._timer - self._prev_timer
        time_left_s = int(float(interval_time) * float(self.iterations - self._i) / float(self._interval))
        time_left_h, time_left_m = formatTime(time_left_s)
        print(f'Time left: {time_left_h}h {time_left_m}m')
        

class Stats():
    def __init__(self, stats_dir='./', start_time=None, end_time=None, n_samples=None, train_percent=None, n_epochs=None, batch_size=None, learning_rate=None, losses=None, n_false_positive=None, n_false_negative=None):
        self.stats_dir = stats_dir if stats_dir[-1] == '/' else stats_dir+'/'
        self.n_samples = n_samples
        self.n_false_positive = n_false_positive
        self.n_false_negative = n_false_negative
        self.train_percent = train_percent
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate  
        self.losses = losses
        self.start_time = start_time
        self.end_time = end_time

    def plotStats(self):
        pass

    def saveLosses(self):
        assert not self.losses == None
        now = datetime.now().strftime('%d%m%Y_%H-%M-%S')
        with open(self.stats_dir + now + '_losses.csv', 'w') as f:
            for item in self.losses:
                f.write(f'{item:.6f}\n')

    def saveStats(self):
        time_h, time_m = formatTime(self.end_time - self.start_time)
        n_wrong = self.n_false_negative + self.n_false_positive
        n_right = self.n_samples - n_wrong
        p_acc = float(n_right)/float(self.n_samples)*100
        now = datetime.now().strftime('%d%m%Y_%H-%M-%S')
        with open(self.stats_dir + 'stats_' + now + '.csv', 'w') as f:
            f.write(f'Epochs, {self.n_epochs}\n')
            f.write(f'Batch size, {self.batch_size}\n')
            f.write(f'Training percentage, {(self.train_percent*100):.2f}\n')
            f.write(f'Training time, {time_h} h {time_m} m\n')
            f.write(f'Learning rate, {self.learning_rate}\n')
            f.write(f'Accuracy, {p_acc:.2f} %\n')
            f.write(f'# false positves, {self.n_false_positive}\n')
            f.write(f'# false negatives, {self.n_false_negative}\n')
            f.write(f'% false positves, {(float(self.n_false_positive)/float(n_wrong)*100.0):.2f} %\n')
            f.write(f'% false negatives, {(float(self.n_false_negative)/float(n_wrong)*100.0):.2f} %\n')

    def plotLosses(self):
        assert not self.losses == None
        x = np.array(range(len(self.losses)), dtype=float)
        x = np.round(x/len(self.losses)*100,3)
        y = np.array(self.losses, dtype=float)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='% of Training', ylabel='Loss', title='Loss progression')
        ax.grid()
        now = datetime.now().strftime('%d%m%Y_%H-%M-%S')
        fig.savefig(self.stats_dir + now + '_loss.png')
        plt.show()

    def getAccuracy(self):
        assert not self.n_false_positive == None and not self.n_false_negative == None and not self.n_samples == None
        n_right = self.n_samples - self.n_false_negative - self.n_false_positive
        return float(n_right)/float(self.n_samples)
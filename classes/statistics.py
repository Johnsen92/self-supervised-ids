import matplotlib.pyplot as plt
import numpy as np
import math
from datetime import datetime

def formatTime(time_s):
    time_h = float(time_s) // 3600.0
    time_m = math.floor((float(time_s) / 3600.0 - time_h) * 60.0)
    return time_h, time_m

class Stats():
    def __init__(self, stats_dir="./", start_time=None, end_time=None, n_samples=None, training_percentage=None, n_epochs=None, batch_size=None, learning_rate=None, losses=None, n_false_positive=None, n_false_negative=None):
        self.stats_dir = stats_dir if stats_dir[-1] == "/" else stats_dir+"/"
        self.n_samples = n_samples
        self.n_false_positive = n_false_positive
        self.n_false_negative = n_false_negative
        self.training_percentage = training_percentage
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
        now = datetime.now().strftime("%d%m%Y_%H-%M-%S")
        with open(self.stats_dir + "losses_" + now + ".csv", "w") as f:
            for item in self.losses:
                f.write(f"{item:.6f}\n")

    def saveStats(self):
        time_h, time_m = formatTime(self.end_time - self.start_time)
        n_wrong = self.n_false_negative + self.n_false_positive
        n_right = self.n_samples - n_wrong
        p_acc = float(n_right)/float(self.n_samples)*100
        now = datetime.now().strftime("%d%m%Y_%H-%M-%S")
        with open(self.stats_dir + "stats_" + now + ".csv", "w") as f:
            f.write(f"Epochs, {self.n_epochs}\n")
            f.write(f"Batch size, {self.batch_size}\n")
            f.write(f"Training percentage, {(self.training_percentage*100):.2f}\n")
            f.write(f"Training time, {time_h} h {time_m} m\n")
            f.write(f"Learning rate, {self.learning_rate}\n")
            f.write(f"Accuracy, {p_acc:.2f} %\n")
            f.write(f"# false positves, {self.n_false_positive}\n")
            f.write(f"# false negatives, {self.n_false_negative}\n")
            f.write(f"% false positves, {(float(self.n_false_positive)/float(n_wrong)*100.0):.2f} %\n")
            f.write(f"% false negatives, {(float(self.n_false_negative)/float(n_wrong)*100.0):.2f} %\n")

    def plotLosses(self):
        assert not self.losses == None
        x = np.array(range(len(self.losses)), dtype=float)
        x = np.round(x/len(self.losses)*100,3)
        y = np.array(self.losses, dtype=float)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel="% of Training", ylabel="Loss", title="Loss progression")
        ax.grid()
        now = datetime.now().strftime("%d%m%Y_%H-%M-%S")
        fig.savefig(self.stats_dir + "loss_" + now + ".png")
        plt.show()

    def getAccuracy(self):
        assert not self.n_false_positive == None and not self.n_false_negative == None and not self.n_samples == None
        n_right = self.n_samples - self.n_false_negative - self.n_false_positive
        return float(n_right)/float(self.n_samples)
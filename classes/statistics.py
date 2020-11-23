import matplotlib.pyplot as plt
import numpy as np

class Stats():
    def __init__(self, start_time=None, end_time=None, n_samples=None, training_percentage=None, n_epochs=None, batch_size=None, learning_rate=None, losses=None, n_false_positive=None, n_false_negative=None):
        self.n_samples = n_samples
        self.n_false_positive = n_false_positive
        self.n_false_negative = n_false_negative
        self.training_percentage = training_percentage
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate 
        self.losses = losses

    def plotStats(self):
        pass

    def saveLosses(self):
        assert not self.losses == None
        with open("_losses.csv", "w") as f:
            for item in self.losses:
                f.write(f"{item:.6f}\n")

    def plotLosses(self):
        assert not self.losses == None
        x = np.array(range(len(self.losses)), dtype=float)
        x = np.round(x/len(self.losses)*100,3)
        y = np.array(self.losses, dtype=float)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel="% of Training", ylabel="Loss", title="Loss progression")
        ax.grid()
        fig.savefig("loss.png")
        plt.show()
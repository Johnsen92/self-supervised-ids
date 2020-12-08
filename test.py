import argparse
import sys
import pickle
from torch.utils.data import random_split, DataLoader
from torch import optim, nn
from timeit import default_timer as timer
from datetime import timedelta
from classes import lstm, statistics, utils, datasets
import math
import torchvision
import torch
import os.path
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, BatchSampler, SequentialSampler, RandomSampler

def testfunction(stats):
    stats.batch_size = 64
    return 0

stats = statistics.Stats(    
    stats_dir = "./stats/",
    n_samples = 2000,
    train_percent = 100.0,
    n_epochs = 1,
    batch_size = 32,
    learning_rate = 0.001
)

print(stats.batch_size)
testfunction(stats)
print(stats.batch_size)

    
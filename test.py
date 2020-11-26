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

class FlowBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for i in self.sampler:
            batch.append(i)
            if len(batch) == self.batch_size:
                batch_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(batch)
                yield batch_padded
                batch = []
        if len(batch) > 0 and not self.drop_last:
            batch_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(batch, batch_first=True)
            yield batch_padded     


print(list(BatchSampler(RandomSampler(range(10)), batch_size=3, drop_last=False)))
print(list(FlowBatchSampler(RandomSampler(range(10)), batch_size=3, drop_last=False)))
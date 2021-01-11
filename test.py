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



mask = torch.ones(5, 5, 5)

print(mask)

mask[:, 2:4, :] = torch.zeros(mask.size())

print(mask)
    
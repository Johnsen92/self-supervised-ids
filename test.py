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

parser = argparse.ArgumentParser(description='Self-seupervised machine learning IDS')
parser.add_argument('-f', '--dataFile', help='Pickle file containing the training data')
parser.add_argument('-g', action='store_true', help='Train on GPU if available')
parser.add_argument('-t', action='store_true', help='Force training even if cache file exists')
parser.add_argument('-d', action='store_true', help='Debug flag')
parser.add_argument('-c', default="./cache/", help='Cache folder')
parser.add_argument('-s', default="./stats/", help='Statistics folder')
parser.add_argument('-e', default=10, help='Number of epochs')
parser.add_argument('-b', default=32, help='Batch size')
parser.add_argument('-p', default=90, help='Training percentage')
parser.add_argument('-l', default=512, help='Size of hidden layers')
parser.add_argument('-n', default=3, help='Number of LSTM layers')
args = parser.parse_args(sys.argv[1:])

print(args.dataFile)
    
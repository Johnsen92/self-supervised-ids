import argparse
import sys
from classes import datasets
from torch.utils.data import random_split, DataLoader
from torch import optim, nn
from timeit import default_timer as timer
from datetime import timedelta
from classes import linear
import math
import torchvision
import torch

# define argument parser
parser = argparse.ArgumentParser(description='Self-seupervised machine learning IDS')
parser.add_argument('-f', help='Pickle file containing the training data')
parser.add_argument('-t', action='store_true', help='Flag that enables training the network')
args = parser.parse_args(sys.argv[1:])

# load dataset
dataset = datasets.Flows(args.f)

data, labels, categories = dataset[0]

print(data.size())
print(labels.size())
print(categories.size())
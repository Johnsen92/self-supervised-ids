import argparse
import sys
import pickle
from torch.utils.data import random_split, DataLoader
from torch import optim, nn
from classes import datasets, lstm, statistics, utils, trainer
import torchvision
import torch
import os.path


# Define argument parser
parser = argparse.ArgumentParser(description='Self-seupervised machine learning IDS')
parser.add_argument('-f', '--data_file', help='Pickle file containing the training data', required=True)
parser.add_argument('-g', '--gpu', action='store_true', help='Train on GPU if available')
parser.add_argument('-t', '--train', action='store_true', help='Force training even if cache file exists')
parser.add_argument('-d', '--debug', action='store_true', help='Debug flag')
parser.add_argument('-c', '--cache_dir', default='./cache/', help='Cache folder')
parser.add_argument('-s', '--stats_dir', default='./stats/', help='Statistics folder')
parser.add_argument('-e', '--n_epochs', default=10, help='Number of epochs')
parser.add_argument('-b', '--batch_size', default=32, help='Batch size')
parser.add_argument('-p', '--train_percent', default=90, help='Training percentage')
parser.add_argument('-l', '--hidden_size', default=512, help='Size of hidden layers')
parser.add_argument('-n', '--n_layers', default=3, help='Number of LSTM layers')
parser.add_argument('--no_cache', action='store_true', help='Flag to disable cache')
parser.add_argument('--sequenceToVector', action='store_true', help='If set, only the output of the last LSTM iteration is considered')
parser.add_argument('--self_supervised', action='store_true', help='If set, self supervised pretraining is performed')
parser.add_argument('--pre_training', default=70, help='Percentage of training data used for self supervised pretraining')

args = parser.parse_args(sys.argv[1:])

# Define hyperparameters
data_filename = os.path.basename(args.data_file)
learning_rate = 0.001
output_size = 2

# Define cache
key_prefix = data_filename[:-7] + f'_hs{args.hidden_size}_bs{args.batch_size}_ep{args.n_epochs}_tp{args.train_percent}'
cache = utils.Cache(cache_dir=args.cache_dir, md5=True, key_prefix=key_prefix, disabled=args.no_cache)

# Load dataset and normalize data, or load from cache
if not cache.exists('dataset'):
    dataset = datasets.Flows(data_pickle=args.data_file, cache=cache)
    cache.save('dataset', dataset)
else:
    print('(Cache) Loading normalized dataset...', end='')
    dataset = cache.load('dataset')
    print('done')

# Create data loaders
n_samples = len(dataset)
training_size = (n_samples*args.train_percent) // 100
validation_size = n_samples - training_size
train, val = random_split(dataset, [training_size, validation_size])
train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True, num_workers=12, collate_fn=datasets.collate_flows, drop_last=True)
val_loader = DataLoader(dataset=val, batch_size=args.batch_size, shuffle=True, num_workers=12, collate_fn=datasets.collate_flows, drop_last=True)

# Decide between GPU and CPU Training
if torch.cuda.is_available() and args.gpu:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Define model
data, labels, categories = dataset[0]
input_size = data.size()[1]
model = lstm.LSTM(input_size, args.hidden_size, output_size, args.n_layers, args.batch_size, device).to(device)

# Define loss
criterion = nn.CrossEntropyLoss()

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Define statistics object
stats = statistics.Stats(
    stats_dir = args.stats_dir,
    n_samples = n_samples,
    train_percent = args.train_percent / 100.0,
    n_epochs = args.n_epochs,
    batch_size = args.batch_size,
    learning_rate = learning_rate
)

# Define trainer
trainer = trainer.SupvervisedTrainer(
    model = model, 
    training_data = train_loader, 
    validation_data = val_loader,
    device = device,
    criterion = criterion, 
    optimizer = optimizer, 
    epochs = args.n_epochs, 
    stats = stats, 
    cache = cache
)

# Train model
trainer.train()

# Validate model
trainer.validate()

    


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
parser.add_argument('-e', '--n_epochs', default=10, type=int, help='Number of epochs')
parser.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('-p', '--train_percent', default=90, type=int, help='Training percentage')
parser.add_argument('-l', '--hidden_size', default=512, type=int, help='Size of hidden layers')
parser.add_argument('-n', '--n_layers', default=3, type=int, help='Number of LSTM layers')
parser.add_argument('-o', '--output_size', default=2, type=int, help='Size of LSTM output vector')
parser.add_argument('-r', '--learning_rate', default=0.001, type=float, help='Initial learning rate for optimizer as decimal number')
parser.add_argument('-m', '--max_sequence_length', default=100, type=int, help='Longer data sequences will be pruned to this length')
parser.add_argument('--remove_changeable', action='store_true', help='If set, remove features an attacker could easily manipulate')
parser.add_argument('--no_cache', action='store_true', help='Flag to disable cache')
parser.add_argument('--selfsupervised', action='store_true', help='If set, self supervised pretraining is performed')
args = parser.parse_args(sys.argv[1:])

debug_size = 512
if args.debug:
    args.n_epochs = 1

# Define hyperparameters
data_filename = os.path.basename(args.data_file)

# Define cache
key_prefix = data_filename[:-7] + f'_hs{args.hidden_size}_bs{args.batch_size}_ep{args.n_epochs}_tp{args.train_percent}_lr{str(args.learning_rate).replace(".", "")}'
if args.selfsupervised:
    key_prefix.join(f'_pr{args.pre_training}')
cache = utils.Cache(cache_dir=args.cache_dir, md5=True, key_prefix=key_prefix, disabled=args.no_cache)

# Load dataset and normalize data, or load from cache
if not cache.exists('dataset'):
    dataset = datasets.Flows(data_pickle=args.data_file, cache=cache, max_length=args.max_sequence_length, remove_changeable=args.remove_changeable)
    cache.save('dataset', dataset, msg='Storing normalized dataset')
else:
    dataset = cache.load('dataset', msg='Loading normalized dataset')

# Create data loaders
n_samples = len(dataset)

# If debug flag is set, reduce size of dataset significantly
if args.debug:
    dataset, _ = random_split(dataset, [debug_size, n_samples - debug_size])
    n_samples = debug_size

training_size = (n_samples*args.train_percent) // 100
validation_size = n_samples - training_size
train, val = random_split(dataset, [training_size, validation_size])

train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True, num_workers=24, collate_fn=datasets.collate_flows, drop_last=True)
val_loader = DataLoader(dataset=val, batch_size=args.batch_size, shuffle=True, num_workers=24, collate_fn=datasets.collate_flows, drop_last=True)

# Decide between GPU and CPU Training
if torch.cuda.is_available() and args.gpu:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Define model
data, _, _ = dataset[0]
input_size = data.size()[1]
output_size = args.output_size
if args.selfsupervised:
    model = lstm.PretrainableLSTM(input_size, args.hidden_size, output_size, args.n_layers, args.batch_size, device).to(device)
else:
    model = lstm.LSTM(input_size, args.hidden_size, output_size, args.n_layers, args.batch_size, device).to(device)

# Define loss
if args.selfsupervised:
    pretraining_criterion = nn.L1Loss()
training_criterion = nn.CrossEntropyLoss()

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  

# Define statistics objects
if args.selfsupervised:
    stats_pretraining = statistics.Stats(
        stats_dir = args.stats_dir,
        n_samples = n_samples,
        train_percent = args.train_percent,
        val_percent = 100 - args.train_percent,
        n_epochs = args.n_epochs,
        batch_size = args.batch_size,
        learning_rate = args.learning_rate,
        gpu = args.gpu
    )

stats_training = statistics.Stats(
    stats_dir = args.stats_dir,
    n_samples = n_samples,
    train_percent = args.train_percent,
    val_percent = 100 - args.train_percent,
    n_epochs = args.n_epochs,
    batch_size = args.batch_size,
    learning_rate = args.learning_rate,
    gpu = args.gpu
)

# Define pretrainer
pretrainer = trainer.PredictPacket(
    model = model, 
    training_data = train_loader, 
    validation_data = val_loader,
    device = device,
    criterion = pretraining_criterion, 
    optimizer = optimizer, 
    epochs = args.n_epochs, 
    stats = stats_pretraining, 
    cache = cache
)

# Define trainer
trainer = trainer.Supvervised(
    model = model, 
    training_data = train_loader, 
    validation_data = val_loader,
    device = device,
    criterion = training_criterion, 
    optimizer = optimizer, 
    epochs = args.n_epochs, 
    stats = stats_training, 
    cache = cache
)

# Pretrain, if flag is set, then train model
if args.selfsupervised:
    pretrainer.train()
trainer.train()

# Validate model
trainer.validate()

    


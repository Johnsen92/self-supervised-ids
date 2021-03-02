import argparse
import sys
import pickle
from torch.utils.data import random_split, DataLoader
from torch import optim, nn
from classes import datasets, lstm, statistics, utils, lstm_trainer, trainer
import torchvision
import torch
import os.path
import jsons
import json
from enum import Enum
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class ProxyTask(Enum):
    NONE = 1,
    PREDICT = 2,
    OBSCURE = 3

    def __str__(self):
        return self.name

# Init argument parser
parser = argparse.ArgumentParser(description='Self-seupervised machine learning IDS')
parser.add_argument('-f', '--data_file', help='Pickle file containing the training data', required=True)
parser.add_argument('-t', '--train', action='store_true', help='Force training even if cache file exists')
parser.add_argument('-d', '--debug', action='store_true', help='Debug flag')
parser.add_argument('-C', '--cache_dir', default='./cache/', help='Cache folder')
parser.add_argument('-S', '--stats_dir', default='./stats/', help='Statistics folder')
parser.add_argument('-J', '--json_dir', default='./json/', help='Json exports folder')
parser.add_argument('-e', '--n_epochs', default=10, type=int, help='Number of epochs')
parser.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('-p', '--train_percent', default=90, type=int, help='Training percentage of data')
parser.add_argument('-v', '--val_percent', default=10, type=int, help='Validation percentage of data')
parser.add_argument('-c', '--benign_category', default=10, type=int, help='Normal/Benign category in class/category mapping')
parser.add_argument('-l', '--hidden_size', default=512, type=int, help='Size of hidden states and cell states')
parser.add_argument('-n', '--n_layers', default=3, type=int, help='Number of LSTM layers')
parser.add_argument('-o', '--output_size', default=1, type=int, help='Size of LSTM output vector')
parser.add_argument('-r', '--learning_rate', default=0.001, type=float, help='Initial learning rate for optimizer as decimal number')
parser.add_argument('-m', '--max_sequence_length', default=100, type=int, help='Longer data sequences will be pruned to this length')
parser.add_argument('-y', '--proxy_task', default=ProxyTask.PREDICT, type=lambda proxy_task: ProxyTask[proxy_task], choices=list(ProxyTask))
parser.add_argument('--remove_changeable', action='store_true', help='If set, remove features an attacker could easily manipulate')
parser.add_argument('--no_cache', action='store_true', help='Flag to ignore existing cache entries')
parser.add_argument('-s', '--self_supervised', default=0, type=int, help='Percentage of training data to be used in pretraining in respect to training percentage')
args = parser.parse_args(sys.argv[1:])

assert args.train_percent + args.self_supervised + args.val_percent <= 100

# Serialize arguments and store them in json export folder
with open(args.json_dir + '/args.json', 'w') as f:
    f.write(jsons.dumps(args))

# If debug flag is set, minimize dataset and epochs
debug_size = 2048
if args.debug:
    args.n_epochs = 1

# Init hyperparameters
data_filename = os.path.basename(args.data_file)[:-7]

# Init cache
key_prefix = data_filename + f'_hs{args.hidden_size}_bs{args.batch_size}_ep{args.n_epochs}_tp{args.train_percent}_sp{args.self_supervised}_lr{str(args.learning_rate).replace(".", "")}'
if args.self_supervised > 0:
    key_prefix.join(f'_pr{args.self_supervised}')
cache = utils.Cache(cache_dir=args.cache_dir, md5=True, key_prefix=key_prefix, disabled=args.no_cache)

# Load dataset and normalize data, or load from cache
if not cache.exists(data_filename + "_normalized", no_prefix=True) or args.no_cache:
    dataset = datasets.Flows(data_pickle=args.data_file, cache=cache, max_length=args.max_sequence_length, remove_changeable=args.remove_changeable)
    cache.save(data_filename + "_normalized", dataset, no_prefix=True, msg='Storing normalized dataset')
else:
    dataset = cache.load(data_filename + "_normalized", no_prefix=True, msg='Loading normalized dataset')

# Get category mapping from dataset 
category_mapping = dataset.mapping

# Number of samples
n_samples = len(dataset)

# If debug flag is set, only take debug_size samples
if args.debug:
    dataset, _ = random_split(dataset, [debug_size, n_samples - debug_size])
    n_samples = debug_size

# Split dataset into training and validation parts
unallocated_size = n_samples
validation_size = (n_samples * args.val_percent) // 100
supervised_size = (n_samples * args.train_percent) // 100
pretraining_size = (n_samples * args.self_supervised) // 100
unallocated_size -= supervised_size
train_data, unallocated = random_split(dataset, [supervised_size, unallocated_size])
unallocated_size -= pretraining_size
pretrain_data, unallocated = random_split(unallocated, [pretraining_size, unallocated_size])
unallocated_size -= validation_size
val_data, unallocated = random_split(unallocated, [validation_size, unallocated_size])

# Init data loaders
if args.self_supervised > 0:
    pretrain_loader = DataLoader(dataset=pretrain_data, batch_size=args.batch_size, shuffle=True, num_workers=12, collate_fn=datasets.collate_flows_batch_first, drop_last=True)
train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=12, collate_fn=datasets.collate_flows_batch_first, drop_last=True)
val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True, num_workers=12, collate_fn=datasets.collate_flows_batch_first, drop_last=True)

# Won't get far without GPU, so I assume you have one...
device = torch.device('cuda:0')

# Init model
data, _, _ = dataset[0]
input_size = data.size()[1]
output_size = args.output_size
model = lstm.PretrainableLSTM(input_size, args.hidden_size, args.output_size, args.n_layers, args.batch_size, device).to(device)

# Init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# Init class stats
class_stats_training = statistics.ClassStats(
    stats_dir = args.stats_dir,
    mapping = category_mapping,
    benign = args.benign_category
)

# Gather model parameters for statistic
model_parameters = {
    'Hidden size' : args.hidden_size,
    '# Layers' : args.n_layers
}

# Init stats
stats_training = statistics.Stats(
    stats_dir = args.stats_dir,
    class_stats = class_stats_training,
    proxy_task = f'{args.proxy_task}',
    pretrain_percent = round(pretraining_size * 100.0 / n_samples),
    train_percent = round(supervised_size * 100.0 / n_samples),
    val_percent = args.val_percent,
    n_epochs = args.n_epochs,
    batch_size = args.batch_size,
    learning_rate = args.learning_rate,
    model_parameters = model_parameters
)

# Init summary writer for TensorBoard
writer = SummaryWriter(f'runs/lstm_nl{args.n_layers}_hs{args.hidden_size}_{stats_training}_{datetime.now().strftime("%d%m%Y_%H%M%S")}')

# Pretraining if enabled
if args.self_supervised > 0:
    # Init pretraining criterion
    pretraining_criterion = nn.L1Loss()
    
    # Init pretrainer
    if(args.proxy_task == ProxyTask.PREDICT):
        pretrainer = trainer.LSTM.PredictPacket(
            model = model, 
            training_data = pretrain_loader, 
            validation_data = val_loader,
            device = device,
            criterion = pretraining_criterion, 
            optimizer = optimizer, 
            epochs = args.n_epochs, 
            stats = stats_training, 
            cache = cache,
            json = args.json_dir,
            writer = writer
        )
    elif(args.proxy_task == ProxyTask.OBSCURE):
        pretrainer = trainer.LSTM.ObscureFeature(
            model = model, 
            training_data = pretrain_loader, 
            validation_data = val_loader,
            device = device,
            criterion = pretraining_criterion, 
            optimizer = optimizer, 
            epochs = args.n_epochs, 
            stats = stats_training, 
            cache = cache,
            json = args.json_dir,
            writer = writer
        )
    else:
        print(f'Proxy task can not be {args.proxy_task} for self supervised training')
    
    # Pretrain
    pretrainer.train()

# Disable pretraining mode
model.pretraining = False

# Init criterion
training_criterion = nn.BCEWithLogitsLoss(reduction="mean")

# Init trainer
trainer = trainer.LSTM.Supervised(
    model = model, 
    training_data = train_loader, 
    validation_data = val_loader,
    device = device, 
    criterion = training_criterion, 
    optimizer = optimizer, 
    epochs = args.n_epochs, 
    stats = stats_training, 
    cache = cache,
    json = args.json_dir,
    writer = writer
)

# Train model
trainer.train()

# Evaluate model
if not args.debug:
    trainer.evaluate()

    


import argparse
import sys
import pickle
from torch.utils.data import random_split, DataLoader
from torch import optim, nn
from classes import datasets, lstm, statistics, utils, transformer, trainer
import torchvision
import torch
import os.path
import jsons
import json
from enum import Enum
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class ProxyTask(Enum):
    NONE = 0,
    AUTO = 1,
    PREDICT = 2,
    OBSCURE = 3,
    MASK = 4,
    INTER = 5

    def __str__(self):
        return self.name

# Init argument parser
parser = argparse.ArgumentParser(description='Self-seupervised machine learning IDS')
parser.add_argument('-f', '--data_file', help='Pickle file containing the training data', required=True)
parser.add_argument('-d', '--debug', action='store_true', help='Debug flag')
parser.add_argument('-C', '--cache_dir', default='./cache/', help='Cache folder')
parser.add_argument('-S', '--stats_dir', default='./stats/', help='Statistics folder')
parser.add_argument('-J', '--json_dir', default='./json/', help='Json exports folder')
# ---------------------- Model parameters ----------------------
parser.add_argument('-x', '--forward_expansion', default=20, type=int, help='Multiplier for input_size for transformer internal data width')
parser.add_argument('-n', '--n_heads', default=3, type=int, help='Number of attention heads')
parser.add_argument('-l', '--n_layers', default=10, type=int, help='Number of transformer layers')
parser.add_argument('-o', '--dropout', default=0.0, type=float, help='Dropout rate')
parser.add_argument('--output_size', default=1, type=int, help='Size of LSTM output vector')
# ---------------------- Hyper parameters ----------------------
parser.add_argument('-b', '--batch_size', default=512, type=int, help='Batch size')
parser.add_argument('-e', '--n_epochs', default=10, type=int, help='Number of epochs')
parser.add_argument('-r', '--learning_rate', default=0.001, type=float, help='Initial learning rate for optimizer as decimal number')
parser.add_argument('-m', '--max_sequence_length', default=100, type=int, help='Longer data sequences will be pruned to this length')
# ---------------------- Training config -----------------------
parser.add_argument('-y', '--proxy_task', default=ProxyTask.NONE, type=lambda proxy_task: ProxyTask[proxy_task], choices=list(ProxyTask))
parser.add_argument('-p', '--train_percent', default=90, type=int, help='Training percentage of data')
parser.add_argument('-s', '--self_supervised', default=0, type=int, help='Pretraining percentage of data')
parser.add_argument('-v', '--val_percent', default=10, type=int, help='Validation percentage of data')
parser.add_argument('--remove_changeable', action='store_true', help='If set, remove features an attacker could easily manipulate')
# ---------------------- Stats & cache -------------------------
parser.add_argument('--no_cache', action='store_true', help='Flag to ignore existing cache entries')
parser.add_argument('-c', '--benign_category', default=10, type=int, help='Normal/Benign category in class/category mapping')
args = parser.parse_args(sys.argv[1:])

assert args.train_percent + args.self_supervised + args.val_percent <= 100

# Serialize arguments and store them in json export folder
with open(args.json_dir + '/args.json', 'w') as f:
    f.write(jsons.dumps(args))

# Datafile basename
data_filename = os.path.basename(args.data_file)[:-7]

# Identifier for current parameters
run_id = f'transformer_{data_filename}_do{str(args.dropout*10).replace(".", "")}_nl{args.n_layers}_nh{args.n_heads}_fx{args.forward_expansion}_bs{args.batch_size}_ep{args.n_epochs}_lr{str(args.learning_rate*10).replace(".", "")}_tp{args.train_percent}_sp{args.self_supervised}_xy{args.proxy_task}'
if args.debug:
    run_id += '_debug'
    
# Timestamp
timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")

# Unique identifier for this run
run_uid = f'{run_id}_{timestamp}'

# Init cache
cache = utils.Cache(cache_dir=args.cache_dir, md5=True, key_prefix=run_id, disabled=args.no_cache)

# Extended stats directory for this run
extended_stats_dir = (args.stats_dir if args.stats_dir[-1] == '/' else args.stats_dir + '/') + run_uid + '/'

# Load dataset and normalize data, or load from cache
cache_filename = 'dataset_normalized'
if not cache.exists(cache_filename, no_prefix=True):
    dataset = datasets.Flows(data_pickle=args.data_file, cache=cache, max_length=args.max_sequence_length, remove_changeable=args.remove_changeable)
    cache.save(cache_filename, dataset, no_prefix=True, msg='Storing normalized dataset')
else:
    dataset = cache.load(cache_filename, no_prefix=True, msg='Loading normalized dataset')

# Get category mapping from dataset 
category_mapping = dataset.mapping

# Number of samples
n_samples = len(dataset)

# Won't get far without GPU, so I assume you have one...
device = torch.device('cuda:0')

# Split dataset into pretraining, training and validation parts
unallocated_size = n_samples
validation_size = (n_samples * args.val_percent) // 100
supervised_size = (n_samples * args.train_percent) // 100
pretraining_size = (n_samples * args.self_supervised) // 100

# If debug flag is set, use exactly one batch for pretraining, training and validation
if args.debug:
    validation_size = supervised_size = pretraining_size = args.batch_size
    args.n_epochs = 1

# Split dataset into pretraining, training and validation set
unallocated_size -= supervised_size
train_data, unallocated = random_split(dataset, [supervised_size, unallocated_size])
unallocated_size -= pretraining_size
pretrain_data, unallocated = random_split(unallocated, [pretraining_size, unallocated_size])
unallocated_size -= validation_size
val_data, unallocated = random_split(unallocated, [validation_size, unallocated_size])

# Init data loaders
if args.self_supervised > 0:
    pretrain_loader = DataLoader(dataset=pretrain_data, batch_size=args.batch_size, shuffle=True, num_workers=24, collate_fn=datasets.collate_flows, drop_last=True)
train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=24, collate_fn=datasets.collate_flows, drop_last=True)
val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True, num_workers=24, collate_fn=datasets.collate_flows, drop_last=True)

# Define input and output data width
data, _, _ = dataset[0]
input_size = data.size()[1]
output_size = args.output_size

# Init model
transformer_model = transformer.Transformer(
    input_size = input_size,
    num_heads = args.n_heads,
    num_encoder_layers = args.n_layers,
    num_decoder_layers = args.n_layers,
    forward_expansion = input_size * args.forward_expansion,
    dropout = args.dropout,
    max_len = args.max_sequence_length
).to(device)

# Init transformer encoder
model = transformer.TransformerEncoder(
    encoder = transformer_model.transformer.encoder,
    input_size = input_size,
    output_size = args.output_size,
    dropout = args.dropout,
    max_len = args.max_sequence_length
).to(device)

# Init optimizer
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Init class stats
class_stats_training = statistics.ClassStats(
    stats_dir = extended_stats_dir,
    mapping = category_mapping,
    benign = args.benign_category
)

# Gather model parameters for statistics
model_parameters = {
    '# Layers' : args.n_layers,
    '# Heads' : args.n_heads,
    'Forward expansion' : args.forward_expansion,
    'Dropout' : args.dropout
}

# Init statistics object
stats_training = statistics.Stats(
    stats_dir = extended_stats_dir,
    class_stats = class_stats_training,
    proxy_task = f'{args.proxy_task}',
    pretrain_percent = args.self_supervised,
    train_percent = args.train_percent,
    val_percent = args.val_percent,
    n_epochs = args.n_epochs,
    batch_size = args.batch_size,
    learning_rate = args.learning_rate,
    model_parameters = model_parameters
)

# Init summary writer for TensorBoard
writer = SummaryWriter(f'runs/{run_uid}')

# Pretraining if enabled
if args.self_supervised > 0:
    # Init pretraining criterion
    pretraining_criterion = nn.L1Loss()
    if(args.proxy_task == ProxyTask.INTER):
        trainer.Transformer.Interpolation(
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
    elif(args.proxy_task == ProxyTask.AUTO):
        pretrainer = trainer.Transformer.Autoencode(
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
        pretrainer = trainer.Transformer.ObscureFeature(
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
    elif(args.proxy_task == ProxyTask.MASK):
        pretrainer = trainer.Transformer.MaskPacket(
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

# Init training criterion
training_criterion = nn.BCEWithLogitsLoss(reduction="mean")

# Switch model into supervised fine-tuning mode
model.tune()

# Init trainer for supervised training
trainer = trainer.Transformer.Supervised(
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

# Print and save stats
if not args.debug:
    trainer.evaluate()
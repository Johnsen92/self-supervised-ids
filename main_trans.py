import argparse
import sys
import pickle
from torch.utils.data import random_split, DataLoader
from torch import optim, nn
from classes import datasets, lstm, statistics, utils, transformer, trainer
from classes.datasets import Flows, FlowsSubset
import torchvision
import torch
import os.path
import jsons
import json
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import random

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
parser.add_argument('-E', '--n_epochs_pretraining', default=0, type=int, help='Number of epochs for pretraining. If 0 n_epochs is used')
parser.add_argument('-r', '--learning_rate', default=0.001, type=float, help='Initial learning rate for optimizer as decimal number')
parser.add_argument('--min_sequence_length', default=2, type=int, help='Shorter sequences will no be included')
parser.add_argument('-m', '--max_sequence_length', default=100, type=int, help='Longer data sequences will be pruned to this length')
# ---------------------- Training config -----------------------
parser.add_argument('-y', '--proxy_task', default=trainer.Transformer.ProxyTask.NONE, type=lambda proxy_task: trainer.Transformer.ProxyTask[proxy_task], choices=list(trainer.Transformer.ProxyTask))
parser.add_argument('-p', '--train_percent', default=900, type=int, help='Training per-mill of data')
parser.add_argument('-G', '--subset_config', default=None, help='Path to config file for specialized subset')
parser.add_argument('-s', '--self_supervised', default=0, type=int, help='Pretraining per-mill of data')
parser.add_argument('-v', '--val_percent', default=100, type=int, help='Validation per-mill of data')
parser.add_argument('-V', '--val_epochs', default=0, type=int, help='Validate model after every val_epochs of supervised training. 0 disables periodical validation')
parser.add_argument('--subset', action='store_true', help='If set, only use a specialized subset for supervised learning')
parser.add_argument('--remove_changeable', action='store_true', help='If set, remove features an attacker could easily manipulate')
# ---------------------- Stats & cache -------------------------
parser.add_argument('--no_cache', action='store_true', help='Flag to ignore existing cache entries')
parser.add_argument('--random_seed', default=0, type=int, help='Seed for random initialization of NP, Torch and Python randomizers')
parser.add_argument('-P', '--pdp_config', default=None, help='Path to PD plot config file')
parser.add_argument('-N', '--neuron_config', default=None, help='Path to neuron activation plot config file')
parser.add_argument('-c', '--benign_category', default=10, type=int, help='Normal/Benign category in class/category mapping')
args = parser.parse_args(sys.argv[1:])

assert args.train_percent + args.self_supervised + args.val_percent <= 1000

# Set random seed
if args.random_seed == 0:
    SEED = random.randint(1, pow(2,16)-1)
else:
    SEED = args.random_seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
random_seed = SEED

# Serialize arguments and store them in json export folder
with open(args.json_dir + '/args.json', 'w') as f:
    f.write(jsons.dumps(args))

# Datafile basename
data_filename = os.path.basename(args.data_file)[:-7]

# Identifier for current parameters
run_id = f'transformer_{data_filename}_do{str(args.dropout*10).replace(".", "")}_nl{args.n_layers}_nh{args.n_heads}_fx{args.forward_expansion}_bs{args.batch_size}_lr{str(args.learning_rate*10).replace(".", "")}'
if not args.subset_config is None:
    run_id += '_subset|' + os.path.basename(args.subset_config)[:-5]
if args.debug:
    run_id += '_debug'
    
# Timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Pretraining ID
pretraining_id = f'sp{args.self_supervised}_sep{args.n_epochs_pretraining}_xy{args.proxy_task}'

# Training ID
training_id = f'tep{args.n_epochs}_tp{args.train_percent}'

# ID and unique ID (with timestamp) for this run
id = f'{run_id}_{training_id}_{pretraining_id}'
uid = f'{timestamp}_{id}'

# Init cache
general_cache = utils.Cache(cache_dir=args.cache_dir, key_prefix=id, disabled=args.no_cache, label='Transformer Cache')
training_cache = utils.Cache(cache_dir=args.cache_dir + '/training', key_prefix=id, disabled=args.no_cache, label='Transformer Training Cache')
pretraining_cache = utils.Cache(cache_dir=args.cache_dir + '/pretraining', key_prefix=run_id + '_' + pretraining_id, disabled=args.no_cache, label='Transformer Pretraining Cache')

# Extended stats directory for this run
extended_stats_dir = (args.stats_dir if args.stats_dir[-1] == '/' else args.stats_dir + '/') + uid + '/'

# Load dataset and normalize data, or load from cache
cache_filename = f'dataset_normalized_{data_filename}'
if not general_cache.exists(cache_filename, no_prefix=True):
    dataset = Flows(data_pickle=args.data_file, cache=general_cache, max_length=args.max_sequence_length, remove_changeable=args.remove_changeable)
    #dataset = FlowsSubset(dataset_all, dataset_all.mapping, min_flow_length=args.min_sequence_length)
    general_cache.save(cache_filename, dataset, no_prefix=True, msg='Storing normalized dataset')
else:
    dataset = general_cache.load(cache_filename, no_prefix=True, msg='Loading normalized dataset')

# Get category mapping from dataset 
category_mapping = dataset.mapping

# Number of samples
n_samples = len(dataset)

# Won't get far without GPU, so I assume you have one...
device = torch.device('cuda:0')

# Split dataset into training and validation parts
validation_size = int(round((n_samples * args.val_percent) / 1000.0))
supervised_size = int(round((n_samples * args.train_percent) / 1000.0))
pretraining_size = int(round((n_samples * args.self_supervised) / 1000.0))

# correct for rounding error
if (validation_size + supervised_size + pretraining_size) - n_samples == 1:
    validation_size -= 1

# If debug flag is set, use exactly one batch for pretraining, training and validation
if args.debug:
    validation_size = supervised_size = pretraining_size = args.batch_size
    args.n_epochs = 1
    args.n_epochs_pretraining = 1

# Split dataset into pretraining, training and validation set
if args.self_supervised > 0:
    train_data, pretrain_data, val_data = dataset.split([supervised_size, pretraining_size, validation_size], stratify=True)
else:
    train_data, val_data = dataset.split([supervised_size, validation_size], stratify=True)

# If the subset flag is set, only use this small selected dataset for supervised learning
if not args.subset_config is None:
    train_data = FlowsSubset(train_data, category_mapping, config_file=args.subset_config, key='TRAIN', config_index=args.subset_config_index)
    val_data = FlowsSubset(val_data, category_mapping, config_file=args.subset_config, key='VALIDATE', config_index=args.subset_config_index)
    if args.self_supervised > 0:
        pretrain_data = FlowsSubset(pretrain_data, category_mapping, config_file=args.subset_config, key='PRETRAIN', config_index=args.subset_config_index)

# Init data loaders
if args.self_supervised > 0:
    pretrain_loader = DataLoader(dataset=pretrain_data, batch_size=args.batch_size, shuffle=True, num_workers=12, collate_fn=datasets.collate_flows, drop_last=True)
train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=12, collate_fn=datasets.collate_flows, drop_last=True)
val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True, num_workers=12, collate_fn=datasets.collate_flows, drop_last=True)
if args.debug:
    test_loader = val_loader
else:
    test_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, collate_fn=datasets.collate_flows, drop_last=True)

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
class_stats = statistics.ClassStats(
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

# If pretraining epochs argument is set, use it, else default to supervised epochs argument
epochs_pretraining = args.n_epochs if args.n_epochs_pretraining == 0 else args.n_epochs_pretraining

# Init statistics object
stats = statistics.Stats(
    stats_dir = extended_stats_dir,
    class_stats = class_stats,
    proxy_task = f'{args.proxy_task}',
    pretrain_percent = args.self_supervised,
    train_percent = args.train_percent,
    val_percent = args.val_percent,
    n_epochs = args.n_epochs,
    n_epochs_pretraining = epochs_pretraining,
    batch_size = args.batch_size,
    learning_rate = args.learning_rate,
    model_parameters = model_parameters,
    random_seed = random_seed,
    subset = os.path.basename(args.subset_config)[:-5] if not args.subset_config is None else ''
)

# Init summary writer for TensorBoard
writer = SummaryWriter(f'runs/{uid}')

# Pretraining if enabled
if args.self_supervised > 0:
    # Init pretraining criterion
    pretraining_criterion = nn.L1Loss()
    if(args.proxy_task == trainer.Transformer.ProxyTask.INTER):
        trainer.Transformer.Interpolation(
            model = model, 
            training_data = pretrain_loader, 
            device = device, 
            criterion = pretraining_criterion, 
            optimizer = optimizer, 
            epochs = epochs_pretraining, 
            val_epochs = args.val_epochs,
            stats = stats, 
            cache = pretraining_cache,
            json = args.json_dir,
            writer = writer,
            title = 'Interpolation',
            test_data = test_loader
        )
    elif(args.proxy_task == trainer.Transformer.ProxyTask.AUTO):
        # Introduce dropout for denoising autoencoder
        #model.dropout = nn.Dropout(0.2)
        pretrainer = trainer.Transformer.AutoEncode(
            model = model, 
            training_data = pretrain_loader, 
            device = device, 
            criterion = pretraining_criterion, 
            optimizer = optimizer, 
            epochs = epochs_pretraining, 
            val_epochs = args.val_epochs,
            stats = stats, 
            cache = pretraining_cache,
            json = args.json_dir,
            writer = writer,
            title = 'AutoEncoder',
            test_data = test_loader
        )
    elif(args.proxy_task == trainer.Transformer.ProxyTask.OBSCURE):
        pretrainer = trainer.Transformer.ObscureFeature(
            model = model, 
            training_data = pretrain_loader, 
            device = device, 
            criterion = pretraining_criterion, 
            optimizer = optimizer, 
            epochs = epochs_pretraining, 
            val_epochs = args.val_epochs,
            stats = stats, 
            cache = pretraining_cache,
            json = args.json_dir,
            writer = writer,
            title = 'ObscureFeature',
            test_data = test_loader
        )
    elif(args.proxy_task == trainer.Transformer.ProxyTask.MASK):
        pretrainer = trainer.Transformer.MaskPacket(
            model = model, 
            training_data = pretrain_loader, 
            device = device, 
            criterion = pretraining_criterion, 
            optimizer = optimizer, 
            epochs = epochs_pretraining, 
            val_epochs = args.val_epochs,
            stats = stats, 
            cache = pretraining_cache,
            json = args.json_dir,
            writer = writer,
            title = 'MaskPacket',
            test_data = test_loader
        )
    else:
        print(f'Proxy task can not be {args.proxy_task} for self supervised training')

    # Pretrain
    pretrainer.train()

    if args.proxy_task == trainer.Transformer.ProxyTask.AUTO:
        model.dropout = nn.Dropout(args.dropout)      

    if not args.neuron_config is None:
        pretrainer.neuron_activation(run_id, args.neuron_config, postfix='Pretraining')


# Init training criterion
training_criterion = nn.BCEWithLogitsLoss(reduction="mean")

# Switch model into supervised fine-tuning mode
model.tune()

# Init trainer for supervised training
finetuner = trainer.Transformer.Supervised(
    model = model, 
    training_data = train_loader, 
    validation_data = val_loader,
    device = device, 
    criterion = training_criterion, 
    optimizer = optimizer, 
    epochs = args.n_epochs, 
    val_epochs = args.val_epochs,
    stats = stats, 
    cache = training_cache,
    json = args.json_dir,
    writer = writer,
    title = 'Supervised',
    test_data = test_loader
)

# Train model
finetuner.train()

# Partial dependency data
if args.proxy_task == trainer.LSTM.ProxyTask.NONE and not args.pdp_config is None:
    finetuner.pdp(uid, args.pdp_config)

# Neuron activation data
if not args.neuron_config is None:
    finetuner.neuron_activation(uid, args.neuron_config, postfix='Supervised')

# Remove temp directories
general_cache.clean()
training_cache.clean()
pretraining_cache.clean()

# Print and save stats
if not args.debug:
    finetuner.evaluate()
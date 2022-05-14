import pickle
import os.path
import hashlib
import os
import errno
import torch
from enum import Enum
import numpy as np
import shutil
from  argparse import ArgumentParser
import random
import time

def make_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def rm_dir(dir):
    shutil.rmtree(dir, ignore_errors=True)

def numpy_sigmoid(x):
    return 1/(1+np.exp(-x))

class ProxyTask(Enum):
    NONE = 0,
    AUTO = 1,
    PREDICT = 2,
    OBSCURE = 3,
    MASK = 4,
    ID = 5,
    INTER = 6,
    COMPOSITE = 7

    def __str__(self):
        return self.name

class ModelArgumentParser(ArgumentParser):
    def __init__(self, description):
        super(ModelArgumentParser, self).__init__(description = description)
        self.add_argument('-f', '--data_file', help='Pickle file containing the training data', required=True)
        self.add_argument('-d', '--debug', action='store_true', help='Debug flag')
        self.add_argument('-C', '--cache_dir', default='./cache/', help='Cache folder')
        self.add_argument('-S', '--stats_dir', default='./stats/', help='Statistics folder')
        self.add_argument('-J', '--json_dir', default='./json/', help='Json exports folder')
        self.add_argument('-L', '--log_dir', default='./runs/', help='Tensorboard logdir')
        # ---------------------- Hyper parameters ----------------------
        self.add_argument('-e', '--n_epochs', default=10, type=int, help='Number of epochs for supervised training')
        self.add_argument('-E', '--n_epochs_pretraining', default=0, type=int, help='Number of epochs for pretraining. If 0, n_epochs is used')
        self.add_argument('-b', '--batch_size', default=128, type=int, help='Batch size')
        self.add_argument('-r', '--learning_rate', default=0.001, type=float, help='Initial learning rate for optimizer as decimal number')
        self.add_argument('--min_sequence_length', default=1, type=int, help='Shorter sequences will no be included')
        self.add_argument('-m', '--max_sequence_length', default=100, type=int, help='Longer data sequences will be pruned to this length')
        self.add_argument('--output_size', default=1, type=int, help='Size of output tensor')
        # ---------------------- Training config -----------------------
        self.add_argument('-p', '--train_percent', default=900, type=int, help='Training per-mill of data')
        self.add_argument('-s', '--self_supervised', default=0, type=int, help='Pretraining per-mill of data')
        self.add_argument('-v', '--val_percent', default=100, type=int, help='Validation per-mill of data')
        self.add_argument('-V', '--val_epochs', default=0, type=int, help='Validate model after every val_epochs of supervised training. 0 disables periodical validation. -1 choose a reasonable value')
        self.add_argument('--val_batch_size', default=8192, type=int, help='Batch size used for validation. Selected to not strain GPU too much')
        self.add_argument('-y', '--proxy_task', default=ProxyTask.NONE, type=lambda proxy_task: ProxyTask[proxy_task], choices=list(ProxyTask))
        self.add_argument('-G', '--subset_config', default=None, help='Path to config file for specialized subset')
        self.add_argument('-i', '--subset_config_index', default=-1, type=int, help='If the subset configuration file contains multiple configurations, this index is needed')
        self.add_argument('--remove_changeable', action='store_true', help='If set, remove features an attacker could easily manipulate')
        self.add_argument('--feature_expansion', default=1, type=int, help='Factor by which the number of input features is extended by random data')
        self.add_argument('--random_seed', default=0, type=int, help='Seed for random initialization of NP, Torch and Python randomizers')
        # ---------------------- Stats & Cache -------------------------
        self.add_argument('--id_only', action='store_true', help='If set only print the ID and return. Used for scripting purposes')
        self.add_argument('-c', '--benign_category', default=10, type=int, help='Normal/Benign category in class/category mapping')
        self.add_argument('-P', '--pdp_config', default=None, help='Path to PD plot config file')
        self.add_argument('-N', '--neuron_config', default=None, help='Path to neuron activation plot config file')
        self.add_argument('--no_cache', action='store_true', help='Flag to ignore existing cache entries')
        # ---------------------- Pytorch & Numpy -----------------------
        self.add_argument('--n_threads', default=2, type=int, help='Number of threads used by PyTorch')
        self.add_argument('--n_worker_threads', default=4, type=int, help='Number of worker threads to load dataset')

    def parse_args(self, args=None, namespace=None):
        args = super(ModelArgumentParser, self).parse_args(args, namespace)
        if args.proxy_task == ProxyTask.NONE:
            args.self_supervised = 0
            args.n_epochs_pretraining = 0
        assert args.train_percent + args.self_supervised + args.val_percent <= 1000
        # If val_epochs is set to auto mode, calculate reasonable value
        if args.val_epochs == -1:
            args.val_epochs = max(1, args.n_epochs // 100)

        if args.random_seed == 0:
            args.random_seed = random.randint(1, pow(2,16)-1)
        else:
            args.random_seed = args.random_seed

        if not os.path.exists(os.path.dirname(args.log_dir)):
            try:
                os.makedirs(os.path.dirname(args.log_dir))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        return args

class DTArgumentParser(ArgumentParser):
    def __init__(self, description):
        super(DTArgumentParser, self).__init__(description = description)
        self.add_argument('-f', '--data_file', help='Pickle file containing the training data', required=True)
        self.add_argument('-d', '--debug', action='store_true', help='Debug flag')
        self.add_argument('-C', '--cache_dir', default='./cache/', help='Cache folder')
        self.add_argument('-S', '--stats_dir', default='./stats/', help='Output folder for statistics')
        self.add_argument('-T', '--trees_dir', default='./trees/', help='Output folder for trees in text')
        self.add_argument('-P', '--plots_dir', default='./plots/', help='Output folder for tree plots')
        # ---------------------- Model parameters ----------------------
        self.add_argument('-m', '--max_depth', default=6, type=int, help='Maximum depth of decision tree')
        # ---------------------- Training config -----------------------
        self.add_argument('-p', '--train_percent', default=900, type=int, help='Training per-mill of data')
        self.add_argument('-v', '--val_percent', default=100, type=int, help='Validation per-mill of data')
        self.add_argument('-c', '--benign_category', default=10, type=int, help='Normal/Benign category in class/category mapping')
        self.add_argument('-t', '--target_category', default=-1, type=int, help='Decision tree tries to differentiate between benign category and this category. If -1, use all categories')
        self.add_argument('--random_seed', default=0, type=int, help='Seed for random initialization of NP, Torch and Python randomizers')
        self.add_argument('--max_sequence_length', default=100, type=int, help='Longer data sequences will be pruned to this length')
        # ---------------------- Stats & Cache -------------------------
        self.add_argument('--val_batch_size', type=int, default=4096, help='Batch size used for validation data loader')
        self.add_argument('--id_only', action='store_true', help='If set only print the ID and return. Used for scripting purposes')
        self.add_argument('--no_cache', action='store_true', help='Flag to ignore existing cache entries')
        self.add_argument('--plot', default=False, action='store_true', help='If set, plots the decision tree')
        self.add_argument('--depth_analysis', action='store_true', help='If set, fit the tree with depths 1-max depth and save a summary to stats')
        self.add_argument('-x', type=int, default=25, help='X dimension of plot')
        self.add_argument('-y', type=int, default=20, help='Y dimension of plot')
        self.add_argument('-o', '--output_file', default='', help='Output file name')
        # ---------------------- Pytorch & Numpy -----------------------
        self.add_argument('--n_threads', default=2, type=int, help='Number of threads used by PyTorch')
        self.add_argument('--n_worker_threads', default=4, type=int, help='Number of worker threads to load dataset')

    def parse_args(self, args=None, namespace=None):
        args = super(DTArgumentParser, self).parse_args(args, namespace)
        assert args.train_percent + args.val_percent <= 1000
        if args.random_seed == 0:
            args.random_seed = random.randint(1, pow(2,16)-1)
        else:
            args.random_seed = args.random_seed
        return args

class LSTMArgumentParser(ModelArgumentParser):
    def __init__(self, description):
        super(LSTMArgumentParser, self).__init__(description = description)
        # ---------------------- Model parameters ----------------------
        self.add_argument('-l', '--hidden_size', default=512, type=int, help='Size of hidden states and cell states')
        self.add_argument('-n', '--n_layers', default=3, type=int, help='Number of LSTM layers')

    def parse_args(self, args=None, namespace=None):
        return super(LSTMArgumentParser, self).parse_args(args, namespace)

class TransformerArgumentParser(ModelArgumentParser):
    def __init__(self, description):
        super(TransformerArgumentParser, self).__init__(description = description)
        # ---------------------- Model parameters ----------------------
        self.add_argument('-x', '--forward_expansion', default=2, type=int, help='Multiplier for input_size for transformer internal data width')
        self.add_argument('-n', '--n_heads', default=3, type=int, help='Number of attention heads')
        self.add_argument('-l', '--n_layers', default=10, type=int, help='Number of transformer layers')
        self.add_argument('-o', '--dropout', default=0.0, type=float, help='Dropout rate')

    def parse_args(self, args=None, namespace=None):
        return super(TransformerArgumentParser, self).parse_args(args, namespace)

class Cache():

    index = 0

    def __init__(self, cache_dir, md5=False, key_prefix='', disabled=False, label=None, verbose=False):
        self.cache_dir = cache_dir if cache_dir[-1] == '/' else cache_dir + '/'
        self.cache_tmp_dir = self.cache_dir + f'tmp_{key_prefix}/'
        self.make_cache_dir()
        self.make_tmp_dir()
        self.md5 = md5
        self.label = label if not label is None else f'Cache #{Cache.index}'
        Cache.index += 1
        self.key_prefix = key_prefix
        self.disabled = disabled
        self.verbose = verbose
        if self.verbose:
            self.log(f'Initialize Cache \'{self.label}\' with {"md5" if self.md5 else ""} key prefix \'{self.get_real_key("", False)}\'...done.')

    def log(self, string, end='\n'):
        if self.verbose:
            print(f'{string}{end}')

    def make_cache_dir(self):
        if not os.path.exists(os.path.dirname(self.cache_dir)):
            try:
                os.makedirs(os.path.dirname(self.cache_dir))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    def make_tmp_dir(self):
        if not os.path.exists(os.path.dirname(self.cache_tmp_dir)):
            try:
                os.makedirs(os.path.dirname(self.cache_tmp_dir))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    def exists(self, key, no_prefix=False, tmp=False):
        key = self.get_real_key(key, no_prefix)
        cache_file = (self.cache_tmp_dir if tmp else self.cache_dir) + key
        return os.path.isfile(cache_file + '.pickle') or os.path.isfile(cache_file + '.sdc') if not self.disabled else False

    def load(self, key, no_prefix=False, msg=None, tmp=False):
        key = self.get_real_key(key, no_prefix)
        cache_file = (self.cache_tmp_dir if tmp else self.cache_dir) + key + '.pickle'
        assert os.path.isfile(cache_file), f'File not found: {cache_file}'
        self.log(f'(Cache) Loading {key} from{" temp" if tmp else ""} cache' if msg == None else f'(Cache) {msg}', end='')
        with open (cache_file, 'rb') as f:
            pkl = pickle.load(f)
            self.log('...done')
        return pkl

    def save(self, key, obj, no_prefix=False, msg=None, tmp=False):
        key = self.get_real_key(key, no_prefix)
        cache_file = (self.cache_tmp_dir if tmp else self.cache_dir) + key + '.pickle'
        self.log(f'(Cache) Storing {key} to{" temp" if tmp else ""} cache' if msg == None else f'(Cache) {msg}', end='')
        with open (cache_file, 'wb') as f:
            f.write(pickle.dumps(obj))
            self.log('...done')

    def save_model(self, key, model, no_prefix=False, msg=None, tmp=False):
        key = self.get_real_key(key, no_prefix)
        cache_file = (self.cache_tmp_dir if tmp else self.cache_dir) + key + '.sdc'
        self.log(f'(Cache) Storing model {key} to{" temp" if tmp else ""} cache' if msg == None else f'(Cache) {msg}', end='')
        torch.save(model.state_dict(), cache_file)
        self.log('...done')

    def load_model(self, key, model, no_prefix=False, msg=None, tmp=False):
        key = self.get_real_key(key, no_prefix)
        cache_file = (self.cache_tmp_dir if tmp else self.cache_dir) + key + '.sdc'
        assert os.path.isfile(cache_file), f'File not found: {cache_file}'
        self.log(f'(Cache) Loading model {key} from{" temp" if tmp else ""} cache' if msg == None else f'(Cache) {msg}', end='')
        model.load_state_dict(torch.load(cache_file))
        self.log('...done')

    def get_real_key(self, key, no_prefix):
        prefix = hashlib.md5(self.key_prefix.encode('utf-8')).hexdigest() if self.md5 else self.key_prefix
        return key if no_prefix else prefix + '_' + key

    def clean(self):
        try:
            shutil.rmtree(self.cache_tmp_dir)
        except OSError as e:
            self.log(f'Error: {e.filename} - {e.strerror}.')
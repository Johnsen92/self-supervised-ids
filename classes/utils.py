import math
import pickle
import os.path
import hashlib
import os
import errno
import torch
from enum import Enum

def make_dir(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

class Cache():
    def __init__(self, cache_dir, md5=False, key_prefix='', disabled=False):
        self.cache_dir = cache_dir if cache_dir[-1] == '/' else cache_dir + '/'
        self.make_cache_dir()
        self.md5 = md5
        self.key_prefix = key_prefix
        self.disabled = disabled

    def make_cache_dir(self):
        if not os.path.exists(os.path.dirname(self.cache_dir)):
            try:
                os.makedirs(os.path.dirname(self.cache_dir))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    def exists(self, key, no_prefix=False):
        key = self.get_real_key(key, no_prefix)
        cache_file = self.cache_dir + key + '.pickle'
        return os.path.isfile(cache_file) if not self.disabled else False

    def load(self, key, no_prefix=False, msg=None):
        key = self.get_real_key(key, no_prefix)
        cache_file = self.cache_dir + key + '.pickle'
        assert os.path.isfile(cache_file)
        print(f'(Cache) Loading {key} from cache' if msg == None else f'(Cache) {msg}', end='')
        with open (cache_file, 'rb') as f:
            pkl = pickle.load(f)
            print('...done')
        return pkl

    def save(self, key, obj, no_prefix=False, msg=None):
        key = self.get_real_key(key, no_prefix)
        cache_file = self.cache_dir + key + '.pickle'
        print(f'(Cache) Storing {key} to cache' if msg == None else f'(Cache) {msg}', end='')
        with open (cache_file, 'wb') as f:
            f.write(pickle.dumps(obj))
            print('...done')

    def save_model(self, key, model, no_prefix=False, msg=None):
        key = self.get_real_key(key, no_prefix)
        cache_file = self.cache_dir + key + '.sdc'
        print(f'(Cache) Storing model {key} to cache' if msg == None else f'(Cache) {msg}', end='')
        torch.save(model.state_dict(), cache_file)
        print('...done')

    def load_model(self, key, model, no_prefix=False, msg=None):
        key = self.get_real_key(key, no_prefix)
        cache_file = self.cache_dir + key + '.sdc'
        assert os.path.isfile(cache_file)
        print(f'(Cache) Loading model {key} from cache' if msg == None else f'(Cache) {msg}', end='')
        model.load_state_dict(torch.load(cache_file))
        print('...done')

    def get_real_key(self, key, no_prefix):
        prefix = hashlib.md5(self.key_prefix.encode('utf-8')).hexdigest() if self.md5 else self.key_prefix
        return key if no_prefix else prefix + '_' + key

class RunControl():
    class Controls(Enum):
        NONE = 1,
        OVERWRITE = 2,
        LOAD_PRETRAINED_MODEL = 3,
        LOAD_PRETRAINING_EPOCH = 4,
        LOAD_TRAINING_EPOCH = 5,
        LOAD_TRAINED_MODEL = 6

        def __str__(self):
            return self.name
    
    def __init__(self, cmd, start_epoch, n_epochs, cache, stats, model):
        self.run = Run(cache, stats, model)

    def checkpoint(self, epoch):
        pass

    @property
    def random_seed(self):
        if not self.stats is None:
            return self.stats.random_seed
        else:
            return 0

class Run():
    def __init__(self, cache, stats, model, random_seed):
        self.cache = cache
        self.stats = stats
        self.model = model
        self.seed = random_seed

    def exists(self, key, epoch):
        return self.cache.exists(f'{key}_stats_{epoch}') and self.cache.exists(f'{key}_model_{epoch}')

    def save(self, key, epoch):
        self.cache.save(f'{key}_stats_{epoch}', self.stats)
        self.cache.save_model(f'{key}_model_{epoch}', self.model)

    def load(self, key, epoch):
        self.cache.load(f'{key}_stats_{epoch}', self.stats)
        self.cache.load_model(f'{key}_model_{epoch}', self.model)

    def get_latest_epoch(self, key):
        epoch = 0
        while self.exists(key, epoch):
            epoch += 1
        return epoch - 1
import math
import pickle
import os.path
import hashlib
import os
import errno
import torch

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
        
    def get_real_key(self, key, no_prefix):
        prefix = hashlib.md5(self.key_prefix.encode('utf-8')).hexdigest() if self.md5 else self.key_prefix
        return key if no_prefix else prefix + '_' + key

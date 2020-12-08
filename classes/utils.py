import math
import pickle
import os.path
import hashlib  

class Cache():
    def __init__(self, cache_dir, md5=False, key_prefix='', disabled=False):
        self.cache_dir = cache_dir if cache_dir[-1] == '/' else cache_dir+'/'
        self.md5 = md5
        self.key_prefix = key_prefix
        self.disabled = disabled

    def exists(self, key, noPrefix=False):
        key = self.getRealKey(key, noPrefix)
        cache_file = self.cache_dir + key + '.pickle'
        return os.path.isfile(cache_file) if not self.disabled else False

    def load(self, key, noPrefix=False, msg=None):
        key = self.getRealKey(key, noPrefix)
        cache_file = self.cache_dir + key + '.pickle'
        assert os.path.isfile(cache_file)
        print(f'(Cache) Loading {key} from cache' if msg == None else f'(Cache) {msg}', end='')
        with open (cache_file, 'rb') as f:
            pkl = pickle.load(f)
            print('...done')
        return pkl
        

    def save(self, key, obj, noPrefix=False, msg=None):
        key = self.getRealKey(key, noPrefix)
        cache_file = self.cache_dir + key + '.pickle'
        print(f'(Cache) Storing {key} to cache' if msg == None else f'(Cache) {msg}', end='')
        with open (cache_file, 'wb') as f:
            f.write(pickle.dumps(obj))
            print('...done')
        
    def getRealKey(self, key, noPrefix):
        prefix = hashlib.md5(self.key_prefix.encode('utf-8')).hexdigest() if self.md5 else self.key_prefix
        return key if noPrefix else prefix + '_' + key

def getTimeLeft(sample_time, samples_left):
    time_left = (sample_time * float(samples_left))/3600.0
    time_left_h = math.floor(time_left)
    time_left_m = math.floor((time_left - time_left_h)*60.0)
    return time_left_h, time_left_m
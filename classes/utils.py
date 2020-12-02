import math
import pickle
import os.path
import hashlib  

class Cache():
    def __init__(self, cache_dir, md5=False, key_prefix='', no_cache=False):
        self.cache_dir = cache_dir if cache_dir[-1] == '/' else cache_dir+'/'
        self.md5 = md5
        self.key_prefix = key_prefix
        self.no_cache = no_cache

    def exists(self, key, noPrefix=False):
        key = self.getRealKey(key, noPrefix)
        cache_file = self.cache_dir + key + '.pickle'
        return os.path.isfile(cache_file) if not self.no_cache else False

    def load(self, key, noPrefix=False):
        key = self.getRealKey(key, noPrefix)
        cache_file = self.cache_dir + key + '.pickle'
        assert os.path.isfile(cache_file)
        with open (cache_file, 'rb') as f:
            return pickle.load(f)

    def save(self, key, obj, noPrefix=False):
        key = self.getRealKey(key, noPrefix)
        cache_file = self.cache_dir + key + '.pickle'
        with open (cache_file, 'wb') as f:
            f.write(pickle.dumps(obj))

    def getRealKey(self, key, noPrefix):
        prefix = hashlib.md5(self.key_prefix.encode('utf-8')).hexdigest() if self.md5 else self.key_prefix
        return key if noPrefix else prefix + '_' + key

def getTimeLeft(sample_time, samples_left):
    time_left = (sample_time * float(samples_left))/3600.0
    time_left_h = math.floor(time_left)
    time_left_m = math.floor((time_left - time_left_h)*60.0)
    return time_left_h, time_left_m
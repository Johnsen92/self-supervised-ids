import math
import pickle
import os.path
import hashlib  

class Cache():
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir if cache_dir[-1] == "/" else cache_dir+"/"

    def exists(self, key):
        md5key = hashlib.md5(key)
        cache_file = self.cache_dir + md5key + ".pickle"
        return os.path.isfile(cache_file)

    def load(self, key):
        md5key = hashlib.md5(key)
        cache_file = self.cache_dir + md5key + ".pickle"
        assert os.path.isfile(cache_file)
        with open (cache_file, "rb") as f:
            return pickle.load(f)

    def save(self, key, obj):
        md5key = hashlib.md5(key)
        cache_file = self.cache_dir + md5key + ".pickle"
        with open (cache_file, "wb") as f:
            f.write(pickle.dumps(obj))

def getTimeLeft(sample_time, samples_left):
    time_left = (sample_time * float(samples_left))/3600.0
    time_left_h = math.floor(time_left)
    time_left_m = math.floor((time_left - time_left_h)*60.0)
    return time_left_h, time_left_m
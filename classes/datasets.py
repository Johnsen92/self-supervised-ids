import torch
import pandas as pd
from torch.utils.data import Dataset, Subset
from torch.utils.data.sampler import Sampler, RandomSampler
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import json
import os.path
import re
from sklearn.model_selection import train_test_split
from torch._utils import _accumulate
from enum import Enum

def overwrite_manipulable_entries(seq, filler=-1):
    forward_direction = seq[0,5]
    wrong_direction = (seq[:,5]==forward_direction)
    seq[wrong_direction,:][:,3:5] = filler
    return seq

# Stack and compress sequences of different length in batch
def collate_flows(seqs):    
    flows, labels, categories = zip(*seqs)
    flows_lens = [len(flow) for flow in flows]
    assert not 0 in flows_lens
    padded_flows = torch.nn.utils.rnn.pad_sequence(flows)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels)
    padded_categories = torch.nn.utils.rnn.pad_sequence(categories)
    return (padded_flows, flows_lens), padded_labels, padded_categories

# Stack and compress sequences of different length in batch
def collate_flows_batch_first(seqs):    
    flows, labels, categories = zip(*seqs)
    flows_lens = [len(flow) for flow in flows]
    assert not 0 in flows_lens
    padded_flows = torch.nn.utils.rnn.pad_sequence(flows, batch_first=True)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    padded_categories = torch.nn.utils.rnn.pad_sequence(categories, batch_first=True)
    return (padded_flows, flows_lens), padded_labels, padded_categories

class Flows(Dataset):
    def __init__(self, data_pickle, cache=None, max_length=100, remove_changeable=False, expansion_factor=1):
        super().__init__()
        print('Loading data file...', end='')

        self.cache = cache
        self.data_pickle = data_pickle
        self.max_length = max_length
        self.remove_changeable = remove_changeable

        # Load pickled dataset
        with open(data_pickle, 'rb') as f:
            all_data = pickle.load(f)

        with open(data_pickle[:-7]+'_categories_mapping.json', 'r') as f:
            categories_mapping_content = json.load(f)
        categories_mapping, mapping = categories_mapping_content['categories_mapping'], categories_mapping_content['mapping']
        assert min(mapping.values()) == 0

        # Remove flows witch invalid IATs
        all_data = [item[:max_length,:] for item in all_data if np.all(item[:,4]>=0)]
        if remove_changeable:
            all_data = [overwrite_manipulable_entries(item) for item in all_data]

        X = [item[:, :-2] for item in all_data]
        Y = [item[:, -1:] for item in all_data]

        # If feature expansion is enabled, expand feature size by expansion_factor with random data
        if expansion_factor > 1:
            X = [np.concatenate((item, np.random.rand(item.shape[0], item.shape[1] * (expansion_factor - 1))), axis=1) for item in X]

        print('done')

        # Normalize data between -1 and 1
        cache_filename = 'normalization_data'
        if not self.cache == None and not self.cache.exists(cache_filename):
            print('Calculating normalization data...', end='')
            catted_x = np.concatenate(X, axis=0)
            means = np.mean(catted_x, axis=0)
            stds = np.std(catted_x, axis=0)
            stds[stds==0.0] = 1.0
            print('done')
            
            # Store for future use
            cache.save(cache_filename, (means, stds), msg='Storing normalization data')
        else:
            (means, stds) = cache.load(cache_filename, msg='Loading normalization data')

        assert means.shape[0] == X[0].shape[-1], 'means.shape: {}, x.shape: {}'.format(means.shape, X[0].shape)
        assert stds.shape[0] == X[0].shape[-1], 'stds.shape: {}, x.shape: {}'.format(stds.shape, X[0].shape)
        assert not (stds==0).any(), 'stds: {}'.format(stds)

        # Store in class members
        self.x = [(item-means)/stds for item in X]
        self.y = Y
        self.categories = [item[:, -2:-1] for item in all_data]
        self.categories_mapping = categories_mapping
        self.mapping = mapping
        self.n_samples = len(self.x)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        tensor_categories = torch.LongTensor(self.categories[i])
        tensor_labels = torch.FloatTensor(self.y[i])
        tensor_data = torch.FloatTensor(self.x[i])
        return tensor_data, tensor_labels, tensor_categories

    def split(self, split_sizes, stratify=False):
        X_idx_rest = np.arange(self.n_samples)
        y_rest = np.squeeze(np.array([c[0] for c in self.categories], dtype=np.intc))
        splits = []
        for i, size in enumerate(split_sizes):
            X_idx_split, X_idx_rest, _, y_rest = train_test_split(X_idx_rest, y_rest, train_size=size, stratify=(y_rest if stratify else None))
            splits.append(Subset(self, X_idx_split))
            # If the whole dataset is used, we have to stop splitting one iteration early, otherwise one split would produce an empty-set which upsets sklearn
            if i == len(split_sizes)-2 and sum(split_sizes) == self.n_samples:
                splits.append(Subset(self, X_idx_rest))
                break
        assert sum([len(s) for s in splits]) == sum(split_sizes) 
        return tuple(splits)

# dist: contains the dictionary of (category, value) pairs for how many samples to keep in the set (if available).  
# (-1, def) def is the new default value for all categories that are not listed in the dictionary
# If key -1 does not appear in the dictionary, the default value is ALL (large int)
# ditch: contains a list of all categories to be removed from the set completely. 
# If the list contains -1, the selection is inverted so all categories that do not appear 
# in the list are ditched. ditch and dist can be used simultaneously
class FlowsSubset(Subset):
    # Parse subset configuration from json file
    def parse(config_file, key, index=-1):
        assert os.path.isfile(config_file)
        with open(config_file, 'r') as f:
            config = json.load(f)

        # If config file contains multiple configs, select given index
        if index == -1:
            dist_string = config[key]["dist"]
            ditch = config[key]["ditch"]
        elif index >= 0:
            dist_string = config[str(index)][key]["dist"]
            ditch = config[str(index)][key]["ditch"]
        else:
            print(f'Invalid subset config index: {index}')

        # Cast string to int for dist dictionary
        dist = {}
        for k, v in dist_string.items():
            dist[int(k)] = v

        return dist, ditch

    # parse config from file an stringify config
    def subset_string(dist={}, ditch=[], config_file=None, config_index=-1):
        if not config_file is None:
            dist, ditch = FlowsSubset.parse(config_file, config_index)
        return FlowsSubset.string(dist, ditch)

    # stringify config
    def string(dist, ditch):
        subset_string = ''

        # stringify class distribution
        if len(dist.items()) > 0:
            subset_string += '_dist['
            for c,v in dist.items():
                subset_string += f'{c}:{v},'
            subset_string = subset_string[:-1] + ']'

        # stringify ditch list
        if len(ditch) > 0:
            subset_string += '_ditch['
            for c in ditch:
                subset_string += f'{c},'
            subset_string = subset_string[:-1] + ']'

        return subset_string

    def __init__(self, flows_dataset, mapping, dist={}, ditch=[], config_file=None, key="TRAIN", config_index=-1):
        self.mapping = mapping
        if not config_file is None:
            dist, ditch = FlowsSubset.parse(config_file, key, config_index)
        
        if -1 in [c for c, _ in dist.items()]:
            default = dist[-1]
        else:
            # set to max int (there is no max int in python though so just a very big number)
            default = pow(2,32)

        # If ditch contains -1, ditch all categories which are not in the ditch list (inverted operation)
        subset_ditch = ditch
        if -1 in ditch:
            subset_ditch = [v for _,v in self.mapping.items()]
            ditch.remove(-1)
            for c in ditch:
                subset_ditch.remove(c)

        # Parse how many samples from each category should be collected
        self._dist = dist
        self._ditch = ditch
        self.subset_num = {}
        self.subset_count = {}
        self.subset_samples = []
        for _, val in self.mapping.items():
            if val in [c for c, _ in dist.items()]:
                self.subset_num[val] = dist[val]
            else:
                self.subset_num[val] = default
            self.subset_count[val] = 0

        # Set all categories that appear in the ditch list to 0
        for c in subset_ditch:
            self.subset_num[c] = 0

        assert sum([v for _, v in self.subset_num.items()]) > 0

        # Gather number of flows of each category from dataset according to subset_num
        print(f'Loading subset {key} with config {str(self)[1:]}...',end='')
        for idx, (_, _, cat) in enumerate(flows_dataset):
            c = cat[0].item()
            if self.subset_count[c] < self.subset_num[c]:
                self.subset_count[c] += 1
                self.subset_samples.append(idx)
        super().__init__(flows_dataset, self.subset_samples)
        print('done')
        print(f'Gathered for {key}: {self.subset_count}')

    def __str__(self):
        return FlowsSubset.string(self._dist, self._ditch)

    
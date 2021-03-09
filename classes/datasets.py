import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, RandomSampler
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import json
import os.path

def overwrite_manipulable_entries(seq, filler=-1):
    forward_direction = seq[0,5]
    wrong_direction = (seq[:,5]==forward_direction)
    seq[wrong_direction,:][:,3:5] = filler
    return seq

# Stack and compress sequences of different length in batch
def collate_flows(seqs):    
    flows, labels, categories = zip(*seqs)
    flows_len = [len(flow) for flow in flows]
    assert not 0 in flows_len
    padded_flows = torch.nn.utils.rnn.pad_sequence(flows)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels)
    padded_categories = torch.nn.utils.rnn.pad_sequence(categories)
    packed_padded_flows = torch.nn.utils.rnn.pack_padded_sequence(padded_flows, flows_len, enforce_sorted=False)
    return (padded_flows, packed_padded_flows), padded_labels, padded_categories

# Stack and compress sequences of different length in batch
def collate_flows_batch_first(seqs):    
    flows, labels, categories = zip(*seqs)
    flows_len = [len(flow) for flow in flows]
    assert not 0 in flows_len
    padded_flows = torch.nn.utils.rnn.pad_sequence(flows, batch_first=True)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    padded_categories = torch.nn.utils.rnn.pad_sequence(categories, batch_first=True)
    packed_padded_flows = torch.nn.utils.rnn.pack_padded_sequence(padded_flows, flows_len, batch_first=True, enforce_sorted=False)
    return (padded_flows, packed_padded_flows), padded_labels, padded_categories

class CAIA(Dataset):
    def __init__(self, csv_file):
        super().__init__()

        print('Loading csv file...', end='')

        # read csv file and load data into variables
        csv = pd.read_csv(csv_file)
        x = csv.iloc[2:, 8:34].values
        x[x != x] = -1
        y = csv.iloc[2:, 36].values

        # scale
        sc = StandardScaler()
        x = sc.fit_transform(x)

        # convert to torch tensors
        self.n_samples = len(y)
        self.X = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y)

        print('done')
        

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return self.X[i], self.y[i]

class Flows(Dataset):
    def __init__(self, data_pickle, cache=None, max_length=100, remove_changeable=False):
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
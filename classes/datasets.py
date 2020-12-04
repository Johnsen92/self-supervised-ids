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

def pad_label_sequence(labels, categories):
    max_seq_length = max([len(seq) for seq in labels])
    padded_labels = []
    padded_categories = []
    for label, category in zip(labels, categories):
        pad_length = max_seq_length - label.size()[0]
        if not pad_length == 0:
            lab_pad = torch.zeros(pad_length, dtype=torch.long)
            lab_pad.fill_(label[0].item())
            cat_pad = torch.zeros(pad_length, dtype=torch.long)
            cat_pad.fill_(category[0].item())
            padded_labels.append(torch.cat((label, lab_pad), 0))
            padded_categories.append(torch.cat((category, cat_pad), 0))
        else:
            padded_labels.append(label)
            padded_categories.append(category)

    return torch.stack(padded_labels), torch.stack(padded_categories)

# Stack and compress sequences of different length in batch
def collate_flows(seqs, things=(True, True, True)):
    batch_data = []
    batch_labels = []
    batch_categories = []
    for data, label, categorie in seqs:
        batch_data.append(data)
        batch_labels.append(label)
        batch_categories.append(categorie)

    padded_data = torch.nn.utils.rnn.pad_sequence(batch_data, batch_first=True)
    #padded_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True)
    #padded_categories = torch.nn.utils.rnn.pad_sequence(batch_categories, batch_first=True)
    padded_labels, padded_categories = pad_label_sequence(batch_labels, batch_categories)
    return padded_data, padded_labels, padded_categories


class FlowBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch_data = []
        batch_labels = []
        batch_categories = []
        for (data, label, categorie) in self.dataset:
            batch_data.append(data)
            batch_labels.append(label)
            batch_categories.append(categorie)
            if len(batch_data) == self.batch_size:
                batch_data_padded = torch.nn.utils.rnn.pad_sequence(batch_data, batch_first=True)
                yield (batch_data_padded, torch.stack(batch_labels), torch.stack(batch_categories))
                batch_data = batch_labels = batch_categories = []
        if len(batch_data) > 0 and not self.drop_last:
            batch_data_padded = torch.nn.utils.rnn.pad_sequence(batch_data, batch_first=True)
            yield (batch_data_padded, torch.stack(batch_labels), torch.stack(batch_categories))
    
    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size 

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
    def __init__(self, data_pickle, cache=None):
        super().__init__()
        print('Loading data file...', end='')

        self.cache = cache
        self.data_pickle = data_pickle

        # Parameters
        maxLength = 100
        removeChangeable = False

        # Load pickled dataset
        with open (data_pickle, 'rb') as f:
            all_data = pickle.load(f)

        with open(data_pickle[:-7]+'_categories_mapping.json', 'r') as f:
            categories_mapping_content = json.load(f)
        categories_mapping, mapping = categories_mapping_content['categories_mapping'], categories_mapping_content['mapping']
        assert min(mapping.values()) == 0

        # Remove flows witch invalid IATs
        all_data = [item[:maxLength,:] for item in all_data if np.all(item[:,4]>=0)]
        if removeChangeable:
            all_data = [overwrite_manipulable_entries(item) for item in all_data]

        X = [item[:, :-2] for item in all_data]
        Y = [item[:, -1:] for item in all_data]
        print('done')

        print('Normalizing data...', end='')

        # Normalize data between -1 and 1
        if not self.cache == None and not self.cache.exists('norm'):
            catted_x = np.concatenate(X, axis=0)
            means = np.mean(catted_x, axis=0)
            stds = np.std(catted_x, axis=0)
            stds[stds==0.0] = 1.0
            
            # Store for future use
            cache.save('norm', (means, stds))
        else:
            (means, stds) = cache.load('norm')

        assert means.shape[0] == X[0].shape[-1], 'means.shape: {}, x.shape: {}'.format(means.shape, X[0].shape)
        assert stds.shape[0] == X[0].shape[-1], 'stds.shape: {}, x.shape: {}'.format(stds.shape, X[0].shape)
        assert not (stds==0).any(), 'stds: {}'.format(stds)
        
        print('done')
        

        # Store in class members
        self.x = [(item-means)/stds for item in X]
        #self.y = [item[:, -1:] for item in all_data]
        self.y = [np.zeros(len(item)) if item[0]==0.0 else np.ones(len(item)) for item in Y]
        self.categories = [item[:, -2:-1] for item in all_data]
        #self.y = [0 if item[0]==0.0 else 1 for item in Y]
        #self.categories = [item[0, -2:-1] for item in all_data]
        self.categories_mapping = categories_mapping
        self.n_samples = len(self.x)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        #tensor_categories = torch.LongTensor(self.categories[i]).squeeze() if len(self.categories[i]) > 1 else torch.LongTensor(self.categories[i])
        tensor_categories = torch.reshape(torch.LongTensor(self.categories[i]), (-1,))
        tensor_labels = torch.LongTensor(self.y[i])
        tensor_data = torch.FloatTensor(self.x[i])
        data, labels, categories = tensor_data, tensor_labels, tensor_categories
        return data, labels, categories

    def getCategories(self):
        return self.categories
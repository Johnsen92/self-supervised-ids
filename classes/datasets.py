import torch
import pandas as pd
from torch.utils.data import Dataset
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

        print("done")
        

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return self.X[i], self.y[i]

class Flows(Dataset):
    def __init__(self, pickle_file):
        super().__init__()
        print('Loading data file...', end='')

        # parameters
        maxLength = 100
        removeChangeable = False

        # load pickled dataset
        with open (pickle_file, "rb") as f:
            all_data = pickle.load(f)

        with open(pickle_file[:-7]+"_categories_mapping.json", "r") as f:
            categories_mapping_content = json.load(f)
        categories_mapping, mapping = categories_mapping_content["categories_mapping"], categories_mapping_content["mapping"]
        assert min(mapping.values()) == 0

        # a few flows have have invalid IATs due to dataset issues. Sort those out.
        all_data = [item[:maxLength,:] for item in all_data if np.all(item[:,4]>=0)]
        if removeChangeable:
            all_data = [overwrite_manipulable_entries(item) for item in all_data]
        #random.shuffle(all_data)
        # print("lens", [len(item) for item in all_data])
        X = [item[:, :-2] for item in all_data]
        Y = [item[:, -1:] for item in all_data]
        print('done')

        print('Normalizing data...', end='')

        # normalize data
        chache_file_name = pickle_file[:-7]+"_normalization_data.pickle"
        if not os.path.isfile(chache_file_name):
            catted_x = np.concatenate(X, axis=0)
            means = np.mean(catted_x, axis=0)
            stds = np.std(catted_x, axis=0)
            stds[stds==0.0] = 1.0
            
            # store for future use
            with open(chache_file_name, "wb") as f:
                f.write(pickle.dumps((means, stds)))
        else:
            with open(chache_file_name, "rb") as f:
                means, stds = pickle.load(f)

        assert means.shape[0] == X[0].shape[-1], "means.shape: {}, x.shape: {}".format(means.shape, X[0].shape)
        assert stds.shape[0] == X[0].shape[-1], "stds.shape: {}, x.shape: {}".format(stds.shape, X[0].shape)
        assert not (stds==0).any(), "stds: {}".format(stds)
        
        print('done')
        

        # store in object members
        self.x = [(item-means)/stds for item in X]
        self.y = Y
        self.categories = [item[:, -2:-1] for item in all_data]
        self.categories_mapping = categories_mapping
        self.n_samples = len(self.x)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        data, labels, categories = torch.FloatTensor(self.x[i]), torch.FloatTensor(self.y[i]), torch.FloatTensor(self.categories[i])
        return data, labels, categories
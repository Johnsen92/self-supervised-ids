import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class CAIA(Dataset):
    def __init__(self, csv_file):
        super().__init__()
        
        # read csv file and load data into variables
        csv = pd.read_csv(csv_file)
        x = csv.iloc[1:, 1:35].values
        y = csv.iloc[1:, 37].values

        # scale
        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = y

        # convert to torch tensors
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train)
        self.n_samples = len(self.y_train)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return self.X_train[i], self.y_train[i]
import torch as torch
from torch import nn
from torch.nn import functional as F

class LinResNet(nn.Module):
    def __init__(self, nb_features):
        super().__init__()
        self.l1 = nn.Linear(nb_features, nb_features)
        self.l2 = nn.Linear(nb_features, nb_features)
        self.l3 = nn.Linear(nb_features, 10)
        self.do = nn.Dropout(0.1)

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        do = self.do(h2 + h1)
        logits = self.l3(do)
        return logits
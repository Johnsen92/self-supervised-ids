import torch as torch
from torch import nn
from torch.nn import functional as F

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_size, device):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # x is of shape batch_size x seq_len x input_size
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_init = torch.zeros(num_layers, batch_size, hidden_size).to(device)
        self.cell_init = torch.zeros(num_layers, batch_size, hidden_size).to(device)

    def forward(self, x):
        out, _ = self.lstm(x, (self.hidden_init, self.cell_init))
        # out is of shape batch_size x seq_len x input_size
        out = self.fc(out)
        return out
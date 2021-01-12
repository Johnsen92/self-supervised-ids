import torch as torch
from torch import nn
from torch.nn import functional as F

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_size, device):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self._lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # x is of shape (batch_size x seq_len x input_size)
        self._fc = nn.Linear(hidden_size, output_size)
        self._hidden_init = torch.zeros(num_layers, batch_size, hidden_size).to(device)
        self._cell_init = torch.zeros(num_layers, batch_size, hidden_size).to(device)

    def forward(self, x):
        out, _ = self._lstm(x, (self._hidden_init, self._cell_init))
        # out is of shape (batch_size x seq_len x input_size)
        out = self._fc(out)
        return out

class ChainLSTM(LSTM):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_size, device, prev_model):
        super().__init__(input_size, hidden_size, output_size, num_layers, batch_size, device)
        self.prev_model = prev_model

    def forward(self, x):
        s1 = self.prev_model(x)
        s2, _ = self._lstm(s1, (self._hidden_init, self._cell_init))
        out = self._fc(s2)
        return out

class PretrainableLSTM(LSTM):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_size, device):
        super().__init__(input_size, hidden_size, output_size, num_layers, batch_size, device)
        self.pretraining = True

    def forward(self, x):
        if self.pretraining:
            out, _ = self._lstm(x, (self._hidden_init, self._cell_init))
            out = self._fc(out) 
        else:
            out, _ = self._lstm(x, (self._hidden_init, self._cell_init))
        return out
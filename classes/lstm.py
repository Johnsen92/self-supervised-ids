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
        s1, _ = self._lstm(x, (self._hidden_init, self._cell_init))
        s2, _ = torch.nn.utils.rnn.pad_packed_sequence(s1, batch_first=True)
        # out is of shape (batch_size x seq_len x output_size)
        out = self._fc(s2)
        return out

class ChainLSTM(LSTM):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_size, device, prev_model):
        super().__init__(input_size, hidden_size, output_size, num_layers, batch_size, device)
        self.prev_model = prev_model

    def forward(self, x):
        s1 = self.prev_model(x)
        s1_lens = [len(item) for item in s1]
        s2 = torch.nn.utils.rnn.pack_padded_sequence(s1, s1_lens, batch_first=True, enforce_sorted=False)
        s3, _ = self._lstm(s2, (self._hidden_init, self._cell_init))
        s4, _ = torch.nn.utils.rnn.pad_packed_sequence(s3, batch_first=True)
        out = self._fc(s4)
        return out

class PretrainableLSTM(LSTM):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_size, device):
        super().__init__(input_size, hidden_size, output_size, num_layers, batch_size, device)
        self.pretraining = True

    def forward(self, x):
        out, _ = self._lstm(x, (self._hidden_init, self._cell_init))
        if self.pretraining:
            s1, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            out = self._fc(s1) 
        return out
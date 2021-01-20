import torch as torch
from torch import nn
from torch.nn import functional as F

# Fill padded section of output with last unpadded LSTM output in sequence
def pad_packed_output_sequence(packed_output):
    output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
    padded_output = output
    batch_size = output.size()[0]
    for i in range(batch_size):
        padded_output[i, output_lengths[i]:, :] = output[i, output_lengths[i]-1, :]
    return padded_output

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
        # out is of shape (batch_size x seq_len x output_size)
        out = self._fc(s1)
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
        self._fc_pretraining = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self._lstm(x, (self._hidden_init, self._cell_init))
        out = pad_packed_output_sequence(out)
        if self.pretraining:
            out = self._fc_pretraining(out)
        else:
            out = self._fc(out)
        return out
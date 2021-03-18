import torch as torch
from torch import nn

# Fill padded section of output with last unpadded LSTM output in sequence
def pad_packed_output_sequence(packed_output):
    output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
    padded_output = output
    batch_size = output.size()[0]
    for i in range(batch_size):
        padded_output[i, output_lengths[i]:, :] = output[i, output_lengths[i]-1, :]
    return padded_output

class LSTM(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        output_size, 
        num_layers
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self._lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # x is of shape (batch_size x seq_len x input_size)
        self._fc = nn.Linear(hidden_size, output_size)

    def forward(self, src_packed):
        _, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(src_packed, batch_first=True)
        batch_size = len(seq_lens)
        hidden_init = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cell_init = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        s1, _ = self._lstm(src_packed, (hidden_init, cell_init))
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
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        output_size, 
        num_layers
    ):
        super().__init__(input_size, hidden_size, output_size, num_layers)
        self.pretraining = True
        self._fc_pretraining = nn.Linear(hidden_size, input_size)

    def forward(self, src_packed):
        _, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(src_packed, batch_first=True)
        batch_size = len(seq_lens)
        current_device = src_packed.data.get_device()

        hidden_init = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(current_device)
        cell_init = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(current_device)

        out, _ = self._lstm(src_packed, (hidden_init, cell_init))
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        if self.pretraining:
            out = self._fc_pretraining(out)
        else:
            out = self._fc(out)
        return out

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
        self._lstm.flatten_parameters()
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
        #self._lstm.flatten_parameters()
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

class AutoEncoderLSTM(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        output_size, 
        num_layers
    ):
        super().__init__()
        self._encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self._decoder_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self._encoder_fc = nn.Linear(hidden_size, output_size)
        self._decoder_fc = nn.Linear(hidden_size, input_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pretraining = True

    def reverse_seq_order(self, seqs_packed):
        seqs, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(seqs_packed, batch_first=True)
        current_device = seqs.get_device()
        seqs_reversed = torch.zeros(seqs.size(), dtype=torch.float32)
        for i, seq_len in enumerate(seq_lens):
            rev_idx = [i for i in range(seq_len-1, -1, -1)]
            rev_idx = torch.LongTensor(rev_idx).to(current_device)
            seqs_reversed[i, :seq_len, :] = torch.index_select(seqs[i, :seq_len, :],0,rev_idx)
        seqs_reversed_packed = torch.nn.utils.rnn.pack_padded_sequence(seqs_reversed, seq_lens, batch_first=True, enforce_sorted=False).to(current_device)
        return seqs_reversed_packed

    def forward(self, src_packed):

        # Get batch_size
        src, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(src_packed, batch_first=True)
        batch_size = len(seq_lens)

        # Get current device
        current_device = src.get_device()

        # Zero-init cell and hidden state of encoder LSTM
        encoder_hidden_init = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(current_device)
        encoder_cell_init = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(current_device)

        # Forward pass for encoder LSTM
        #self._encoder_lstm.flatten_parameters()
        encoder_out, (h_state, c_state) = self._encoder_lstm(src_packed, (encoder_hidden_init, encoder_cell_init))

        if self.pretraining:
            decoder_hidden_init = torch.zeros(h_state.size()).to(current_device)
            decoder_cell_init = torch.zeros(c_state.size()).to(current_device)
            decoder_hidden_init[0, :, :] = h_state[-1, :, :]
            decoder_cell_init[0, :, :] = c_state[-1, :, :]

            #decoder_hidden_init = h_state[-1,:,:].expand(3, h_state.shape[1], h_state.shape[2]).to(current_device)
            #decoder_cell_init = c_state[-1,:,:].expand(3, c_state.shape[1], c_state.shape[2]).to(current_device)

            src_packed_reverse = self.reverse_seq_order(src_packed)
            self._decoder_lstm.flatten_parameters()
            decoder_out, _ = self._decoder_lstm(src_packed_reverse, (decoder_hidden_init, decoder_cell_init))

            decoder_out_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(decoder_out, batch_first=True)
            out = self._decoder_fc(decoder_out_unpacked)
        else:
            encoder_out_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_out, batch_first=True)
            out = self._encoder_fc(encoder_out_unpacked)
        return out
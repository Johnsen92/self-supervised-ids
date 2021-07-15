import torch as torch
from torch import nn
import math

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
        _, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(src_packed, batch_first=True)
        batch_size = len(seq_lens)
        current_device = src_packed.data.get_device()

        hidden_init = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(current_device)
        cell_init = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(current_device)

        out, (hidden_state, cell_state) = self._lstm(src_packed, (hidden_init, cell_init))
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        neurons = out.detach()
        if self.pretraining:
            out = self._fc_pretraining(out)
        else:
            out = self._fc(out)
        return out, neurons, (hidden_state, cell_state)

class CompositeLSTM(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        output_size, 
        num_layers,
        teacher_forcing=False
    ):
        super().__init__()
        self._encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self._past_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self._future_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self._encoder_fc = nn.Linear(hidden_size, output_size)
        self._decoder_fc = nn.Linear(hidden_size, input_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pretraining = True
        self.teacher_forcing = teacher_forcing

    def split_input(self, seqs, seq_lens):
        current_device = seqs.get_device()
        max_half_seq_len = math.ceil(seqs.size()[1] / 2.)
        seqs_past = torch.zeros(seqs.size()[0], max_half_seq_len, seqs.size()[2], dtype=torch.float32).to(current_device)
        seqs_past_reversed = torch.zeros(seqs.size()[0], max_half_seq_len, seqs.size()[2], dtype=torch.float32).to(current_device)
        seqs_future = torch.zeros(seqs.size()[0], seqs.size()[1] - max_half_seq_len, seqs.size()[2], dtype=torch.float32).to(current_device)
        seq_lens_past = []
        seq_lens_future = []
        for i, seq_len in enumerate(seq_lens):
            # Calculate new sequence lengths
            half_seq_len = math.ceil(seq_len / 2.)
            seq_lens_past.append(half_seq_len)
            seq_lens_future.append(seq_len.item() - half_seq_len)

            # Get indices for past, reverse past and future
            past_idx = [j for j in range(seq_lens_past[i])]
            past_reversed_idx = [j for j in range(seq_lens_past[i]-1, -1, -1)]
            future_idx = [j for j in range(seq_lens_past[i], seq_len)]
            past_idx = torch.LongTensor(past_idx).to(current_device)
            past_reversed_idx = torch.LongTensor(past_reversed_idx).to(current_device)
            future_idx = torch.LongTensor(future_idx).to(current_device)

            # Select indices
            seqs_past[i, :seq_lens_past[i], :] = torch.index_select(seqs[i, :seq_len, :], 0, past_idx)
            seqs_past_reversed[i, :seq_lens_past[i], :] = torch.index_select(seqs[i, :seq_len, :], 0, past_reversed_idx)
            seqs_future[i, :seq_lens_future[i], :] = torch.index_select(seqs[i, :seq_len, :], 0, future_idx)
        
        # Pack sequences
        seqs_past_packed = torch.nn.utils.rnn.pack_padded_sequence(seqs_past, seq_lens_past, batch_first=True, enforce_sorted=False).to(current_device)
        #seqs_past_reversed_packed = torch.nn.utils.rnn.pack_padded_sequence(seqs_past_reversed, seq_lens_past, batch_first=True, enforce_sorted=False).to(current_device)
        #seqs_future_packed = torch.nn.utils.rnn.pack_padded_sequence(seqs_future, seq_lens_future, batch_first=True, enforce_sorted=False).to(current_device)
        return seqs_past_packed, (seqs_past_reversed, seq_lens_past), (seqs_future, seq_lens_future)

    def merge_outputs(self, out_past, neurons_past, seq_lens_past, out_future, neurons_future, seq_lens_future):
        #seqs_past_reversed, seq_lens_past = torch.nn.utils.rnn.pad_packed_sequence(past_reversed_packed, batch_first=True)
        #seqs_future, seq_lens_future = torch.nn.utils.rnn.pad_packed_sequence(future_packed, batch_first=True)
        current_device = out_past.get_device()
        seqs_out = torch.zeros(out_past.size()[0], out_past.size()[1] + out_future.size()[1], out_past.size()[2], dtype=torch.float32)
        neurons_out = torch.zeros(neurons_past.size()[0], neurons_past.size()[1] + neurons_future.size()[1], neurons_past.size()[2], dtype=torch.float32)
        seq_lens_out = []
        assert out_past.size()[1] + out_future.size()[1] == neurons_past.size()[1] + neurons_future.size()[1], 'Seq lengths of neurons and outputs don`t match'
        assert neurons_past.size()[0] == out_past.size()[0], 'Batch size of neurons and output don`t match'
        for i, (seq_len_past, seq_len_future) in enumerate(zip(seq_lens_past, seq_lens_future)):
            # Calculate new sequence lengths
            seq_len = seq_len_past + seq_len_future
            seq_lens_out.append(seq_len)

            # Get indices for past, reverse past and future
            past_idx = [i for i in range(seq_len_past-1, -1, -1)]
            future_idx = [i for i in range(seq_len_future)]
            past_idx = torch.LongTensor(past_idx).to(current_device)
            future_idx = torch.LongTensor(future_idx).to(current_device)

            # Select indices for output sequences
            seqs_past_masked = torch.index_select(out_past[i, :seq_len_past, :], 0, past_idx)
            seqs_future_masked = torch.index_select(out_future[i, :seq_len_future, :], 0, future_idx)
            seqs_out[i, :seq_len, :] = torch.cat((seqs_past_masked, seqs_future_masked), 0)
            
            # Select indices for neurons
            neurons_past_masked = torch.index_select(neurons_past[i, :seq_len_past, :], 0, past_idx)
            neurons_future_masked = torch.index_select(neurons_future[i, :seq_len_future, :], 0, future_idx)
            neurons_out[i, :seq_len, :] = torch.cat((neurons_past_masked, neurons_future_masked), 0)
        
        return seqs_out.to(current_device), neurons_out.to(current_device), seq_lens_out

    def forward(self, src_packed):

        # Get batch_size
        src, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(src_packed, batch_first=True)
        batch_size = len(seq_lens)

        # Split input into past, future and reversed past
        seqs_past_packed, (seqs_past_reversed, seq_lens_past), (seqs_future, seq_lens_future) = self.split_input(src, seq_lens)

        # Get current device
        current_device = src.get_device()

        # Zero-init cell and hidden state of encoder LSTM
        encoder_hidden_init = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(current_device)
        encoder_cell_init = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(current_device)

        # Forward pass for encoder LSTM
        if self.pretraining:
            encoder_out, (hidden_state, cell_state) = self._encoder_lstm(seqs_past_packed, (encoder_hidden_init, encoder_cell_init))
        else:
            encoder_out, (hidden_state, cell_state) = self._encoder_lstm(src_packed, (encoder_hidden_init, encoder_cell_init))

        if self.pretraining:
            past_hidden_init = future_hidden_init = torch.zeros(hidden_state.size()).to(current_device)
            past_cell_init = future_cell_init =  torch.zeros(cell_state.size()).to(current_device)
            past_hidden_init[0, :, :] = future_hidden_init[0, :, :] = hidden_state[-1, :, :]
            past_cell_init[0, :, :] = future_cell_init[0, :, :] = cell_state[-1, :, :]

            #decoder_hidden_init = h_state[-1,:,:].expand(3, h_state.shape[1], h_state.shape[2]).to(current_device)
            #decoder_cell_init = c_state[-1,:,:].expand(3, c_state.shape[1], c_state.shape[2]).to(current_device)

            self._past_lstm.flatten_parameters()
            self._future_lstm.flatten_parameters()

            # Reversed past decoder
            past_in = past_out = torch.zeros((seqs_past_reversed.size()[0],1,seqs_past_reversed.size()[2])).to(current_device)
            past_neurons = torch.zeros((seqs_past_reversed.size()[0],1,self.hidden_size)).to(current_device)
            (past_hidden, past_cell) = (past_hidden_init, past_cell_init)
            for i in range(max(seq_lens_past)):
                out, (past_hidden, past_cell) = self._past_lstm(past_in, (past_hidden, past_cell))
                fc_out = self._decoder_fc(out)
                if i == 0:
                    past_neurons = out.detach()
                    past_out = fc_out
                else:
                    past_neurons = torch.cat((past_neurons, out), 1)
                    past_out = torch.cat((past_out, fc_out), 1)

                # If teacher forcing is active, take ground trough as input, otherwise take prediction of last stage as input
                if self.teacher_forcing:
                    past_in = seqs_past_reversed[:,i,:].unsqueeze(1)
                else:
                    past_in = fc_out.detach()

            # Future decoder
            future_in = future_out = torch.zeros((seqs_future.size()[0], 1, seqs_future.size()[2])).to(current_device)
            future_neurons = torch.zeros((seqs_future.size()[0], 1, self.hidden_size)).to(current_device)
            (future_hidden, future_cell) = (future_hidden_init, future_cell_init)
            for i in range(max(seq_lens_future)):
                out, (future_hidden, future_cell) = self._future_lstm(future_in, (future_hidden, future_cell))
                fc_out = self._decoder_fc(out)
                if i == 0:
                    future_neurons = out.detach()
                    future_out = fc_out
                else:
                    future_neurons = torch.cat((future_neurons, out), 1)
                    future_out = torch.cat((future_out, fc_out), 1)
                
                # If teacher forcing is active, take ground trough as input, otherwise take prediction of last stage as input
                if self.teacher_forcing:
                    future_in = seqs_future[:,i,:].unsqueeze(1)
                else:
                    future_in = fc_out.detach()

            # Get final hidden state of future LSTM
            (hidden_state, cell_state) = (future_hidden, future_cell)

            #past_out, _ = self._past_lstm(src_past_reversed_packed, (past_hidden_init, past_cell_init))
            #future_out, _ = self._future_lstm(src_future_packed, (future_hidden_init, future_cell_init))
            out, neurons, _ = self.merge_outputs(past_out, past_neurons, seq_lens_past, future_out, future_neurons, seq_lens_future)
        else:
            encoder_out_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_out, batch_first=True)
            out = self._encoder_fc(encoder_out_unpacked)
            neurons = encoder_out_unpacked.detach()
        return out, neurons, (hidden_state, cell_state)

class AutoEncoderLSTM(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        output_size, 
        num_layers,
        teacher_forcing=False,
        identity=True
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
        self.teacher_forcing = teacher_forcing
        self.identity = identity

    def reverse_seq_order(self, seqs_packed):
        seqs, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(seqs_packed, batch_first=True)
        current_device = seqs.get_device()
        seqs_reversed = torch.zeros(seqs.size(), dtype=torch.float32)
        for i, seq_len in enumerate(seq_lens):
            rev_idx = [i for i in range(seq_len-1, -1, -1)]
            rev_idx = torch.LongTensor(rev_idx).to(current_device)
            seqs_reversed[i, :seq_len, :] = torch.index_select(seqs[i, :seq_len, :],0,rev_idx)
        #seqs_reversed_packed = torch.nn.utils.rnn.pack_padded_sequence(seqs_reversed, seq_lens, batch_first=True, enforce_sorted=False).to(current_device)
        return seqs_reversed.to(current_device)

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
        encoder_out, (hidden_state, cell_state) = self._encoder_lstm(src_packed, (encoder_hidden_init, encoder_cell_init))

        if self.pretraining:
            decoder_hidden_init = torch.zeros(hidden_state.size()).to(current_device)
            decoder_cell_init = torch.zeros(cell_state.size()).to(current_device)
            decoder_hidden_init[0, :, :] = hidden_state[-1, :, :]
            decoder_cell_init[0, :, :] = cell_state[-1, :, :]

            #decoder_hidden_init = h_state[-1,:,:].expand(3, h_state.shape[1], h_state.shape[2]).to(current_device)
            #decoder_cell_init = c_state[-1,:,:].expand(3, c_state.shape[1], c_state.shape[2]).to(current_device)

            src_reverse = self.reverse_seq_order(src_packed)
            self._decoder_lstm.flatten_parameters()

            decoder_in = decoder_out = torch.zeros((src.size()[0],1,src.size()[2])).to(current_device)
            decoder_neurons = torch.zeros((src.size()[0],1,self.hidden_size)).to(current_device)
            decoder_in = src_reverse[:,0,:].unsqueeze(1)
            (decoder_hidden, decoder_cell) = (decoder_hidden_init, decoder_cell_init)
            for i in range(src.size()[1]):
                if self.identity:
                    decoder_in = src_reverse[:,i,:].unsqueeze(1)
                out, (decoder_hidden, decoder_cell) = self._decoder_lstm(decoder_in, (decoder_hidden, decoder_cell))
                fc_out = self._decoder_fc(out)
                if i == 0:
                    decoder_neurons = out.detach()
                    decoder_out = fc_out.detach()
                else:
                    decoder_neurons = torch.cat((decoder_neurons, out), 1)
                    decoder_out = torch.cat((decoder_out, fc_out), 1)

                # If teacher forcing is active, take ground trough as input, otherwise take prediction of last stage as input
                if self.teacher_forcing and not self.identity:
                    decoder_in = src_reverse[:,i,:].unsqueeze(1)
                elif not self.identity:
                    decoder_in = fc_out.detach()

            (hidden_state, cell_state) = (decoder_hidden, decoder_cell)

            #decoder_out_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(decoder_out, batch_first=True)
            out = decoder_out
            neurons = decoder_neurons
        else:
            encoder_out_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_out, batch_first=True)
            out = self._encoder_fc(encoder_out_unpacked)
            neurons = encoder_out_unpacked.detach()
        return out, neurons, (hidden_state, cell_state)
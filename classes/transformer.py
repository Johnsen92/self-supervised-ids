import torch
from torch import nn

class Transformer(nn.Module):
    def __init__(
        self,
        input_size,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        self.src_position_embedding = nn.Embedding(max_len, input_size)
        self.trg_position_embedding = nn.Embedding(max_len, input_size)

        self.device = device
        self.transformer = nn.Transformer(
            input_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, batch_size, _ = src.shape
        trg_seq_length, batch_size, _ = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, batch_size)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, batch_size)
            .to(self.device)
        )

        embed_src = self.dropout(
            (src + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (trg + self.trg_position_embedding(trg_positions))
        )

        #src_padding_mask = self.make_src_mask(seq_lens)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
             self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            #src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder,
        input_size,
        output_size,
        dropout,
        max_len,
        device,
    ):
        super(TransformerEncoder, self).__init__()
        self.encoder = encoder
        self.output_size = output_size
        self.device = device
        self.input_size = input_size
        self.fc = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.src_position_embedding = nn.Embedding(max_len, input_size)

    def src_mask(self, size, seq_lens):
        mask = torch.ones(size[:2], dtype=torch.bool).transpose(0, 1)
        for index, length in enumerate(seq_lens):
            mask[index, :length] = False
        return mask

    def logits(self, output, seq_lens):
        logits = torch.zeros(output.size()[1], dtype=torch.float)
        for index, length in enumerate(seq_lens):
            logits[index] = torch.sum(output[:length, index, :])/length
        return logits

    def forward(self, src_packed):

        # Unpack data
        src, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(src_packed)

        # Get source mask to only consider non-padded data
        mask = self.src_mask(src.size(), seq_lens).to(self.device)

        # Create positional encoding
        src_seq_length, batch_size, _ = src.shape    
        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, batch_size)
            .to(self.device)
        )
        embed_src = self.dropout((src + self.src_position_embedding(src_positions)))

        # Forward propagation
        out = self.encoder(embed_src, src_key_padding_mask=mask)

        # Filter out NaNs
        out = out.masked_fill(torch.isnan(out), 0)

        # Project input_size to output_size
        out = self.fc(out)
        
        # Create logits as average of seq outputs
        out = self.logits(out, seq_lens).to(self.device)
        return out



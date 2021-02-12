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

    def forward(self, src, mask):
        src_seq_length, batch_size, _ = src.shape

        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, batch_size)
            .to(self.device)
        )

        embed_src = self.dropout((src + self.src_position_embedding(src_positions)))

        out = self.encoder(embed_src, src_key_padding_mask=mask)
        out = out.masked_fill(torch.isnan(out), 0)
        out = self.fc(out)
        return out



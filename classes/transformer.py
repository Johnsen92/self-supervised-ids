import torch
from torch import nn

class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, 1)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, batch_size, src_size = src.shape
        trg_seq_length, batch_size, trg_size = trg.shape

        # src_positions = (
        #     torch.arange(0, batch_size)
        #     .unsqueeze(1)
        #     .unsqueeze(2)
        #     .expand(batch_size, src_seq_length, src_size)
        #     .to(self.device)
        # )

        # trg_positions = (
        #     torch.arange(0, batch_size)
        #     .unsqueeze(1)
        #     .unsqueeze(2)
        #     .expand(batch_size, trg_seq_length, trg_size)
        #     .to(self.device)
        # )

        # print(src.shape)
        # print(src_positions.shape)

        # embed_src = self.dropout(
        #     (src + self.src_position_embedding(src_positions))
        # )
        # embed_trg = self.dropout(
        #     (trg + self.trg_position_embedding(trg_positions))
        # )

        # src_padding_mask = self.make_src_mask(src)
        # trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
        #     self.device
        # )

        out = self.transformer(
            src,
            trg,
            #src_key_padding_mask=src_padding_mask,
            #tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out
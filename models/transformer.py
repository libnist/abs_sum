import torch
from torch import nn

from ..modules.blocks import TripleEmbeddingBlock
from ..models import get_attn_mask


class Transformer(nn.Module):

    def __init__(self,
                 d_model=64,
                 dim_feedforward=128,
                 nhead=4,
                 num_layers=3,
                 source_vs=8_000,
                 target_vs=8_000,
                 src_padding_idx=0,
                 tgt_padding_idx=0,
                 dropout=0.1):
        super().__init__()

        self.src_padding_idx = src_padding_idx
        self.tgt_padding_idx = tgt_padding_idx

        self.enc_embed = TripleEmbeddingBlock(num_word_embeddings=source_vs,
                                              embedding_dim=d_model,
                                              padding_index=src_padding_idx)

        self.dec_embed = TripleEmbeddingBlock(num_word_embeddings=target_vs,
                                              embedding_dim=d_model,
                                              padding_index=tgt_padding_idx)

        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers,
                                          dim_feedforward=dim_feedforward,
                                          batch_first=True,
                                          dropout=dropout)

        self.classifier = nn.Linear(in_features=d_model,
                                    out_features=target_vs)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def key_paddding_mask(self, input, pad_id, device):
        return (input == pad_id).to(device)

    def forward(self, src, tgt):
        device = src.device

        src_key_padding_mask = self.key_paddding_mask(src,
                                                      self.src_padding_idx,
                                                      device=device)

        tgt_mask = get_attn_mask(tgt.shape[-1],
                                 device)

        tgt_key_padding_mask = self.key_paddding_mask(tgt,
                                                      self.tgt_padding_idx,
                                                      device=device)

        enc_embeds = self.enc_embed(src)
        dec_embeds = self.dec_embed(tgt)

        transformer_out = self.transformer(
            src=enc_embeds,
            tgt=dec_embeds,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        return self.classifier(transformer_out)


class Uformer(nn.Module):

    def __init__(self,
                 d_model=64,
                 dim_feedforward=128,
                 nhead=4,
                 num_layers=3,
                 source_vs=8_000,
                 target_vs=8_000,
                 src_padding_idx=0,
                 tgt_padding_idx=0,
                 dropout=0.1):
        super().__init__()

        self.src_padding_idx = src_padding_idx
        self.tgt_padding_idx = tgt_padding_idx

        self.enc_embed = TripleEmbeddingBlock(num_word_embeddings=source_vs,
                                              embedding_dim=d_model,
                                              padding_index=src_padding_idx)

        self.dec_embed = TripleEmbeddingBlock(num_word_embeddings=target_vs,
                                              embedding_dim=d_model,
                                              padding_index=tgt_padding_idx)

        self.encoders = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=d_model,
                                        nhead=nhead,
                                        dim_feedforward=dim_feedforward,
                                        dropout=dropout,
                                        batch_first=True)
             for _ in range(num_layers)]
        )

        self.decoders = nn.ModuleList(
            [nn.TransformerDecoderLayer(d_model=d_model,
                                        nhead=nhead,
                                        dim_feedforward=dim_feedforward,
                                        dropout=dropout,
                                        batch_first=True)
             for _ in range(num_layers)]
        )

        self.classifier = nn.Linear(in_features=d_model,
                                    out_features=target_vs)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def key_paddding_mask(self, input, pad_id, device):
        return (input == pad_id).to(device)

    def forward(self, src, tgt):
        device = src.device

        src_key_padding_mask = self.key_paddding_mask(src,
                                                      self.src_padding_idx,
                                                      device=device)

        tgt_mask = get_attn_mask(tgt.shape[-1],
                                 device)

        tgt_key_padding_mask = self.key_paddding_mask(tgt,
                                                      self.tgt_padding_idx,
                                                      device=device)

        enc_embeds = self.enc_embed(src)
        dec_embeds = self.dec_embed(tgt)

        enc_outs = [self.encoders[0](
            src=enc_embeds,
            src_key_padding_mask=src_key_padding_mask)
        ]

        for encoder in self.encoders[1:]:
            enc_outs.append(encoder(src=enc_outs[-1],
                                    src_key_padding_mask=src_key_padding_mask))

        output = self.decoders[0](tgt=dec_embeds,
                                  memory=enc_outs[0],
                                  tgt_mask=tgt_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=src_key_padding_mask)

        for i, decoder in enumerate(self.decoders[1:]):
            output = decoder(tgt=output,
                             memory=enc_outs[i],
                             tgt_mask=tgt_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=src_key_padding_mask)

        return self.classifier(output)

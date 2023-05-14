# Import libraries
import torch
from torch import nn

import numpy as np

from ..modules.layers import FnetEncoderLayer
from ..modules.blocks import TripleEmbeddingBlock


class FnetModel(nn.Module):
    def __init__(self,
                 model_dim: int,
                 n_head: int,
                 dim_feedforward: int,
                 num_layers: int,
                 src_vs: int,
                 tgt_vs: int,
                 src_padding_index: int,
                 tgt_padding_index: int,
                 dropout: float = 0.1) -> nn.Module:
        super().__init__()
        
        self.src_padding_index = src_padding_index
        self.tgt_padding_index = tgt_padding_index

        self.encoder_embedding = TripleEmbeddingBlock(
            num_word_embeddings=src_vs,
            embedding_dim=model_dim,
            padding_index=src_padding_index,
        )

        # Create the FnetCNNEncoder
        self.encoder = nn.Sequential(
            *[FnetEncoderLayer(model_dim=model_dim,
                               extend_dim=dim_feedforward,
                               dropout=dropout)
              for _ in range(num_layers)]
        )

        # Create the Decoder
        self.decoder_embedding = TripleEmbeddingBlock(
            num_word_embeddings=tgt_vs,
            embedding_dim=model_dim,
            padding_index=tgt_padding_index
        )

        decoder = nn.TransformerDecoderLayer(d_model=model_dim,
                                             nhead=n_head,
                                             dim_feedforward=dim_feedforward,
                                             dropout=dropout,
                                             batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder,
                                             num_layers=num_layers)

        # Create the output layer
        self.output_layer = nn.Linear(in_features=model_dim,
                                      out_features=tgt_vs)

    def forward(self,
                doc_tokens: torch.tensor,
                sum_tokens: torch.tensor) -> torch.tensor:
        
        device = doc_tokens.device
        
        memory_key_padding_mask = (doc_tokens == self.tgt_padding_index).to(device)
        tgt_key_padding_mask = (sum_tokens == self.src_padding_index).to(device)
        
        src_embed = self.encoder_embedding(doc_tokens)
        memory = self.encoder(src_embed)

        dec_embed = self.decoder_embedding(sum_tokens)

        attn_mask = self.get_attn_mask(sum_tokens.shape[-1],
                                       device)

        decoder_outputs = self.decoder(
            dec_embed,
            memory,
            tgt_mask=attn_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        # Generate our predictions.
        return self.output_layer(decoder_outputs)

    @staticmethod
    def get_attn_mask(summary_tokens_shape, device) -> torch.tensor:
        attn_mask = np.triu(
            m=np.ones((summary_tokens_shape,
                       summary_tokens_shape), dtype=bool),
            k=1
        )
        return torch.tensor(attn_mask).to(device)

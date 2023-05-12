import torch
from torch import nn
import torch.nn.functional as F

from ..modules.layers import (
    LightConvEncoderLayer,
    LightConvDecoderLayer
)

from ..modules.blocks import TripleEmbeddingBlock

class LightConvModel(nn.Module):
    def __init__(self,
                 src_vs: int = 8000,
                 tgt_vs: int = 8000,
                 d_model: int = 64,
                 nheads: int = 4,
                 dim_feedforward: int = 128,
                 src_padding_idx: int = None,
                 tgt_padding_idx: int = None,
                 dropout: float = 0.1,
                 encoder_kernels: list = [3, 7, 15, 31, 31, 31, 31],
                 decoder_kernels: list = [3, 7, 15, 31, 31, 31],
                 encoder_dilations: list = None,
                 decoder_dilations: list = None,
                 maxpool: bool = False):
        
        super().__init__()
        
        self.embeddig = TripleEmbeddingBlock(
            num_word_embeddings=src_vs,
            embedding_dim=d_model,
            padding_index=src_padding_idx
        )
        
        self.dec_embeddig = TripleEmbeddingBlock(
            num_word_embeddings=tgt_vs,
            embedding_dim=d_model,
            padding_index=tgt_padding_idx
        )
        
        if encoder_dilations:
            self.encoder = nn.Sequential(
                *[LightConvEncoderLayer(d_model=d_model,
                               n_heads=nheads,
                               dim_feedforward=dim_feedforward,
                               dropout=dropout,
                               dilation=dilation,
                               maxpool=maxpool)
                  for dilation in encoder_dilations]
            )
        else:
            self.encoder = nn.Sequential(
                *[LightConvEncoderLayer(d_model=d_model,
                               n_heads=nheads,
                               dim_feedforward=dim_feedforward,
                               kernel_size=kernel,
                               dropout=dropout,
                               maxpool=maxpool)
                  for kernel in encoder_kernels]
            )
            
        if decoder_dilations:
            self.decoder = nn.ModuleList(
                [LightConvDecoderLayer(d_model=d_model,
                              n_heads=nheads,
                              dim_feedforward=dim_feedforward,
                              dropout=dropout,
                              dilation=dilation)
                 for dilation in decoder_dilations]
            )
        else:
            self.decoder = nn.ModuleList(
                [LightConvDecoderLayer(d_model=d_model,
                              n_heads=nheads,
                              dim_feedforward=dim_feedforward,
                              kernel_size=kernel,
                              dropout=dropout)
                 for kernel in decoder_kernels]
            )
            
        self.classifier = nn.Linear(in_features=d_model,
                                    out_features=tgt_vs)
            
    def forward(self,
                src: torch.tensor,
                tgt: torch.tensor):
        enc_embeddings = self.embeddig(src)
        
        enc_output = self.encoder(enc_embeddings)
        
        output = self.dec_embeddig(tgt)
        
        for decoder in self.decoder:
            output = decoder(output,
                             enc_output)
            
        return self.classifier(output)
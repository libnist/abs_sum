import torch
from torch import nn
import torch.nn.functional as F

from .blocks import LightConvBlock, MLPBlock, MHABlock

class LightConvEncoderLayer(nn.Sequential):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 dim_feedforward: int,
                 kernel_size: int = 3,
                 dropout: float = 0.1,
                 dilation: int = None,
                 maxpool: bool = False):
        super().__init__()
        
        self.add_module("LightConv",
                        LightConvBlock(d_model=d_model,
                                       n_heads=n_heads,
                                       kernel_size=kernel_size,
                                       dropout=dropout,
                                       dilation=dilation,
                                       maxpool=maxpool))
        
        self.add_module("MLP",
                        MLPBlock(extend_dim=dim_feedforward,
                                 output_dim=d_model))
        
class LightConvDecoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 dim_feedforward: int,
                 kernel_size: int = 3,
                 dropout: float = 0.1,
                 dilation: int = None):
        super().__init__()
        
        self.light_conv = LightConvBlock(d_model=d_model,
                                         n_heads=n_heads,
                                         kernel_size=kernel_size,
                                         dropout=dropout,
                                         dilation=dilation)
        
        self.mha = MHABlock(embed_dim=d_model,
                            num_heads=n_heads,
                            dropout=dropout)
        
        self.mlp = MLPBlock(extend_dim=dim_feedforward,
                            output_dim=d_model)
        
    def forward(self,
                x: torch.tensor,
                encoder_out: torch.tensor):
        
        output = self.light_conv(x)
        output = self.mha(query=output,
                          key=encoder_out,
                          value=encoder_out)
        return self.mlp(output)
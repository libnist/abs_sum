import torch
from torch import nn
import torch.nn.functional as F

from .blocks import (LightConvBlock,
                     MLPBlock,
                     MHABlock,
                     FnetBlock)

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
                                 output_dim=d_model,
                                 dropout=dropout))
        
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
                            output_dim=d_model,
                            dropout=dropout)
        
    def forward(self,
                x: torch.tensor,
                encoder_out: torch.tensor,
                attn_mask: torch.tensor = None):
        
        output = self.light_conv(x)
        output = self.mha(query=output,
                          key=encoder_out,
                          value=encoder_out,
                          attn_mask=attn_mask)
        return self.mlp(output)
    
class FnetEncoderLayer(nn.Module):
    def __init__(self,
                 model_dim: int,
                 extend_dim: int,
                 dropout: float = .5) -> torch.nn.Module:
        """Returns an encoder layer comprised of FnetBlock.

        Args:
            model_dim (int): Dimension of the model.
            extend_dim (int): Dimension of the first linear layer in MLP block.
            dropout (float, optional): Dropout rate. Defaults to .5.
        Returns:
            FnetEncoderLayer: torch.nn.Module
        """
        super().__init__()

        self.fnet_block = FnetBlock(
            output_dim=model_dim,
            dropout=dropout)

        self.mlp_block = MLPBlock(extend_dim=extend_dim,
                                  output_dim=model_dim,
                                  dropout=dropout)

    def forward(self,
                x: torch.tensor) -> torch.tensor:
        # Pass the input through FnetBlock.
        output = self.fnet_block(x)

        # Pass the input through MLP block.
        return self.mlp_block(output)
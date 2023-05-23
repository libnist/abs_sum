# Import libraries
import torch
from torch import nn

import numpy as np

from ..modules.layers import FnetEncoderLayer
from ..modules.blocks import TripleEmbeddingBlock, DepthWiseSeparableConv1d

from ..models import get_attn_mask


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
        
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self,
                doc_tokens: torch.tensor,
                sum_tokens: torch.tensor) -> torch.tensor:

        device = doc_tokens.device

        memory_key_padding_mask = (
            doc_tokens == self.tgt_padding_index).to(device)
        tgt_key_padding_mask = (
            sum_tokens == self.src_padding_index).to(device)

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

class UFnetModel(nn.Module):
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
        self.encoders = nn.ModuleList(
            [FnetEncoderLayer(model_dim=model_dim,
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

        self.decoders = nn.ModuleList(
            [nn.TransformerDecoderLayer(d_model=model_dim,
                                        nhead=n_head,
                                        dim_feedforward=dim_feedforward,
                                        dropout=dropout,
                                        batch_first=True)
             for _ in range(num_layers)]
        )

        # Create the output layer
        self.output_layer = nn.Linear(in_features=model_dim,
                                      out_features=tgt_vs)
        
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        
    def key_paddding_mask(self, input, pad_id, device):
        return (input == pad_id).to(device)

    def forward(self,
                src: torch.tensor,
                tgt: torch.tensor) -> torch.tensor:

        device = src.device

        tgt_mask = get_attn_mask(tgt.shape[-1],
                                 device)

        tgt_key_padding_mask = self.key_paddding_mask(tgt,
                                                      self.tgt_padding_index,
                                                      device=device)

        enc_embeds = self.encoder_embedding(src)
        dec_embeds = self.decoder_embedding(tgt)

        enc_outs = [self.encoders[0](enc_embeds)]

        for encoder in self.encoders[1:]:
            enc_outs.append(encoder(enc_outs[-1]))

        output = self.decoders[0](tgt=dec_embeds,
                                  memory=enc_outs[0],
                                  tgt_mask=tgt_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask)

        for i, decoder in enumerate(self.decoders[1:]):
            output = decoder(tgt=output,
                             memory=enc_outs[i+1],
                             tgt_mask=tgt_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask)

        return self.output_layer(output)
    
class UFnetConvModel(nn.Module):
    def __init__(self,
                 model_dim: int,
                 n_head: int,
                 dim_feedforward: int,
                 num_layers: int,
                 src_vs: int,
                 tgt_vs: int,
                 src_padding_index: int,
                 tgt_padding_index: int,
                 dropout: float = 0.1,
                 dilation: bool = True) -> nn.Module:
        super().__init__()

        self.src_padding_index = src_padding_index
        self.tgt_padding_index = tgt_padding_index

        self.encoder_embedding = TripleEmbeddingBlock(
            num_word_embeddings=src_vs,
            embedding_dim=model_dim,
            padding_index=src_padding_index,
        )

        # Create the FnetCNNEncoder
        self.encoders = nn.ModuleList(
            [FnetEncoderLayer(model_dim=model_dim,
                              extend_dim=dim_feedforward,
                              dropout=dropout)
             for _ in range(num_layers)]
        )
        
        self.convs = nn.ModuleList(
            [DepthWiseSeparableConv1d(d_model=model_dim,
                                      kernel_size=3,
                                      stride=i,
                                      dilation=i if dilation else 1)
             for i in range(1, num_layers+1)]
        )

        # Create the Decoder
        self.decoder_embedding = TripleEmbeddingBlock(
            num_word_embeddings=tgt_vs,
            embedding_dim=model_dim,
            padding_index=tgt_padding_index
        )

        self.decoders = nn.ModuleList(
            [nn.TransformerDecoderLayer(d_model=model_dim,
                                        nhead=n_head,
                                        dim_feedforward=dim_feedforward,
                                        dropout=dropout,
                                        batch_first=True)
             for _ in range(num_layers)]
        )

        # Create the output layer
        self.output_layer = nn.Linear(in_features=model_dim,
                                      out_features=tgt_vs)
        
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        
    def key_paddding_mask(self, input, pad_id, device):
        return (input == pad_id).to(device)

    def forward(self,
                src: torch.tensor,
                tgt: torch.tensor) -> torch.tensor:

        device = src.device

        tgt_mask = get_attn_mask(tgt.shape[-1],
                                 device)

        tgt_key_padding_mask = self.key_paddding_mask(tgt,
                                                      self.tgt_padding_index,
                                                      device=device)

        enc_embeds = self.encoder_embedding(src)
        dec_embeds = self.decoder_embedding(tgt)

        enc_outs = [self.encoders[0](enc_embeds)]

        for encoder in self.encoders[1:]:
            enc_outs.append(encoder(enc_outs[-1]))
        
        for i in range(len(enc_outs)):
            enc_outs[i] = self.convs[i](enc_outs[i])
        

        output = self.decoders[0](tgt=dec_embeds,
                                  memory=enc_outs[0],
                                  tgt_mask=tgt_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask)

        for i, decoder in enumerate(self.decoders[1:]):
            output = decoder(tgt=output,
                             memory=enc_outs[i+1],
                             tgt_mask=tgt_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask)

        return self.output_layer(output)

class UFnetSharedModel(nn.Module):
    def __init__(self,
                 model_dim: int,
                 n_head: int,
                 dim_feedforward: int,
                 num_layers: int,
                 vs: int,
                 padding_index: int,
                 dropout: float = 0.1,
                 reverse_encoder: bool = False,
                 return_encoder: bool = False) -> nn.Module:
        super().__init__()

        self.padding_index = padding_index
        self.reverse_encoder = reverse_encoder
        self.return_encoder = return_encoder

        self.embedding = TripleEmbeddingBlock(
            num_word_embeddings=vs,
            embedding_dim=model_dim,
            padding_index=padding_index,
        )

        # Create the FnetCNNEncoder
        self.encoders = nn.ModuleList(
            [FnetEncoderLayer(model_dim=model_dim,
                              extend_dim=dim_feedforward,
                              dropout=dropout)
             for _ in range(num_layers)]
        )

        self.decoders = nn.ModuleList(
            [nn.TransformerDecoderLayer(d_model=model_dim,
                                        nhead=n_head,
                                        dim_feedforward=dim_feedforward,
                                        dropout=dropout,
                                        batch_first=True)
             for _ in range(num_layers)]
        )

        # Create the output layer
        self.output_layer = nn.Linear(in_features=model_dim,
                                      out_features=vs)
        
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        
    def key_paddding_mask(self, input, pad_id, device):
        return (input == pad_id).to(device)

    def forward(self,
                src: torch.tensor,
                tgt: torch.tensor) -> torch.tensor:

        device = src.device

        tgt_mask = get_attn_mask(tgt.shape[-1],
                                 device)

        tgt_key_padding_mask = self.key_paddding_mask(tgt,
                                                      self.padding_index,
                                                      device=device)

        enc_embeds = self.embedding(src)
        dec_embeds = self.embedding(tgt)

        enc_outs = [self.encoders[0](enc_embeds)]

        for encoder in self.encoders[1:]:
            enc_outs.append(encoder(enc_outs[-1]))
            
        if self.reverse_encoder:
            enc_outs = enc_outs[::-1]

        output = self.decoders[0](tgt=dec_embeds,
                                  memory=enc_outs[0],
                                  tgt_mask=tgt_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask)

        for i, decoder in enumerate(self.decoders[1:]):
            output = decoder(tgt=output,
                             memory=enc_outs[i+1],
                             tgt_mask=tgt_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask)
            
        if self.return_encoder:
            if self.reverse_encoder:
                enc_last = enc_outs[0]
            else:
                enc_last = enc_outs[-1]
            return enc_last, self.output_layer(output)

        return self.output_layer(output)
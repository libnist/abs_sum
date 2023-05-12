import torch
from torch import nn
import numpy as np


def positional_encoding(length, depth, device, dtype=torch.float):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return torch.tensor(pos_encoding, dtype=dtype, device=device).unsqueeze(0)


class TripleEmbeddingBlock(nn.Module):
    def __init__(self,
                 num_word_embeddings: int,
                 embedding_dim: int,
                 num_type_embeddings: int = None,
                 padding_index: int = None) -> torch.nn.Module:
        """Return an embedding block that also uses positional and
        type embedding.

        Args:
            num_word_embeddings (int): Size of vocabulary.
            num_type_embeddings (int): Number of type embeddings.
            embedding_dim (int): Model dimensions.
            sequence_len (int): Length of the input sequence.

        Returns:
            torch.nn.Module: PyTorch Module.
        """
        super(TripleEmbeddingBlock, self).__init__()

        self.embedding_dim = embedding_dim
        self.padding_index = padding_index
        self.error_message = "Model is defined w/o type embedding."
        self.missing_types = "Missing type inputs."

        # Create word embedding layer.
        self.word_embedding = nn.Embedding(num_embeddings=num_word_embeddings,
                                           embedding_dim=embedding_dim,
                                           padding_idx=padding_index)

        self.sqrt_model_dim = embedding_dim ** (0.5)

        if num_type_embeddings is not None:
            # Create type embedding layer.
            self.type_embedding = nn.Embedding(
                num_embeddings=num_type_embeddings,
                embedding_dim=embedding_dim)

    def forward(self,
                tokens: torch.tensor,
                token_types: torch.tensor = None) -> torch.tensor:

        # Getting the length of the input
        token_length = tokens.shape[-1]

        # Perform word embeddings.
        word_embedding = self.word_embedding(tokens) * self.sqrt_model_dim

        # Positional embedding
        positional_embedding = positional_encoding(token_length,
                                                   self.embedding_dim,
                                                   word_embedding.device,
                                                   word_embedding.dtype)

        # Add all the embeddings to produce the output tensor
        output = (word_embedding +
                  positional_embedding)

        if token_types is not None:
            assert hasattr(self, "type_embedding"), self.error_message
            # Perform type embeddings.
            output += self.type_embedding(token_types) * self.sqrt_model_dim
        elif hasattr(self, "type_embedding"):
            raise ValueError(self.missing_types)

        if self.padding_index:
            paddings = (tokens != self.padding_index).unsqueeze(-1)
            output *= paddings

        return output

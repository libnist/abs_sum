import torch
import numpy as np

def get_attn_mask(summary_tokens_shape: int, 
                  device: str) -> torch.tensor:
    attn_mask = np.triu(
        m=np.ones((summary_tokens_shape,
                   summary_tokens_shape), dtype=bool),
        k=1
    )
    return torch.tensor(attn_mask).to(device)
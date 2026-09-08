"""fusion.py
Modules for fusing embeddings into one embedding.
Author(s): Felix Jimenez"""

import logging
logger = logging.getLogger(__name__)

import torch.nn as nn

class AttentionFusion(nn.Module):
    """
    Transformer-based fusion module that applies multi-layer self-attention to a sequence 
    of embeddings and produces a single fused embedding vector.

    Designed to take as input a batch of sequences of embeddings (e.g., outputs from FullEncoder 
    for multiple spectral bands and auxiliary data), apply layer normalization followed by 
    multiple Transformer encoder layers, and finally flatten the output to a single vector 
    per example.

    Args:
        nlayers (int): Number of Transformer encoder layers to apply. Default is 4.
        nhead (int): Number of attention heads in each Transformer encoder layer. Default is 8.
        d_model (int): Dimensionality of the input embeddings and Transformer layers. Default is 256.

    Forward Input:
        x (Tensor): Input tensor of shape (batch_size, sequence_length, d_model).

    Forward Output:
        Tensor of shape (batch_size, sequence_length * d_model), flattened fused embeddings.
    """
    def __init__(self, nlayers=4, nhead=8, d_model=256):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.layers = nn.Sequential(
            *[nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
               for _ in range(nlayers)]
        )

    def forward(self, x):
        x = self.layers(self.norm(x))
        return x.flatten(1)
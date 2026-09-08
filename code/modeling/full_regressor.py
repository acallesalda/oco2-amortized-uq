"""full_regressor.py
The full regression model using four encoders, 
a fusion layer and an mlp predictor.
Author(s): Felix Jimenez
"""

import logging
logger = logging.getLogger(__name__)

import torch.nn as nn

from .components.encoders import FullEncoder
from .components.fusion import AttentionFusion


class FullRegressor(nn.Module):
    """
    Regression model that encodes spectral bands and auxiliary features,
    fuses embeddings via attention, and outputs a scalar prediction.

    Args:
        aux_dim (int): Dimension of auxiliary features. Default: 32.
        embed_dim (int): Embedding dimension. Default: 256.
        nhead (int): Number of attention heads. Default: 8.
        nlayers (int): Number of Transformer encoder layers. Default: 4.

    Inputs:
        x (tuple): (bands: dict of tensors, aux_cont: tensor, aux_cat: list of tensors)

    Returns:
        Tensor of shape (batch_size, 1) with regression predictions.
    """
    def __init__(self, aux_dim=32, embed_dim=256, nhead=8, nlayers=4,
                 n_cont=None, cat_cardinalities=None, aux_feat_dim=8):
        super().__init__()
        # n_cont / cat_cardinalities left at None reproduces the published four-covariate
        # model exactly; supplying them selects the generalized auxiliary encoder.
        self.encoder = FullEncoder(aux_dim, embed_dim, n_cont=n_cont,
                                   cat_cardinalities=cat_cardinalities,
                                   feat_dim=aux_feat_dim)
        self.fusion = AttentionFusion(nlayers, nhead, embed_dim)
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )
    
    def embed(self, x):
        return self.fusion(self.encoder(x))

    def forward(self, x):
        x = self.encoder(x)
        x = self.fusion(x)
        return self.predictor(x)
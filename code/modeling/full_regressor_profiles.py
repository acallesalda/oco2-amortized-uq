"""full_regressor_profiles.py
Full model (spectral encoders + attention fusion) → 20-d CO2 profile.
Author(s): Alejandor Calle-Saldarriaga
"""
import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn

from .components.encoders import FullEncoder
from .components.fusion import AttentionFusion

class FullRegressorProfiles(nn.Module):
    """
    Same backbone as FullRegressor, but predicts a vector (out_dim=20 by default).
    Input:  x = (bands: dict, aux_cont: Tensor[B,1], aux_cat: list[Tensor[B,1]])
    Output: Tensor[B, out_dim]
    """
    def __init__(self, aux_dim: int = 32, embed_dim: int = 256, nhead: int = 8,
                 nlayers: int = 4, out_dim: int = 20,
                 n_cont: int = None, cat_cardinalities=None, aux_feat_dim: int = 8):
        super().__init__()
        self.out_dim = int(out_dim)
        # n_cont / cat_cardinalities left at None reproduces the published four-covariate
        # model exactly; supplying them selects the generalized auxiliary encoder.
        self.encoder = FullEncoder(aux_dim, embed_dim, n_cont=n_cont,
                                   cat_cardinalities=cat_cardinalities,
                                   feat_dim=aux_feat_dim)        # → [B, 4, E]
        self.fusion = AttentionFusion(nlayers, nhead, embed_dim) # → [B, 4*E]
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, self.out_dim),
        )

    def embed(self, x):
        return self.fusion(self.encoder(x))

    def forward(self, x):
        z = self.encoder(x)      # [B, 4, E]
        z = self.fusion(z)       # [B, 4*E]
        yhat = self.predictor(z) # [B, out_dim]
        assert yhat.ndim == 2 and yhat.shape[-1] == self.out_dim, \
            f"Expected (B,{self.out_dim}), got {tuple(yhat.shape)}"
        return yhat

# modeling/legacy_full_regressor_profiles.py
import torch
import torch.nn as nn

from .legacy_components import FullEncoder, AttentionFusion

class FullRegressorProfilesLegacy(nn.Module):
    """
    Legacy backbone: FullEncoder + AttentionFusion + MLP head → out_dim=20
    Matches old checkpoint parameter names (conv.0/1/2..., mlp.0/1, etc.)
    """
    def __init__(self, aux_dim: int = 32, embed_dim: int = 256, nhead: int = 8,
                 nlayers: int = 4, out_dim: int = 20):
        super().__init__()
        self.out_dim = int(out_dim)
        self.encoder = FullEncoder(aux_dim, embed_dim)           # → [B, 4, E]
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
        assert yhat.ndim == 2 and yhat.shape[-1] == self.out_dim
        return yhat

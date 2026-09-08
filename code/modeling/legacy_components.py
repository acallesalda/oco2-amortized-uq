# modeling/legacy_components.py
# Exact legacy encoder/fusion stack you pasted

import torch
import torch.nn as nn
from typing import Tuple, List

class BandEncoder(nn.Module):
    """
    Length-agnostic 1D spectral band encoder with constant features.
    Inputs:
      x: Tuple[Tensor, Tensor]
         - band:  [B, C, L] or [C, L]  (C = in_channels, typically 2)
         - const: [B, 2] or [2]        (scaling constants)
    Returns:
      Tensor [B, embed_dim]
    """
    def __init__(self, embed_dim: int = 256, in_channels: int = 2, const_dim: int = 2, dropout: float = 0.1):
        super().__init__()
        self.const_dim = const_dim

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # -> [B, 128, 1]

        self.mlp = nn.Sequential(
            nn.Linear(128 + const_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        band, const = x  # band: [B,C,L] or [C,L]; const: [B,2] or [2]
        if band.dim() == 2:   # [C, L] -> [1, C, L]
            band = band.unsqueeze(0)

        h = self.conv(band)           # [B, 128, L']
        h = self.pool(h).squeeze(-1)  # [B, 128]

        if const.dim() == 1:          # [2] -> [1,2] -> [B,2]
            const = const.unsqueeze(0).expand(h.size(0), -1)
        const = const.to(h.dtype).to(h.device)

        return self.mlp(torch.cat([h, const], dim=-1))


class AuxEmbedding(nn.Module):
    """
    One continuous + three categorical → concat 32-dim embedding.
    """
    def __init__(self):
        super().__init__()
        self.cat1 = nn.Embedding(8, 8)
        self.cat2 = nn.Embedding(5, 8)
        self.cat3 = nn.Embedding(5, 8)
        self.fc_cont = nn.Linear(1, 8)
        self.dim = 4 * 8

    def forward(self, cont_feat, cat_feat):
        cont_emb = self.fc_cont(cont_feat)
        cat_embs = [emb(cat).squeeze(1) for emb, cat in zip([self.cat1, self.cat2, self.cat3], cat_feat)]
        all_embs = torch.cat([cont_emb] + cat_embs, dim=-1)
        return all_embs


class AuxEncoder(nn.Module):
    """
    AuxEmbedding (32) → Linear to embed_dim.
    """
    def __init__(self, aux_dim:int=32, embed_dim:int=256):
        super().__init__()
        self.embedding = AuxEmbedding()
        self.encoder = nn.Linear(aux_dim, embed_dim)

    def forward(self, aux_cont, aux_cat):
        if len(aux_cont.shape) < 2:
            aux_cont = aux_cont.unsqueeze(0)
        aux_emb = self.embedding(aux_cont, aux_cat)
        aux = self.encoder(aux_emb)
        return aux


class FullEncoder(nn.Module):
    """
    Per-band BandEncoder + AuxEncoder → stack [B, 4, embed_dim]
    """
    def __init__(self, aux_dim:int=32, embed_dim:int=256):
        super().__init__()
        band_names = ("b1", "b2", "b3")
        self.band_names = band_names
        self.band_encoders = nn.ModuleDict({
            name: BandEncoder(embed_dim, 2) for name in band_names
        })
        self.aux_encoder = AuxEncoder(aux_dim, embed_dim)

    def forward(self,x):
        bands, aux_cont, aux_cat = x
        aux = self.aux_encoder(aux_cont, aux_cat)
        band_embs = [self.band_encoders[name](bands[name]) for name in self.band_names]
        band_embs.append(aux)
        emb = torch.stack(band_embs, axis=1)
        return emb


class AttentionFusion(nn.Module):
    """
    Transformer encoder layers (batch_first) over length-4 sequence → flatten to [B, 4*E]
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

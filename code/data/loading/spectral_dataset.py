"""spectral_dataset.py
PyTorch dataset for OCO-2 spectra.
Author(s): Felix Jimenez + Alejandro Calle-Saldarriaga
"""
from __future__ import annotations

import logging
logger = logging.getLogger(__name__)

from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from .covariates import CovariateEncoder, LEGACY_AEROSOL_VOCAB, get_spec

# # define the vocabulary mapping
vocab = LEGACY_AEROSOL_VOCAB

def encode(x):
    return torch.tensor([vocab[x]], dtype=torch.long)

class SpectralDataset(Dataset):
    def __init__(
        self,
        annotations_file: Path | str | pd.DataFrame,
        spectra_dir: Path | str,
        b1_transform=None,
        b2_transform=None,
        b3_transform=None,
        target_transform=None,
        target_cols: Optional[List[str]] = None,
        retr_cols:   Optional[List[str]] = None,   # NEW: columns for retrieved baseline
        spectra_dir_col: Optional[str] = None,     # NEW: per-row subdirectory of spectra_dir
        cov_spec=None,                             # NEW: CovariateSpec / preset name
        cov_stats: Optional[dict] = None,          # NEW: covariate_stats.json contents
    ):
        # An in-memory frame is accepted so that group-disjoint splits can be built by
        # filtering a single split table, without materializing new annotation CSVs.
        if isinstance(annotations_file, pd.DataFrame):
            self.files = annotations_file.reset_index(drop=True)
        else:
            self.files = pd.read_csv(annotations_file)
        self.spectra_dir = Path(spectra_dir)
        self.transform = {"b1": b1_transform, "b2": b2_transform, "b3": b3_transform}
        self.target_transform = target_transform
        self.target_cols = target_cols
        self.retr_cols = retr_cols
        self.spectra_dir_col = spectra_dir_col
        if spectra_dir_col is not None and spectra_dir_col not in self.files.columns:
            raise KeyError(
                f"spectra_dir_col='{spectra_dir_col}' not found in annotations "
                f"(columns: {list(self.files.columns)[:10]}...)"
            )
        # Auxiliary covariates. The default spec is the published four, encoded exactly as
        # they were before this parameter existed, so nothing that does not pass cov_spec
        # sees any change. Encoding is done once here rather than per __getitem__.
        self.cov_spec = get_spec(cov_spec)
        self.cov_encoder = CovariateEncoder(self.files, self.cov_spec, cov_stats)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        bands = self.load_spectra(idx)

        if self.target_cols is None:
            # scalar target (back-compat)
            label = torch.tensor([self.files.true_xco2.iloc[idx]], dtype=torch.float32)
            retr_label = torch.tensor([self.files.retrieved_xco2.iloc[idx]], dtype=torch.float32)
        else:
            # vector target (profiles)
            label = torch.tensor([self.files[c].iloc[idx] for c in self.target_cols], dtype=torch.float32)
            if self.retr_cols is not None:
                retr_label = torch.tensor([self.files[c].iloc[idx] for c in self.retr_cols], dtype=torch.float32)
            else:
                retr_label = label.clone()

        # band transforms
        for band in bands.keys():
            if self.transform[band] is not None:
                b, const = bands[band]
                bands[band] = (self.transform[band](b), const)

        # standardize targets (apply the SAME transform to both for fair baseline)
        if self.target_transform is not None:
            label = self.target_transform(label)
            retr_label = self.target_transform(retr_label)

        aux_continuous, aux_categorical = self.load_aux(idx)
        return (bands, aux_continuous, aux_categorical), label, retr_label

    def spectra_path(self, idx) -> Path:
        """Resolve the .npy path for row `idx`.

        When `spectra_dir_col` is set, `spectra_dir` is treated as a root and the column
        value selects the subdirectory (e.g. "train"/"test" as written by clean_data.py).
        This lets one split table draw rows from both preprocessed partitions.
        """
        stem = self.files.ID.iloc[idx] + '.npy'
        if self.spectra_dir_col is not None:
            sub = str(self.files[self.spectra_dir_col].iloc[idx])
            return self.spectra_dir / sub / stem
        return self.spectra_dir / stem

    def load_spectra(self, idx) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        x = torch.tensor(np.load(self.spectra_path(idx))).transpose(-1, -2)

        # split by band and scale
        wl = x[0]
        b1, c1 = add_scaling_channels(x[:, wl < 1.2])
        b2, c2 = add_scaling_channels(x[:, (wl < 1.8) * (wl > 1.2)])
        b3, c3 = add_scaling_channels(x[:, wl > 1.8])
        return {"b1": (b1, c1), "b2": (b2, c2), "b3": (b3, c3)}

    def load_aux(self, idx):
        """Auxiliary covariates for row `idx`: (continuous [C], categorical list of [1]).

        With the default (legacy) spec this returns exactly what the hardcoded version
        returned: SlrZen/90 - 0.5, Footprint-1, and the two aerosol-type codes.
        """
        return self.cov_encoder(idx)


def add_scaling_channels(x):
    # x: tensor of shape (2, L), [wavelength, radiance]
    radiance = x[1]
    max_val = radiance.max()
    order_mag = 10 ** torch.floor(torch.log10(max_val))

    # Normalize radiance
    radiance_norm = radiance / max_val

    # constant channels in [0,1]
    order_mag_channel = torch.log10(order_mag) / 100
    max_val_channel = max_val / (order_mag * 10)

    # for recovering original spectra
    const = torch.tensor([order_mag_channel / .01, max_val_channel])

    out = torch.stack([
        x[0],          # wavelength
        radiance_norm, # normalized radiance
    ], dim=0)

    return out, const

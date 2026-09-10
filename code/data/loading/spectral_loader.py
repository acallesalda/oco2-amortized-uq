"""spectral_loader.py
Dataloaders for spectral dataset.
Author(s): Felix Jimenez + Alejandro Calle-Saldarriaga
"""
from __future__ import annotations

import logging
logger = logging.getLogger(__name__)

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.transforms import v2
from torch.utils.data import DataLoader, WeightedRandomSampler
from .spectral_dataset import SpectralDataset
from .covariates import get_spec, load_covariate_stats
from .density_weights import MODES as DENSITY_MODES, scene_density_weights
from typing import Tuple, List, Optional

# `data.balance_by` takes a column name for the categorical sampler; this one reserved
# value selects the covariate-density sampler instead. It is not a column, and the split
# table must not have one called this -- build_density_sampler says so if it does.
DENSITY_KEY = "density"

# Defaults of the density sampler's own settings, keyed by the `data.*` name each is read
# from. The rarity exponent is NOT here: it is `data.balance_alpha`, shared with the
# categorical sampler, because it is the same knob with the same range and meaning.
DENSITY_DEFAULTS = {
    "density_mode": "inverse",
    "density_beta": 1.0,
    "density_k": 15,
    "density_clip": 10.0,
}


def density_settings(data_cfg) -> dict:
    """Read the density sampler's settings out of a `data` config block.

    Returns the kwargs `build_density_sampler` takes, with `DENSITY_DEFAULTS` filling in
    whatever the config does not declare. Works on a plain dict and on an OmegaConf node,
    which is why it is a function and not a dataclass: the deterministic trainer is
    `@hydra.main` and the three flow trainers load a plain YAML.

    Validated here, at config-read time, rather than inside the estimator. A typo in a
    submit file should cost a second and not a GPU slot plus the forty minutes of unpacking
    that precede the first epoch -- the same reason bo_final.sh validates balance_alpha
    before it sources job_common.sh.
    """
    got = {k: data_cfg.get(k, v) for k, v in DENSITY_DEFAULTS.items()}
    mode = str(got["density_mode"])
    if mode not in DENSITY_MODES:
        raise ValueError(f"unknown data.density_mode '{mode}' "
                         f"(expected one of {DENSITY_MODES})")
    return {
        "mode": mode,
        "beta": float(got["density_beta"]),
        "k": int(got["density_k"]),
        "clip": float(got["density_clip"]),
    }

# default place where you saved μ/σ
_DEFAULT_STATS_DIR = Path("/workspace/project/experiments/pipelines/stats")

# column lists
_PROFILE_TRUE_COLS  = [f"true_state_vector_{i:02d}"      for i in range(1, 21)]
_PROFILE_RETR_COLS  = [f"retrieved_state_vector_{i:02d}" for i in range(1, 21)]

# Scalar XCO2 standardization constants used for the original replicate-level split.
# These were computed on that split's training pool and are kept only so the published
# replicate-split results stay reproducible. Any new split must recompute them from its
# own training rows (see experiments/pipelines/compute_split_stats.py), otherwise the
# preprocessing leaks information from held-out soundings.
_LEGACY_XCO2_MU = 0.0004108437
_LEGACY_XCO2_SIGMA = 2.1281e-05

_SPLIT_DIR_COL = "source_dir"


def gaussian_noise(sig):
    mask = torch.Tensor([0.,1.,0.,1.]).unsqueeze(-1)
    return v2.Lambda(lambda x: x + torch.randn_like(x) * sig * mask)

def standardize(mu, sig):
    return v2.Lambda(lambda x: ((x-mu) / sig).squeeze(0))

class VectorStandardizer:
    """Apply (y - μ) / σ elementwise for 1D or 2D tensors."""
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        assert mu.shape == (20,) and sigma.shape == (20,), "mu/sigma must be shape (20,)"
        # store as torch for speed & broadcast; float32 is enough for training
        self.mu = torch.tensor(mu, dtype=torch.float32)
        self.sigma = torch.tensor(np.where(sigma == 0.0, 1e-12, sigma), dtype=torch.float32)
    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        # y can be shape (20,) or (B,20) depending on collate
        return (y - self.mu) / self.sigma

def _load_profile_stats(stats_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    mu_path = stats_dir / "profile_mu.npy"
    sd_path = stats_dir / "profile_sigma.npy"
    if not mu_path.exists() or not sd_path.exists():
        raise FileNotFoundError(f"Expected μ/σ at {mu_path} and {sd_path}")
    mu = np.load(mu_path)
    sd = np.load(sd_path)
    sd = np.where(sd == 0.0, 1.0, sd)
    return mu, sd

def _load_scalar_stats(stats_dir: Optional[Path]) -> Tuple[float, float]:
    """Load XCO2 μ/σ from `xco2_stats.json`, falling back to the legacy constants.

    The fallback keeps the original replicate-split experiments reproducible, but it is
    only correct for that split; it warns so a new split cannot silently reuse it.
    """
    if stats_dir is not None:
        path = Path(stats_dir) / "xco2_stats.json"
        if path.exists():
            with open(path, "r") as fh:
                stats = json.load(fh)
            return float(stats["mu"]), float(stats["sigma"])
    logger.warning(
        "xco2_stats.json not found in %s; falling back to the legacy replicate-split "
        "constants (mu=%.10g, sigma=%.6g). Recompute with compute_split_stats.py for "
        "any new split.", stats_dir, _LEGACY_XCO2_MU, _LEGACY_XCO2_SIGMA
    )
    return _LEGACY_XCO2_MU, _LEGACY_XCO2_SIGMA

def build_scalar_target_transform(mu: float, sigma: float):
    return v2.Compose([
        standardize(mu, sigma),
        v2.Lambda(lambda x: x.unsqueeze(-1)),
    ])

def build_dataset_profiles(
    root: Path,
    annotations_filename: str,
    stats_dir: Optional[Path] = None,
) -> SpectralDataset:
    stats_dir = Path(stats_dir) if stats_dir is not None else _DEFAULT_STATS_DIR
    mu, sd = _load_profile_stats(stats_dir)
    tgt_tf = VectorStandardizer(mu, sd)

    return SpectralDataset(
        annotations_file=root / annotations_filename,
        spectra_dir=root,
        b1_transform=None, b2_transform=None, b3_transform=None,
        target_transform=tgt_tf,
        target_cols=_PROFILE_TRUE_COLS,      # train against true_...
        retr_cols=_PROFILE_RETR_COLS,        # compare against retrieved_...
    )


b1_transform = v2.Compose([
    v2.Lambda(
        lambda x : nn.ConstantPad1d((0,1024-x.shape[-1]),0)(x)
    )
])

b2_transform = v2.Compose([
    v2.Lambda(
        lambda x : nn.ConstantPad1d((0,1024-x.shape[-1]),0)(x)
    )
])

b3_transform = v2.Compose([
    v2.Lambda(
        lambda x : nn.ConstantPad1d((0,1024-x.shape[-1]),0)(x)
    )
])

transform_output = build_scalar_target_transform(_LEGACY_XCO2_MU, _LEGACY_XCO2_SIGMA)

def build_dataset(
    root: Path,
    annotations_filename: str,
    stats_dir: Optional[Path] = None,
) -> SpectralDataset:
    mu, sigma = _load_scalar_stats(stats_dir)
    return SpectralDataset(
        root / annotations_filename,
        root,
        b1_transform=b1_transform,
        b2_transform=b2_transform,
        b3_transform=b3_transform,
        target_transform=build_scalar_target_transform(mu, sigma),
    )


# --------------------------------------------------------------------------------------
# Group-disjoint (reference-sounding-level) splits
#
# The original pipeline assigns train/test independently to each of the 100 replicates of
# a reference sounding, so every reference sounding appears in both partitions. The
# functions below instead read a precomputed split table in which whole reference
# soundings are assigned to train/val/test; see experiments/pipelines/make_group_split.py.
# --------------------------------------------------------------------------------------

def load_split_table(split_csv: Path | str, split: Optional[str] = None) -> pd.DataFrame:
    """Load a group split table, optionally filtered to one partition."""
    df = pd.read_csv(split_csv)
    for col in ("ID", "split", _SPLIT_DIR_COL):
        if col not in df.columns:
            raise KeyError(f"split table {split_csv} is missing required column '{col}'")
    if split is not None:
        df = df[df["split"] == split]
        if len(df) == 0:
            raise ValueError(f"split table {split_csv} has no rows with split == '{split}'")
    return df.reset_index(drop=True)


def build_dataset_grouped(
    files_root: Path | str,
    split_csv: Path | str,
    split: str,
    stats_dir: Optional[Path] = None,
    cov_spec=None,
) -> SpectralDataset:
    """Scalar XCO2 dataset for one partition of a group-disjoint split."""
    mu, sigma = _load_scalar_stats(stats_dir)
    return SpectralDataset(
        annotations_file=load_split_table(split_csv, split),
        spectra_dir=Path(files_root),
        b1_transform=b1_transform,
        b2_transform=b2_transform,
        b3_transform=b3_transform,
        target_transform=build_scalar_target_transform(mu, sigma),
        spectra_dir_col=_SPLIT_DIR_COL,
        cov_spec=cov_spec,
        cov_stats=load_covariate_stats(stats_dir),
    )


def build_dataset_profiles_grouped(
    files_root: Path | str,
    split_csv: Path | str,
    split: str,
    stats_dir: Optional[Path] = None,
    cov_spec=None,
) -> SpectralDataset:
    """20-d profile dataset for one partition of a group-disjoint split."""
    stats_dir = Path(stats_dir) if stats_dir is not None else _DEFAULT_STATS_DIR
    mu, sd = _load_profile_stats(stats_dir)
    return SpectralDataset(
        annotations_file=load_split_table(split_csv, split),
        spectra_dir=Path(files_root),
        b1_transform=None, b2_transform=None, b3_transform=None,
        target_transform=VectorStandardizer(mu, sd),
        target_cols=_PROFILE_TRUE_COLS,
        retr_cols=_PROFILE_RETR_COLS,
        spectra_dir_col=_SPLIT_DIR_COL,
        cov_spec=cov_spec,
        cov_stats=load_covariate_stats(stats_dir),
    )


def build_balanced_sampler(
    dataset,
    balance_by: str = "clst",
    alpha: float = 0.5,
    seed: int = 42,
) -> WeightedRandomSampler:
    """Sample training rows with weight proportional to n_group ** (-alpha).

    Why this exists
    ---------------
    The group-disjoint split is stratified by SOM cluster, and the clusters are very
    unequal: 86 training reference soundings in cluster 1 against 12 and 14 in clusters 8
    and 9. Per-cluster test error of the learned retrievals falls off with the number of
    training reference soundings (Spearman -0.86 to -0.91), while the operational
    retrieval's does not -- so the deficit is a property of the sampling, not of the
    scenes. Uniform shuffling lets the majority regimes set the loss.

    `alpha` tempers the correction. alpha=0 is the current behaviour (no reweighting);
    alpha=1 is full inverse-frequency, which makes 12 reference soundings carry as much
    gradient as 86 and therefore risks memorising those 12; alpha=0.5 (the default) is
    square-root inverse frequency, the usual compromise in long-tailed problems. The
    number of *distinct* scenes in a rare cluster is unchanged whatever alpha is, so this
    can correct the loss weighting but cannot manufacture regime diversity -- see the
    scene-count caveat in the revision notes.

    `dataset` must expose the split table as `.files` (SpectralDataset does); a
    `random_split` Subset does not, so balancing requires an explicit split table.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
    files = getattr(dataset, "files", None)
    if files is None:
        raise TypeError(
            "balanced sampling needs a dataset carrying its split table as `.files`; "
            "pass split_csv= so the grouped datasets are used"
        )
    if balance_by not in files.columns:
        raise KeyError(
            f"split table has no column '{balance_by}' to balance on; "
            f"available: {sorted(files.columns)[:12]}..."
        )
    groups = files[balance_by].to_numpy()
    labels, counts = np.unique(groups, return_counts=True)
    per_group = counts.astype(np.float64) ** (-float(alpha))
    weight_of = dict(zip(labels, per_group))
    weights = np.array([weight_of[g] for g in groups], dtype=np.float64)
    logger.info(
        "balanced sampling on '%s' (alpha=%.2f): %d groups, counts %s, "
        "relative per-row weights %s",
        balance_by, alpha, len(labels), dict(zip(labels.tolist(), counts.tolist())),
        {k: round(float(v / per_group.min()), 2) for k, v in zip(labels.tolist(), per_group)},
    )
    generator = torch.Generator().manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def build_density_sampler(
    dataset,
    cov_spec=None,
    alpha: float = 0.5,
    beta: float = 1.0,
    k: int = 15,
    clip: float = 10.0,
    mode: str = "inverse",
    target_dataset=None,
    seed: int = 42,
) -> WeightedRandomSampler:
    """Sample training rows by the local density of the covariates the model sees.

    The weighting arithmetic, the choice of estimator and what the intervention is and is
    not are all in density_weights.py; this function is the loader-side adapter. Two
    things it adds:

    * the density lives in the covariate space of `cov_spec` -- the same preset the model
      is given, so a weight is a function of the model's own inputs and never of `clst`,
      which covariates.py rules inadmissible as an input;
    * mode="ratio" needs the partition to tilt toward, which for a group-disjoint split is
      the validation dataset. Its *covariates* are read and its targets are not.

    Like `build_balanced_sampler`, `dataset` must carry its split table as `.files`, so
    the caller has to have passed `split_csv=`.
    """
    if cov_spec is None:
        raise ValueError(
            "density weighting needs a covariate preset; pass cov_spec= (the same one the "
            "model is built with) so the density is estimated in the model's input space"
        )
    files = getattr(dataset, "files", None)
    if files is None:
        raise TypeError(
            "density sampling needs a dataset carrying its split table as `.files`; "
            "pass split_csv= so the grouped datasets are used"
        )
    if DENSITY_KEY in files.columns:
        raise KeyError(
            f"the split table has a column called '{DENSITY_KEY}', which collides with "
            f"the reserved value of data.balance_by that selects this sampler"
        )
    target_files = None
    if mode == "ratio":
        target_files = getattr(target_dataset, "files", None)
        if target_files is None:
            raise TypeError(
                "density mode 'ratio' needs the validation dataset to tilt toward; it is "
                "passed by _train_dataloader and requires split_csv="
            )
    weights = scene_density_weights(
        files,
        cov_spec.continuous,
        cov_spec.categorical,
        alpha=alpha,
        beta=beta,
        k=k,
        clip=clip,
        mode=mode,
        target_files=target_files,
    )
    generator = torch.Generator().manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def _train_dataloader(train_ds, batch_size, balance_by, balance_alpha, seed,
                      cov_spec=None, density=None, val_ds=None):
    """Shuffled loader, or a reweighted one when `balance_by` is set.

    `balance_by` is a column name for the categorical sampler, or the reserved value
    DENSITY_KEY for the covariate-density sampler. `density` carries that sampler's own
    settings (mode, beta, k, clip); `balance_alpha` is its rarity exponent, the same knob
    and the same range as for the categorical one.
    """
    if balance_by is None:
        return DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    if balance_by == DENSITY_KEY:
        sampler = build_density_sampler(train_ds, cov_spec=cov_spec, alpha=balance_alpha,
                                        seed=seed, target_dataset=val_ds,
                                        **(density or {}))
    else:
        sampler = build_balanced_sampler(train_ds, balance_by, balance_alpha, seed)
    return DataLoader(train_ds, batch_size=batch_size, sampler=sampler)


def train_val_loader(
        root,
        annotations_filename,
        train_batch_size=256,
        val_batch_size=256,
        seed=42,
        train_size=.9,
        stats_dir: Optional[Path] = None,
        split_csv: Optional[Path] = None,
        cov_spec=None,
        balance_by: Optional[str] = None,
        balance_alpha: float = 0.5,
        density: Optional[dict] = None,
    )-> tuple[DataLoader, DataLoader]:
        torch.manual_seed(seed)
        if split_csv is not None:
            train_ds = build_dataset_grouped(root, split_csv, "train", stats_dir=stats_dir,
                                             cov_spec=cov_spec)
            val_ds = build_dataset_grouped(root, split_csv, "val", stats_dir=stats_dir,
                                           cov_spec=cov_spec)
        else:
            ds = build_dataset(Path(root) / 'train', annotations_filename, stats_dir=stats_dir)
            n_total = len(ds)
            n_train = int(round(train_size * n_total))
            n_val = n_total - n_train
            train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

        train_dl = _train_dataloader(train_ds, train_batch_size, balance_by,
                                     balance_alpha, seed, cov_spec=cov_spec,
                                     density=density, val_ds=val_ds)
        val_dl = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False)

        return train_dl, val_dl

def test_loader(
        root,
        annotations_filename,
        val_batch_size=256,
        seed=42,
        stats_dir: Optional[Path] = None,
        split_csv: Optional[Path] = None,
        cov_spec=None,
    )-> tuple[DataLoader]:
        torch.manual_seed(seed)
        if split_csv is not None:
            ds = build_dataset_grouped(root, split_csv, "test", stats_dir=stats_dir,
                                       cov_spec=cov_spec)
        else:
            ds = build_dataset(Path(root) / 'test', annotations_filename, stats_dir=stats_dir)

        test_dl = DataLoader(
            ds,
            batch_size=val_batch_size,
            shuffle=False
        )

        return test_dl

def train_val_loader_profiles(
         root,
        annotations_filename,
        train_batch_size=128,
        val_batch_size=256,
        seed=42,
        train_size=.9,
        stats_dir: Optional[Path] = None,
        split_csv: Optional[Path] = None,
        cov_spec=None,
        balance_by: Optional[str] = None,
        balance_alpha: float = 0.5,
        density: Optional[dict] = None,
    ) -> Tuple[DataLoader, DataLoader]:
    torch.manual_seed(seed)
    if split_csv is not None:
        train_ds = build_dataset_profiles_grouped(root, split_csv, "train",
                                                  stats_dir=stats_dir, cov_spec=cov_spec)
        val_ds = build_dataset_profiles_grouped(root, split_csv, "val",
                                                stats_dir=stats_dir, cov_spec=cov_spec)
    else:
        ds = build_dataset_profiles(Path(root) / 'train', annotations_filename, stats_dir=stats_dir)
        n_total = len(ds)
        n_train = int(round(train_size * n_total))
        n_val = n_total - n_train
        train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    train_dl = _train_dataloader(train_ds, train_batch_size, balance_by,
                                 balance_alpha, seed, cov_spec=cov_spec,
                                 density=density, val_ds=val_ds)
    val_dl = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False)
    return train_dl, val_dl

def test_loader_profiles(
        root,
        annotations_filename,
        val_batch_size=256,
        seed=42,
        stats_dir: Optional[Path] = None,
        split_csv: Optional[Path] = None,
        cov_spec=None,
    ) -> DataLoader:
    torch.manual_seed(seed)
    if split_csv is not None:
        ds = build_dataset_profiles_grouped(root, split_csv, "test", stats_dir=stats_dir,
                                            cov_spec=cov_spec)
    else:
        ds = build_dataset_profiles(Path(root) / 'test', annotations_filename, stats_dir=stats_dir)
    return DataLoader(ds, batch_size=val_batch_size, shuffle=False)

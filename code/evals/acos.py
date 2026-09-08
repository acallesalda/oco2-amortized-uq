"""acos.py

The ACOS L2FP baseline: its retrieved state and its posterior covariance, aligned with
whatever split table the models were evaluated on.

Two traps live here, both of which the notebooks fall into and neither of which announces
itself at runtime.

**The covariance files are not all in `test/`.** `processing.save_data_with_covariance`
writes `<ID>_cov.npy` beside the spectra, so a covariance sits in `files/train/` or
`files/test/` according to where `clean_data.py` put that sounding. Under the original
replicate split the test partition was exactly `files/test/`, and
`full_profiles.ipynb` could therefore hardcode `TEST_DIR / f"{sid}_cov.npy"`. Under the
reference-sounding-disjoint split a *test* sounding is very often in `files/train/`,
because the two partitionings are unrelated. Reading the fixed directory silently drops
most of the baseline and leaves a table computed on whichever soundings happened to
survive. The split table's `source_dir` column is the only correct answer, and it is
required below rather than defaulted.

**Alignment is positional.** `SpectralDataset` returns no identifier, so predictions line
up with the annotation frame only because the loader is unshuffled. Everything here is
indexed off the same frame, in the same order, and `ids` is written into the artifact so
the assumption can be checked after the fact instead of trusted.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROFILE_RETR_COLS = [f"retrieved_state_vector_{i:02d}" for i in range(1, 21)]
SOURCE_DIR_COL = "source_dir"


def _cov_path(files_root: Path, row: pd.Series, source_dir_col: Optional[str]) -> Path:
    stem = f"{row['ID']}_cov.npy"
    if source_dir_col is None:
        return files_root / stem
    return files_root / str(row[source_dir_col]) / stem


def load_profiles(
    frame: pd.DataFrame,
    files_root: Path | str,
    y_mu: np.ndarray,
    y_sigma: np.ndarray,
    levels: Optional[Sequence[int]] = None,
    source_dir_col: Optional[str] = SOURCE_DIR_COL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ACOS profile mean and covariance, standardized.

    Returns `(mu_std [N,D], cov_std [N,D,D], has_cov [N])`. The mean is always available
    (it is a column of `aux_data.csv`); the covariance is a separate file and is NaN-filled
    where it is missing or malformed, with `has_cov` recording which.

    `levels` selects 1-based pressure levels for the near-surface task; the covariance is
    sub-blocked accordingly, which is the right marginal of a joint Gaussian.
    """
    files_root = Path(files_root)
    if source_dir_col is not None and source_dir_col not in frame.columns:
        raise KeyError(
            f"'{source_dir_col}' is not in the annotation frame, so the covariance files "
            "cannot be located. For a group split this column is written by "
            "make_group_split.py; pass source_dir_col=None only when every sounding "
            "really does live directly under files_root."
        )

    idx0 = None if levels is None else np.asarray([int(l) - 1 for l in levels])
    d_full = len(PROFILE_RETR_COLS)
    d = d_full if idx0 is None else len(idx0)

    mu_orig = frame[PROFILE_RETR_COLS].to_numpy(dtype=np.float64)
    if idx0 is not None:
        mu_orig = mu_orig[:, idx0]

    n = len(frame)
    cov_std = np.full((n, d, d), np.nan)
    has_cov = np.zeros(n, dtype=bool)
    n_missing_file = 0
    n_bad_shape = 0

    outer = np.outer(np.asarray(y_sigma), np.asarray(y_sigma))
    for i in range(n):
        p = _cov_path(files_root, frame.iloc[i], source_dir_col)
        if not p.exists():
            n_missing_file += 1
            continue
        cov = np.load(p).astype(np.float64)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] != d_full:
            n_bad_shape += 1
            continue
        if not np.all(np.isfinite(cov)):
            n_bad_shape += 1
            continue
        cov = 0.5 * (cov + cov.T)
        if idx0 is not None:
            cov = cov[np.ix_(idx0, idx0)]
        cov_std[i] = cov / outer
        has_cov[i] = True

    logger.info(
        "ACOS covariance: %d/%d available (%d file missing, %d malformed)",
        int(has_cov.sum()), n, n_missing_file, n_bad_shape,
    )
    if has_cov.mean() < 0.5:
        logger.warning(
            "fewer than half the ACOS covariances were found under %s. If this is a "
            "group split, check that the split table's '%s' column is being used -- "
            "reading a single fixed directory is the usual cause.",
            files_root, source_dir_col,
        )

    mu_std = (mu_orig - np.asarray(y_mu)) / np.asarray(y_sigma)
    return mu_std, cov_std, has_cov


def load_xco2(
    frame: pd.DataFrame, y_mu: float, y_sigma: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ACOS XCO2 mean and variance, standardized, as `[N,1]`, `[N,1,1]`, `[N]`.

    The scalar task has no covariance file: `retrieved_xco2_variance` in `aux_data.csv`
    is the posterior variance and is already in squared mole fraction, so it scales by
    `1/sigma^2` exactly as the profile covariance does.
    """
    for col in ("retrieved_xco2", "retrieved_xco2_variance"):
        if col not in frame.columns:
            raise KeyError(f"'{col}' missing from the annotation frame")
    mu = ((frame["retrieved_xco2"].to_numpy(dtype=np.float64) - float(y_mu))
          / float(y_sigma)).reshape(-1, 1)
    var = (frame["retrieved_xco2_variance"].to_numpy(dtype=np.float64)
           / float(y_sigma) ** 2).reshape(-1, 1, 1)
    valid = np.isfinite(mu).ravel() & np.isfinite(var).reshape(-1) & (var.reshape(-1) > 0)
    return mu, var, valid

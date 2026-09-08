"""losses.py
Per-level loss weighting, and why the profile models need it.

Targets are standardized per level. The training-set standard deviation runs from about
1.4e-6 at pressure level 1 to 2.0e-5 at level 20 -- a factor of 15 -- so an unweighted MSE
in standardized coordinates weights every level equally, while the RMSE reported in
Tables 2 and 3 is in ppm and weights level `d` by sigma_d^2. Levels 17-20 carry roughly
three quarters of the ppm error budget and one fifth of the standardized loss. The
objective and the reported metric are therefore not the same function, and the same
mismatch appears in checkpoint selection, which used the standardized validation RMSE.

`level_weights(sigma, "ppm", idxs)` returns weights proportional to sigma_d^2, normalized
to mean 1 so the effective learning rate is unchanged. Weighting the standardized squared
error by them is exactly MSE in ppm up to the constant mean(sigma_d^2).

At D=1 (the XCO2 task) the two coincide up to a constant and this module is not needed.

Author(s): Alejandro Calle-Saldarriaga
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODES = ("std", "ppm")


def level_weights(sigma, mode: str, idxs: Sequence[int]) -> Optional[torch.Tensor]:
    """Per-level loss weights for the levels `idxs`, or None for an unweighted loss.

    :param sigma: per-level target standard deviations of the FULL 20-level vector, as
        written by compute_split_stats.py (mole fraction, not ppm -- the scale cancels).
    :param mode: "std" (or None/"none") for the published unweighted MSE; "ppm" for
        weights proportional to sigma_d^2.
    :param idxs: indices of the levels the model predicts.
    """
    if mode in (None, "std", "none"):
        return None
    if mode != "ppm":
        raise ValueError(f"unknown loss_weight '{mode}' (expected one of {MODES})")
    s = np.asarray(sigma, dtype=np.float64)[list(idxs)]
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError(f"non-positive or non-finite target sigma for levels {list(idxs)}: {s}")
    w = s ** 2
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def weighted_mse(y_hat: torch.Tensor, y: torch.Tensor,
                 weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """MSE over the batch and the level axis, optionally weighted per level."""
    d2 = (y_hat - y) ** 2
    if weights is None:
        return d2.mean()
    return (d2 * weights).mean()

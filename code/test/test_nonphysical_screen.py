"""test_nonphysical_screen.py
One non-physical sounding must not blank a whole table cell.

ACOS's July near-surface RMSE and MAE came out empty in `table_last4.csv` on 2026-08-26:
one sounding of 44,327 carries a fill sentinel of order -1e19 in the RETRIEVED state
vector, `np.mean` over it is NaN, and the pooled RMSE and MAE were the only columns not
computed on a subset. The NLL and the coverage survived because they use `valid` and
`nanmean`.

The fix screens on the same physical rule `prepare_month.py` applies to the truth vectors.
These tests pin both halves of it: the bad sounding is excluded and counted, and a clean
artifact is left numerically untouched.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from evals import metrics as M  # noqa: E402

# Standardization of a near-surface task: ~400 ppm mean, ~15 ppm scale, 4 levels.
Y_MU = np.full(4, 4.0e-4)
Y_SIGMA = np.full(4, 1.5e-5)
SENTINEL_STD = (-1e19 - 4.0e-4) / 1.5e-5   # what -1e19 mole fraction looks like standardized


def _clean(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((n, 4))
    mu = y + 0.3 * rng.standard_normal((n, 4))
    cov = np.broadcast_to(np.eye(4) * 0.5, (n, 4, 4)).copy()
    return y, mu, cov


def test_physical_mask_flags_the_sentinel_and_nothing_else():
    y, mu, _ = _clean()
    mu[7, 2] = SENTINEL_STD
    ok = M.physical_mask(y, mu, Y_MU, Y_SIGMA)
    assert ok.sum() == len(y) - 1
    assert not ok[7]


def test_physical_mask_flags_nan_too():
    """`isfinite` was never the problem, but it must still be caught."""
    y, mu, _ = _clean()
    mu[3, 0] = np.nan
    assert not M.physical_mask(y, mu, Y_MU, Y_SIGMA)[3]


def test_one_sentinel_no_longer_blanks_the_gaussian_row():
    y, mu, cov = _clean()
    row = M.summarize_gaussian(y, mu, cov, Y_MU, Y_SIGMA, "ACOS L2FP (v10)")
    mu_bad = mu.copy()
    mu_bad[7, 2] = SENTINEL_STD
    bad = M.summarize_gaussian(y, mu_bad, cov, Y_MU, Y_SIGMA, "ACOS L2FP (v10)")

    assert np.isfinite(bad["rmse_ppm"]), "the cell this test exists for is still blank"
    assert np.isfinite(bad["mae_ppm"])
    assert bad["n_nonphysical"] == 1
    assert bad["n_point"] == len(y) - 1
    # Dropping one of 200 moves the number a little; dropping it must not move it much.
    assert bad["rmse_ppm"] == pytest.approx(row["rmse_ppm"], rel=0.02)


def test_one_sentinel_no_longer_blanks_the_flow_row():
    """The flow is screened on the same rule, so it cannot blank silently either."""
    rng = np.random.default_rng(1)
    y, _, _ = _clean()
    samples = y[:, None, :] + 0.4 * rng.standard_normal((len(y), 64, 4))
    logp = -np.ones(len(y))
    bad = samples.copy()
    bad[7, :, 2] = SENTINEL_STD
    row = M.summarize_samples(y, bad, Y_MU, Y_SIGMA, "Normalizing flow", logprob_std=logp)
    assert np.isfinite(row["rmse_ppm"]) and np.isfinite(row["mae_ppm_median"])
    assert row["n_nonphysical"] == 1


def test_a_clean_artifact_is_unchanged():
    """The screen must be inert on data that has nothing wrong with it.

    Every published number was produced without it; if it moves them, it is not a fix.
    """
    y, mu, cov = _clean(seed=3)
    row = M.summarize_gaussian(y, mu, cov, Y_MU, Y_SIGMA, "Laplace")
    assert row["n_nonphysical"] == 0
    assert row["n_point"] == len(y)
    err = (y - mu) * Y_SIGMA * 1e6
    assert row["rmse_ppm"] == pytest.approx(float(np.sqrt(np.mean(err ** 2))), rel=1e-12)
    assert row["mae_ppm"] == pytest.approx(float(np.mean(np.abs(err))), rel=1e-12)

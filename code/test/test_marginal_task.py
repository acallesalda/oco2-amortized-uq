"""The `last4_marginal` task: does the slice pick the right levels, and is it a marginal?

Two things can go wrong and neither raises. The slice can take the wrong four columns --
levels 17-20 are at output positions 16-19, and an off-by-one is a plausible-looking
table. And the "marginal" can be computed by refitting on four levels instead of slicing
a twenty-level predictive, which is a different object: refitting conditions on the subset,
marginalizing integrates the rest out. For a Gaussian the marginal is exactly the
submatrix, so that identity is pinned here.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "scoring"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import predict as P  # noqa: E402


def test_native_tasks_do_not_slice():
    for task in ("xco2", "profiles", "last4"):
        assert P.model_slice(P.TASKS[task]) is None, task


def test_marginal_slice_is_the_last_four_output_positions():
    spec = P.TASKS["last4_marginal"]
    assert spec["levels"] == [17, 18, 19, 20]
    assert P.model_slice(spec) == [16, 17, 18, 19]


def test_marginal_task_reports_itself_into_table_last4():
    """Otherwise the rows land in their own table instead of beside the native ones."""
    spec = P.TASKS["last4_marginal"]
    assert spec["table_task"] == "last4"
    assert spec["split_suffix"] == "__marginal"
    assert spec["det"] == "det_profiles" and spec["flow"] == "nf_profiles"


def test_marginal_uses_the_same_checkpoints_as_the_profiles_task():
    """The marginal row must be the 20-level model, or it is not marginalizing anything."""
    assert P.TASKS["last4_marginal"]["det"] == P.TASKS["profiles"]["det"]
    assert P.TASKS["last4_marginal"]["flow"] == P.TASKS["profiles"]["flow"]
    assert P.TASKS["last4_marginal"]["model_levels"] == P.TASKS["profiles"]["levels"]


def test_gaussian_marginal_is_the_submatrix():
    """Sanity on the mathematical claim the implementation rests on.

    Slicing mean and covariance of a joint Gaussian gives the exact marginal, so the
    marginalized Laplace row needs no approximation -- unlike the flow row, which has to
    fall back to a KDE because a flow has no closed-form marginal.
    """
    rng = np.random.default_rng(0)
    d, k = 20, [16, 17, 18, 19]
    A = rng.normal(size=(d, d))
    cov = A @ A.T + d * np.eye(d)
    mu = rng.normal(size=d)

    n = 200_000
    x = rng.multivariate_normal(mu, cov, size=n)
    sub = np.ix_(k, k)

    # Tolerances from the Monte Carlo standard error, not guessed.
    #
    # This used to assert allclose(..., rtol=0.05, atol=0.05) on the sample covariance.
    # For an off-diagonal element the sampling sd is sqrt((s_ii s_jj + s_ij^2)/n), which
    # here is about 0.078 -- larger than the 0.068 tolerance that element was allowed. So
    # the test failed or passed on the RNG, and it does differ across numpy versions:
    # it passes under numpy 1.x and fails under 2.0 on the same seed. Five sigma is a
    # 1-in-3.5-million false-failure rate per element, which is what a test in a suite
    # someone runs before every submit should be aiming at.
    se_mean = np.sqrt(np.diag(cov)[k] / n)
    assert np.allclose(x[:, k].mean(axis=0), mu[k], atol=5 * se_mean.max(), rtol=0)

    s_ii = np.diag(cov)[k][:, None]
    se_cov = np.sqrt((s_ii * s_ii.T + cov[sub] ** 2) / n)
    assert np.allclose(np.cov(x[:, k], rowvar=False), cov[sub],
                       atol=5 * se_cov.max(), rtol=0)


def test_flow_marginal_falls_back_to_kde():
    """`summarize_samples` must not silently reuse a joint log_prob as a marginal one."""
    M = pytest.importorskip("evals.metrics")
    rng = np.random.default_rng(1)
    n, s, d = 64, 400, 4
    y = rng.normal(size=(n, d))
    samples = rng.normal(size=(n, s, d))

    row = M.summarize_samples(y, samples, np.zeros(d), np.ones(d) * 1e-6, "flow",
                              logprob_std=None, latent_std=None)
    assert row["nll_estimator"] == "gaussian KDE"
    # No latent means no joint coverage claim, rather than a wrong one.
    assert not np.isfinite(row["cov95_joint"])
    assert np.isfinite(row["cov95_marginal"])

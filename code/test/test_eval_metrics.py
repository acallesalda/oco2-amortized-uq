"""Tests for evals.metrics.

Pure numpy/scipy: no torch, no checkpoints, no data, so this runs anywhere the container
runs. The point is to pin the two things the notebooks got wrong -- the units of the NLL,
and the difference between marginal and joint coverage -- against closed forms rather
than against previous output.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals import metrics as M


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _random_cov(rng, d, n=1):
    A = rng.normal(size=(n, d, d))
    return A @ np.swapaxes(A, -1, -2) + d * np.eye(d)


# ---------------------------------------------------------------------------------
# Gaussian NLL
# ---------------------------------------------------------------------------------
def test_mvn_nll_matches_scipy(rng):
    d, n = 5, 20
    cov = _random_cov(rng, d, n)
    mu = rng.normal(size=(n, d))
    y = mu + rng.normal(size=(n, d))

    got = M.mvn_nll(y, mu, cov)
    want = np.array([
        -stats.multivariate_normal(mu[i], cov[i]).logpdf(y[i]) for i in range(n)
    ])
    assert np.allclose(got, want, atol=1e-9)


def test_mvn_nll_drops_singular_rather_than_regularizing():
    """A rank-deficient covariance must not silently become a finite NLL.

    ACOS posterior covariances are often near-singular; quietly adding enough jitter to
    make the Cholesky succeed turns an undefined density into a plausible number. The
    jitter ladder is bounded, and what it cannot factor is reported as NaN so the table
    can carry an honest denominator.
    """
    d = 4
    C = np.zeros((1, d, d))
    C[0, 0, 0] = 1.0            # rank 1, and the mean diagonal is tiny
    y = np.zeros((1, d))
    mu = np.zeros((1, d))
    out = M.mvn_nll(y, mu, C, jitters=(0.0,))
    assert np.isnan(out[0])


def test_nan_covariance_gives_nan():
    C = np.full((2, 3, 3), np.nan)
    out = M.mvn_nll(np.zeros((2, 3)), np.zeros((2, 3)), C)
    assert np.all(np.isnan(out))


# ---------------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------------
def test_nll_ppm_is_the_exact_change_of_variables(rng):
    """`nll_std_to_ppm` must equal the NLL of the same predictive expressed in ppm."""
    d, n = 3, 200
    y_mu = np.array([4.0e-4, 4.1e-4, 4.2e-4])
    y_sigma = np.array([2.0e-5, 2.1e-5, 1.9e-5])

    cov_std = _random_cov(rng, d, n) * 0.1
    mu_std = rng.normal(size=(n, d))
    y_std = mu_std + rng.normal(size=(n, d)) * 0.3

    nll_std = np.nanmean(M.mvn_nll(y_std, mu_std, cov_std))
    shifted = M.nll_std_to_ppm(nll_std, y_sigma)

    # The same quantity computed from scratch in ppm.
    y_ppm = M.to_ppm(y_std, y_mu, y_sigma)
    mu_ppm = M.to_ppm(mu_std, y_mu, y_sigma)
    cov_ppm = np.stack([M.cov_to_ppm(c, y_sigma) for c in cov_std])
    direct = np.nanmean(M.mvn_nll(y_ppm, mu_ppm, cov_ppm))

    assert shifted == pytest.approx(direct, rel=1e-9)


def test_ppm_shift_is_method_independent():
    """The correction is additive and identical for every method, so it cannot change a
    ranking -- which is the argument for reporting it rather than debating it."""
    y_sigma = np.array([2e-5] * 20)
    a, b = 3.0, 5.0
    assert (M.nll_std_to_ppm(a, y_sigma) - M.nll_std_to_ppm(b, y_sigma)) == pytest.approx(a - b)


# ---------------------------------------------------------------------------------
# Whitening and joint calibration
# ---------------------------------------------------------------------------------
def test_whiten_of_a_correct_gaussian_is_standard_normal(rng):
    d, n = 6, 4000
    cov = _random_cov(rng, d, 1)[0]
    L = np.linalg.cholesky(cov)
    mu = rng.normal(size=d)
    y = mu + (L @ rng.normal(size=(d, n))).T

    z = M.whiten(y, np.repeat(mu[None, :], n, 0), np.repeat(cov[None], n, 0))
    assert np.allclose(z.mean(axis=0), 0.0, atol=0.08)
    assert np.allclose(z.std(axis=0), 1.0, atol=0.08)


@pytest.mark.parametrize("level", [0.68, 0.95])
def test_joint_coverage_hits_nominal_when_calibrated(rng, level):
    d, n = 20, 20000
    z = rng.normal(size=(n, d))
    assert M.joint_coverage_from_z(z, level) == pytest.approx(level, abs=0.02)


def test_marginal_coverage_can_be_right_while_joint_is_wrong(rng):
    """The failure mode the manuscript's coverage column cannot see.

    A predictive with exactly correct marginals that assumes independence, against a
    truth whose 20 levels are strongly correlated. Averaged marginal coverage sits on
    nominal by construction; joint coverage is nowhere near it. This is the objection
    made concrete, in one line.

    The direction is worth being precise about, because it is the alarming one. The
    predictive has the right *total* variance -- tr(I) = tr(C) = 20 -- but spreads it
    isotropically, while the truth concentrates it along one direction: the eigenvalues
    of the equicorrelated C are 1+(d-1)rho = 18.1 once and 1-rho = 0.1 nineteen times.
    So ||z||^2 is 18.1 chi2_1 + 0.1 chi2_19, which is usually *smaller* than a chi2_20
    draw and occasionally far larger. The 95% ball, sized for chi2_20, therefore misses
    the heavy tail and the predictive **under**-covers: 79.8% against a nominal 95%.
    """
    d, n = 20, 40000
    rho = 0.9
    corr = rho * np.ones((d, d)) + (1 - rho) * np.eye(d)
    L = np.linalg.cholesky(corr)
    y = (L @ rng.normal(size=(d, n))).T          # truth: correlated, unit marginals
    mu = np.zeros((n, d))
    cov_indep = np.repeat(np.eye(d)[None], n, 0)  # predictive: right marginals, no correlation

    z = M.whiten(y, mu, cov_indep)
    marg = np.mean(M.gaussian_marginal_coverage(y, mu, np.ones((n, d)), 0.95))
    joint95 = M.joint_coverage_from_z(z, 0.95)
    joint99 = M.joint_coverage_from_z(z, 0.99)

    # Marginally indistinguishable from perfect calibration.
    assert marg == pytest.approx(0.95, abs=0.01)
    # Jointly 15 points short, against the closed-form value for this covariance.
    assert joint95 == pytest.approx(0.798, abs=0.01)
    # And it does not recover at a higher nominal level -- the 99% region covers 84%.
    assert joint99 == pytest.approx(0.839, abs=0.01)
    assert abs(joint95 - 0.95) > 0.1        # the gap the reported column cannot show


def test_chi2_pit_is_uniform_when_calibrated(rng):
    z = rng.normal(size=(5000, 8))
    u = M.chi2_pit(z)
    assert stats.kstest(u, "uniform").statistic < 0.03


# ---------------------------------------------------------------------------------
# Sample-based predictives
# ---------------------------------------------------------------------------------
def test_sample_and_gaussian_coverage_agree_on_a_gaussian(rng):
    d, n, s = 3, 2000, 4000
    mu = rng.normal(size=(n, d))
    y = mu + rng.normal(size=(n, d))
    samples = mu[:, None, :] + rng.normal(size=(n, s, d))

    from_samples = M.sample_marginal_coverage(y, samples, 0.95)
    closed_form = M.gaussian_marginal_coverage(y, mu, np.ones((n, d)), 0.95)
    assert np.allclose(from_samples, closed_form, atol=0.02)


def test_kde_joint_nll_recovers_the_true_density(rng):
    """The KDE fallback should land near the analytic NLL for a Gaussian predictive.

    It is only used where a flow's exact `log_prob` is unavailable, so what matters is
    that it is not systematically off -- a KDE over 2000 draws is biased high by a little,
    and the tolerance says how much.
    """
    d, n, s = 3, 40, 2000
    mu = rng.normal(size=(n, d))
    y = mu + rng.normal(size=(n, d)) * 0.5
    samples = mu[:, None, :] + rng.normal(size=(n, s, d))

    kde = M.kde_joint_nll(y, samples)
    exact = M.mvn_nll(y, mu, np.repeat(np.eye(d)[None], n, 0))
    assert np.nanmean(kde) == pytest.approx(np.nanmean(exact), abs=0.25)


def test_kde_joint_nll_handles_one_dimension(rng):
    """d=1 is the `last1` task, and np.cov of an [s,1] block returns a scalar.

    The 0-d covariance reached _symmetrize before np.atleast_2d and raised
    AxisError, so the flow NLL of any single-level artifact -- the surface task"s
    headline column -- died inside score.py, which only warns and moves on.
    """
    n, s = 40, 2000
    mu = rng.normal(size=(n, 1))
    y = mu + rng.normal(size=(n, 1)) * 0.5
    samples = mu[:, None, :] + rng.normal(size=(n, s, 1))

    kde = M.kde_joint_nll(y, samples)
    exact = M.mvn_nll(y, mu, np.ones((n, 1, 1)))
    assert np.isfinite(kde).all()
    assert np.nanmean(kde) == pytest.approx(np.nanmean(exact), abs=0.25)


def test_sample_mean_cov_matches_numpy(rng):
    s = rng.normal(size=(7, 500, 4))
    mu, cov = M.sample_mean_cov(s)
    assert np.allclose(mu, s.mean(axis=1))
    assert np.allclose(cov[3], np.cov(s[3], rowvar=False), atol=1e-10)


# ---------------------------------------------------------------------------------
# Table rows
# ---------------------------------------------------------------------------------
def test_summarize_gaussian_reports_both_unit_systems(rng):
    d, n = 4, 500
    y_mu = np.full(d, 4.0e-4)
    y_sigma = np.full(d, 2.0e-5)
    mu = rng.normal(size=(n, d)) * 0.1
    y = mu + rng.normal(size=(n, d))
    cov = np.repeat(np.eye(d)[None], n, 0)

    row = M.summarize_gaussian(y, mu, cov, y_mu, y_sigma, "test")
    assert row["nll_ppm"] == pytest.approx(M.nll_std_to_ppm(row["nll_std"], y_sigma))
    assert row["cov95_marginal"] == pytest.approx(0.95, abs=0.03)
    assert row["cov95_joint"] == pytest.approx(0.95, abs=0.04)
    # RMSE in ppm is the standardized RMSE times sigma*1e6, here 1 * 20 ppm.
    assert row["rmse_ppm"] == pytest.approx(20.0, rel=0.1)


def test_summarize_samples_prefers_the_exact_log_prob(rng):
    d, n, s = 2, 300, 400
    y_mu, y_sigma = np.zeros(d), np.full(d, 2.0e-5)
    mu = rng.normal(size=(n, d)) * 0.1
    y = mu + rng.normal(size=(n, d))
    samples = mu[:, None, :] + rng.normal(size=(n, s, d))
    logp = np.array([stats.multivariate_normal(mu[i], np.eye(d)).logpdf(y[i]) for i in range(n)])

    row = M.summarize_samples(y, samples, y_mu, y_sigma, "nf", logprob_std=logp)
    assert row["nll_estimator"] == "flow log_prob"
    assert row["nll_std"] == pytest.approx(np.mean(-logp))

    row_kde = M.summarize_samples(y, samples, y_mu, y_sigma, "nf")
    assert row_kde["nll_estimator"] == "gaussian KDE"

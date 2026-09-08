"""metrics.py

Every number that appears in Tables 1-3, computed once, in one place, from an artifact.

Two things this module fixes about the notebook versions.

**Units.** The published NLLs are in standardized coordinates and the RMSEs are in ppm,
so the D=1, D=4 and D=20 columns of Tables 1-3 cannot be compared with each other.
The change of variables is exact and constant --
`NLL_ppm = NLL_std + sum_d log(sigma_d * 1e6)` -- so both are reported here and the
conversion cannot be forgotten.

**Coverage.** "Coverage" in the manuscript is the per-level marginal rate averaged over
levels, which says nothing about the joint 20-dimensional distribution and is exactly
what Reviewer 2 objected to. Marginal coverage is kept, because it is what the earlier
tables report, but every function that computes it has a joint counterpart next to it,
and `summarize_*` returns both.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------------
PPM = 1e6


def to_ppm(x_std: np.ndarray, y_mu: np.ndarray, y_sigma: np.ndarray) -> np.ndarray:
    """Standardized values -> ppm. Broadcasts over any leading dimensions."""
    return (np.asarray(x_std) * y_sigma + y_mu) * PPM


def cov_to_ppm(cov_std: np.ndarray, y_sigma: np.ndarray) -> np.ndarray:
    """Standardized covariance -> ppm^2."""
    s = np.asarray(y_sigma) * PPM
    return np.asarray(cov_std) * np.outer(s, s)


def nll_std_to_ppm(nll_std: float | np.ndarray, y_sigma: np.ndarray) -> float | np.ndarray:
    """Shift an NLL from standardized coordinates to ppm.

    Densities transform by the Jacobian of y_ppm = (y_std * sigma + mu) * 1e6, which is
    diagonal and constant, so the whole correction is the additive constant below. Same
    for every method and every sounding -- which is why it changes no ranking, and why
    reporting it is free.
    """
    return np.asarray(nll_std) + float(np.sum(np.log(np.asarray(y_sigma) * PPM)))


# ---------------------------------------------------------------------------------
# Gaussian predictors (ACOS, Laplace)
# ---------------------------------------------------------------------------------
def _symmetrize(cov: np.ndarray) -> np.ndarray:
    return 0.5 * (cov + np.swapaxes(cov, -1, -2))


def mvn_nll(
    y: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    jitters: Sequence[float] = (0.0, 1e-8, 1e-6, 1e-4, 1e-3),
) -> np.ndarray:
    """Per-sounding joint Gaussian NLL, `[N]`, NaN where the covariance is unusable.

    ACOS posterior covariances are frequently near-singular at this resolution, so the
    Cholesky is retried up a ladder of jitters and the sounding is dropped rather than
    silently regularized into significance. The number of drops is reported by
    `summarize_gaussian` so a table can carry an honest denominator.
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    cov = _symmetrize(np.asarray(cov, dtype=np.float64))
    n, d = y.shape
    out = np.full(n, np.nan)
    const = d * np.log(2.0 * np.pi)

    for i in range(n):
        C, r = cov[i], y[i] - mu[i]
        if not np.all(np.isfinite(C)) or not np.all(np.isfinite(r)):
            continue
        scale = max(float(np.mean(np.diag(C))), 1e-30)
        for j in jitters:
            try:
                L = np.linalg.cholesky(C + j * scale * np.eye(d))
            except np.linalg.LinAlgError:
                continue
            z = np.linalg.solve(L, r)
            out[i] = 0.5 * (const + 2.0 * np.sum(np.log(np.diag(L))) + float(z @ z))
            break
    return out


def whiten(y: np.ndarray, mu: np.ndarray, cov: np.ndarray,
           jitters: Sequence[float] = (0.0, 1e-8, 1e-6, 1e-4, 1e-3)) -> np.ndarray:
    """`z = L^-1 (y - mu)`, `[N,D]`, NaN rows where the Cholesky failed.

    Under a correct joint Gaussian the pooled `z` are i.i.d. standard normal, which is
    the basis of every joint diagnostic here and the Gaussian counterpart of the flow's
    latent map.
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    cov = _symmetrize(np.asarray(cov, dtype=np.float64))
    n, d = y.shape
    z = np.full((n, d), np.nan)
    for i in range(n):
        C, r = cov[i], y[i] - mu[i]
        if not np.all(np.isfinite(C)) or not np.all(np.isfinite(r)):
            continue
        scale = max(float(np.mean(np.diag(C))), 1e-30)
        for j in jitters:
            try:
                L = np.linalg.cholesky(C + j * scale * np.eye(d))
            except np.linalg.LinAlgError:
                continue
            z[i] = np.linalg.solve(L, r)
            break
    return z


def gaussian_marginal_coverage_mask(
    y: np.ndarray, mu: np.ndarray, var: np.ndarray, level: float
) -> np.ndarray:
    """`[N,D]` boolean: is the truth inside the central `level` interval, per level?

    The per-sounding form, so that coverage can be aggregated by something other than
    "all of them" -- by held-out reference sounding, for instance, which is the unit the
    group split makes meaningful.
    """
    half = stats.norm.ppf(0.5 + level / 2.0) * np.sqrt(np.clip(var, 1e-30, None))
    return np.abs(y - mu) <= half


def gaussian_marginal_coverage(
    y: np.ndarray, mu: np.ndarray, var: np.ndarray, level: float
) -> np.ndarray:
    """Per-level empirical coverage of the central `level` interval, `[D]`."""
    return np.mean(gaussian_marginal_coverage_mask(y, mu, var, level), axis=0)


def gaussian_marginal_pit(y: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    return np.clip(stats.norm.cdf(y, loc=mu, scale=np.sqrt(np.clip(var, 1e-30, None))),
                   1e-6, 1 - 1e-6)


def joint_coverage_from_z(z: np.ndarray, level: float) -> float:
    """Fraction of soundings inside the joint `level` highest-density region.

    For whitened residuals that region is the ball `||z||^2 <= chi2_D(level)`. This is
    the quantity the manuscript should be reporting where it currently averages marginal
    rates: with 20 correlated levels the two can differ by tens of percentage points.
    """
    z = np.asarray(z, dtype=np.float64)
    ok = np.all(np.isfinite(z), axis=1)
    if not ok.any():
        return float("nan")
    d = z.shape[1]
    m2 = np.sum(z[ok] ** 2, axis=1)
    return float(np.mean(m2 <= stats.chi2.ppf(level, df=d)))


def chi2_pit(z: np.ndarray) -> np.ndarray:
    """`F_chi2_D(||z||^2)`; uniform on [0,1] iff the joint predictive is calibrated."""
    z = np.asarray(z, dtype=np.float64)
    ok = np.all(np.isfinite(z), axis=1)
    out = np.full(len(z), np.nan)
    out[ok] = stats.chi2.cdf(np.sum(z[ok] ** 2, axis=1), df=z.shape[1])
    return out


# ---------------------------------------------------------------------------------
# Sample-based predictors (normalizing flow)
# ---------------------------------------------------------------------------------
def sample_marginal_coverage_mask(
    y: np.ndarray, samples: np.ndarray, level: float
) -> np.ndarray:
    """`[N,D]` boolean counterpart of `gaussian_marginal_coverage_mask` for samples."""
    lo = np.quantile(samples, (1 - level) / 2.0, axis=1)
    hi = np.quantile(samples, 1 - (1 - level) / 2.0, axis=1)
    return (y >= lo) & (y <= hi)


def sample_marginal_coverage(y: np.ndarray, samples: np.ndarray, level: float) -> np.ndarray:
    """`y: [N,D]`, `samples: [N,S,D]` -> per-level coverage `[D]` of the central interval."""
    return np.mean(sample_marginal_coverage_mask(y, samples, level), axis=0)


def sample_marginal_pit(y: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Rank of the truth among the predictive samples, per level, `[N,D]`."""
    return np.clip(np.mean(samples <= y[:, None, :], axis=1), 1e-6, 1 - 1e-6)


def sample_mean_cov(samples: np.ndarray):
    """Sample mean `[N,D]` and covariance `[N,D,D]` of `samples: [N,S,D]`."""
    mu = samples.mean(axis=1)
    c = samples - mu[:, None, :]
    return mu, np.einsum("nsd,nse->nde", c, c) / max(samples.shape[1] - 1, 1)


def kde_joint_nll(
    y: np.ndarray,
    samples: np.ndarray,
    bandwidth: str | float = "scott",
    jitters: Sequence[float] = (1e-8, 1e-6, 1e-4, 1e-3),
) -> np.ndarray:
    """Joint NLL of a sample-based predictive via a Gaussian KDE, `[N]`.

    Only needed where the flow's own `log_prob` is unavailable -- for a normalizing flow
    it is not, so `nf_logprob_std` in the artifact is the number to report and this is a
    cross-check on it. Kept because the near-surface notebook used it to put the flow and
    the Gaussians on one joint scale, and because it is the only option for a predictive
    given as samples (e.g. a Laplace posterior propagated through a nonlinearity).
    """
    y = np.asarray(y, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    n, s, d = samples.shape
    out = np.full(n, np.nan)
    const = d * np.log(2.0 * np.pi)

    for i in range(n):
        S = samples[i]
        # np.atleast_2d BEFORE _symmetrize: np.cov of an [s,1] block returns a 0-d
        # array, and swapping the last two axes of that raises AxisError. d=1 is not a
        # corner case here -- it is the `last1` task, where a single pressure level's
        # flow NLL can only come from a KDE over the sliced samples.
        cov = _symmetrize(np.atleast_2d(np.cov(S, rowvar=False)))
        if isinstance(bandwidth, str):
            if bandwidth != "scott":
                raise ValueError(f"unknown bandwidth rule {bandwidth!r}")
            h2 = s ** (-2.0 / (d + 4))          # Scott's factor, squared
        else:
            h2 = float(bandwidth) ** 2
        H = cov * h2
        scale = max(float(np.mean(np.diag(H))), 1e-30)
        L = None
        for j in jitters:
            try:
                L = np.linalg.cholesky(H + j * scale * np.eye(d))
                break
            except np.linalg.LinAlgError:
                continue
        if L is None:
            continue
        z = np.linalg.solve(L, (y[i] - S).T)                     # [d,s]
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        logk = -0.5 * (const + logdet + np.sum(z ** 2, axis=0))  # [s]
        m = logk.max()
        out[i] = -(m + np.log(np.mean(np.exp(logk - m))))
    return out


# ---------------------------------------------------------------------------------
# Table rows
# ---------------------------------------------------------------------------------
# A CO2 mole fraction outside this range is not a measurement. There is a
# SECOND, undeclared fill sentinel of order -1e19 in the state vectors, on roughly one
# sounding in a thousand, which `isfinite` does not catch -- and a single one of them
# turned a scene's profile RMSE into 5e8 ppm. `prepare_month.py` screens the TRUTH
# vectors on exactly this rule; nothing screened ACOS's RETRIEVED vector, so on July's
# near-surface task one contaminated sounding made the pooled mean NaN and the whole
# control cell went blank.
_PHYSICAL_PPM = (1.0, 10_000.0)


def physical_mask(y_std: np.ndarray, mu_std: np.ndarray, y_mu, y_sigma) -> np.ndarray:
    """`[N]` bool: soundings whose truth AND prediction are physically possible CO2.

    Applied to every method, not just ACOS. A flow that ever produced a non-physical
    mean would otherwise blank its own row with no indication of why, which is the
    failure this exists to have already happened once.

    Dropping a sounding is not free -- it makes the point-error columns and the NLL
    column describe marginally different sets -- so the count is reported alongside as
    `n_nonphysical` rather than absorbed silently.
    """
    lo, hi = _PHYSICAL_PPM
    ok = np.ones(len(y_std), dtype=bool)
    for a in (y_std, mu_std):
        v = to_ppm(np.asarray(a, dtype=np.float64), y_mu, y_sigma)
        ok &= np.isfinite(v).all(axis=1) & (v >= lo).all(axis=1) & (v <= hi).all(axis=1)
    return ok


def rmse_ppm(y_std: np.ndarray, mu_std: np.ndarray, y_mu, y_sigma) -> float:
    """Pooled RMSE over soundings and levels, in ppm."""
    err = to_ppm(y_std, y_mu, y_sigma) - to_ppm(mu_std, y_mu, y_sigma)
    return float(np.sqrt(np.mean(err ** 2)))


def rmse_ppm_per_level(y_std: np.ndarray, mu_std: np.ndarray, y_mu, y_sigma) -> np.ndarray:
    err = to_ppm(y_std, y_mu, y_sigma) - to_ppm(mu_std, y_mu, y_sigma)
    return np.sqrt(np.mean(err ** 2, axis=0))


def mae_ppm(y_std: np.ndarray, mu_std: np.ndarray, y_mu, y_sigma) -> float:
    """Pooled mean absolute error over soundings and levels, in ppm.

    Reported next to the RMSE because the two answer different questions about the same
    residuals: the RMSE is dominated by the tail of the error distribution and the MAE is
    not. The residuals here are heavy-tailed -- a handful of soundings carry a
    disproportionate share of the squared error -- so a method can win on one and lose on
    the other, and reporting only the RMSE hides which of the two a difference comes from.

    Unlike the RMSE, the MAE is minimized by the predictive *median*, not the mean. For
    ACOS and for the Laplace approximation those coincide; for the flow they do not, so
    `summarize_samples` reports both point estimates.
    """
    err = to_ppm(y_std, y_mu, y_sigma) - to_ppm(mu_std, y_mu, y_sigma)
    return float(np.mean(np.abs(err)))


def mae_ppm_per_level(y_std: np.ndarray, mu_std: np.ndarray, y_mu, y_sigma) -> np.ndarray:
    err = to_ppm(y_std, y_mu, y_sigma) - to_ppm(mu_std, y_mu, y_sigma)
    return np.mean(np.abs(err), axis=0)


def _sounding_abs_err(y_std, mu_std, y_mu, y_sigma) -> np.ndarray:
    """Per-sounding mean absolute error in ppm, `[N]`; what the bootstrap resamples.

    The MAE is a plain mean over soundings, so unlike the RMSE it needs no special
    bootstrap: `_boot_se` of this array is the standard error of the reported number.
    """
    err = to_ppm(y_std, y_mu, y_sigma) - to_ppm(mu_std, y_mu, y_sigma)
    return np.mean(np.abs(err), axis=1)


def _group_index(groups: np.ndarray) -> list:
    """Row indices of each group, in a fixed order, for the cluster bootstrap."""
    g = np.asarray(groups)
    order = np.argsort(g, kind="stable")
    keys, starts = np.unique(g[order], return_index=True)
    return np.split(order, starts[1:])


def _boot_replicates(n: int, groups: Optional[np.ndarray], n_boot: int, seed: int):
    """Yield `n_boot` index arrays: soundings if `groups` is None, whole groups if not.

    The sounding bootstrap treats the 4,489 test soundings as 4,489 independent tests.
    They are not: they are 96-100 stochastic replicates of each of 45 reference
    soundings, and every quantity here is strongly correlated within a reference. The
    cluster bootstrap resamples the 45 reference soundings with replacement and takes
    each one whole, which is the standard error of a statistic computed on 45 scenes --
    the number the reference-sounding-disjoint tables should carry.

    It is wider, sometimes by a factor of several, and that is the point.
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        for _ in range(n_boot):
            yield rng.integers(0, n, size=n)
        return
    blocks = _group_index(groups)
    n_g = len(blocks)
    for _ in range(n_boot):
        yield np.concatenate([blocks[j] for j in rng.integers(0, n_g, size=n_g)])


def _boot_se(values: np.ndarray, n_boot: int = 500, seed: int = 0,
             groups: Optional[np.ndarray] = None) -> float:
    """Bootstrap standard error of the mean over soundings, or over groups.

    The notebooks reported the standard deviation *across mini-batches*, which depends on
    the batch size and is not a standard error of anything. Resampling soundings is what
    the parenthesised numbers in Tables 1-3 should have been; resampling whole reference
    soundings (`groups`) is what they should be under a group-disjoint split.
    """
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    g = None if groups is None else np.asarray(groups)[ok]
    v = v[ok]
    if v.size < 2:
        return float("nan")
    return float(np.std([v[i].mean() for i in _boot_replicates(v.size, g, n_boot, seed)],
                        ddof=1))


def _boot_se_rmse(sounding_mse: np.ndarray, n_boot: int = 500, seed: int = 0,
                  groups: Optional[np.ndarray] = None) -> float:
    """Bootstrap standard error of the *RMSE*, given per-sounding mean squared error.

    Passing `sounding_mse` to `_boot_se` would give the standard error of the mean
    squared error -- a number in ppm^2, roughly `2 * RMSE` times too large, printed in a
    column headed ppm. The statistic has to be recomputed inside each bootstrap replicate,
    square root and all, which is what this does.
    """
    v = np.asarray(sounding_mse, dtype=np.float64)
    ok = np.isfinite(v)
    g = None if groups is None else np.asarray(groups)[ok]
    v = v[ok]
    if v.size < 2:
        return float("nan")
    return float(np.std([np.sqrt(v[i].mean())
                         for i in _boot_replicates(v.size, g, n_boot, seed)], ddof=1))


def _var_ratio(y: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
    """Mean squared residual over mean predictive variance. 1 is calibrated in scale."""
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).reshape(y.shape)
    var = np.asarray(var, dtype=np.float64).reshape(y.shape)
    denom = float(np.nanmean(var))
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float(np.nanmean((y - mu) ** 2) / denom)


def summarize_gaussian(
    y_std: np.ndarray, mu_std: np.ndarray, cov_std: np.ndarray,
    y_mu, y_sigma, name: str, levels: Iterable[float] = (0.95, 0.68),
    valid: Optional[np.ndarray] = None, seed: int = 0,
    groups: Optional[np.ndarray] = None,
) -> Dict:
    """One table row for a Gaussian predictive.

    `groups` is the bootstrap resampling unit -- one label per sounding, typically the
    reference sounding. Omit it for the sounding bootstrap the published tables used.
    """
    y_mu = np.asarray(y_mu, dtype=np.float64)
    y_sigma = np.asarray(y_sigma, dtype=np.float64)
    if valid is None:
        valid = np.ones(len(y_std), dtype=bool)

    nll_joint = mvn_nll(y_std[valid], mu_std[valid], cov_std[valid])
    z = whiten(y_std[valid], mu_std[valid], cov_std[valid])
    # Coverage is computed on the same subset as the NLL. Leaving the invalid rows in
    # would not raise -- a NaN interval half-width compares False -- it would just quietly
    # count every sounding with no ACOS covariance as "not covered" and drag the baseline
    # column down by however many of those there are.
    var_v = np.diagonal(cov_std[valid], axis1=-2, axis2=-1)
    y_v, mu_v = y_std[valid], mu_std[valid]

    # The point-error columns are computed on physically possible soundings only, and
    # the count of what that excluded is reported. Everything below already used
    # `valid` (ACOS covariance present) or `nanmean`; these two did not, so one bad
    # retrieved state vector took the whole cell out.
    fin = physical_mask(y_std, mu_std, y_mu, y_sigma)
    g_fin = None if groups is None else np.asarray(groups)[fin]
    y_f, mu_f = y_std[fin], mu_std[fin]

    row = {
        "method": name,
        "n": int(valid.sum()),
        "n_point": int(fin.sum()),
        "n_nonphysical": int((~fin).sum()),
        "n_joint_ok": int(np.isfinite(nll_joint).sum()),
        "rmse_ppm": rmse_ppm(y_f, mu_f, y_mu, y_sigma),
        "rmse_ppm_se": _boot_se_rmse(
            np.mean((to_ppm(y_f, y_mu, y_sigma) - to_ppm(mu_f, y_mu, y_sigma)) ** 2, axis=1),
            seed=seed, groups=g_fin,
        ),
        "mae_ppm": mae_ppm(y_f, mu_f, y_mu, y_sigma),
        "mae_ppm_se": _boot_se(_sounding_abs_err(y_f, mu_f, y_mu, y_sigma), seed=seed,
                               groups=g_fin),
        # A Gaussian's median is its mean, so the MAE-optimal point estimate is the one
        # used above. The column is written anyway, so the flow's median row is compared
        # against a number rather than against a blank.
        "mae_ppm_median": mae_ppm(y_f, mu_f, y_mu, y_sigma),
        "nll_std": float(np.nanmean(nll_joint)),
        "nll_ppm": float(nll_std_to_ppm(np.nanmean(nll_joint), y_sigma)),
        # The NLL is averaged over the valid rows only, so its bootstrap unit has to be
        # subset the same way -- otherwise the labels and the values are misaligned and
        # the resampling is of the wrong rows entirely.
        "nll_se": _boot_se(nll_joint, seed=seed,
                           groups=None if groups is None else np.asarray(groups)[valid]),
        "bootstrap": "sounding" if groups is None else "group",
        # Residual variance over mean predictive variance: 1 is calibrated in scale. It is
        # the cheapest possible check that a Gaussian predictive describes the errors it
        # is placed against, and it exists because a predictive that does not is invisible
        # in every column beside it -- the NLL is a number, the coverage is a percentage,
        # and both look like findings. The scalar Laplace shipped at 27.6 for the whole
        # project because its predictive carried no observation-noise term.
        #
        # It is a ratio of second moments, so it is not a coverage statement and will not
        # be 1 exactly for heavy-tailed errors. Only gross departures mean anything, and
        # `score.py` flags those in the LaTeX it writes rather than leaving them to be read
        # out of a CSV column nobody opens.
        "var_ratio": _var_ratio(y_v, mu_v, var_v),
    }
    for lv in levels:
        tag = f"{int(round(lv * 100))}"
        row[f"cov{tag}_marginal"] = float(np.mean(gaussian_marginal_coverage(y_v, mu_v, var_v, lv)))
        row[f"cov{tag}_joint"] = joint_coverage_from_z(z, lv)
    return row


def summarize_samples(
    y_std: np.ndarray, samples_std: np.ndarray, y_mu, y_sigma, name: str,
    logprob_std: Optional[np.ndarray] = None,
    latent_std: Optional[np.ndarray] = None,
    levels: Iterable[float] = (0.95, 0.68), seed: int = 0,
    groups: Optional[np.ndarray] = None,
) -> Dict:
    """One table row for a sample-based predictive.

    `logprob_std` is the flow's own exact density at the truth; when present it is used
    for the NLL rather than a KDE over its samples, because it is exact.
    `latent_std` is `z = f_x(y)` and gives the joint coverage without a Gaussian
    assumption -- for a correctly calibrated flow those z are standard normal, so the
    same chi-squared region applies.
    """
    y_mu = np.asarray(y_mu, dtype=np.float64)
    y_sigma = np.asarray(y_sigma, dtype=np.float64)
    mu = samples_std.mean(axis=1)
    med = np.median(samples_std, axis=1)

    nll = -np.asarray(logprob_std, dtype=np.float64) if logprob_std is not None \
        else kde_joint_nll(y_std, samples_std)

    # Screened on the same rule as the Gaussian rows, so the two are comparable and a
    # non-physical flow mean could never blank this row without saying so.
    fin = physical_mask(y_std, mu, y_mu, y_sigma)
    g_fin = None if groups is None else np.asarray(groups)[fin]
    y_f, mu_f, med_f = y_std[fin], mu[fin], med[fin]

    row = {
        "method": name,
        "n": int(len(y_std)),
        "n_point": int(fin.sum()),
        "n_nonphysical": int((~fin).sum()),
        "n_joint_ok": int(np.isfinite(nll).sum()),
        "rmse_ppm": rmse_ppm(y_f, mu_f, y_mu, y_sigma),
        "rmse_ppm_se": _boot_se_rmse(
            np.mean((to_ppm(y_f, y_mu, y_sigma) - to_ppm(mu_f, y_mu, y_sigma)) ** 2, axis=1),
            seed=seed, groups=g_fin,
        ),
        "mae_ppm": mae_ppm(y_f, mu_f, y_mu, y_sigma),
        "mae_ppm_se": _boot_se(_sounding_abs_err(y_f, mu_f, y_mu, y_sigma), seed=seed,
                               groups=g_fin),
        # The MAE-optimal point estimate of a sample-based predictive is its median, and
        # for a skewed flow posterior that is not its mean. Both are reported: `mae_ppm`
        # keeps the same point estimate as the RMSE column, so the two columns describe
        # the same predictor, and `mae_ppm_median` is the number the flow should be judged
        # on when absolute error is the loss.
        "mae_ppm_median": mae_ppm(y_f, med_f, y_mu, y_sigma),
        "mae_ppm_median_se": _boot_se(_sounding_abs_err(y_f, med_f, y_mu, y_sigma),
                                      seed=seed, groups=g_fin),
        "nll_std": float(np.nanmean(nll)),
        "nll_ppm": float(nll_std_to_ppm(np.nanmean(nll), y_sigma)),
        "nll_se": _boot_se(nll, seed=seed, groups=groups),
        "nll_estimator": "flow log_prob" if logprob_std is not None else "gaussian KDE",
        # The same scale check as the Gaussian rows, against the flow's own sample
        # variance, so the three rows of a table are comparable on it.
        "var_ratio": _var_ratio(y_std, mu, samples_std.var(axis=1, ddof=1)),
        "bootstrap": "sounding" if groups is None else "group",
    }
    for lv in levels:
        tag = f"{int(round(lv * 100))}"
        row[f"cov{tag}_marginal"] = float(np.mean(sample_marginal_coverage(y_std, samples_std, lv)))
        row[f"cov{tag}_joint"] = (
            joint_coverage_from_z(latent_std, lv) if latent_std is not None else float("nan")
        )
    return row

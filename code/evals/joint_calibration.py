"""joint_calibration.py
Joint (whole-profile) calibration diagnostics for probabilistic retrievals.

The metrics reported in the manuscript are marginal: coverage is computed per pressure
level and then averaged, and the PIT histograms are one-dimensional. Near-nominal
marginal coverage says nothing about whether the *joint* D-dimensional posterior is
calibrated -- the cross-level correlation structure is never tested. This module supplies
that test.

The idea is the same for both model families: map the observation to a space in which a
correctly specified model would produce standard normal residuals.

    Gaussian predictors (Laplace, ACOS):  z = L^-1 (x - mu),  Sigma = L L^T
    Normalizing flow:                     z = f_x(x)          (the flow's own base map)

Under correct joint calibration the pooled z are i.i.d. N(0, I_D). Two summaries follow:

    * per-coordinate z should be N(0,1)          -> QQ plots, KS statistics
    * ||z||^2 should be chi-squared with D d.o.f. -> chi2-PIT histogram, joint coverage

Note that z is invariant to the choice of target units: standardizing the targets
rescales mu and Sigma consistently, so unlike NLL these diagnostics are directly
comparable across the scalar, near-surface and full-profile tasks.

Author(s): Alejandro Calle-Saldarriaga
"""

import logging
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
from scipy import stats

# torch is imported lazily inside whiten_flow so that the array-based diagnostics
# (which are the bulk of this module) can be used from a plain numpy/scipy session.

logger = logging.getLogger(__name__)

__all__ = [
    "whiten_gaussian",
    "whiten_flow",
    "mahalanobis_sq",
    "chi2_pit",
    "normal_pit",
    "joint_coverage",
    "qq_points",
    "summarize",
    "plot_joint_calibration",
]


def whiten_gaussian(
    x: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    jitters: Sequence[float] = (0.0, 1e-10, 1e-8, 1e-6, 1e-4),
) -> np.ndarray:
    """z = L^-1 (x - mu) for each observation, with L the Cholesky factor of Sigma.

    Args:
        x:   [N, D] observed states.
        mu:  [N, D] predictive means.
        cov: [N, D, D] predictive covariances.
        jitters: escalating diagonal loading, tried in order, for covariances that are
            not numerically positive definite. Rows that fail at every level -- and rows
            with non-finite entries, e.g. ACOS soundings with no reported covariance --
            come back as NaN so callers can drop them explicitly rather than silently
            absorbing them into the statistics.

    Returns:
        [N, D] whitened residuals, NaN on rows where the factorization failed.
    """
    x = np.asarray(x, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    if x.shape != mu.shape:
        raise ValueError(f"x {x.shape} and mu {mu.shape} must match")
    n, d = x.shape
    if cov.shape != (n, d, d):
        raise ValueError(f"cov must be [{n},{d},{d}], got {cov.shape}")

    z = np.full((n, d), np.nan, dtype=np.float64)
    n_failed = 0
    for i in range(n):
        ci, di = cov[i], x[i] - mu[i]
        if not (np.all(np.isfinite(ci)) and np.all(np.isfinite(di))):
            n_failed += 1
            continue
        scale = float(np.mean(np.diag(ci))) or 1.0
        for eps in jitters:
            try:
                L = np.linalg.cholesky(ci + eps * scale * np.eye(d))
            except np.linalg.LinAlgError:
                continue
            z[i] = np.linalg.solve(L, di)
            break
        else:
            n_failed += 1
    if n_failed:
        logger.warning("whiten_gaussian: %d/%d rows not factorizable -> NaN", n_failed, n)
    return z


def whiten_flow(nf, loader, device, target_idxs: Optional[Sequence[int]] = None,
                move_batch=None) -> np.ndarray:
    """Collect z = f_x(x) over a loader for a conditional flow.

    Args:
        nf: an NFProfiles (or any object exposing `.latent(x_tuple, y)`).
        loader: yields ((bands, aux_cont, aux_cat), y, y_retr).
        device: torch device.
        target_idxs: column subset of y to use, for models trained on a slice.
        move_batch: batch -> (x_tuple, y, y_retr) mover; defaults to the profile mover.
    """
    import torch

    if move_batch is None:
        move_batch = _default_move_batch
    out = []
    was_training = nf.training
    nf.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                x_tuple, y, _ = move_batch(batch, device)
                if target_idxs is not None:
                    y = y[:, list(target_idxs)]
                out.append(nf.latent(x_tuple, y).detach().cpu().numpy())
    finally:
        nf.train(was_training)
    return np.concatenate(out, axis=0)


def _default_move_batch(batch, device):
    x_tuple, y, y_retr = batch
    bands, aux_cont, aux_cat = x_tuple
    bands = {k: (b.to(device), c.to(device)) for k, (b, c) in bands.items()}
    return (bands, aux_cont.to(device), [t.to(device) for t in aux_cat]), y.to(device), y_retr


def mahalanobis_sq(z: np.ndarray) -> np.ndarray:
    """||z||^2 per observation; chi^2_D distributed under correct calibration."""
    return np.sum(np.asarray(z, dtype=np.float64) ** 2, axis=1)


def chi2_pit(z: np.ndarray) -> np.ndarray:
    """PIT values of ||z||^2 under chi^2_D; Uniform(0,1) when jointly calibrated."""
    z = np.asarray(z, dtype=np.float64)
    return stats.chi2.cdf(mahalanobis_sq(z), df=z.shape[1])


def normal_pit(z: np.ndarray) -> np.ndarray:
    """Per-coordinate PIT values Phi(z); Uniform(0,1) when marginally calibrated."""
    return stats.norm.cdf(np.asarray(z, dtype=np.float64))


def joint_coverage(z: np.ndarray, levels: Iterable[float] = (0.68, 0.95)) -> Dict[float, float]:
    """Fraction of observations inside the joint highest-density region at each level.

    For a Gaussian this region is the ellipsoid {||z||^2 <= chi2_D.ppf(level)}, so this
    is genuine joint coverage rather than the average of D marginal coverages.
    """
    z = np.asarray(z, dtype=np.float64)
    m = mahalanobis_sq(z)
    m = m[np.isfinite(m)]
    d = z.shape[1]
    return {float(a): float(np.mean(m <= stats.chi2.ppf(a, df=d))) for a in levels}


def qq_points(z_col: np.ndarray):
    """Theoretical vs empirical standard-normal quantiles for one coordinate."""
    v = np.sort(np.asarray(z_col, dtype=np.float64))
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return np.array([]), np.array([])
    q = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    return q, v


def summarize(z: np.ndarray, name: str = "") -> Dict:
    """Compact numeric summary of a set of whitened residuals."""
    z = np.asarray(z, dtype=np.float64)
    ok = np.all(np.isfinite(z), axis=1)
    zf = z[ok]
    n, d = zf.shape if zf.size else (0, z.shape[1])
    if n == 0:
        return {"name": name, "n": 0, "n_dropped": int((~ok).sum())}

    m = mahalanobis_sq(zf)
    per_dim_ks = [float(stats.kstest(zf[:, j], "norm").statistic) for j in range(d)]
    return {
        "name": name,
        "n": int(n),
        "n_dropped": int((~ok).sum()),
        "dim": int(d),
        # Under calibration: mean 0, sd 1, mahalanobis mean D.
        "z_mean": float(zf.mean()),
        "z_sd": float(zf.std(ddof=1)),
        "per_dim_mean": zf.mean(axis=0).tolist(),
        "per_dim_sd": zf.std(axis=0, ddof=1).tolist(),
        "per_dim_ks": per_dim_ks,
        "max_per_dim_ks": float(np.max(per_dim_ks)),
        "mahalanobis_mean": float(m.mean()),
        "mahalanobis_expected": float(d),
        # KS of the chi2-PIT against Uniform(0,1): the single headline joint number.
        "chi2_pit_ks": float(stats.kstest(stats.chi2.cdf(m, df=d), "uniform").statistic),
        "joint_coverage": joint_coverage(zf),
    }


def plot_joint_calibration(results: Dict[str, np.ndarray], out_path=None,
                           max_qq_dims: int = 4, title: str = ""):
    """Three-panel figure: chi2-PIT histograms, joint coverage curves, per-coordinate QQ.

    Args:
        results: {method name -> [N, D] whitened residuals}.
        out_path: if given, save here.
        max_qq_dims: how many coordinates to overlay in the QQ panel.
    """
    import matplotlib.pyplot as plt

    n_methods = len(results)
    fig, axes = plt.subplots(3, n_methods, figsize=(4.6 * n_methods, 11), squeeze=False)
    levels = np.linspace(0.01, 0.99, 99)

    for col, (name, z) in enumerate(results.items()):
        z = np.asarray(z, dtype=np.float64)
        z = z[np.all(np.isfinite(z), axis=1)]
        d = z.shape[1]

        ax = axes[0][col]
        if len(z):
            ax.hist(chi2_pit(z), bins=20, range=(0, 1), density=True,
                    edgecolor="black", linewidth=0.5)
        ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
        ax.set_title(f"{name}\nchi$^2$ PIT (joint)", fontsize=12)
        ax.set_xlabel("PIT", fontsize=11)
        ax.set_ylabel("density", fontsize=11)

        ax = axes[1][col]
        cov = joint_coverage(z, levels)
        ax.plot(levels, [cov[a] for a in levels], linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_title("Joint coverage (HDR)", fontsize=12)
        ax.set_xlabel("nominal", fontsize=11)
        ax.set_ylabel("observed", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax = axes[2][col]
        step = max(1, d // max_qq_dims)
        for j in range(0, d, step):
            q, v = qq_points(z[:, j])
            if len(q):
                ax.plot(q, v, marker=".", linestyle="none", markersize=2,
                        label=f"level {j + 1}")
        lim = 4.0
        ax.plot([-lim, lim], [-lim, lim], linestyle="--", color="gray", linewidth=1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title("QQ of $z$ vs $N(0,1)$", fontsize=12)
        ax.set_xlabel("theoretical", fontsize=11)
        ax.set_ylabel("empirical", fontsize=11)
        ax.legend(fontsize=8, markerscale=4)

    for row in axes:
        for ax in row:
            ax.tick_params(labelsize=10)
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97 if title else 1])
    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig

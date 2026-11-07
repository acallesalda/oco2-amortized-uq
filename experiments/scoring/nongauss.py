#!/usr/bin/env python3
"""nongauss.py -- how much of the flow's advantage is shape, and where the shape comes from.

    python3 experiments/scoring/nongauss.py results/group_split_v1/*.npz \
        --split-csv group_split_v1_cov.csv --out tables/

Section 4.5 of the manuscript argues non-Gaussianity from six pictures. The open question is
the obvious next question -- whether the non-Gaussian structure was put into the simulation
or came out of the inference -- and whether the extra flexibility buys anything.
Both are answerable from the artifacts `predict.py` already wrote, with no GPU and no
retraining. Four tables:

  1. `nongauss_shape`    -- the flow against **its own moment-matched Gaussian**: the
     Gaussian N(m_i, C_i) whose mean and covariance are those of the flow's samples for
     that sounding. The two predictives agree exactly on the first two moments, so every
     difference in NLL or coverage between them is shape beyond second order and nothing
     else. This is the number that separates "the flow is better calibrated in scale"
     (which a rescaled Laplace could also achieve) from "the posterior is not Gaussian".
     Its uncertainty is reported twice: resampling rows, as Tables 1-3 do, and resampling
     reference soundings, which is the unit of independence -- the test partition has
     4,489 rows but 45 scenes.

  2. `nongauss_levels`   -- per level: the skewness and excess kurtosis of the flow's
     marginal posterior, averaged over soundings and, more to the point, their spread
     *within a cluster*. The state ensemble is fixed inside a cluster, so a prior cannot
     make posterior shape vary from sounding to sounding; whatever spread survives the
     Monte Carlo error sqrt(6/S) is the likelihood doing the shaping. The same table
     carries the prior's own sd/skewness/kurtosis on the scored soundings next to the
     flow's RMSE, which is what says how much of the posterior is prior.

  3. `nongauss_clusters` -- every headline number broken down by geophysical cluster. The
     ensemble is fitted separately in each, so the nine clusters are the only variation in
     the state-generating mechanism this experiment contains, and the per-cluster spread of
     calibration is the evidence that the result is not an artifact of one particular p(x).

  4. `nongauss_mixture`  -- what the mechanism actually is: the number of mixture components
     in use per cluster and their shares, read from `mixture_component` in the split table,
     plus the ACOS degrees of freedom for the CO2 profile. Descriptive, but it is the fact
     the question really is, and it is ours to state.

Everything is reported in standardized units and in ppm, as in `score.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
for p in (str(_ROOT / "code"), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evals import metrics as M                      # noqa: E402
from evals.artifacts import Artifact                # noqa: E402
from evals.tables import md_table                  # noqa: E402
# The numpy-only half of the sampling module: the stage schedule, the plug-in penalty and
# the 1/S fit. It imports torch only if asked to sample, which this never does.
from evals.gaussianize import extrapolate as gz_extrapolate   # noqa: E402

PPM = 1e6
CHUNK = 256          # soundings per chunk; [CHUNK, S, D] in float64 is the working set


# ---------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------
def _cluster_labels(art: Artifact, split: pd.DataFrame | None):
    """Cluster label per scored sounding, from the artifact or joined from the split table.

    Artifacts written before the column-name fix in `predict.py` carry no cluster at all --
    the split table calls it `clst` and `predict.py` was looking for `cluster` -- so the
    join by ID is not a convenience, it is the only route for those files.
    """
    for key in ("clst", "cluster"):
        v = art.get(key)
        if v is not None:
            return np.asarray(v)
    if split is None:
        return None
    ids = np.asarray(art["ids"]).astype(str)
    for key in ("clst", "cluster"):
        if key in split.columns:
            m = split.set_index(split["ID"].astype(str))[key]
            return m.reindex(ids).to_numpy()
    return None


def _ref_labels(art: Artifact, split: pd.DataFrame | None):
    """Reference-sounding label per scored sounding -- the unit of independence.

    The test partition of the group split holds 4,489 rows but only 45 reference soundings,
    100 replicates of each. A bootstrap that resamples rows treats replicates of one scene
    as independent draws and reports a standard error that is far too small; the honest one
    resamples scenes. Both are reported here so the size of the difference is visible.

    A reference sounding is identified by its cluster AND its ref number: ref numbers restart
    in every cluster, so grouping on `ref` alone merges distinct reference soundings that
    share a number (37 groups instead of 45 on the reference-disjoint test partition). The
    sounding ID `clstNNNN_refNNNN_repNNNN` carries both, so its prefix is the label; the
    `clst`/`ref` columns are the fallback when an ID does not have that form.
    """
    ids = np.asarray(art["ids"]).astype(str)
    if all("_ref" in i and "_rep" in i for i in ids):
        return np.array([i.rsplit("_rep", 1)[0] for i in ids])
    clusters = _cluster_labels(art, split)
    ref = None
    for key in ("ref", "reference_sounding"):
        v = art.get(key)
        if v is not None:
            ref = np.asarray(v)
            break
    if ref is None and split is not None and "ID" in split.columns:
        idx = split.set_index(split["ID"].astype(str))
        for key in ("ref", "reference_sounding"):
            if key in idx.columns:
                ref = idx[key].reindex(ids).to_numpy()
                break
    if ref is None or clusters is None:
        return None
    return np.char.add(np.asarray(clusters).astype(str),
                       np.char.add("/", np.asarray(ref).astype(str)))


def _boot_se_grouped(values, groups, n_boot: int = 500, seed: int = 0) -> float:
    """Bootstrap standard error of a mean, resampling whole groups."""
    v = np.asarray(values, dtype=np.float64)
    g = np.asarray(groups)
    ok = np.isfinite(v) & pd.notna(g)
    v, g = v[ok], g[ok]
    if v.size < 2:
        return float("nan")
    _, inv = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]
    if len(members) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(members), size=len(members))
        means[b] = v[np.concatenate([members[p] for p in pick])].mean()
    return float(np.std(means, ddof=1))


def _joined_truth(art: Artifact, split: pd.DataFrame | None, levels):
    """True states in ppm for the scored soundings, taken from the split table.

    The artifact already holds the truth in standardized coordinates; this exists so the
    prior moments are read off the same file the simulation ensemble is described from, and
    so a mismatch between the two shows up as a disagreement rather than staying hidden.
    """
    if split is None or "ID" not in split.columns:
        return None
    ids = np.asarray(art["ids"]).astype(str)
    idx = split.set_index(split["ID"].astype(str))
    if not levels:
        if "true_xco2" not in idx.columns:
            return None
        return idx["true_xco2"].reindex(ids).to_numpy(dtype=np.float64)[:, None] * PPM
    cols = [f"true_state_vector_{int(lv):02d}" for lv in levels]
    if any(c not in idx.columns for c in cols):
        return None
    return idx[cols].reindex(ids).to_numpy(dtype=np.float64) * PPM


def _moments(samples, chunk: int = CHUNK):
    """Per-sounding mean, covariance, skewness and excess kurtosis of `[N,S,D]` samples.

    Chunked over soundings: the 20-level artifact is [4489, 512, 20] in float16, which is
    0.7 GB once promoted to float64, and the covariance einsum would want a second copy.
    """
    n, s, d = samples.shape
    mu = np.empty((n, d))
    cov = np.empty((n, d, d))
    skew = np.empty((n, d))
    exk = np.empty((n, d))
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        x = np.asarray(samples[a:b], dtype=np.float64)
        m = x.mean(axis=1)
        c = x - m[:, None, :]
        mu[a:b] = m
        cov[a:b] = np.einsum("nsd,nse->nde", c, c) / max(s - 1, 1)
        m2 = np.mean(c ** 2, axis=1)
        sd = np.sqrt(np.clip(m2, 1e-300, None))
        skew[a:b] = np.mean(c ** 3, axis=1) / sd ** 3
        exk[a:b] = np.mean(c ** 4, axis=1) / np.clip(m2 ** 2, 1e-300, None) - 3.0
    return mu, cov, skew, exk


def _mc_se(s: int):
    """Monte Carlo sd of the sample skewness / excess kurtosis under normality."""
    return float(np.sqrt(6.0 / s)), float(np.sqrt(24.0 / s))


def _describe(art: Artifact) -> str:
    return f"{art.meta.task}/{art.meta.split_name}"


# ---------------------------------------------------------------------------------
# 1. the moment-matched Gaussian control
# ---------------------------------------------------------------------------------
def shape_row(art: Artifact, split: pd.DataFrame | None = None, seed: int = 0):
    """Flow vs its own moment-matched Gaussian: the part of the gain that is shape."""
    if not art.has("nf_samples_std"):
        return None
    y = np.asarray(art["y_std"], dtype=np.float64)
    y_sigma = np.asarray(art.y_sigma, dtype=np.float64)
    # Cast once. The samples are stored in float16 and `score.py` promotes them the same
    # way before it computes anything, so this is what keeps the coverage columns here
    # numerically identical to the ones in Tables 1-3 rather than merely close.
    samples = np.asarray(art["nf_samples_std"], dtype=np.float64)
    n, s, d = samples.shape

    mu, cov, _, _ = _moments(samples)
    nll_ctrl = M.mvn_nll(y, mu, cov)

    if art.has("nf_logprob_std"):
        nll_flow = -np.asarray(art["nf_logprob_std"], dtype=np.float64)
        estimator = "flow log_prob"
    else:
        # A marginal of the flow has no closed-form density; score.py falls back to the
        # same KDE and records the fact. The comparison is then KDE-vs-Gaussian, which is
        # not a clean shape test -- the KDE carries its own bias -- so the row is kept but
        # flagged.
        nll_flow = M.kde_joint_nll(y, np.asarray(samples, dtype=np.float64))
        estimator = "gaussian KDE (not a clean shape test)"

    gap = nll_ctrl - nll_flow                        # > 0 means the flow's shape helps
    ok = np.isfinite(gap)
    shift = float(np.sum(np.log(y_sigma * PPM)))
    var = np.diagonal(cov, axis1=-2, axis2=-1)
    z = M.whiten(y, mu, cov)
    refs = _ref_labels(art, split)

    return {
        "task": art.meta.task,
        "split": art.meta.split_name,
        "n": int(n), "dim": int(d), "n_samples": int(s),
        "nll_estimator": estimator,
        "nll_flow_std": float(np.nanmean(nll_flow)),
        "nll_gauss_control_std": float(np.nanmean(nll_ctrl)),
        "shape_gap_nats": float(np.nanmean(gap[ok])),
        "shape_gap_se": M._boot_se(gap[ok], seed=seed),
        # The one to quote. 100 replicates of a reference sounding are not 100 independent
        # soundings, and the row bootstrap above does not know that.
        "shape_gap_se_by_ref": (float("nan") if refs is None
                                else _boot_se_grouped(gap, refs, seed=seed)),
        "n_ref": (int(pd.Series(refs).nunique()) if refs is not None else -1),
        "shape_gap_frac_positive": float(np.mean(gap[ok] > 0)),
        # A Gaussian fitted by plug-in from S samples pays about d(d+3)/(4S) nats for the
        # estimation itself. That handicap is charged to the control, so it inflates the
        # gap: 0.22 nats at d=20, S=512, and 0.002 at d=1. Subtract before believing.
        "plugin_handicap_nats": float(d * (d + 3) / (4.0 * s)),
        "nll_flow_ppm": float(np.nanmean(nll_flow)) + shift,
        "nll_gauss_control_ppm": float(np.nanmean(nll_ctrl)) + shift,
        "cov95_flow": float(np.mean(M.sample_marginal_coverage(y, samples, 0.95))),
        "cov95_gauss_control": float(np.mean(M.gaussian_marginal_coverage(y, mu, var, 0.95))),
        "cov68_flow": float(np.mean(M.sample_marginal_coverage(y, samples, 0.68))),
        "cov68_gauss_control": float(np.mean(M.gaussian_marginal_coverage(y, mu, var, 0.68))),
        "cov95_joint_gauss_control": M.joint_coverage_from_z(z, 0.95),
        "cov95_joint_flow": (M.joint_coverage_from_z(np.asarray(art["nf_latent_std"]), 0.95)
                             if art.has("nf_latent_std") else float("nan")),
    }


# ---------------------------------------------------------------------------------
# 1b. the same control, from the streamed pass  (predict.py --gaussianize)
# ---------------------------------------------------------------------------------
def _stages(art: Artifact) -> list:
    return list((art.meta.nf or {}).get("gaussianize_stages") or [])


def _pit_coverage(pit: np.ndarray, level: float) -> float:
    """Mean marginal coverage from PIT values, `pit: [N,D]`.

    The truth lies inside the central `level` interval exactly when its PIT is inside
    `((1-level)/2, 1-(1-level)/2)`. This differs from `metrics.sample_marginal_coverage`
    only in the quantile convention -- that one interpolates between order statistics,
    this one is the empirical CDF -- so the two agree to O(1/S) and neither is wrong. The
    PIT form is used here because the streamed pass never keeps the samples that
    `np.quantile` would need.
    """
    lo = (1.0 - level) / 2.0
    return float(np.mean((pit > lo) & (pit < 1.0 - lo)))


def shape_row_streamed(art: Artifact, split: pd.DataFrame | None = None, seed: int = 0):
    """The moment-matched Gaussian control from `--gaussianize`, with the 1/S intercept.

    Preferred over `shape_row` whenever the artifact carries the streamed pass: the control
    there is fitted from thousands of samples rather than 512, and the stage schedule makes
    its remaining estimation bias measurable rather than assumed.
    """
    stages = _stages(art)
    if not stages or not art.has("nfg_nll_std"):
        return None

    y = np.asarray(art["y_std"], dtype=np.float64)
    y_mu, y_sigma = art.y_mu, art.y_sigma
    n, d = y.shape
    nll_ctrl_stage = np.asarray(art["nfg_nll_std"], dtype=np.float64)      # [N,K]
    mu_stage = np.asarray(art["nfg_mu_std"], dtype=np.float64)             # [N,K,D]
    var_stage = np.asarray(art["nfg_var_std"], dtype=np.float64)           # [N,K,D]
    pit_stage = np.asarray(art["nfg_pit_std"], dtype=np.float64)           # [N,K,D]
    last = len(stages) - 1

    if art.has("nf_logprob_std"):
        nll_flow = -np.asarray(art["nf_logprob_std"], dtype=np.float64)
        estimator = "flow log_prob"
    else:
        # `last4_marginal` only. The flow has no closed-form marginal density, so there is
        # no exact number to put on the flow's side and the gap is not a clean shape test.
        # The control side is unaffected -- it is a Gaussian either way -- so the row is
        # computed and flagged rather than dropped.
        nll_flow = M.kde_joint_nll(y, np.asarray(art["nf_samples_std"], dtype=np.float64))
        estimator = "gaussian KDE (not a clean shape test)"

    gap = nll_ctrl_stage[:, last] - nll_flow
    ok = np.isfinite(gap)
    refs = _ref_labels(art, split)
    shift = float(np.sum(np.log(np.asarray(y_sigma, dtype=np.float64) * PPM)))

    # The control's mean NLL as a function of budget, and the a + b/S fit through it. `a`
    # is what the control would score with its moments known exactly; that is the number
    # the shape gap should be measured against, because the flow pays no such penalty.
    per_stage = np.nanmean(nll_ctrl_stage, axis=0)
    fit = gz_extrapolate(stages, per_stage)

    row = {
        "task": art.meta.task,
        "split": art.meta.split_name,
        "n": int(n), "dim": int(d),
        "control": "streamed",
        "n_samples_control": int(stages[last]),
        "n_stages": len(stages),
        "nll_estimator": estimator,
        "nll_flow_std": float(np.nanmean(nll_flow)),
        "nll_gauss_control_std": float(per_stage[last]),
        "nll_gauss_control_extrap_std": fit["a"],
        "shape_gap_nats": float(np.nanmean(gap[ok])),
        # The one to quote: the control's estimation bias is removed by the fit rather
        # than merely made small, so this does not depend on the budget that was affordable.
        "shape_gap_extrap_nats": fit["a"] - float(np.nanmean(nll_flow)),
        "shape_gap_se": M._boot_se(gap[ok], seed=seed),
        "shape_gap_se_by_ref": (float("nan") if refs is None
                                else _boot_se_grouped(gap, refs, seed=seed)),
        "n_ref": (int(pd.Series(refs).nunique()) if refs is not None else -1),
        "shape_gap_frac_positive": float(np.mean(gap[ok] > 0)),
        # A fitted slope near d(d+3)/4 is the sign that the trend really is the plug-in
        # penalty and the intercept means what it is being read to mean.
        "fit_slope_b": fit["b"],
        "fit_slope_expected": d * (d + 3) / 4.0,
        "fit_r2": fit["r2"],
        "plugin_handicap_nats": d * (d + 3) / (4.0 * stages[last]),
        "nll_flow_ppm": float(np.nanmean(nll_flow)) + shift,
        "nll_gauss_control_ppm": float(per_stage[last]) + shift,
        "cov95_flow": _pit_coverage(pit_stage[:, last, :], 0.95),
        "cov95_gauss_control": float(np.mean(M.gaussian_marginal_coverage(
            y, mu_stage[:, last, :], var_stage[:, last, :], 0.95))),
        "cov68_flow": _pit_coverage(pit_stage[:, last, :], 0.68),
        "cov68_gauss_control": float(np.mean(M.gaussian_marginal_coverage(
            y, mu_stage[:, last, :], var_stage[:, last, :], 0.68))),
        "cov95_joint_flow": (M.joint_coverage_from_z(np.asarray(art["nf_latent_std"]), 0.95)
                             if art.has("nf_latent_std") else float("nan")),
        # The flow and the control share a mean by construction, so this is one number for
        # both and it is a check, not a result: it must sit on top of the RMSE that
        # score.py reports from the 512 stored samples, up to the Monte Carlo error of the
        # sample mean. If it does not, the two passes are not looking at the same model.
        "rmse_ppm_shared": M.rmse_ppm(y, mu_stage[:, last, :], y_mu, y_sigma),
    }
    if art.has("nfg_cov_std"):
        # The final stage recomputed on the CPU from the stored covariance, through the
        # same `metrics.mvn_nll` the tables use. It must match `nll_gauss_control_std` to
        # rounding: the in-job torch version and this one are independent implementations
        # of the same quantity, and that is the point of computing it twice.
        recomputed = M.mvn_nll(y, mu_stage[:, last, :],
                               np.asarray(art["nfg_cov_std"], dtype=np.float64))
        row["nll_control_recomputed_std"] = float(np.nanmean(recomputed))
        row["recompute_max_abs_diff"] = float(
            np.nanmax(np.abs(recomputed - nll_ctrl_stage[:, last])))
    return row


def stage_rows(art: Artifact) -> pd.DataFrame:
    """One row per sample budget: what the control costs, and what converges with it."""
    stages = _stages(art)
    if not stages or not art.has("nfg_nll_std"):
        return pd.DataFrame()
    y = np.asarray(art["y_std"], dtype=np.float64)
    y_mu, y_sigma = art.y_mu, art.y_sigma
    d = y.shape[1]
    nll = np.asarray(art["nfg_nll_std"], dtype=np.float64)
    mu = np.asarray(art["nfg_mu_std"], dtype=np.float64)
    var = np.asarray(art["nfg_var_std"], dtype=np.float64)
    pit = np.asarray(art["nfg_pit_std"], dtype=np.float64)

    out = []
    for k, s in enumerate(stages):
        out.append(dict(
            task=art.meta.task, split=art.meta.split_name, n_samples=int(s),
            nll_gauss_control_std=float(np.nanmean(nll[:, k])),
            plugin_penalty_nats=d * (d + 3) / (4.0 * s),
            # The posterior mean is itself a Monte Carlo average, so this falls as the
            # budget rises. It is how much of the published RMSE was sampling noise.
            rmse_ppm=M.rmse_ppm(y, mu[:, k, :], y_mu, y_sigma),
            cov95_gauss_control=float(np.mean(
                M.gaussian_marginal_coverage(y, mu[:, k, :], var[:, k, :], 0.95))),
            cov95_flow=_pit_coverage(pit[:, k, :], 0.95),
            cov68_flow=_pit_coverage(pit[:, k, :], 0.68),
        ))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------
# 2. per-level shape, and the prior beside it
# ---------------------------------------------------------------------------------
def level_rows(art: Artifact, split: pd.DataFrame | None) -> pd.DataFrame:
    if not art.has("nf_samples_std"):
        return pd.DataFrame()
    y = np.asarray(art["y_std"], dtype=np.float64)
    y_mu, y_sigma = art.y_mu, art.y_sigma
    samples = np.asarray(art["nf_samples_std"], dtype=np.float64)
    n, s, d = samples.shape
    levels = list(art.meta.levels or [0])

    mu, _, skew, exk = _moments(samples)
    se_skew, se_kurt = _mc_se(s)

    clusters = _cluster_labels(art, split)
    truth_ppm = _joined_truth(art, split, art.meta.levels)
    if truth_ppm is None or not np.all(np.isfinite(truth_ppm)):
        truth_ppm = M.to_ppm(y, y_mu, y_sigma)

    rmse = M.rmse_ppm_per_level(y, mu, y_mu, y_sigma)
    out = []
    for j, lv in enumerate(levels):
        # Spread of per-sounding skewness inside a cluster. The state ensemble is fixed
        # within a cluster, so a prior cannot produce this; anything above the Monte Carlo
        # error is the likelihood shaping the posterior sounding by sounding.
        within = np.nan
        if clusters is not None:
            sds = [float(np.std(skew[clusters == c, j], ddof=1))
                   for c in pd.unique(clusters[pd.notna(clusters)])
                   if int(np.sum(clusters == c)) > 5]
            within = float(np.mean(sds)) if sds else np.nan
        prior = truth_ppm[:, j]
        prior = prior[np.isfinite(prior)]
        p_sd = float(np.std(prior))
        p_c = prior - prior.mean()
        out.append(dict(
            task=art.meta.task, split=art.meta.split_name, level=lv,
            nf_skew_mean=float(np.mean(skew[:, j])),
            nf_skew_absmean=float(np.mean(np.abs(skew[:, j]))),
            nf_skew_sd_across_soundings=float(np.std(skew[:, j], ddof=1)),
            nf_skew_sd_within_cluster=within,
            nf_exkurt_mean=float(np.mean(exk[:, j])),
            mc_se_skew=se_skew, mc_se_exkurt=se_kurt,
            prior_sd_ppm=p_sd,
            prior_skew=float(np.mean(p_c ** 3) / p_sd ** 3) if p_sd > 0 else np.nan,
            prior_exkurt=float(np.mean(p_c ** 4) / p_sd ** 4 - 3.0) if p_sd > 0 else np.nan,
            nf_rmse_ppm=float(rmse[j]),
            rmse_over_prior_sd=float(rmse[j] / p_sd) if p_sd > 0 else np.nan,
        ))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------
# 3. by cluster
# ---------------------------------------------------------------------------------
def cluster_rows(art: Artifact, split: pd.DataFrame | None, seed: int = 0) -> pd.DataFrame:
    clusters = _cluster_labels(art, split)
    if clusters is None:
        return pd.DataFrame()
    y = np.asarray(art["y_std"], dtype=np.float64)
    y_mu, y_sigma = art.y_mu, art.y_sigma
    # `_LazyArrays` re-reads from the .npz on every lookup, so the samples are pulled once
    # here rather than once per cluster inside the loop.
    nf = (np.asarray(art["nf_samples_std"], dtype=np.float64)
          if art.has("nf_samples_std") else None)
    out = []
    for c in pd.unique(clusters[pd.notna(clusters)]):
        m = np.asarray(clusters == c)
        if int(m.sum()) < 5:
            continue
        yc = y[m]
        base = dict(task=art.meta.task, split=art.meta.split_name, cluster=c,
                    n_cluster=int(m.sum()),
                    prior_sd_ppm=float(np.mean(np.std(M.to_ppm(yc, y_mu, y_sigma), axis=0))))

        if art.has("acos_mu_std") and art.has("acos_cov_std"):
            valid = art.get("acos_valid")
            valid = None if valid is None else np.asarray(valid, dtype=bool)[m]
            out.append({**base, **M.summarize_gaussian(
                yc, np.asarray(art["acos_mu_std"], dtype=np.float64)[m],
                np.asarray(art["acos_cov_std"], dtype=np.float64)[m],
                y_mu, y_sigma, "ACOS L2FP (v10)", valid=valid, seed=seed)})
        if art.has("lap_mu_std") and art.has("lap_cov_std"):
            out.append({**base, **M.summarize_gaussian(
                yc, np.asarray(art["lap_mu_std"], dtype=np.float64)[m],
                np.asarray(art["lap_cov_std"], dtype=np.float64)[m],
                y_mu, y_sigma, "Laplace", seed=seed)})
        if nf is not None:
            lp, lat = art.get("nf_logprob_std"), art.get("nf_latent_std")
            out.append({**base, **M.summarize_samples(
                yc, nf[m], y_mu, y_sigma, "Normalizing flow",
                logprob_std=None if lp is None else np.asarray(lp)[m],
                latent_std=None if lat is None else np.asarray(lat)[m], seed=seed)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------
# 4. what the one mechanism is
# ---------------------------------------------------------------------------------
def mixture_table(split: pd.DataFrame) -> pd.DataFrame:
    """Components in use per cluster, their shares, and the ACOS CO2 degrees of freedom.

    Descriptive, and deliberately so: the question is what the state-generating mechanism is, and
    the honest answer is one mechanism, delivered, whose component structure we can read off
    the files rather than inherit from the published method.
    """
    if "clst" not in split.columns or "mixture_component" not in split.columns:
        return pd.DataFrame()
    out = []
    for c, g in split.groupby("clst"):
        comp = g["mixture_component"].dropna()
        counts = comp.value_counts().sort_index()
        shares = counts / max(len(comp), 1)
        row = dict(cluster=c, n=int(len(g)), n_components=int(counts.size),
                   min_share=float(shares.min()) if len(shares) else np.nan,
                   max_share=float(shares.max()) if len(shares) else np.nan,
                   components=",".join(str(int(k)) for k in counts.index))
        if "dof_co2" in g.columns:
            row["dof_co2_mean"] = float(g["dof_co2"].mean())
            row["dof_co2_sd"] = float(g["dof_co2"].std())
        # Reference soundings per partition, so the per-cluster table's denominators -- and
        # any cluster with no test soundings at all -- are visible in the same place.
        if {"split", "ref"} <= set(g.columns):
            for sp in ("train", "val", "test"):
                row[f"n_ref_{sp}"] = int(g.loc[g["split"] == sp, "ref"].nunique())
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------
def _md(df: pd.DataFrame) -> str:
    # Kept as a name so the call sites below do not change; the implementation moved
    # to evals/tables.py when spectral_attribution.py needed the same fallback and the
    # attribution jobs died on 4 Sep 2026 for want of it. See that module's docstring.
    return md_table(df)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("artifacts", nargs="+")
    ap.add_argument("--split-csv", default=None,
                    help="the split table the artifacts were scored on. Supplies the "
                         "cluster labels that artifacts written before the predict.py "
                         "column-name fix do not carry, plus mixture_component and "
                         "dof_co2 for the mechanism table.")
    ap.add_argument("--out", default="tables")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(args.split_csv) if args.split_csv else None

    shape, levels, clusters, budgets = [], [], [], []
    for path in args.artifacts:
        art = Artifact.load(path, lazy=True)
        print(f"-- {_describe(art)}  ({path})")
        # The streamed control when the artifact has one, the 512-sample control when it
        # does not. Never both: two rows for one task, differing only in how well the
        # control was estimated, is an invitation to quote the wrong one.
        row = shape_row_streamed(art, split, seed=args.seed)
        if row is None:
            row = shape_row(art, split, seed=args.seed)
            if row is not None:
                row["control"] = "stored samples"
        if row is not None:
            shape.append(row)
        budgets.append(stage_rows(art))
        levels.append(level_rows(art, split))
        clusters.append(cluster_rows(art, split, seed=args.seed))

    def _cat(frames):
        keep = [d for d in frames if not d.empty]
        return pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()

    shape_df, levels_df, clusters_df = pd.DataFrame(shape), _cat(levels), _cat(clusters)
    budgets_df = _cat(budgets)
    mixture_df = mixture_table(split) if split is not None else pd.DataFrame()

    for name, df in (("nongauss_shape", shape_df), ("nongauss_budgets", budgets_df),
                     ("nongauss_levels", levels_df),
                     ("nongauss_clusters", clusters_df), ("nongauss_mixture", mixture_df)):
        if not df.empty:
            df.to_csv(out_dir / f"{name}.csv", index=False)
            print(f"wrote {out_dir / (name + '.csv')}  ({len(df)} rows)")

    md = ["# Non-Gaussianity diagnostics\n",
          "\n## 1. The flow against its own moment-matched Gaussian\n\n",
          "Same mean, same covariance, per sounding, so `shape_gap_nats` is what shape "
          "alone is worth. Subtract `plugin_handicap_nats` before believing it.\n\n",
          _md(shape_df),
          "\n### 1b. The control as a function of sample budget\n\n",
          "`nll_gauss_control_std` should fall like `plugin_penalty_nats` and flatten; the "
          "intercept of that trend is `shape_gap_extrap_nats` above. `rmse_ppm` falling "
          "with the budget is the Monte Carlo error of the posterior mean, and says how "
          "much of the published RMSE was sampling noise.\n\n",
          _md(budgets_df),
          "\n## 2. Per level: posterior shape, and the prior beside it\n\n",
          "`nf_skew_sd_within_cluster` is the argument that the shaping is done by the "
          "likelihood: the state ensemble is fixed inside a cluster, so a prior cannot "
          "make this number exceed `mc_se_skew`. `rmse_over_prior_sd` near 1 means the "
          "level is prior-dominated and the posterior shape there is largely inherited.\n\n",
          _md(levels_df),
          "\n## 3. By cluster\n\n",
          "Nine separately fitted state ensembles are the only variation in the "
          "state-generating mechanism this experiment contains.\n\n",
          _md(clusters_df),
          "\n## 4. The mechanism itself\n\n",
          _md(mixture_df)]
    (out_dir / "nongauss.md").write_text("".join(md), encoding="utf-8")
    print(f"wrote {out_dir / 'nongauss.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

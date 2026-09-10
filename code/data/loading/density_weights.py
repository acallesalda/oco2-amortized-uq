"""density_weights.py
Per-row training weights from the local density of the *covariates the model sees*.

Why this exists, and what it can and cannot do
----------------------------------------------
`build_balanced_sampler` equalises a categorical column -- in practice `clst`, the SOM
cluster -- and a cluster-balanced training run measures what that buys: cluster 9 improved
29.5 % on XCO2, cluster 8 moved 0.2 %, and the pooled 20-level number moved -1.0 % (not
resolved). The asymmetry has an explanation that is not about counts at all. Cluster 9 is
*identifiable from the network's inputs* -- its prior surface pressure is 554-749 hPa
against 734-1024 for every other cluster -- and cluster 8 is not: the nearest training
scenes to `8/4` and `8/15` in the model's own covariate space belong to clusters 6 and 4.
Upweighting a regime the network cannot recognise cannot buy that regime an intercept.

That says the right thing to weight by is *support in covariate space*, not membership of
a label the network never sees. `clst` is in the `simulation` tier of covariates.py and is
inadmissible as an input for exactly that reason, so using it to shape the loss is a back
door to the same information. A density estimated over the admissible covariates has no
such problem: it is a function of what the model is given.

What this module does NOT claim
-------------------------------
Weighting sparse regions is not the same intervention as adding scenes there, and on this
data it is measurably weaker. The estimator below, at alpha=0.5 on `group_split_v1`'s 361
training scenes with the `scene_prior` preset, moves the two rare clusters like this:

    cluster 9   share of loss 3.9 % -> 6.8 %   (weight 2.19x the lightest cluster)
    cluster 8   share of loss 3.3 % -> 3.3 %   (weight 1.22x)
    cluster 3   share of loss 19.4 % -> 24.4 % (the widest spread in covariate space)

So it reproduces the cluster-balanced run's asymmetry from the input geometry rather than
from cluster counts -- cluster 9 gets weight because it is isolated, cluster 8 gets none because it is
not -- and it hands the largest single increase to a cluster that is already at 0.62x
ACOS. Its Spearman correlation with `n_cluster ** -alpha` is 0.36, so it is *not* a
smoothed version of the balanced run; it is a different intervention that happens to agree
about cluster 9. Read the two together and the honest statement is that the 20-level
task's deficit is not a weighting problem: its flow over-fits (validation-minus-training
gap +17.4 nats, joint 95 % coverage 27.9 %), and no reweighting of a correctly-specified
loss addresses that.

The two exponents
-----------------
    alpha   rarity. w_scene proportional to p_hat(x_scene) ** -alpha. 0 = off, 1 = full
            inverse density, 0.5 = square root, the same tempering `balance_alpha` applies
            to the categorical sampler and for the same reason: alpha=1 lets a handful of
            isolated scenes carry the gradient and risks memorising them.
    beta    scene-proportionality. Rows of a scene are downweighted by n_rows_in_scene
            ** beta, so beta=1 gives every training *scene* equal expected weight whatever
            its replicate count and beta=0 keeps the published replicate-proportional
            behaviour.

beta exists because of the February extension. `ext_v1` folded in 200 replicates of each
extension scene against February's 100, which put clusters 8 and 9 at 42.5 % of the
training rows while they hold 25.1 % of the training scenes and 8.8 % of validation and
test. Capping the replicates in the prep job fixes that upstream, at the cost of rebuilding
the split; beta=1 fixes the same thing in the sampler, on the split that is already built
and staged, which is the only reason a density arm can be submitted without a prep job.

Scene-level by construction
---------------------------
Every covariate in the `scene`/`scene_prior` presets except `prior_psfc_pa` is constant
across a scene's replicates -- verified on `group_split_v1_cov.csv`: 361 scenes, one
distinct value per scene for all of SlrZen_fix, sec_sza, vza_deg, sec_vza, airmass,
XCO2Prior, AODPrior1, AODPrior2, Footprint_fix, AerType1_fix, AerType2_fix. A replicate is
a fresh draw of the discrepancy and the noise, not a new point in covariate space. So the
density is estimated over *scenes* (361 or 447 points), not over rows (36,003 or 58,115),
which is both the honest effective sample size and three orders of magnitude cheaper.

`prior_psfc_pa` does vary within a scene, and on the training partition it is rebuilt from
the truth (`prep_ext.sh`'s verification block says so). Its scene *median* is used, which
is a scene-level quantity; the per-replicate values are not, because a weight that tracked
them would be a function of the target.

The estimator
-------------
A Gaussian kernel on standardized scene coordinates, bandwidth set to the median k-th
nearest-neighbour distance:

    p_hat(x_i) = (1/n) sum_j exp(-||z_i - z_j||^2 / (2 h^2)),  h = median_i r_k(x_i)

Relative values are all that is used, so the kernel's normalizing constant -- the part
that carries the curse of dimensionality at n=361 in 24 dimensions -- cancels. Continuous
covariates are standardized across scenes; categoricals are one-hot columns scaled by
1/sqrt(2) so that one mismatched level contributes exactly 1.0 to the squared distance.
An upper bound on the ratio of heaviest to lightest weight (`clip`) keeps one isolated
scene from taking the run over, which is the failure alpha=1 invites.

Author(s): Alejandro Calle-Saldarriaga
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODES = ("inverse", "ratio")

# The columns that identify a scene. A reference sounding is unique within a cluster, so
# the pair is the key; `ID` is per-row and `ref` alone collides across clusters.
SCENE_KEYS = ("clst", "ref")

# Covariates known to vary across a scene's replicates. Reduced by median rather than by
# `first`, and listed here so the constancy check below can exempt them by name instead of
# silently tolerating any variation it happens to find.
_WITHIN_SCENE_VARYING = ("prior_psfc_pa",)


def _scene_table(files: pd.DataFrame, continuous: Sequence[str],
                 categorical: Sequence[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    """One row per scene, plus the row count of each scene.

    Continuous covariates are reduced by median and categoricals by first value; a
    categorical or an unlisted continuous covariate that is not constant within a scene is
    an error, because it means the table is not the scene-replicate structure this module
    assumes and the resulting weights would not mean what the docstring says they mean.
    """
    missing = [c for c in tuple(continuous) + tuple(categorical) + SCENE_KEYS
               if c not in files.columns]
    if missing:
        raise KeyError(
            f"the split table is missing {missing}; density weighting needs the covariate "
            f"columns in the split table, which is what augment_split.py merges in -- "
            f"point data.split_csv at the *_cov.csv table"
        )

    grouped = files.groupby(list(SCENE_KEYS), sort=True)
    nonconstant = []
    for col in tuple(continuous) + tuple(categorical):
        if col in _WITHIN_SCENE_VARYING:
            continue
        if int(grouped[col].nunique(dropna=False).max()) > 1:
            nonconstant.append(col)
    if nonconstant:
        raise ValueError(
            f"{nonconstant} vary within a scene but are not in _WITHIN_SCENE_VARYING. "
            f"A scene is one point in covariate space -- a replicate is a fresh draw of "
            f"the discrepancy, not a new scene -- so either the table is not what this "
            f"module assumes or the list needs updating deliberately."
        )

    agg = {c: "median" for c in continuous}
    agg.update({c: "first" for c in categorical})
    scenes = grouped.agg(agg).reset_index()
    counts = grouped.size().reset_index(name="_n_rows")
    scenes = scenes.merge(counts, on=list(SCENE_KEYS), validate="one_to_one")
    return scenes, scenes["_n_rows"].to_numpy(dtype=np.float64)


def _design(scenes: pd.DataFrame, continuous: Sequence[str], categorical: Sequence[str],
            ref: Optional[pd.DataFrame] = None) -> np.ndarray:
    """Standardized scene coordinates: continuous z-scores plus scaled one-hot levels.

    `ref` supplies the centring, scaling and level vocabulary when the returned design has
    to live in the same space as another one (mode="ratio" embeds the target scenes in the
    training scenes' coordinates). Levels absent from `ref` become all-zero rows for that
    covariate, which places them at 1/sqrt(2) from every known level rather than dropping
    the scene.
    """
    basis = scenes if ref is None else ref
    X = scenes[list(continuous)].to_numpy(dtype=np.float64)
    B = basis[list(continuous)].to_numpy(dtype=np.float64)

    # NaNs come from augment_split.py's left join when a split row has no covariate row.
    # Impute at the basis median so an unmatched scene sits in the middle of the cloud
    # rather than at infinity, and say how many.
    n_nan = int(np.isnan(X).sum())
    if n_nan:
        med = np.nanmedian(B, axis=0)
        logger.warning("density weighting: %d missing covariate values imputed at the "
                       "reference median", n_nan)
        X = np.where(np.isnan(X), med, X)
        B = np.where(np.isnan(B), med, B)

    mu = B.mean(axis=0)
    sd = B.std(axis=0)
    # A covariate that is constant across scenes carries no information about support and
    # would divide by zero. Keep the column at zero rather than dropping it, so the design
    # matrix's shape is a function of the preset alone.
    sd = np.where(sd > 0, sd, 1.0)
    blocks = [(X - mu) / sd]

    for col in categorical:
        levels = pd.Index(sorted(basis[col].astype(str).unique()))
        codes = scenes[col].astype(str).to_numpy()
        idx = levels.get_indexer(codes)
        onehot = np.zeros((len(scenes), len(levels)), dtype=np.float64)
        rows = np.nonzero(idx >= 0)[0]
        onehot[rows, idx[rows]] = 1.0 / np.sqrt(2.0)
        n_unknown = int((idx < 0).sum())
        if n_unknown:
            logger.warning("density weighting: %d scenes carry a level of '%s' unseen in "
                           "the reference set; they are placed off every axis",
                           n_unknown, col)
        blocks.append(onehot)

    return np.hstack(blocks)


def _sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between the rows of A and the rows of B."""
    a2 = (A * A).sum(axis=1)[:, None]
    b2 = (B * B).sum(axis=1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0)


def _kernel_density(Z: np.ndarray, basis: np.ndarray, h: float) -> np.ndarray:
    """Unnormalized Gaussian kernel density of each row of Z against `basis`."""
    return np.exp(-_sq_dists(Z, basis) / (2.0 * h * h)).mean(axis=1)


def _bandwidth(Z: np.ndarray, k: int) -> float:
    """Median k-th nearest-neighbour distance, each point excluded from its own count."""
    n = len(Z)
    if n < 2:
        raise ValueError(f"a density needs at least two scenes, got {n}")
    k_eff = int(min(max(k, 1), n - 1))
    if k_eff != k:
        logger.warning("density weighting: k=%d is not usable with %d scenes; using k=%d",
                       k, n, k_eff)
    d = np.sqrt(_sq_dists(Z, Z))
    r_k = np.sort(d, axis=1)[:, k_eff]
    h = float(np.median(r_k))
    if not np.isfinite(h) or h <= 0:
        raise ValueError(
            f"degenerate bandwidth (median {k_eff}-NN distance = {h}); the scene "
            f"covariates are probably all identical, which makes a density meaningless"
        )
    return h


def scene_density_weights(
    files: pd.DataFrame,
    continuous: Sequence[str],
    categorical: Sequence[str],
    alpha: float = 0.5,
    beta: float = 1.0,
    k: int = 15,
    clip: float = 10.0,
    mode: str = "inverse",
    target_files: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Per-ROW training weights from the scene-level covariate density. Mean 1.

    :param files: the training partition of an augmented split table (`*_cov.csv`), i.e.
        one row per sounding, carrying `clst`, `ref` and the covariate columns.
    :param continuous: continuous covariate column names -- `CovariateSpec.continuous`.
    :param categorical: categorical covariate column names.
    :param alpha: rarity exponent in [0, 1]. 0 disables the density term entirely.
    :param beta: scene-proportionality exponent in [0, 1]. 1 gives every scene equal
        expected weight regardless of its replicate count.
    :param k: neighbour rank whose median distance sets the kernel bandwidth.
    :param clip: upper bound on the ratio of heaviest to lightest scene weight. Each
        weight is confined to a factor sqrt(clip) either side of the geometric mean, so
        the realised ratio is at most `clip` and is usually smaller -- clamping a skewed
        set of weights around its centre need not saturate both ends. Must be >= 1.
    :param mode: "inverse" for p_hat ** -alpha; "ratio" for
        (p_hat_target / p_hat_train) ** alpha, which needs `target_files`.
    :param target_files: rows whose covariate distribution the training loss should be
        tilted toward, for mode="ratio". Use the VALIDATION partition -- its covariates
        only. This is covariate-shift correction, it couples training to the validation
        *inputs*, and it has to be reported as such.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must lie in [0, 1], got {beta}")
    if clip < 1.0:
        raise ValueError(f"clip must be >= 1 (it is a max/min ratio), got {clip}")
    if mode not in MODES:
        raise ValueError(f"unknown density mode '{mode}' (expected one of {MODES})")
    if mode == "ratio" and target_files is None:
        raise ValueError("mode='ratio' needs target_files (the validation partition)")

    scenes, n_rows = _scene_table(files, continuous, categorical)
    Z = _design(scenes, continuous, categorical)
    h = _bandwidth(Z, k)
    p_train = _kernel_density(Z, Z, h)

    if alpha == 0.0:
        w_scene = np.ones(len(scenes), dtype=np.float64)
    elif mode == "inverse":
        w_scene = p_train ** (-alpha)
    else:
        tgt_scenes, _ = _scene_table(target_files, continuous, categorical)
        Zt = _design(tgt_scenes, continuous, categorical, ref=scenes)
        p_target = _kernel_density(Z, Zt, h)
        # A training scene with no target support at all would get weight 0 and stop
        # contributing; floor the ratio at the smallest positive value instead, so the
        # weighting tilts the loss without deleting rows from it.
        floor = float(np.min(p_target[p_target > 0])) if np.any(p_target > 0) else 1.0
        w_scene = (np.maximum(p_target, floor) / p_train) ** alpha

    # Scene-proportionality: divide out the replicate count so a scene's total expected
    # weight does not depend on how many times it was simulated.
    w_scene = w_scene * n_rows ** (-beta)

    # Clip on the SCENE weights before broadcasting, so `clip` means what it says about
    # the ratio between two scenes and is not a function of replicate counts.
    lo, hi = float(w_scene.min()), float(w_scene.max())
    if lo > 0 and hi / lo > clip:
        geo = float(np.exp(np.log(w_scene).mean()))
        bound = np.sqrt(clip)
        w_scene = np.clip(w_scene, geo / bound, geo * bound)
        logger.info("density weighting: scene weight ratio clipped from %.1f to %.1f",
                    hi / lo, w_scene.max() / w_scene.min())

    scene_of_row = files[list(SCENE_KEYS)].merge(
        scenes[list(SCENE_KEYS)].assign(_w=w_scene), on=list(SCENE_KEYS), how="left",
    )
    w = scene_of_row["_w"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("density weighting produced a non-positive or non-finite weight")
    w = w / w.mean()

    logger.info(
        "density weighting: %d scenes, %d rows, dim=%d, h=%.3f, mode=%s, alpha=%.2f, "
        "beta=%.2f | scene weight ratio max/min=%.2f | row weight ratio max/min=%.2f",
        len(scenes), len(files), Z.shape[1], h, mode, alpha, beta,
        w_scene.max() / w_scene.min(), w.max() / w.min(),
    )
    log_group_shares(files, w)
    return w


def log_group_shares(files: pd.DataFrame, w: np.ndarray, by: str = "clst") -> None:
    """Log the share of expected loss each cluster takes before and after weighting.

    Not used by the sampler. It is here because the share table is the only thing that
    says what the weighting actually did, and a run that does not print it leaves no
    record of the intervention it applied.
    """
    if by not in files.columns:
        return
    df = pd.DataFrame({by: files[by].to_numpy(), "w": w})
    share = df.groupby(by)["w"].agg(["size", "sum"])
    before = 100.0 * share["size"] / share["size"].sum()
    after = 100.0 * share["sum"] / share["sum"].sum()
    logger.info("density weighting: share of expected loss by %s, before -> after:", by)
    for g in share.index:
        logger.info("    %s=%s  %.1f %% -> %.1f %%", by, g, before[g], after[g])

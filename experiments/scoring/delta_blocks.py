#!/usr/bin/env python3
"""delta_blocks.py -- how much of each method's error is the model discrepancy.

    python3 experiments/scoring/delta_blocks.py experiments/scoring/*_bo.npz \
        --out experiments/scoring/tables

Reviewer 1 asks whether the normalizing flow's extra flexibility buys anything, whether the
data are "generated under a Gaussian assumption" that makes the Laplace approximation
sufficient, and whether the discrepancy contributed to the Laplace approximation's
efficiency. All three are the same question about where each method's residual error comes
from, and the delivered ensemble answers it directly, because the discrepancy is *not*
redrawn per sounding.

The block structure, which is measured and not assumed
-----------------------------------------------------
`characterize_discrepancy.py` recovers delta + epsilon exactly from the delivered files
(measured minus noiseless radiance, divided by the reported noise sd) and finds that the
whitened residual has unit within-block standard deviation for blocks of ten consecutive
replicates and not for any coarser grouping. So a reference sounding carries **ten** draws
of delta, each shared by ten replicates that differ only in state and instrument noise, and
the ensemble carries 4,510 draws rather than 45,100. The block index is therefore
`(rep - 1) // 10`, recoverable from the sounding ID alone.

That gives a three-level nested design -- scene / discrepancy block / replicate -- and a
variance decomposition of each method's error in it:

  between scenes            what a scene-level covariate could in principle remove
  between blocks, in scene  the discrepancy: which draw of delta this sounding received
  within a block            state and instrument noise

The middle row is the number that matters. For the operational retrieval it is large (the
retrieval assumes delta = 0 and absorbs it into the state estimate); for a network trained
on data containing delta it is what the network has had to learn to express as predictive
spread rather than to remove.

Estimation is unbalanced nested ANOVA (Searle, Casella & McCulloch, Ch. 3) rather than the
variance of group means, which is biased upward by exactly the within-group noise this is
trying to separate from. Components are estimated per output dimension and combined by
variance weighting; standard errors resample whole scenes, which is the only independent
unit here.

Reads nothing but the artifacts: no GPU, no checkpoint, no raw file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
for p in (str(_ROOT / "code"), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evals import metrics as M                       # noqa: E402
from evals.artifacts import Artifact, scene_labels   # noqa: E402

METHOD_ORDER = ["ACOS L2FP (v10)", "Laplace", "Normalizing flow"]
BLOCK = 10


def method_predictives(art: Artifact):
    """`{method -> (mu_std, var_std or None)}` for every method the artifact carries."""
    out = {}
    if art.has("acos_mu_std"):
        cov = np.asarray(art["acos_cov_std"], dtype=np.float64)
        out["ACOS L2FP (v10)"] = (np.asarray(art["acos_mu_std"], dtype=np.float64),
                                  _diag(cov))
    if art.has("lap_mu_std"):
        cov = np.asarray(art["lap_cov_std"], dtype=np.float64)
        out["Laplace"] = (np.asarray(art["lap_mu_std"], dtype=np.float64), _diag(cov))
    if art.has("nf_samples_std"):
        s = np.asarray(art["nf_samples_std"], dtype=np.float64)
        out["Normalizing flow"] = (s.mean(axis=1), s.var(axis=1, ddof=1))
    return out


def _diag(cov: np.ndarray) -> np.ndarray:
    """Per-dimension predictive variance from a stored covariance of any stored shape."""
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim == 1:                      # [N], already a variance
        return cov[:, None]
    if cov.ndim == 2:                      # [N, 1] or [N, D] diagonal
        return cov
    return np.einsum("nii->ni", cov)       # [N, D, D]


def block_labels(art: Artifact, block: int = BLOCK):
    """`(scene, block)` per sounding. `None` if the IDs do not carry a replicate index."""
    scenes = scene_labels(art)
    if scenes is None or not art.has("rep"):
        return None, None
    rep = np.asarray(art["rep"]).ravel().astype(int)
    return scenes, np.array([f"{s}#{(r - 1) // block}" for s, r in zip(scenes, rep)])


def nested_components(y: np.ndarray, scene: np.ndarray, blk: np.ndarray) -> dict:
    """Unbalanced two-level nested ANOVA variance components for one 1-d quantity.

    Returns the three components in the units of `y` squared. Negative estimates are
    reported as they come out and clipped only when shares are formed: a negative component
    is information (the level is not resolved by this many groups), and silently zeroing it
    hides that.
    """
    y = np.asarray(y, dtype=np.float64)
    N = len(y)
    grand = y.mean()

    s_ids, s_idx = np.unique(scene, return_inverse=True)
    b_ids, b_idx = np.unique(blk, return_inverse=True)
    S, B = len(s_ids), len(b_ids)
    if B <= S or N <= B or S < 2:
        return dict(scene=np.nan, block=np.nan, within=np.nan, n=N, n_scenes=S, n_blocks=B)

    n_b = np.bincount(b_idx, minlength=B).astype(float)
    sum_b = np.bincount(b_idx, weights=y, minlength=B)
    mean_b = sum_b / n_b
    n_s = np.bincount(s_idx, minlength=S).astype(float)
    sum_s = np.bincount(s_idx, weights=y, minlength=S)
    mean_s = sum_s / n_s

    # Which scene each block belongs to; blocks never straddle scenes by construction.
    b_scene = np.zeros(B, dtype=int)
    b_scene[b_idx] = s_idx

    ss_within = float(np.sum((y - mean_b[b_idx]) ** 2))
    ss_block = float(np.sum(n_b * (mean_b - mean_s[b_scene]) ** 2))
    ss_scene = float(np.sum(n_s * (mean_s - grand) ** 2))

    ms_within = ss_within / (N - B)
    ms_block = ss_block / (B - S)
    ms_scene = ss_scene / (S - 1)

    # Expected mean squares for the unbalanced nested design.
    sum_nb2_per_s = np.bincount(b_scene, weights=n_b ** 2, minlength=S)
    t1 = float(np.sum(sum_nb2_per_s / n_s))          # sum_s (sum_b n_sb^2) / n_s
    t2 = float(np.sum(n_b ** 2)) / N                 # (sum_sb n_sb^2) / N
    t3 = float(np.sum(n_s ** 2)) / N                 # (sum_s n_s^2) / N

    k1 = (N - t1) / (B - S)
    k2 = (t1 - t2) / (S - 1)
    k3 = (N - t3) / (S - 1)

    v_within = ms_within
    v_block = (ms_block - ms_within) / k1
    v_scene = (ms_scene - ms_within - k2 * v_block) / k3
    return dict(scene=v_scene, block=v_block, within=v_within,
                n=N, n_scenes=S, n_blocks=B)


def decompose(err_ppm: np.ndarray, scene: np.ndarray, blk: np.ndarray) -> dict:
    """Variance-weighted combination of the per-dimension components."""
    err_ppm = np.atleast_2d(err_ppm.T).T if err_ppm.ndim == 1 else err_ppm
    parts = [nested_components(err_ppm[:, d], scene, blk) for d in range(err_ppm.shape[1])]
    out = {}
    for key in ("scene", "block", "within"):
        out[key] = float(np.nansum([p[key] for p in parts]))
    out["total"] = out["scene"] + out["block"] + out["within"]
    out["n_scenes"] = parts[0]["n_scenes"]
    out["n_blocks"] = parts[0]["n_blocks"]
    out["dims"] = err_ppm.shape[1]
    return out


def rows_for(art: Artifact, block: int = BLOCK, n_boot: int = 200, seed: int = 0):
    scene, blk = block_labels(art, block)
    if scene is None:
        print(f"  {art.meta.task}: no replicate index in the IDs; skipped")
        return pd.DataFrame()

    y = np.asarray(art["y_std"], dtype=np.float64)
    y = y[:, None] if y.ndim == 1 else y
    sigma = np.asarray(art.y_sigma, dtype=np.float64) * float(art.meta.ppm_scale)

    preds = method_predictives(art)
    # Same physical screen as `metrics.summarize_gaussian`, intersected over methods so
    # that the three rows of a task decompose the same soundings. One undeclared fill
    # sentinel in a retrieved state vector is of order -1e19 and would land entirely in
    # the within-block component, which is the denominator of every share below.
    fin = np.ones(len(y), dtype=bool)
    for _, (mu, _v) in preds.items():
        fin &= M.physical_mask(y, mu[:, None] if mu.ndim == 1 else mu,
                               art.y_mu, art.y_sigma)
    n_dropped = int((~fin).sum())
    if n_dropped:
        print(f"  {art.meta.task}: dropped {n_dropped} non-physical soundings")
    y, scene, blk = y[fin], scene[fin], blk[fin]

    rng = np.random.default_rng(seed)
    uniq = np.unique(scene)
    boots = [rng.choice(uniq, size=len(uniq), replace=True) for _ in range(n_boot)]

    rows = []
    for name, (mu, var) in preds.items():
        mu = mu[:, None] if mu.ndim == 1 else mu
        mu, var = mu[fin], var[fin]
        err = (y - mu) * sigma                                    # ppm
        pred_sd2 = var * sigma ** 2                               # ppm^2, per dimension
        d = decompose(err, scene, blk)

        # The share of the within-scene error variance carried by the discrepancy: the
        # quantity measured as 69 % for the operational retrieval on one scene.
        denom = d["block"] + d["within"]
        share = d["block"] / denom if denom > 0 else np.nan

        # A predictive distribution cannot know which delta a sounding received, so within a
        # scene it should be as wide as the block and replicate variation together. This
        # ratio is that comparison, and it is the sense in which the discrepancy "contributes
        # to" a Gaussian posterior's width.
        pred = float(np.sum(pred_sd2.mean(axis=0)))
        ratio = pred / denom if denom > 0 else np.nan

        shares, ratios = [], []
        for pick in boots:
            m = np.concatenate([np.flatnonzero(scene == s) for s in pick])
            # Relabel the resampled scenes so a scene drawn twice is two scenes, not one
            # with 200 replicates -- otherwise the block level absorbs the duplication.
            # The separator matters: without it, draw 1 of scene "12/3" and draw 11 of
            # scene "2/3" concatenate to the same label and are merged into one block.
            tag = np.concatenate([np.full((scene == s).sum(), f"{k}|", dtype=object)
                                  for k, s in enumerate(pick)])
            db = decompose(err[m], tag, np.array([t + b for t, b in zip(tag, blk[m])],
                                                 dtype=object))
            den = db["block"] + db["within"]
            if den > 0:
                shares.append(db["block"] / den)
                ratios.append(float(np.sum(pred_sd2[m].mean(axis=0))) / den)
        rows.append(dict(
            task=art.meta.task, split=art.meta.split_name, method=name,
            n=len(y), n_nonphysical=n_dropped, dims=d["dims"],
            n_scenes=d["n_scenes"], n_blocks=d["n_blocks"],
            var_scene=d["scene"], var_block=d["block"], var_within=d["within"],
            rmse_ppm=float(np.sqrt(np.mean(np.sum(err ** 2, axis=1) / err.shape[1]))),
            delta_share=share, delta_share_se=float(np.std(shares)) if shares else np.nan,
            pred_var=pred, pred_over_within_scene=ratio,
            pred_over_within_scene_se=float(np.std(ratios)) if ratios else np.nan,
        ))
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    out = ["# Where each method's error comes from", "",
           f"Discrepancy blocks of {BLOCK} replicates, verified in "
           "`characterize_discrepancy.py`. Variance components from unbalanced nested "
           "ANOVA; standard errors resample whole reference soundings.", ""]
    for (task, split), g in df.groupby(["task", "split"], sort=False):
        out += [f"## {task} ({split})", "",
                "| method | RMSE [ppm] | between scenes | **discrepancy** | state + noise "
                "| discrepancy share of within-scene | predictive var / within-scene var |",
                "|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            tot = r.var_scene + r.var_block + r.var_within
            pc = lambda v: f"{100 * v / tot:.0f} %" if tot > 0 else "--"   # noqa: E731
            out.append(
                f"| {r.method} | {r.rmse_ppm:.2f} | {pc(r.var_scene)} | "
                f"**{pc(r.var_block)}** | {pc(r.var_within)} | "
                f"{r.delta_share:.3f} ({r.delta_share_se:.3f}) | "
                f"{r.pred_over_within_scene:.2f} ({r.pred_over_within_scene_se:.2f}) |")
        out.append("")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("artifacts", nargs="+")
    ap.add_argument("--out", default="experiments/scoring/tables")
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    frames = []
    for path in args.artifacts:
        art = Artifact.load(path)
        print(f"=== {path}  task={art.meta.task} split={art.meta.split_name}")
        frames.append(rows_for(art, args.block, args.boot, args.seed))
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "delta_blocks.csv", index=False)
    md = to_markdown(df)
    (out_dir / "delta_blocks.md").write_text(md, encoding="utf-8")
    print()
    print(md)
    print(f"wrote {out_dir / 'delta_blocks.csv'}, {out_dir / 'delta_blocks.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

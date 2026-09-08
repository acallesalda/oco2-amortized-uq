#!/usr/bin/env python3
"""figures.py -- every figure in the results section, drawn from an artifact.

Like `score.py`, this needs nothing but the `.npz`: no GPU, no checkpoint, no data. That
is what makes it possible to answer "can we have larger fonts and integer pressure ticks"
 in thirty seconds rather than by re-queuing a GPU job.

    python3 experiments/scoring/figures.py results/group_split_v1/profiles.npz --out figures/

Figures produced, by task dimension:

  pit_<task>          per-level PIT histograms, one panel per method
  coverage_<task>     observed vs nominal coverage -- marginal *and* joint
  joint_<task>        the three-panel joint diagnostic (chi2-PIT, HDR coverage, QQ)
  scenes_<task>       per-held-out-reference-sounding RMSE, coverage and NLL, ordered
                      by the flow and labelled by cluster -- the group-split figure
  levels_<task>       RMSE and coverage against pressure level        (D > 1)
  pairs_<task>        bivariate posteriors for individual soundings   (D > 1)
  levels_overlay_<t>  two artifacts of ONE task in one per-level figure -- published
                      inputs against corrected inputs, or any other pair that
                      differs in one thing                            (--overlay)

`pairs` is the honest version of Figure 7. The published panel says "six randomly
selected test soundings" but the notebook selected cases where the truth falls outside the
ACOS 95% ellipse. Both are available here -- `--cases random` with a
recorded seed, and `--cases outside95` -- so the figure and its caption can finally agree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
for p in (str(_ROOT / "code"), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from scipy import stats                              # noqa: E402

from evals import joint_calibration as JC            # noqa: E402
from evals import metrics as M                       # noqa: E402
from evals.artifacts import Artifact, scene_labels   # noqa: E402

# The submitted figures were unreadable at column width. This is the fix, applied
# once for every figure rather than per-axis as the notebooks did.
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "figure.dpi": 120, "savefig.dpi": 300, "axes.grid": True,
    "grid.alpha": 0.25,
})
# Deliberately not setting `figure.constrained_layout.use` globally: evals.joint_calibration
# builds its own figure with tight_layout, and the two engines are mutually exclusive --
# matplotlib warns and silently drops one. Each figure below asks for the layout it wants.
_LAYOUT = {"layout": "constrained"}

# The half of the legibility problem that the point sizes above do NOT fix. Every figure here is
# included at `width=\linewidth`, so LaTeX rescales the canvas to the text block and
# rescales every glyph with it. A three-panel row built 16.5 in wide and printed into a
# 6.5 in text block shrinks by 2.5x, and the 13 pt above reaches the page at 5 pt -- which
# is what a reader measures. Raising the point size on a canvas that wide only makes
# the ratio worse.
#
# So the width is fixed once, here, and every `figsize` is derived from it. `OVERSIZE` is
# the only slack: the canvas is built slightly larger than it is printed so that raster
# detail survives, and the effective point size on the page is the value above divided by
# it. At 1.30 the 13 pt body reaches the page at 10 pt, the 12 pt ticks at 9.2 pt, and the
# 14 pt axis labels at 10.8 pt -- i.e. at or above the 10 pt of the manuscript's own text.
#
# PRINT_WIDTH_IN is `\the\textwidth` for agujournal2025 on letter paper, in inches. If the
# class or the paper size changes, put `\the\textwidth` in the preamble, read the value
# out of the log (in pt, 72.27 pt to the inch) and change this one number.
PRINT_WIDTH_IN = 6.5
OVERSIZE = 1.30
FIG_W = PRINT_WIDTH_IN * OVERSIZE


def _size(aspect: float, w_frac: float = 1.0):
    """`figsize` for a figure that LaTeX will include at `w_frac` of `\\linewidth`.

    `aspect` is height over width. `w_frac` matters: `coverage_xco2.png` goes in at
    `width=0.5\\linewidth`, so building it at the full width would halve its type again.
    """
    w = FIG_W * w_frac
    return (w, w * aspect)


def _tick_levels(ax, levels):
    """Integer pressure-level ticks, labelled sparsely.

    Levels are integers and the axis must show no half-integers; `set_xticks` on the levels
    themselves gives exactly that, and the first version of this got it by ticking every
    level and blanking every other label. That still puts twenty tick marks and twenty
    gridlines under a panel a third of the text width wide, which is what the R3 pass
    asked to be rid of. So the labelled (major) ticks are thinned to at most six, and
    every level keeps an unlabelled minor tick. Nothing is invented between the minor
    ticks, so no half-integer can appear, and the labels stay regularly spaced.

    Thinning runs from the surface end rather than from level 1: level 20 is the surface,
    it is the level the near-surface task and half of Section 4 are about, and an axis
    whose last label is 19 invites the reader to count. A four-level axis is under the
    threshold and keeps all four, so `levels_last4` is unchanged by this.
    """
    levels = np.asarray(levels, dtype=int)
    step = max(1, int(np.ceil(len(levels) / 6)))
    major = levels[::-1][::step][::-1]
    ax.set_xticks(major)
    ax.set_xticklabels([str(l) for l in major])
    ax.set_xticks(levels, minor=True)
    ax.tick_params(axis="x", which="minor", length=2)
    ax.set_xlim(levels.min() - 0.5, levels.max() + 0.5)


# Per-figure type sizes, because the three figure families have very different panel
# sizes and one global setting cannot serve all of them. Divide by OVERSIZE for the size
# on the printed page: 13 pt -> 10.0 pt, 16 pt -> 12.3 pt, 8 pt -> 6.2 pt.
#
# The scene panels are full width and get the large labels. The per-level row puts three
# panels across the text width, so each drawing area is about 2.3 in and the same 16 pt
# label swamps it -- those are set a size where the label reads as a label and not as a
# heading. The bivariate grid is six panels and is smaller again.
LABEL_FS = 16        # axis labels on the stacked scene panels
SUPLABEL_FS = 17     # the figure-wide caption under the scene panels
LEVELS_LABEL_FS = 12   # axis labels, three-panel per-level row
LEVELS_TICK_FS = 10    # and its tick labels
# The legend takes the deepest cut of anything here. It is three method names inside a
# panel about 2.3 in wide, and on the four-level version -- where the curves use a fraction
# of the panel and the box has the rest to itself -- it reads as the largest object in the
# figure. 8 pt is 6.2 pt on the page, small, but it is a key whose three entries are also
# named in every caption and every table.
LEVELS_LEGEND_FS = 8
# Markers, not the 6 pt default. Twenty of them across a panel a third of the text width
# wide leaves about 0.11 in per level, and a 6 pt disc is most of that: the discs touch,
# the line between them disappears, and the series reads as a string of blobs rather than
# a curve. At 4 pt the curve is the thing seen and the points still mark the levels.
LEVELS_MS = 4
PAIRS_LABEL_FS = 12    # axis labels on the bivariate panels
PAIRS_TICK_FS = 10
PAIRS_TITLE_FS = 11
PAIRS_LEGEND_FS = 10   # a figure-level key below the panels, so it covers no data


COLORS = {"ACOS L2FP (v10)": "#d95f02", "Laplace": "#1b9e77", "Normalizing flow": "#7570b3"}


# ---------------------------------------------------------------------------------
def predictives(art: Artifact):
    """`{method -> dict}` with whatever that method provides: mu, cov, samples, z."""
    y = np.asarray(art["y_std"], dtype=np.float64)
    n = len(y)
    out = {}
    if art.has("acos_mu_std"):
        mu = np.asarray(art["acos_mu_std"], dtype=np.float64)
        cov = np.asarray(art["acos_cov_std"], dtype=np.float64)
        # `valid` marks the soundings whose ACOS covariance was found on disk. Every
        # uncertainty panel is restricted to those; the mean-based panels are not, since
        # the retrieved state itself is always available.
        valid = art.get("acos_valid")
        valid = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        out["ACOS L2FP (v10)"] = dict(mu=mu, cov=cov, valid=valid,
                                      nll=M.mvn_nll(y, mu, cov),
                                      z=M.whiten(y[valid], mu[valid], cov[valid]))
    if art.has("lap_mu_std"):
        mu = np.asarray(art["lap_mu_std"], dtype=np.float64)
        cov = np.asarray(art["lap_cov_std"], dtype=np.float64)
        out["Laplace"] = dict(mu=mu, cov=cov, valid=np.ones(n, dtype=bool),
                              nll=M.mvn_nll(y, mu, cov),
                              z=M.whiten(y, mu, cov))
    if art.has("nf_samples_std"):
        s = np.asarray(art["nf_samples_std"], dtype=np.float64)
        out["Normalizing flow"] = dict(
            samples=s, mu=s.mean(axis=1), valid=np.ones(n, dtype=bool),
            # The flow's own exact density where predict.py stored it; a KDE over its
            # samples otherwise, which is what score.py falls back to for the same reason.
            nll=(-np.asarray(art["nf_logprob_std"], dtype=np.float64)
                 if art.has("nf_logprob_std") else M.kde_joint_nll(y, s)),
            # The flow's own latent map is the exact analogue of Cholesky whitening and
            # needs no Gaussian assumption; fall back to whitening the sample moments
            # only if predict.py was run without it.
            z=(np.asarray(art["nf_latent_std"], dtype=np.float64)
               if art.has("nf_latent_std")
               else M.whiten(y, s.mean(axis=1), M.sample_mean_cov(s)[1])),
        )
    return y, out


def check_scale(art, preds, y) -> list:
    """Name any predictive that is grossly inconsistent with its own residuals.

    `score.py` refuses to write such a row without a warning (its `NOT REPORTABLE` block).
    The figures had no equivalent, which is how an XCO2 Laplace predictive fitted without
    its observation-noise term reached a PIT histogram and a coverage curve that both look
    like results: a predictive whose scale is 27x too small draws a perfectly
    plausible hard-U histogram. Same test, same threshold, same definition of the ratio as
    `score.py` -- `metrics._var_ratio`, imported rather than re-implemented so the figure and
    the table cannot disagree about which rows are bad.

    ACOS is excluded: its ratio is the paper's finding, not a defect.
    """
    bad = []
    for name, p in preds.items():
        if name.startswith("ACOS"):
            continue
        v = p["valid"]
        if "samples" in p:
            var = p["samples"].var(axis=1, ddof=1)
        else:
            var = np.diagonal(p["cov"][v], axis1=-2, axis2=-1)
        ratio = M._var_ratio(y[v], p["mu"][v], var)
        if np.isfinite(ratio) and not 0.25 <= ratio <= 4.0:
            bad.append((name, ratio))
    for name, ratio in bad:
        print(f"  !! {name}: residual variance is {ratio:.1f}x the mean predictive "
              f"variance. Its PIT, coverage and NLL panels are NOT reportable. "
              f"Re-score the artifact, or draw the interim figure "
              f"with --drop '{name}'.")
    return bad


def drop_methods(preds: dict, patterns) -> dict:
    """Remove named predictives, so an interim figure can omit a known-bad one.

    Substring match, case-insensitive, so `--drop laplace` is enough. A pattern that
    matches nothing is an error rather than a no-op: silently drawing the method you meant
    to drop is the failure this option exists to prevent.
    """
    if not patterns:
        return preds
    lowered = [q.lower() for q in patterns]
    kept, dropped = {}, []
    for name, p in preds.items():
        if any(q in name.lower() for q in lowered):
            dropped.append(name)
        else:
            kept[name] = p
    unmatched = [q for q in lowered if not any(q in n.lower() for n in preds)]
    if unmatched:
        raise SystemExit(f"--drop {unmatched} matched no method in this artifact; it "
                         f"carries {list(preds)}.")
    print(f"  dropped from every figure: {dropped}")
    return kept


def _stem(art) -> str:
    """Filename stem for this artifact's figures.

    `task` alone is not unique: a marginalized artifact reports itself under the native
    task's name so that score.py merges their rows into one table (last4_marginal ->
    table_last4, last1_marginal -> table_last1). For a table that is what you want; for a
    file name it means the marginalized run silently overwrites the native one's PNGs,
    which is what happened to every `*_last4.png` whenever both were scored in one
    invocation. The split name carries the distinguishing suffix, so it goes in the stem.
    """
    suffix = art.meta.split_name.split("__", 1)
    return art.meta.task if len(suffix) == 1 else f"{art.meta.task}__{suffix[1]}"


def fig_pit(art, preds, y, out_dir: Path):
    names = list(preds)
    fig, axes = plt.subplots(1, len(names), figsize=_size(0.95 / max(len(names), 1)),
                             squeeze=False, **_LAYOUT)
    for ax, name in zip(axes[0], names):
        p = preds[name]
        v = p["valid"]
        if "samples" in p:
            u = M.sample_marginal_pit(y, p["samples"]).ravel()
        else:
            var = np.diagonal(p["cov"][v], axis1=-2, axis2=-1)
            u = M.gaussian_marginal_pit(y[v], p["mu"][v], var).ravel()
        u = u[np.isfinite(u)]
        ax.hist(u, bins=20, range=(0, 1), density=True, color=COLORS.get(name, "C0"),
                edgecolor="black", linewidth=0.5)
        ax.axhline(1.0, ls="--", color="gray")
        ax.set_title(name)
        ax.set_xlabel("PIT value")
        ax.set_ylabel("density")
        ax.set_xlim(0, 1)
    path = out_dir / f"pit_{_stem(art)}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_coverage(art, preds, y, out_dir: Path):
    """Observed against nominal, marginal and joint side by side.

    Splitting the two panels is the whole argument: a predictive can sit on the diagonal
    marginally and be badly wrong jointly, which is exactly the worry about the
    20-level results.
    """
    levels = np.linspace(0.05, 0.99, 25)
    # At D = 1 the two panels are the same estimator. The marginal curve averages over one
    # level and the HDR region of a scalar predictive is its central interval, so the
    # figure would be a claim printed twice -- and a reader is entitled to assume that two
    # panels shown side by side are two measurements. One axis, and the caption says the
    # distinction does not arise here.
    scalar = int(art.meta.dim) == 1
    if scalar:
        # One panel, and the manuscript includes it at `width=0.5\linewidth`.
        fig, ax_m = plt.subplots(1, 1, figsize=_size(0.90, w_frac=0.5), **_LAYOUT)
        ax_j = None
        panels = ((ax_m, "Observed against nominal coverage"),)
    else:
        fig, (ax_m, ax_j) = plt.subplots(1, 2, figsize=_size(0.46), **_LAYOUT)
        panels = ((ax_m, "Marginal (averaged over levels)"), (ax_j, "Joint (HDR)"))
    for ax, title in panels:
        ax.plot([0, 1], [0, 1], ls="--", color="gray", label="ideal")
        ax.set_xlabel("nominal coverage")
        ax.set_ylabel("observed coverage")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)

    for name, p in preds.items():
        c = COLORS.get(name, None)
        v = p["valid"]
        if "samples" in p:
            marg = [np.mean(M.sample_marginal_coverage(y, p["samples"], l)) for l in levels]
        else:
            var = np.diagonal(p["cov"][v], axis1=-2, axis2=-1)
            marg = [np.mean(M.gaussian_marginal_coverage(y[v], p["mu"][v], var, l))
                    for l in levels]
        ax_m.plot(levels, marg, marker="o", ms=3, color=c, label=name)
        if ax_j is not None:
            ax_j.plot(levels, [M.joint_coverage_from_z(p["z"], l) for l in levels],
                      marker="o", ms=3, color=c, label=name)
    ax_m.legend()
    path = out_dir / f"coverage_{_stem(art)}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_joint(art, preds, out_dir: Path):
    path = out_dir / f"joint_{_stem(art)}.png"
    fig = JC.plot_joint_calibration(
        {k: v["z"] for k, v in preds.items()}, out_path=path,
        title=f"{art.meta.task} · {art.meta.split_name}",
    )
    plt.close(fig)
    return path


def fig_levels(art, preds, y, out_dir: Path):
    """RMSE, MAE and 95% coverage against pressure level.

    The MAE panel is next to the RMSE one rather than replacing it: where the two
    disagree about the ranking of two methods, the difference between them is carried by
    a small number of large residuals, and that is worth seeing per level rather than
    only in the pooled table.
    """
    levels = np.asarray(art.meta.levels, dtype=int)
    fig, (ax_r, ax_a, ax_c) = plt.subplots(1, 3, figsize=_size(0.34), **_LAYOUT)
    for name, p in preds.items():
        c = COLORS.get(name, None)
        # The MAE point estimate is the predictive median for the flow, as in score.py:
        # it is what absolute error is minimized by, and for a skewed flow it is not the
        # mean. `_per_level` is shared with `fig_levels_overlay`.
        rmse, mae, cov = _per_level(art, p, y)
        ax_r.plot(levels, rmse, marker="o", ms=LEVELS_MS, color=c, label=name)
        ax_a.plot(levels, mae, marker="o", ms=LEVELS_MS, color=c, label=name)
        ax_c.plot(levels, cov, marker="o", ms=LEVELS_MS, color=c, label=name)
    ax_c.axhline(95.0, ls="--", color="gray", label="nominal 95%")
    ax_r.set_ylabel("RMSE [ppm]", fontsize=LEVELS_LABEL_FS)
    ax_a.set_ylabel("MAE [ppm]", fontsize=LEVELS_LABEL_FS)
    # "marginal" moves to the caption, which already carries it: a rotated label is bound
    # by the panel height, and the full phrase does not fit one panel of a three-panel row.
    ax_c.set_ylabel("95% coverage [%]", fontsize=LEVELS_LABEL_FS)
    for ax in (ax_r, ax_a, ax_c):
        ax.set_xlabel("pressure level", fontsize=LEVELS_LABEL_FS)
        ax.tick_params(labelsize=LEVELS_TICK_FS)
        _tick_levels(ax, levels)   # integer ticks, no half-integers.
    ax_r.legend(fontsize=LEVELS_LEGEND_FS)
    path = out_dir / f"levels_{_stem(art)}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _per_level(art, p, y):
    """RMSE, MAE and 95% marginal coverage per level, for one method of one artifact."""
    v = p["valid"]
    rmse = M.rmse_ppm_per_level(y, p["mu"], art.y_mu, art.y_sigma)
    if "samples" in p:
        point = np.median(p["samples"], axis=1)
        cov = M.sample_marginal_coverage(y, p["samples"], 0.95)
    else:
        point = p["mu"]
        cov = M.gaussian_marginal_coverage(
            y[v], p["mu"][v], np.diagonal(p["cov"][v], axis1=-2, axis2=-1), 0.95)
    return rmse, M.mae_ppm_per_level(y, point, art.y_mu, art.y_sigma), 100 * cov


def fig_levels_overlay(items, out_dir: Path, stem: str):
    """Two artifacts of the same task, per level, in one set of axes.

    `items` is `[(label, art, preds, y), ...]`, in the order they should be drawn.

    For any comparison where the two artifacts differ in exactly one thing and the answer
    is a vertical gap between two lines: published inputs against corrected inputs, one
    split against another, a model against its own ablation. `fig_levels`
    cannot draw those, because it takes one artifact and the comparison is between two;
    scoring them into two files loses the gap, which is the whole result.

    Written for the published Figure 6 -- the *native* four-output models against the
    *marginals* of the twenty-level models -- which was then cut from the paper:
    the marginalized evaluation on the reference-disjoint partition never completed, and on
    the revised models the comparison no longer says one thing. The figure mode outlived
    the figure.

    Method is colour and family is line style. The operational retrieval is drawn once:
    it does not train, so its rows are identical in both artifacts by construction, and
    that is asserted rather than assumed.
    """
    if len(items) < 2:
        return None
    levels = [list(map(int, it[1].meta.levels or [])) for it in items]
    if not all(levels):
        # A scalar artifact has no level axis, so there is nothing to overlay against.
        # Silently drawing an empty figure here is worse than refusing: the file would
        # exist, be included, and be blank.
        raise SystemExit("--overlay: one of the artifacts carries no pressure levels "
                         "(a scalar task). There is no per-level figure to overlay.")
    if len({tuple(l) for l in levels}) != 1:
        raise SystemExit(f"--overlay: the artifacts cover different levels: {levels}. "
                         f"An overlay of two different targets is not a comparison.")
    lv = np.asarray(levels[0], dtype=int)

    styles = ["-", "--", ":", "-."]
    fig, (ax_r, ax_a, ax_c) = plt.subplots(1, 3, figsize=_size(0.34), **_LAYOUT)
    acos_seen = None
    for k, (label, art, preds, y) in enumerate(items):
        for name, p in preds.items():
            rmse, mae, cov = _per_level(art, p, y)
            if name.startswith("ACOS"):
                # Same retrieval, same soundings: if the two artifacts disagree they are
                # not the same test partition and nothing else in the figure is readable.
                if acos_seen is None:
                    acos_seen = rmse
                    ax_r.plot(lv, rmse, marker="o", ms=LEVELS_MS, color=COLORS[name],
                              ls="-", label=name)
                    ax_a.plot(lv, mae, marker="o", ms=LEVELS_MS, color=COLORS[name], ls="-")
                    ax_c.plot(lv, cov, marker="o", ms=LEVELS_MS, color=COLORS[name], ls="-")
                elif not np.allclose(acos_seen, rmse, rtol=1e-6, atol=1e-6):
                    print("  WARNING: the operational retrieval differs between the two "
                          "artifacts; they are not the same soundings.")
                continue
            ls = styles[k % len(styles)]
            ax_r.plot(lv, rmse, marker="o", ms=LEVELS_MS, color=COLORS.get(name), ls=ls,
                      label=f"{name} ({label})")
            ax_a.plot(lv, mae, marker="o", ms=LEVELS_MS, color=COLORS.get(name), ls=ls)
            ax_c.plot(lv, cov, marker="o", ms=LEVELS_MS, color=COLORS.get(name), ls=ls)

    ax_c.axhline(95.0, ls="--", color="gray", label="nominal 95%")
    # Same labels and sizes as `fig_levels`: this is the same panel with a second artifact
    # drawn into it, and the two should not disagree about what the axes are called.
    ax_r.set_ylabel("RMSE [ppm]", fontsize=LEVELS_LABEL_FS)
    ax_a.set_ylabel("MAE [ppm]", fontsize=LEVELS_LABEL_FS)
    ax_c.set_ylabel("95% coverage [%]", fontsize=LEVELS_LABEL_FS)
    for ax in (ax_r, ax_a, ax_c):
        ax.set_xlabel("pressure level", fontsize=LEVELS_LABEL_FS)
        ax.tick_params(labelsize=LEVELS_TICK_FS)
        _tick_levels(ax, lv)                 # levels are integers.
    ax_r.legend(fontsize=LEVELS_LEGEND_FS)
    path = out_dir / f"levels_overlay_{stem}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _ellipse(mu2, cov2, level=0.95, n=300):
    """Boundary of the `level` credible ellipse of a bivariate Gaussian."""
    vals, vecs = np.linalg.eigh(0.5 * (cov2 + cov2.T))
    vals = np.clip(vals, 1e-30, None)
    t = np.linspace(0, 2 * np.pi, n)
    r = np.sqrt(stats.chi2.ppf(level, df=2))
    circle = np.stack([np.cos(t), np.sin(t)], axis=0)
    return (mu2[:, None] + vecs @ (np.sqrt(vals)[:, None] * r * circle)).T


def _misses_2d(p, y, i1, i2, level=0.95):
    """Per sounding: does this Gaussian's `level` region at (i1, i2) exclude the truth?

    NaN where the method has no usable covariance there, so a singular ACOS covariance is
    "unknown" rather than "missed" -- counting it as a miss would inflate exactly the
    statistic the selection rule is built on.
    """
    n = len(y)
    out = np.full(n, np.nan)
    thr = stats.chi2.ppf(level, df=2)
    mu, cov, valid = p["mu"], p.get("cov"), p["valid"]
    if cov is None:
        return out
    for i in range(n):
        if not valid[i]:
            continue
        C = cov[i][np.ix_([i1, i2], [i1, i2])]
        r = y[i, [i1, i2]] - mu[i, [i1, i2]]
        if not np.all(np.isfinite(C)):
            continue
        try:
            out[i] = float(r @ np.linalg.solve(C, r)) > thr
        except np.linalg.LinAlgError:
            continue
    return out


def select_cases(art, preds, y, i1, i2, how: str, n_panels: int, seed: int,
                 n_hard: int = 2):
    """`(indices, base_rates)` for one panel of `fig_pairs`.

    Three rules, and the caption has to name the one used:

      random     -- uniform over the test partition at a recorded seed. The honest default
                    and what the submitted caption claimed.
      outside95  -- the truth outside the ACOS 95% region. This is what the submitted
                    figure actually did, kept so the disclosure in the
                    letter can be checked.
      hard       -- the truth outside the ACOS 95% region OR outside the Laplace's. The
                    cases where *a* Gaussian posterior misses, which is what the figure is
                    an argument about; selecting on ACOS alone quietly makes it an argument
                    about the operational retrieval instead.
      mixed      -- `n_hard` panels from the `hard` pool and the rest from `random`, in
                    that order. The figure the paper leads with: an all-hard panel is an
                    argument with no control in it, and a reader cannot tell from six
                    difficult cases whether they are typical. The panels drawn from the
                    hard pool are the SAME ones `hard` would draw at this seed, and the
                    rest the same ones `random` would, so a dense-sample artifact built
                    for those two rules covers this one.

    `base_rates` is returned rather than printed here because the caption needs it: a
    deliberately selected panel is only honest beside the frequency of what it selects on.
    It also carries `hard_flags`, aligned with the returned indices, which is what marks
    the panels and what lets a `random` caption say whether one of its draws happens to be
    a case where a Gaussian misses -- at a 10 % base rate, sometimes one is.
    """
    n = len(y)
    acos = preds.get("ACOS L2FP (v10)")
    lap = preds.get("Laplace")
    rates = {"n": n}

    # Always computed, for every rule: `random` needs it for `hard_flags` and the caption,
    # and it is a couple of seconds on 4,489 soundings.
    m_acos = _misses_2d(acos, y, i1, i2) if acos is not None else np.full(n, np.nan)
    m_lap = _misses_2d(lap, y, i1, i2) if lap is not None else np.full(n, np.nan)
    known = np.isfinite(m_acos) | np.isfinite(m_lap)
    a = np.nan_to_num(m_acos) > 0
    l = np.nan_to_num(m_lap) > 0
    hard = a | l
    rates.update({
        "n_scored": int(known.sum()),
        "acos_misses": int(a.sum()), "laplace_misses": int(l.sum()),
        "both_miss": int((a & l).sum()), "either_misses": int(hard.sum()),
        "neither_misses": int((known & ~hard).sum()),
    })

    def draw(pool, k, rng):
        return np.sort(rng.choice(pool, size=min(k, len(pool)), replace=False))

    if how == "mixed":
        if acos is None and lap is None:
            print("  skipping pairs: --cases mixed needs at least one Gaussian.")
            return None, rates
        k = max(0, min(int(n_hard), n_panels))
        # Each pool is drawn at FULL `n_panels` first and then thinned, which is what makes
        # the mixed panel a subset of what `hard` and `random` draw at this seed -- and
        # therefore of whatever a dense-sample job was run for. Sampling `k` directly from
        # the pool would not: `rng.choice(pool, 2)` is not the first two of
        # `rng.choice(pool, 6)`, since sampling without replacement is not prefix-stable.
        # Three of six ids fell outside the dense artifact before this was fixed.
        #
        # The thinning is a seeded subset rather than the first `k`, because the draws are
        # returned sorted and sorted means ordered by cluster and reference sounding: the
        # first two of six would always come from the lowest-numbered clusters.
        full_hard = draw(np.flatnonzero(hard), n_panels, np.random.default_rng(seed))
        full_rand = draw(np.arange(n), n_panels, np.random.default_rng(seed))
        picked_hard = draw(full_hard, k, np.random.default_rng(seed + 1))
        picked_rand = draw(full_rand, n_panels - k, np.random.default_rng(seed + 2))
        cases = np.concatenate([picked_hard, picked_rand]).astype(int)
        rates["hard_flags"] = hard[cases]
        rates["n_hard_panels"] = int(len(picked_hard))
        return cases, rates

    rng = np.random.default_rng(seed)
    if how == "random":
        pool = np.arange(n)
    elif how == "outside95":
        if acos is None:
            print("  skipping pairs: --cases outside95 needs the ACOS covariance, which "
                  "this artifact does not carry.")
            return None, rates
        pool = np.flatnonzero(a)
    elif how == "hard":
        if acos is None and lap is None:
            print("  skipping pairs: --cases hard needs at least one Gaussian.")
            return None, rates
        pool = np.flatnonzero(hard)
    else:
        raise SystemExit(f"unknown --cases {how!r}")

    if len(pool) == 0:
        # A result in itself, and worth saying rather than crashing on: it means no
        # sounding in the partition has the truth outside the region being tested.
        print(f"  skipping pairs: no sounding satisfies --cases {how} at these levels.")
        return None, rates
    cases = draw(pool, n_panels, rng)
    rates["hard_flags"] = hard[cases]
    return cases, rates


def fig_pairs(art, preds, y, out_dir: Path, dims, n_panels: int, how: str, seed: int,
              dense=None, contours: bool = False, n_hard: int = 2):
    """Bivariate posteriors for individual soundings -- Figure 7.

    `how` is the selection rule; see `select_cases`, whose `base_rates` this prints so the
    caption can quote them. `dense`, if given, is `{id -> [S, D] samples}` from a
    high-sample-count run over these soundings alone: 512 draws is enough for a moment and
    not enough for a picture of a shape, and re-running the whole partition at 20,000 would
    be a three-gigabyte artifact for six panels.
    """
    d1, d2 = dims
    levels = art.meta.levels or [1]
    missing = [d for d in (d1, d2) if d not in levels]
    if missing:
        raise SystemExit(
            f"--pair-dims {d1} {d2}: level(s) {missing} are not in this artifact, which "
            f"covers levels {levels}. For the near-surface task pick from those."
        )
    i1, i2 = levels.index(d1), levels.index(d2)

    cases, rates = select_cases(art, preds, y, i1, i2, how, n_panels, seed, n_hard)
    if "either_misses" in rates:
        n_s = rates["n_scored"]
        print(f"  selection base rates at levels {d1},{d2} over {n_s} scored soundings: "
              f"ACOS misses {rates['acos_misses']} ({100 * rates['acos_misses'] / n_s:.1f}%), "
              f"Laplace misses {rates['laplace_misses']} "
              f"({100 * rates['laplace_misses'] / n_s:.1f}%), "
              f"both {rates['both_miss']} ({100 * rates['both_miss'] / n_s:.1f}%), "
              f"either {rates['either_misses']} "
              f"({100 * rates['either_misses'] / n_s:.1f}%)")
    if cases is None:
        return None

    ncol = min(3, len(cases))
    nrow = int(np.ceil(len(cases) / ncol))
    # Square-ish panels: the aspect is the grid's, not a constant, so a 3x1 panel does
    # not come out as tall as a 3x2 one.
    fig, axes = plt.subplots(nrow, ncol, figsize=_size(0.93 * nrow / ncol),
                             squeeze=False, **_LAYOUT)

    mu_ppm = art.y_mu * M.PPM
    sig_ppm = art.y_sigma * M.PPM

    def ppm(v, j):
        return v * sig_ppm[j] + mu_ppm[j]

    # `"<cluster>/<reference>"` per sounding, or None if the ids are not in that format --
    # in which case the panel falls back to naming the sounding.
    scenes = scene_labels(art)
    n_drawn = []
    for ax, i in zip(axes.ravel(), cases):
        nf = preds.get("Normalizing flow")
        if nf is not None:
            sid = str(art["ids"][i])
            s = (dense[sid] if dense is not None and sid in dense
                 else nf["samples"][i])[:, [i1, i2]]
            n_drawn.append(len(s))
            xs, ys = ppm(s[:, 0], i1), ppm(s[:, 1], i2)
            # A cloud of 512 points at alpha 0.15 reads as sparse noise and a cloud of
            # 20,000 at the same alpha reads as a solid blob. Both hide the shape, so the
            # two are traded off against the count rather than fixed.
            ax.scatter(xs, ys, s=(4.0 if len(s) < 2000 else 2.0),
                       alpha=float(np.clip(700.0 / len(s), 0.02, 0.30)),
                       linewidths=0, color=COLORS["Normalizing flow"],
                       label=f"flow, {len(s):,} draws", rasterized=True)
            if contours:
                # The like-for-like comparison the ellipses deserve: the flow's own 68 and
                # 95 % highest-density regions, at the same two levels, from a Gaussian KDE
                # over its draws. Noisy below a few thousand draws, which is why it is off
                # unless asked for.
                try:
                    kde = stats.gaussian_kde(np.vstack([xs, ys]))
                    gx = np.linspace(xs.min(), xs.max(), 140)
                    gy = np.linspace(ys.min(), ys.max(), 140)
                    GX, GY = np.meshgrid(gx, gy)
                    dens = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
                    at_pts = kde(np.vstack([xs, ys]))
                    thr = [np.quantile(at_pts, 1 - q) for q in (0.95, 0.68)]
                    ax.contour(GX, GY, dens, levels=sorted(thr),
                               colors=[COLORS["Normalizing flow"]], linewidths=1.6,
                               linestyles=["--", "-"])
                except (np.linalg.LinAlgError, ValueError) as exc:
                    print(f"    KDE contour skipped for sounding {sid}: {exc}")
        for name in ("ACOS L2FP (v10)", "Laplace"):
            p = preds.get(name)
            if p is None or not np.all(np.isfinite(p["cov"][i])):
                continue
            C = p["cov"][i][np.ix_([i1, i2], [i1, i2])]
            e = _ellipse(p["mu"][i, [i1, i2]], C, 0.95)
            ax.plot(ppm(e[:, 0], i1), ppm(e[:, 1], i2), color=COLORS[name], lw=2,
                    label=f"{name} 95%")
        # s=220 was set for a 13.9 in canvas; on 8.5 in the same star covers a visible
        # part of the posterior it is meant to sit in. Scaled by the area ratio.
        ax.scatter([ppm(y[i, i1], i1)], [ppm(y[i, i2], i2)], marker="*", s=85,
                   color="black", zorder=5, label="truth")
        ax.set_xlabel(f"level {d1} [ppm]", fontsize=PAIRS_LABEL_FS)
        ax.set_ylabel(f"level {d2} [ppm]", fontsize=PAIRS_LABEL_FS)
        ax.tick_params(labelsize=PAIRS_TICK_FS)
        # The geophysical cluster rather than the sounding id: the id is a filename, it is
        # long enough to set the panel width, and the caption lists them for
        # reproducibility. The cluster is the thing a reader can actually use, because
        # cluster membership is what the posterior geometry varies with.
        flags = rates.get("hard_flags")
        tag = ""
        if flags is not None and how == "mixed":
            k = list(cases).index(i)
            tag = ("\na Gaussian $95\\%$ region misses the truth" if flags[k]
                   else "\nboth Gaussian $95\\%$ regions contain the truth")
        head = (f"From Cluster {scenes[i].split('/')[0]}" if scenes is not None
                else f"{art['ids'][i]}")
        ax.set_title(f"{head}{tag}", fontsize=PAIRS_TITLE_FS)
    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")

    # An explicit legend, because the automatic one is unreadable here: `markerscale=2`
    # applied to a star drawn at s=220 gives a legend entry the size of a data point's
    # neighbourhood, and the automatic version says nothing at all about the two contour
    # line styles, which are the whole added content of the dense version.
    L2D = matplotlib.lines.Line2D
    n_lab = (f"{n_drawn[0]:,}" if len(set(n_drawn)) == 1 else
             f"{min(n_drawn):,}–{max(n_drawn):,}") if n_drawn else "0"
    handles = [L2D([], [], marker="o", ls="none", ms=5, alpha=0.6,
                   color=COLORS["Normalizing flow"], label=f"flow, {n_lab} draws")]
    if contours:
        handles += [L2D([], [], ls="-", lw=1.6, color=COLORS["Normalizing flow"],
                        label="flow $68\\%$ HDR"),
                    L2D([], [], ls="--", lw=1.6, color=COLORS["Normalizing flow"],
                        label="flow $95\\%$ HDR")]
    for name in ("ACOS L2FP (v10)", "Laplace"):
        if name in preds:
            handles.append(L2D([], [], ls="-", lw=2, color=COLORS[name],
                               label=f"{name} $95\\%$"))
    handles.append(L2D([], [], marker="*", ls="none", ms=7, color="black",
                       label="simulated truth"))
    # Deliberately the smallest type in any figure here: five entries with labels this
    # long sit inside one data panel, and at anything larger the box covers the posterior
    # it is describing. It is a key, not an axis, and every
    # entry is also named in the caption. 6.2 pt on the page.
    #
    # Below the panels, not inside one. The submitted figure puts this key in the corner of
    # the first panel, and it fits there only because that canvas is 13.9 in wide: the same
    # six entries on the 8.5 in canvas take a far larger fraction of a panel, and there is
    # then no corner of panel 1 that is free -- `loc="best"` lands on the truth star (one
    # marker, so it costs the solver almost nothing) and `loc="upper left"` lands on the
    # flow's HDR contours. Outside, nothing is covered, no panel has to be read around a
    # box, and the type can go back up to a size that is legible in print rather than being
    # squeezed to fit a gap.
    fig.legend(handles=handles, loc="outside lower center", ncol=3,
               fontsize=PAIRS_LEGEND_FS, framealpha=0.92, labelspacing=0.3,
               handlelength=1.8, borderpad=0.4, handletextpad=0.6, columnspacing=1.8)

    path = out_dir / f"pairs_{_stem(art)}_{how}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    # Both are caption material: the indices make the panel reproducible, and the draw
    # count is the difference between a picture of a shape and a picture of 512 points.
    print(f"  case indices ({how}, seed={seed}): {list(map(int, cases))}")
    print(f"  sounding ids: {[str(art['ids'][i]) for i in cases]}")
    if n_drawn:
        print(f"  flow draws per panel: {sorted(set(n_drawn))}"
              f"{'  (--dense applied)' if dense is not None else ''}")
    return path

# ---------------------------------------------------------------------------------
# Generalization to unseen reference soundings
# ---------------------------------------------------------------------------------
def _by_scene(scene: np.ndarray, v: np.ndarray, how=np.mean):
    """Group means of `v` (n, ...) by scene, as (labels, means) with labels sorted.

    `how` is the reducer: `np.nanmean` where a method may have no value for some
    soundings, so that one unusable row costs its own row and not its whole scene.
    """
    labs = np.unique(scene)
    return labs, np.stack([how(v[scene == g], axis=0) for g in labs])


def error_decomposition(y_ppm, mu_ppm, scene):
    """Mean squared error split into offset, scene and within-scene parts, in ppm^2.

    `e = b + (m_g - b) + (e - m_g)` with `b` the per-level mean error and `m_g` the mean
    error over the replicates of scene `g`. The three terms are orthogonal under
    sounding-weighted averaging, so the parts sum to the total exactly -- which is what
    makes the figure a decomposition rather than three numbers side by side.

    The middle term is the one this experiment exists to measure: it is the error a model
    makes about *which scene this is*, and it is invisible to a replicate-level split,
    where every scene's offset was already learned from its other replicates.
    """
    e = mu_ppm - y_ppm
    b = e.mean(axis=0)
    labs, m = _by_scene(scene, e)
    idx = {g: i for i, g in enumerate(labs)}
    m_i = m[np.array([idx[g] for g in scene])]
    return dict(total=float((e ** 2).mean()),
                offset=float((b ** 2).mean()),
                scene=float(((m_i - b) ** 2).mean()),
                within=float(((e - m_i) ** 2).mean()))


def fig_scenes(art, preds, y, out_dir: Path):
    """Accuracy, calibration and log-density per held-out reference sounding.

    One panel each for RMSE, 95% coverage and NLL, over the reference soundings of the
    test partition -- the unit the group split is disjoint in, and therefore the unit in
    which "did it generalize" is a question with an answer. Under the replicate split
    these panels would be near-flat by construction: every scene's offset was already
    visible in its own training replicates.

    All three panels share one ordering -- by geophysical cluster, and by the flow's RMSE
    inside a cluster -- so a point at position *k* is the same scene in each. Sorting each
    method by its own value would turn the panels into three unrelated quantile functions
    and make a vertical comparison meaningless.

    Grouping by cluster is what makes the tail readable: the scenes the learned models
    retrieve worst are the ones from the clusters with fewest training scenes, and with
    each cluster contiguous the axis says so in one label per block rather than in
    forty-five alternating digits.
    """
    scene = scene_labels(art)
    if scene is None or len(np.unique(scene)) < 5:
        return None
    y_ppm = M.to_ppm(y, art.y_mu, art.y_sigma)
    names = list(preds)

    # The ordering method: the flow if it is here, else whatever came first, so the figure
    # still draws for a --skip-flow artifact.
    lead = next((n for n in names if "flow" in n.lower()), names[0])

    per_scene = {}
    for name in names:
        p = preds[name]
        v = p["valid"]
        mu_ppm = M.to_ppm(p["mu"], art.y_mu, art.y_sigma)
        labs, mse = _by_scene(scene, ((mu_ppm - y_ppm) ** 2).mean(axis=1, keepdims=True))

        if "samples" in p:
            hit = M.sample_marginal_coverage_mask(y, p["samples"], 0.95)
        else:
            hit = M.gaussian_marginal_coverage_mask(
                y, p["mu"], np.diagonal(p["cov"], axis1=-2, axis2=-1), 0.95)
        _, cov = _by_scene(scene, np.where(v[:, None], hit.astype(float), np.nan))

        # NaN where a method has no density for a sounding (a singular ACOS covariance),
        # so one unusable row costs its own scene nothing but that row.
        nll = np.asarray(p["nll"], dtype=np.float64)
        nll = np.where(v, nll, np.nan)
        _, nll_s = _by_scene(scene, nll[:, None], how=np.nanmean)

        per_scene[name] = dict(labs=labs, rmse=np.sqrt(mse.ravel()),
                               cov=100 * np.nanmean(cov, axis=1), nll=nll_s.ravel())

    labs = per_scene[lead]["labs"]
    # Cluster-major ordering (R3). The previous version sorted every scene by the flow's
    # RMSE, which put the clusters in no order at all: a run of one cluster was broken by
    # a scene from another every second or third position, and the tick labels alternated
    # the length of the axis. Sorting by cluster first and by the flow's RMSE only inside
    # a cluster makes each cluster one contiguous block, so the axis carries one label per
    # cluster and the blocks can be compared against each other -- which is the question
    # the group split exists to ask. The within-block rise is still the flow's RMSE, so
    # nothing about a single block's shape changes.
    #
    # Clusters are numbered, so they are ordered numerically rather than as strings, which
    # would put 10 between 1 and 2. Anything non-numeric sorts after, alphabetically.
    cl_all = np.array([g.split("/")[0] for g in labs])
    uniq = sorted(set(cl_all), key=lambda c: (0, int(c), "") if c.isdigit() else (1, 0, c))
    rank = np.array([uniq.index(c) for c in cl_all])
    # `lexsort` takes the primary key last.
    order = np.lexsort((per_scene[lead]["rmse"], rank))
    # The diagnostics printed at the end are about the worst scenes overall, which is no
    # longer the right-hand end of `order`, so they keep their own RMSE-only ranking.
    by_rmse = np.argsort(per_scene[lead]["rmse"])
    x = np.arange(len(labs))
    clusters = [cl_all[i] for i in order]

    # Stacked, not side by side, and this is legibility rather than taste. There are 45 held-out
    # reference soundings, so a three-panel row printed at the text width gives each tick
    # label 0.06 in -- a single digit does not fit in it at any legible point size, which is
    # why the earlier version had to drop to 8 pt and still overlapped. One column gives
    # each label 0.17 in and the axis can be read. The three panels share one ordering, so
    # stacking them also aligns each scene vertically across the three measurements, which
    # is the comparison the figure exists to support.
    fig, (ax_r, ax_c, ax_n) = plt.subplots(3, 1, figsize=_size(1.02), sharex=True,
                                           **_LAYOUT)
    for name in names:
        c = COLORS.get(name, None)
        d = per_scene[name]
        ax_r.plot(x, d["rmse"][order], marker="o", ms=4, color=c, label=name)
        ax_c.plot(x, d["cov"][order], marker="o", ms=4, color=c, label=name)
        ax_n.plot(x, d["nll"][order], marker="o", ms=4, color=c, label=name)

    # Linear, not logarithmic. The RMSE spans about one decade here, which is the awkward
    # range for a log axis: not wide enough to need one, but wide enough that a log axis
    # labels only the powers of ten unless it is given extra subdivisions, and it
    # compresses the tail that the figure exists to show. On a linear axis the worst
    # clusters stand as far above the rest as their error actually is, and the default
    # locator produces a handful of round ticks with no help. The cost is the bulk: the
    # forty-odd well-retrieved scenes are pressed into the lower part of the panel, and a
    # ratio between two of them can no longer be read off as a distance.
    ax_r.set_ylim(bottom=0)
    # Bare quantity names. A rotated y-label has only the panel height to live in, and
    # everything these used to carry beyond the quantity itself -- that the RMSE is within
    # a scene, that the coverage is marginal, that the log-density is in standardized
    # target coordinates -- is caption material, stated once in prose instead of three
    # times in 16 pt type down the side of the figure. NLL is left as the acronym: the
    # panel title above it already says "log-density", and the tables use NLL throughout.
    ax_r.set_ylabel("RMSE [ppm]", fontsize=LABEL_FS)
    nominal = ax_c.axhline(95.0, ls="--", color="gray", label="nominal 95%")
    ax_c.set_ylabel("95% cov.", fontsize=LABEL_FS)
    ax_c.set_ylim(-2, 102)
    ax_n.set_ylabel("NLL", fontsize=LABEL_FS)
    # ACOS can be an order of magnitude off the other two on a bad scene, which on a
    # linear axis flattens the flow into the baseline. A symmetric log keeps the flow's
    # structure visible without clipping anything away -- but only when the spread is
    # large enough to need it, and it must be symmetric because a 20-level NLL is
    # negative. Where everything already fits, linear is easier to read.
    all_nll = np.concatenate([per_scene[n]["nll"] for n in names])
    spread = float(np.nanmax(all_nll) - np.nanmin(all_nll))
    if spread > 30:
        ax_n.set_yscale("symlog", linthresh=max(1.0, 0.05 * spread))
    # A tick exactly where the cluster changes, labelled with the cluster that starts
    # there, and nowhere else. Now that the ordering makes each cluster contiguous there
    # are only eight or nine such positions, so the forty-five alternating digits of the
    # submitted version become one number per cluster, and the gridline that comes with
    # each tick marks the block boundary it sits on. No minor ticks: a tick that is not a
    # cluster change is the thing this axis is being cleared of.
    starts = [0] + [k for k in range(1, len(clusters))
                    if clusters[k] != clusters[k - 1]]
    for ax, t in ((ax_r, "(a) accuracy"), (ax_c, "(b) coverage"),
                  (ax_n, "(c) log-density")):
        ax.set_title(f"{t} per held-out reference sounding")
        ax.set_xticks(x[starts])
        ax.set_xticks([], minor=True)
        ax.tick_params(axis="x", which="major", length=4, pad=3)
    # `sharex` draws the labels on the bottom panel only, which is the point of stacking.
    ax_n.set_xticklabels([clusters[a] for a in starts])
    # An axis label, not a caption. What the ordering is and what a position means belong
    # to the manuscript caption, which can say it in prose and be typeset with the rest of
    # the paper; repeating it inside the PNG sets it in a second typeface at a second size
    # and pins the figure to one wording. The axis says what the axis is.
    fig.supxlabel("geophysical cluster", fontsize=SUPLABEL_FS)
    ax_r.legend(loc="upper left")
    # Only the nominal line is labelled in (b): the methods are named in (a), and a
    # second three-entry legend has no corner that is empty in all three tasks.
    # `loc="best"` for the same reason -- which corner is free depends on the task.
    ax_c.legend(handles=[nominal], loc="best")

    path = out_dir / f"scenes_{_stem(art)}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    # The tail, named, because it is the sentence the figure exists to support and it
    # should be in the log rather than read off the picture.
    k = min(4, len(labs))
    worst = by_rmse[-k:][::-1]
    print(f"  [{lead}] worst {k} scenes: " + ", ".join(
        f"{labs[i]} (cluster {labs[i].split('/')[0]}, {per_scene[lead]['rmse'][i]:.2f} ppm)"
        for i in worst))
    print(f"  [{lead}] best of the rest: "
          f"{per_scene[lead]['rmse'][by_rmse[:-k]].max():.2f} ppm worst, "
          f"{np.median(per_scene[lead]['rmse']):.2f} ppm median over all {len(labs)}")
    for name in names:
        d = error_decomposition(y_ppm[preds[name]["valid"]],
                                M.to_ppm(preds[name]["mu"], art.y_mu, art.y_sigma)[
                                    preds[name]["valid"]],
                                scene[preds[name]["valid"]])
        print(f"  [{name}] RMSE {np.sqrt(d['total']):.2f} ppm = offset "
              f"{100 * d['offset'] / d['total']:.0f}% + scene "
              f"{100 * d['scene'] / d['total']:.0f}% + within "
              f"{100 * d['within'] / d['total']:.0f}%")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("artifacts", nargs="*")
    ap.add_argument("--overlay", nargs="+", metavar="LABEL=ARTIFACT", default=None,
                    help="two or more artifacts of the same task drawn in one per-level "
                         "figure, e.g. --overlay 'native 4D=last4_borep.npz' "
                         "'marginal=last4_marginal_borep.npz'. This is Figure 6.")
    ap.add_argument("--drop", nargs="+", metavar="METHOD", default=None,
                    help="omit these methods from every figure (substring, "
                         "case-insensitive). For drawing an interim figure without a "
                         "predictive the scale check has flagged.")
    ap.add_argument("--overlay-stem", default=None,
                    help="filename stem for the overlay; defaults to the first "
                         "artifact's task")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--cases", default="outside95",
                    choices=["outside95", "random", "hard", "mixed", "both", "all",
                             "paper"],
                    help="selection rule for the bivariate panel: 'random' (uniform, "
                         "seeded), 'outside95' (truth outside the ACOS 95%% region -- the "
                         "submitted figure's rule), 'hard' (outside ACOS's OR the "
                         "Laplace's), 'mixed' (--n-hard from the hard pool, the rest "
                         "random), 'both' = outside95+random, 'all' = hard+outside95+"
                         "random, 'paper' = mixed+random, the two floats the paper uses")
    ap.add_argument("--n-hard", type=int, default=2,
                    help="panels drawn from the hard pool under --cases mixed (default 2)")
    ap.add_argument("--dense", default=None, metavar="NPZ",
                    help="a small artifact carrying many more flow draws for the panel "
                         "soundings (predict.py --ids ... --nf-samples 20000). Matched by "
                         "sounding id; soundings absent from it keep the 512 draws.")
    ap.add_argument("--flow-contours", action="store_true",
                    help="also draw the flow's own 68/95%% highest-density contours, so "
                         "the comparison against the Gaussian ellipses is like for like. "
                         "Needs a few thousand draws to be smooth -- use with --dense.")
    ap.add_argument("--pair-dims", type=int, nargs=2, default=[17, 20],
                    metavar=("LEVEL", "LEVEL"), help="1-based pressure levels to plot")
    ap.add_argument("--n-panels", type=int, default=6)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args(argv)

    if not args.artifacts and not args.overlay:
        ap.error("give at least one artifact, or --overlay LABEL=ARTIFACT ...")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # `{sounding id -> [S, D] draws}`. Loaded once: the same panel soundings are used for
    # every rule, so the dense file is read whichever --cases is asked for.
    dense = None
    if args.dense:
        d = np.load(args.dense, allow_pickle=False)
        if "nf_samples_std" not in d or "ids" not in d:
            raise SystemExit(f"{args.dense} carries {list(d)}; a dense-sample artifact "
                             f"needs `nf_samples_std` and `ids`.")
        dense = {str(k): np.asarray(v, dtype=np.float64)
                 for k, v in zip(d["ids"], d["nf_samples_std"])}
        n_each = {v.shape[0] for v in dense.values()}
        print(f"  dense draws: {len(dense)} soundings x {sorted(n_each)} samples "
              f"from {args.dense}")

    if args.overlay:
        items = []
        for spec in args.overlay:
            if "=" not in spec:
                ap.error(f"--overlay takes LABEL=ARTIFACT, got {spec!r}")
            label, path = spec.split("=", 1)
            art = Artifact.load(path)
            print(f"\n=== overlay [{label}] {path}\n{art.summary()}")
            y, preds = predictives(art)
            check_scale(art, preds, y)
            preds = drop_methods(preds, args.drop)
            items.append((label, art, preds, y))
        stem = args.overlay_stem or items[0][1].meta.task
        p = fig_levels_overlay(items, out_dir, stem)
        if p is not None:
            print(f"  wrote {p}")

    for path in args.artifacts:
        art = Artifact.load(path)
        print(f"\n=== {path}\n{art.summary()}")
        y, preds = predictives(art)
        check_scale(art, preds, y)
        preds = drop_methods(preds, args.drop)
        made = [fig_pit(art, preds, y, out_dir),
                fig_coverage(art, preds, y, out_dir),
                fig_joint(art, preds, out_dir),
                fig_scenes(art, preds, y, out_dir)]
        if art.meta.dim > 1:
            made.append(fig_levels(art, preds, y, out_dir))
            hows = {"both": ["outside95", "random"],
                    "all": ["hard", "outside95", "random"],
                    "paper": ["mixed", "random"]}.get(args.cases, [args.cases])
            for how in hows:
                made.append(fig_pairs(art, preds, y, out_dir, args.pair_dims,
                                      args.n_panels, how, args.seed,
                                      dense=dense, contours=args.flow_contours,
                                      n_hard=args.n_hard))
        for p in [p for p in made if p is not None]:
            print(f"  wrote {p}")

        # The numbers behind the joint panel.
        for name, p in preds.items():
            s = JC.summarize(p["z"], name)
            print(f"  [{name}] mahalanobis mean {s.get('mahalanobis_mean', float('nan')):.2f} "
                  f"(expected {s.get('mahalanobis_expected', float('nan')):.0f}), "
                  f"chi2-PIT KS {s.get('chi2_pit_ks', float('nan')):.3f}, "
                  f"dropped {s.get('n_dropped', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

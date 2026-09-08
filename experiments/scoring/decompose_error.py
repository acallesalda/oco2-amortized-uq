#!/usr/bin/env python3
"""One decomposition, applied identically to the three held-out experiments.

    MSE  =  (scene-placement term)^2  +  (within-scene term)^2

where the scene-placement term is the per-reference-sounding mean error and the
within-scene term is what is left after removing it.  The two are orthogonal by
construction, so the identity is exact.

The within-scene term has a floor: an oracle that knew each reference sounding's mean true
state exactly, and nothing else about the individual sounding, pays the truth's own
within-scene spread.  Hence

    within / floor  = 1   ->  the method reads nothing out of the individual spectrum
    within / floor  < 1   ->  it reads something
    RMSE   / floor        ->  total error in units of that bound

This is the unit that makes the three experiments comparable.  Raw ppm are not: the floor
is a property of the population, and it differs by 20 % between February and July, so a
larger RMSE on July is not by itself evidence of a transfer failure.

Reads only artifacts already in the checkout; no
queue, no GPU, about a minute.

    python3 experiments/scoring/decompose_error.py --out experiments/scoring/tables \
        --fig latex/figures

Outputs
    decomposition_summary.csv      task x experiment x method, pooled and at level 20
    decomposition_by_cluster.csv   the group split, per February cluster, both tasks
    decomposition_by_scene.csv     the group split, per reference sounding, both tasks
    decomposition_bounds.csv       what a perfect scene placement would buy
    decomposition.png (--fig)      the three claims of the temporal experiment: what moves
                                   between months (the placement term), what does not (the
                                   within-sounding term, pinned to its floor), and why the
                                   coverage falls (a constant width against a growing floor)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
# `why_profiles` also configures matplotlib (Agg backend, the paper's rcParams) and owns
# the method colours, so the figure below matches every other figure in the revision.
from why_profiles import Art, COLORS, _LAYOUT, plt  # noqa: E402

# Training reference soundings per cluster in the reference-disjoint partition.
# Cluster 2's five scenes are all in training, so it has no test row.
N_TRAIN = {1: 86, 2: 5, 3: 70, 4: 54, 5: 29, 6: 54, 7: 37, 8: 12, 9: 14}


# ---------------------------------------------------------------------------------
def decompose(a: Art, pred, idx=None, cols=None):
    """The decomposition above, on a subset of soundings and/or of output columns."""
    sl = slice(None) if cols is None else cols
    err = (pred - a.y)[:, sl]
    off = (a.scene_mean(pred) - a.scene_mean(a.y))[:, sl]
    fl = (a.scene_mean(a.y) - a.y)[:, sl]
    if idx is not None:
        err, off, fl = err[idx], off[idx], fl[idx]
    mse = float(np.mean(err ** 2))
    off_mse = float(np.mean(off ** 2))
    within_mse = max(mse - off_mse, 0.0)
    floor_mse = float(np.mean(fl ** 2))
    return dict(
        rmse=np.sqrt(mse),
        placement=np.sqrt(off_mse),
        within=np.sqrt(within_mse),
        floor=np.sqrt(floor_mse),
        over_floor=np.sqrt(mse / floor_mse),
        within_over_floor=np.sqrt(within_mse / floor_mse),
        placement_share=off_mse / mse if mse > 0 else np.nan,
        oracle_placement=np.sqrt(within_mse),
        oracle_gain_pct=100.0 * (1.0 - np.sqrt(within_mse / mse)) if mse > 0 else np.nan,
    )


def widths(a: Art, name, idx=None, cols=None):
    """Predictive width and coverage, in the same units, for the calibration half."""
    sl = slice(None) if cols is None else cols
    S = {"flow": a.flow_sd, "Laplace": a.lap_sd, "ACOS": a.acos_sd}[name][:, sl]
    M = {"flow": a.flow, "Laplace": a.lap, "ACOS": a.acos}[name]
    e = (M - a.y)[:, sl]
    if idx is not None:
        S, e = S[idx], e[idx]
    sd = float(np.sqrt(np.mean(S ** 2)))
    with np.errstate(invalid="ignore"):
        cov = float(np.nanmean(np.abs(e) <= 1.96 * S))
    return dict(pred_sd=sd, cov95=100 * cov)


def epistemic_only(path: Path) -> bool:
    """Is this artifact's Laplace predictive the Note-14 defective one?

    `predict.py` adds the calibrated observation-noise diagonal to the epistemic covariance
    when it assembles `lap_cov_std`; under the old `marglik` path there was no such term and
    the two arrays are bit-identical. Only the *width* columns are affected -- `pred_sd`,
    `z`, `cov95` -- and only for the Laplace. Every other column here is a point-estimate
    quantity and is unaffected.

    This is checked because the three evaluations are no longer scored from the same
    generation of artifact: after the 31 August re-score February's `xco2_bo.npz` carries the
    observation-noise term and the two month files do not, so panel (c) of the figure would
    otherwise show a corrected February width beside two uncorrected ones and read as a
    finding -- a failure mode this project has had once already.
    """
    with np.load(path, allow_pickle=False) as d:
        if "lap_cov_std" not in d.files or "lap_cov_epi_std" not in d.files:
            return False
        return bool(np.array_equal(np.asarray(d["lap_cov_std"], dtype=np.float64),
                                   np.asarray(d["lap_cov_epi_std"], dtype=np.float64)))


def rows_for(a: Art, task, experiment, cols=None, level=None, lap_void=False):
    out = []
    for name, M in a.methods():
        r = dict(task=task, experiment=experiment, level=level or "pooled",
                 n=len(a.ids), n_scenes=len(a.scenes))
        r.update(decompose(a, M, cols=cols))
        r.update(widths(a, name, cols=cols))
        r["method"] = name
        r["z"] = r["rmse"] / r["pred_sd"] if r["pred_sd"] else np.nan
        # Carried in the CSV rather than applied to it: the numbers stay inspectable, and
        # every consumer (the figure below, and anything written later) can see that the
        # three width columns of this row are not reportable.
        r["width_void"] = bool(lap_void and name == "Laplace")
        out.append(r)
    return out


# ---------------------------------------------------------------------------------
METHOD_LABEL = {"flow": "Normalizing flow", "Laplace": "Laplace", "ACOS": "ACOS L2FP"}


def fig_decomposition(S: pd.DataFrame, out_dir: Path, experiments, stem="decomposition"):
    """The three claims of the temporal experiment, in one figure.

    One row per task, three panels across:

      (a) the scene-placement term. This is the only part of the error that generalization
          can touch, and it is where the whole difference between the three experiments
          lives: ours doubles, the operational retrieval's -- which does not train -- does
          not move. Reading it beside the raw RMSE is what stops "July is worse" from being
          mistaken for "the network failed to transfer".
      (b) the within-sounding error in units of its own floor. Unity means the method reads
          nothing out of the individual spectrum. The amortized posterior sits on that line
          near the surface in all three evaluations and below it on the column, which is the
          same fact that appears as a correlation of -0.01, here in ppm and exact.
      (c) the mean predictive standard deviation against that floor (dashed). The
          calibration failure out of distribution is one comparison: the reported width is
          essentially constant across the three evaluations while the floor grows, so on
          March the flow claims a precision no estimator without per-sounding information
          can have.

    Bars are grouped by evaluation so that a method is one colour across the whole figure,
    and the legend is drawn once: with three panels per row and nine bars per panel, a
    legend in every axes covers the bars it is explaining. The two reference lines are
    annotated in place for the same reason.
    """
    tasks = [t for t in ("near surface", "XCO2") if t in set(S.task)]
    exps = [e for e in experiments if e in set(S.experiment)]
    if not tasks or len(exps) < 2:
        return None
    methods = [m for m in ("flow", "Laplace", "ACOS") if m in set(S.method)]
    w = 0.8 / len(methods)
    fig, axes = plt.subplots(len(tasks), 3, figsize=(16.5, 4.8 * len(tasks)), **_LAYOUT)
    axes = np.atleast_2d(axes)

    def cell(task, exp, method, col, default=np.nan):
        r = S[(S.task == task) & (S.experiment == exp) & (S.level == "pooled")
              & (S.method == method)]
        # `width_void` is absent from a summary written before it existed; a stale CSV
        # should draw the old figure rather than crash `--fig-only`.
        if not len(r) or col not in r.columns:
            return default
        return float(r.iloc[0][col])

    def pad(ax, values, bottom=None, head=0.18):
        """Leave room above the bars for the legend or the annotation."""
        v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
        if not v.size:
            return
        lo = 0.0 if bottom is None else bottom
        ax.set_ylim(lo, v.max() + head * (v.max() - lo))

    for i, task in enumerate(tasks):
        x = np.arange(len(exps))
        floors = [cell(task, e, methods[0], "floor") for e in exps]
        series = {0: [], 1: [], 2: []}
        # Precomputed because the "pending re-score" marker needs a height before the bars
        # it sits among have all been drawn.
        wmax = max([v for v in
                    [cell(task, e, m, "pred_sd") for e in exps for m in methods] + floors
                    if np.isfinite(v)] or [1.0])
        for j, m in enumerate(methods):
            off = (j - (len(methods) - 1) / 2) * w
            c = COLORS.get(m, f"C{j}")
            lab = METHOD_LABEL.get(m, m)
            for col, key in ((0, "placement"), (1, "within_over_floor"), (2, "pred_sd")):
                vals = [cell(task, e, m, key) for e in exps]
                # A width that is not reportable is not drawn. `width_void` marks the
                # Laplace rows of any artifact scored before the observation-noise term
                # existed; the bar would otherwise sit beside a corrected one at a third of
                # its height and be read as a result. The two point-estimate panels are
                # unaffected, which is why the flag is applied here and only here.
                if col == 2:
                    void = [bool(cell(task, e, m, "width_void", 0.0)) for e in exps]
                    for k, bad in enumerate(void):
                        if bad:
                            axes[i, col].text(x[k] + off, 0.02 * wmax, "pending re-score",
                                              rotation=90, fontsize=9, ha="center",
                                              va="bottom", color="#777777")
                    vals = [np.nan if bad else v for v, bad in zip(vals, void)]
                series[col] += [v for v in vals if np.isfinite(v)]
                axes[i, col].bar(x + off, vals, w, color=c,
                                 label=lab if (i, col) == (0, 0) else None)
        # The floor is a property of the evaluation set, not of a method, so it is drawn
        # once per group rather than as a fourth bar.
        for k, f in enumerate(floors):
            axes[i, 2].plot([k - 0.45, k + 0.45], [f, f], ls="--", lw=2.5,
                            color="#444444")
        axes[i, 1].axhline(1.0, ls="--", lw=2.5, color="#444444")

        task_name = "Near-surface, levels 17-20" if task == "near surface" \
            else r"Column average, $X$CO$_2$"
        axes[i, 0].set_ylabel("scene placement [ppm]")
        axes[i, 1].set_ylabel("within sounding / bound")
        axes[i, 2].set_ylabel("mean predictive sd [ppm]")
        pad(axes[i, 0], series[0], head=0.30 if i == 0 else 0.12)
        # Zoomed, and deliberately: every value is between 0.78 and 1.09, and an axis
        # anchored at zero turns the one finding this panel carries -- that our ratio is
        # 1.00 and the physical retrieval's is not -- into nine bars of the same height.
        v1 = np.asarray([v for v in series[1] if np.isfinite(v)])
        axes[i, 1].set_ylim(max(0.0, v1.min() - 0.10), v1.max() + 0.10)
        pad(axes[i, 2], list(series[2]) + list(floors), head=0.16)
        axes[i, 1].annotate("reads nothing per sounding", xy=(len(exps) - 1.45, 1.0),
                            xytext=(0, 6), textcoords="offset points", fontsize=11,
                            color="#444444")
        # Above everything in the first group, whether that is the bound or a bar, so the
        # label never lands on top of the line it names.
        head = max([floors[0]] + [cell(task, exps[0], m, "pred_sd") for m in methods])
        axes[i, 2].annotate("within-sounding bound", xy=(-0.45, head),
                            xytext=(0, 8), textcoords="offset points", fontsize=11,
                            color="#444444")
        for k, ax in enumerate(axes[i]):
            ax.set_xticks(x)
            ax.set_xticklabels([e.replace("group split (Feb)", "February\n(held out)")
                                for e in exps])
            ax.set_title(f"({'abcdef'[3 * i + k]}) {task_name}")
    axes[0, 0].legend(fontsize=11, loc="upper left")
    path = Path(out_dir) / f"{stem}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


# ---------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="experiments/scoring/tables")
    ap.add_argument("--xco2-group", default="experiments/scoring/xco2_bo.npz")
    ap.add_argument("--last4-group", default="experiments/scoring/last4_bo.npz")
    ap.add_argument("--month-root", default="experiments/scoring/results",
                    help="holds month_<tag>/{xco2,last4}.npz")
    ap.add_argument("--months", nargs="*", default=["202003_full", "202007_full"],
                    help="pass with no values to skip the months entirely")
    ap.add_argument("--fig", default=None,
                    help="write decomposition.png here (e.g. latex/figures)")
    ap.add_argument("--fig-stem", default="decomposition")
    ap.add_argument("--fig-only", action="store_true",
                    help="redraw the figure from decomposition_summary.csv and stop. The "
                         "decomposition reads about a gigabyte of artifacts and takes a "
                         "minute; changing a legend should not.")
    ap.add_argument("--label", default="group split (Feb)",
                    help="name for the February rows; use e.g. 'balanced' when pointing "
                         "--xco2-group / --last4-group at the _bal artifacts, and give "
                         "--out a directory of its own so the published tables survive")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    GS = args.label
    MONTH_NAME = {"202003_full": "March", "202007_full": "July"}

    if args.fig_only:
        if not args.fig:
            print("--fig-only needs --fig DIR")
            return 2
        S = pd.read_csv(out / "decomposition_summary.csv")
        order = [GS] + [MONTH_NAME.get(t, t) for t in args.months]
        fig_decomposition(S, Path(args.fig), order, stem=args.fig_stem)
        return 0

    cases = [("XCO2", GS, Path(args.xco2_group)),
             ("near surface", GS, Path(args.last4_group))]
    for tag in args.months:
        label = MONTH_NAME.get(tag, tag)
        root = Path(args.month_root) / f"month_{tag}"
        cases += [("XCO2", label, root / "xco2.npz"),
                  ("near surface", label, root / "last4.npz")]

    summary, arts = [], {}
    for task, label, path in cases:
        if not path.exists():
            print(f"!! missing, skipped: {path}")
            continue
        print(f"== {task:13s} {label:18s} {path}")
        a = Art(path)
        arts[(task, label)] = a
        void = epistemic_only(path)
        if void:
            print("   !! this artifact's Laplace predictive has no observation-noise term;"
                  " its width columns are marked width_void and left out of the figure.")
        summary += rows_for(a, task, label, lap_void=void)
        if task == "near surface" and 20 in a.levels:      # the surface level alone
            summary += rows_for(a, task, label, cols=[a.levels.index(20)], level="20",
                                lap_void=void)

    S = pd.DataFrame(summary)[
        ["task", "experiment", "level", "method", "n", "n_scenes", "rmse", "placement",
         "within", "floor", "over_floor", "within_over_floor", "placement_share",
         "oracle_placement", "oracle_gain_pct", "pred_sd", "z", "cov95", "width_void"]]
    print("\n### the decomposition\n")
    print(S.round(4).to_string(index=False))
    S.to_csv(out / "decomposition_summary.csv", index=False)

    if args.fig:
        print("\n### figure\n")
        order = [GS] + [MONTH_NAME.get(t, t) for t in args.months]
        fig_decomposition(S, Path(args.fig), order, stem=args.fig_stem)

    # ---- per cluster and per scene, the group split
    per_c, per_s = [], []
    for task in ("XCO2", "near surface"):
        a = arts.get((task, GS))
        if a is None:
            continue
        for c in np.unique(a.clst):
            idx = a.clst == c
            r = dict(task=task, cluster=int(c), n_train=N_TRAIN.get(int(c)),
                     n_test_scenes=len(np.unique(a.scene[idx])), n=int(idx.sum()))
            for name, M in a.methods():
                for k, v in decompose(a, M, idx=idx).items():
                    r[f"{name}_{k}"] = v
            r["floor"] = decompose(a, a.flow, idx=idx)["floor"]
            per_c.append(r)
        for s in a.scenes:
            idx = a.scene == s
            r = dict(task=task, scene=s, cluster=int(a.clst[idx][0]), n=int(idx.sum()))
            for name, M in a.methods():
                d = decompose(a, M, idx=idx)
                for k in ("rmse", "placement", "over_floor", "within_over_floor",
                          "placement_share"):
                    r[f"{name}_{k}"] = d[k]
            r["floor"] = decompose(a, a.flow, idx=idx)["floor"]
            per_s.append(r)
    C = pd.DataFrame(per_c)
    C.to_csv(out / "decomposition_by_cluster.csv", index=False)
    pd.DataFrame(per_s).to_csv(out / "decomposition_by_scene.csv", index=False)
    for task in C.task.unique():
        print(f"\n### group split, {task}, per February cluster\n")
        cols = ["cluster", "n_train", "n", "floor", "flow_rmse", "flow_placement",
                "flow_over_floor", "flow_within_over_floor", "ACOS_rmse",
                "ACOS_over_floor", "ACOS_within_over_floor"]
        print(C[C.task == task][cols].round(4).to_string(index=False))

    # ---- the two under-sampled clusters against the rest
    print("\n### group split: clusters 8+9 against 1-7\n")
    rows = []
    for task in ("XCO2", "near surface"):
        a = arts.get((task, GS))
        if a is None:
            continue
        for name, sel in (("clusters 1-7", ~np.isin(a.clst, [8, 9])),
                          ("clusters 8+9", np.isin(a.clst, [8, 9]))):
            for m, M in a.methods():
                d = decompose(a, M, idx=sel)
                rows.append(dict(task=task, group=name, method=m, n=int(sel.sum()), **d))
    G = pd.DataFrame(rows)
    print(G[["task", "group", "method", "n", "rmse", "placement", "floor", "over_floor",
             "within_over_floor"]].round(4).to_string(index=False))

    # ---- what any per-cluster remedy can buy at most
    print("\n### upper bound on a per-cluster remedy: zero those scenes' placement error\n")
    bounds = []
    for task in ("XCO2", "near surface"):
        a = arts.get((task, GS))
        if a is None:
            continue
        err = a.flow - a.y
        off = a.scene_mean(a.flow) - a.scene_mean(a.y)
        base = np.sqrt(np.mean(err ** 2))
        for nm, sel in (("cluster 8", a.clst == 8),
                        ("clusters 8+9", np.isin(a.clst, [8, 9])),
                        ("every scene", np.ones(len(a.ids), bool))):
            e2 = err.copy()
            e2[sel] -= off[sel]
            v = float(np.sqrt(np.mean(e2 ** 2)))
            bounds.append(dict(task=task, zeroed=nm, rmse=base, bound=v,
                               gain_pct=100 * (1 - v / base)))
            print(f"  {task:13s} {nm:13s}: {base:7.4f} -> {v:7.4f} ppm "
                  f"({100*(1-v/base):.1f} %)")
    pd.DataFrame(bounds).to_csv(out / "decomposition_bounds.csv", index=False)

    # ---- attribution of each month's rise
    print("\n### how much of each month's rise is the month, and how much is lost skill\n")
    for task in ("XCO2", "near surface"):
        base = S[(S.task == task) & (S.experiment == GS)
                 & (S.level == "pooled") & (S.method == "flow")]
        if base.empty:
            continue
        b = base.iloc[0]
        for label in ("March", "July"):
            m = S[(S.task == task) & (S.experiment == label) & (S.level == "pooled")
                  & (S.method == "flow")]
            if m.empty:
                continue
            r = m.iloc[0]
            same = b.over_floor * r.floor        # February's skill, this month's floor
            rise = r.rmse - b.rmse
            print(f"  {task:13s} {label:6s}: floor {b.floor:.2f} -> {r.floor:.2f}; "
                  f"at February's skill the flow would reach {same:.2f}, it reaches "
                  f"{r.rmse:.2f}. Rise {rise:.2f} = {same-b.rmse:.2f} "
                  f"({100*(same-b.rmse)/rise:.0f} %) the month "
                  f"+ {r.rmse-same:.2f} ({100*(r.rmse-same)/rise:.0f} %) lost skill; "
                  f"placement {b.placement:.2f} -> {r.placement:.2f}, "
                  f"within/floor {b.within_over_floor:.3f} -> {r.within_over_floor:.3f}")

    print(f"\nwrote 4 CSVs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

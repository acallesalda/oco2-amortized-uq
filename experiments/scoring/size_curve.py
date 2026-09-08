#!/usr/bin/env python3
"""size_curve.py -- held-out error against the number of training reference soundings.

The reading half of chtc/size.dag. It adds no metric of its own: every number comes from
`score.py`'s own tables, so a point on this curve is the same quantity as the corresponding
cell of Table~\\ref{tab:splitcompare}. Rows are matched by split name -- `size_<k>_v1[_cov]`
for the subsets, and the published split for the k = 361 point.

    # after scoring every artifact into one table directory
    python3 experiments/scoring/score.py \
        experiments/scoring/results/size_90_v1/xco2_s90.npz \
        experiments/scoring/results/size_180_v1/xco2_s180.npz \
        experiments/scoring/results/size_270_v1/xco2_s270.npz \
        experiments/scoring/results/group_split_v1_cov/xco2_bo.npz \
        --out experiments/scoring/tables

    python3 experiments/scoring/size_curve.py --tables experiments/scoring/tables \
        --task xco2 --out experiments/scoring/tables --fig latex/figures

Writes `size_curve_<task>.csv` and `size_curve_<task>.png`: RMSE, NLL and 95% coverage
against k, one line per method. ACOS L2FP is the control -- it does not train, so its line
must be flat, and any slope in it is a bug in which soundings were scored rather than a
result.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import ticker as mticker             # noqa: E402

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "figure.dpi": 120, "savefig.dpi": 300, "axes.grid": True,
    "grid.alpha": 0.25,
})

RE_SIZE = re.compile(r"^size_(\d+)_v1")
METHOD_ORDER = ["ACOS L2FP (v10)", "Laplace", "Normalizing flow"]


def collect(tables: Path, task: str, full_split: str, full_k: int) -> pd.DataFrame:
    path = tables / f"table_{task}.csv"
    if not path.exists():
        raise SystemExit(f"{path} does not exist. Run score.py over the artifacts first.")
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        split = str(r["split"])
        m = RE_SIZE.match(split)
        if m:
            k = int(m.group(1))
        elif split.startswith(full_split):
            k = int(full_k)
        else:
            continue
        rows.append(dict(k=k, split=split, method=r["method"],
                         rmse=float(r["rmse_ppm"]),
                         rmse_se=float(r.get("rmse_ppm_se", np.nan)),
                         mae=float(r["mae_ppm"]),
                         nll=float(r["nll_std"]),
                         cov95=float(r.get("cov95_marginal", np.nan)),
                         n=int(r["n"])))
    if not rows:
        raise SystemExit(
            f"no rows of {path} belong to a size_<k>_v1 split or to '{full_split}'. "
            f"Splits present: {sorted(set(df['split']))}")
    out = pd.DataFrame(rows).sort_values(["method", "k"]).reset_index(drop=True)
    # Every point must be scored on the same soundings, or the curve is comparing
    # populations rather than training sizes.
    for method, sub in out.groupby("method"):
        if sub["n"].nunique() != 1:
            print(f"[warn] {method} was scored on {sorted(set(sub['n']))} soundings at "
                  "different sizes; the points are not comparable.")
    return out


def figure(df: pd.DataFrame, task: str, out_png: Path) -> None:
    panels = [("rmse", "RMSE [ppm]"), ("nll", "NLL [std]"), ("cov95", "95% coverage")]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6))
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    methods += [m for m in sorted(set(df["method"])) if m not in methods]
    for ax, (col, label) in zip(axes, panels):
        for method in methods:
            sub = df[df["method"] == method].sort_values("k")
            if sub[col].isna().all():
                continue
            style = dict(marker="o", lw=2)
            if method.startswith("ACOS"):
                style.update(ls="--", color="0.45")      # the control
            ax.plot(sub["k"], sub[col], label=method, **style)
            if col == "rmse" and sub["rmse_se"].notna().any():
                ax.fill_between(sub["k"], sub["rmse"] - sub["rmse_se"],
                                sub["rmse"] + sub["rmse_se"], alpha=0.15, lw=0)
        if col == "cov95":
            ax.axhline(0.95, color="k", ls=":", lw=1.2)
        ax.set_xlabel("training reference soundings")
        ax.set_ylabel(label)
        ax.set_xscale("log")
        ax.set_xticks(sorted(set(df["k"])))
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        # matplotlib keeps its own minor labels on a log axis ("2 x 10^2") and they land on
        # top of the four hand-set ticks.
        ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    axes[0].legend(frameon=False)
    fig.suptitle(f"held-out error against training scenes -- {task}", y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def slope(df: pd.DataFrame) -> pd.DataFrame:
    """log-log slope of RMSE against k, fitted per method, and the last increment.

    The slope is the number the conclusion's claim rests on: a power law in the number of
    scenes with exponent b means going from 361 to 3610 scenes buys a factor 10**b. The
    last increment (270 -> 361 here) says whether the curve has flattened where the data
    end, which a single fitted exponent can hide.
    """
    out = []
    for method, sub in df.groupby("method"):
        sub = sub.sort_values("k")
        if len(sub) < 2:
            continue
        b, a = np.polyfit(np.log10(sub["k"]), np.log10(sub["rmse"]), 1)
        last = sub.iloc[-2:]
        out.append(dict(
            method=method, points=len(sub),
            loglog_slope=float(b),
            rmse_at_min_k=float(sub["rmse"].iloc[0]), min_k=int(sub["k"].iloc[0]),
            rmse_at_max_k=float(sub["rmse"].iloc[-1]), max_k=int(sub["k"].iloc[-1]),
            last_step_delta=float(last["rmse"].iloc[1] - last["rmse"].iloc[0]),
            predicted_at_10x=float(10 ** (a + b * np.log10(10 * sub["k"].iloc[-1]))),
        ))
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", type=Path, default=Path("experiments/scoring/tables"))
    ap.add_argument("--task", default="xco2", choices=["xco2", "profiles", "last4"])
    ap.add_argument("--out", type=Path, default=Path("experiments/scoring/tables"))
    ap.add_argument("--fig", type=Path, default=Path("latex/figures"))
    ap.add_argument("--full-split", default="group_split_v1_cov",
                    help="split name of the k = all point (default: the published run)")
    ap.add_argument("--full-k", type=int, default=361,
                    help="its number of training reference soundings")
    args = ap.parse_args()

    df = collect(args.tables, args.task, args.full_split, args.full_k)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / f"size_curve_{args.task}.csv", index=False)
    s = slope(df)
    s.to_csv(args.out / f"size_curve_{args.task}_slopes.csv", index=False)
    figure(df, args.task, args.fig / f"size_curve_{args.task}.png")

    print(df.to_string(index=False))
    print()
    print(s.to_string(index=False))
    print(f"\n-> {args.out / f'size_curve_{args.task}.csv'}")
    print(f"-> {args.fig / f'size_curve_{args.task}.png'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""learning_curves.py -- the training and validation curves of the six trained models.

The model diagnostics that speak to over-fitting. The first one an ML reader
looks for is the pair of curves: the loss on the rows the optimizer sees and the loss on
rows it does not, epoch by epoch. If the second turns upward while the first keeps falling,
the model is over-fitting and the checkpoint should have been taken earlier; if it flattens
and stays flat, it is not, and the early-stopping rule has nothing left to buy.

Two sources, in this order of preference:

1. `history.json` / `*.history.json`, written next to each checkpoint by the five trainers.
   Complete, machine-readable, and the only source that carries the deterministic models'
   *training* loss -- they logged validation only until this was added, so a run from
   before that change has to come from source 2 and will have no training curve.

2. The trainers' stdout, i.e. the `chtc/logs/*.out` files a queued run leaves behind. Every
   trainer prints one line per epoch; the five formats are parsed here. This is what makes
   the curves of the *published* runs available without re-training anything.

    # on the access point, where the .out files are
    python3 experiments/scoring/learning_curves.py chtc/logs/*final*.out \
        --out experiments/scoring/tables --fig latex/figures

    # or from fetched archives, which carry the history files
    python3 experiments/scoring/learning_curves.py \
        experiments/weights/group_split_v1/**/history.json --out ... --fig ...

Writes `learning_curves.csv` (one row per model per epoch) and `learning_curves.png`
(one panel per model, training and validation on the same axes, the selected epoch marked).
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "figure.dpi": 120, "savefig.dpi": 300, "axes.grid": True,
    "grid.alpha": 0.25,
})

# --------------------------------------------------------------------------------------
# stdout formats, one per trainer
# --------------------------------------------------------------------------------------
# train.py, after the history patch:  "epoch = 7, train loss = 5.44E-01, val loss = 6.02E-01, retr_loss = ..."
# train.py, before it:                "epoch = 7, val loss = 6.02E-01, retr_loss = ..."
RE_DET_XCO2 = re.compile(
    r"epoch\s*=\s*(?P<epoch>\d+),\s*(?:train loss\s*=\s*(?P<train>[-\d.eE+]+),\s*)?"
    r"val loss\s*=\s*(?P<val>[-\d.eE+]+)")
# train_profiles.py:  "epoch=7  val_RMSE[std]=0.9123  retr_RMSE[std]=..."
RE_DET_PROF = re.compile(
    r"epoch=(?P<epoch>\d+)\s+val_RMSE\[std\]=(?P<val>[-\d.eE+]+)")
# train_profiles.py, the line added with the history:  "epoch=7  train_loss=0.812345"
RE_DET_PROF_TR = re.compile(r"epoch=(?P<epoch>\d+)\s+train_loss=(?P<train>[-\d.eE+]+)")
# train_nf.py:            "[ep 007] train NLL: 0.81 | val NLL: 0.91"
# train_nf_profiles.py:   "[ep 007] train NLL=0.81 | val NLL=0.91 | val RMSE~0.81234"
#
# The third field appears only when eval.rmse_samples > 0, which the published runs did not
# set and the extension run does. It is the diagnostic the run exists to produce: the flow
# selects its checkpoint on NLL and the paper reports RMSE, so if the two curves bottom out
# at different epochs then the reported RMSE is not the best the configuration reached.
#
# `[^-\d]{1,3}` rather than a literal separator: the trainer prints U+2248 (almost-equal),
# and a log read back on a machine whose default encoding is not UTF-8 turns that into one
# or more replacement characters. Excluding digits and the sign from the separator class is
# what stops it swallowing the front of the number -- a plain `.{1,3}` is greedy, matched
# "~0." and turned 0.65247 into 65247.
RE_NF = re.compile(
    r"\[ep\s*(?P<epoch>\d+)\]\s*train NLL[:=]\s*(?P<train>[-\d.eE+]+)\s*\|\s*"
    r"val NLL[:=]\s*(?P<val>[-\d.eE+]+)"
    r"(?:\s*\|\s*val RMSE[^-\d]{1,3}(?P<val2>[-\d.eE+]+))?")
# train_nf_profiles_4d.py: "[ep 07] train_nll=0.81 | val_nll=0.91 | best=..." (or val only)
RE_NF4 = re.compile(
    r"\[ep\s*(?P<epoch>\d+)\]\s*(?:train_nll=(?P<train>[-\d.eE+]+)\s*\|\s*)?"
    r"val_nll=(?P<val>[-\d.eE+]+)")
# The same trainer's ITERATION line: "[ep 07 it 0200] train_nll=9.4691".
#
# Before this revision its per-epoch line carried the validation NLL only, so for the
# published nf_last4 run this is the only record of the training loss. The printed number
# is the epoch's running mean (train_loss_sum / n_train), so the LAST iteration line of an
# epoch is that epoch's mean training NLL -- not an approximation of it.
RE_NF4_IT = re.compile(
    r"\[ep\s*(?P<epoch>\d+)\s+it\s*(?P<it>\d+)\]\s*train_nll=(?P<train>[-\d.eE+]+)")


# Panel titles. The file a CHTC job leaves behind is named after the study
# (`nf_xco2_final.out`, `det_last4_final_s90.out`), which is the right key for a CSV and the
# wrong one for a figure that goes in the paper. Anything not listed keeps its file stem,
# and any run suffix -- _bal, _full, _rep, _s90 -- is appended so two runs of one study
# stay distinguishable in the same figure.
STUDY_LABELS = {
    "det_xco2":     r"Deterministic, $X\mathrm{CO}_2$",
    "det_profiles": "Deterministic, 20-level profile",
    "det_last4":    "Deterministic, near-surface",
    "nf_xco2":      r"Normalizing flow, $X\mathrm{CO}_2$",
    "nf_profiles":  "Normalizing flow, 20-level profile",
    "nf_last4":     "Normalizing flow, near-surface",
}


def _panel_label(model: str) -> str:
    """`nf_xco2_final_s90` -> `Normalizing flow, XCO2 (s90)`."""
    for study, pretty in STUDY_LABELS.items():
        if model.startswith(study + "_final") or model == study:
            rest = model[len(study) + len("_final"):].strip("_")
            return f"{pretty} ({rest})" if rest else pretty
    return model


def expand(patterns) -> list[Path]:
    """Expand shell wildcards ourselves.

    On Linux the shell has already done this and every argument is a real path. PowerShell
    does not glob the arguments of a native executable -- `python x.py logs/*.out` arrives
    as the literal string -- and neither does cmd.exe, so on Windows the script has to do
    it or the only user-visible symptom is "does not exist" on a file that plainly does.
    """
    out: list[Path] = []
    seen: set[str] = set()
    for pat in patterns:
        text = str(pat)
        if any(ch in text for ch in "*?["):
            hits = sorted(globmod.glob(text))
            if not hits:
                print(f"[warn] {text} matched nothing")
        else:
            hits = [text]
        for h in hits:
            if h not in seen:
                seen.add(h)
                out.append(Path(h))
    return out


def _model_name(path: Path) -> str:
    """A label for the panel, from the file name a CHTC job left behind."""
    stem = path.name
    for suffix in (".out", ".er", ".log", ".history.json", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem or path.parent.name


def parse_log(path: Path) -> pd.DataFrame:
    """Per-epoch rows from one trainer's stdout, whichever of the five wrote it."""
    # utf-8 explicitly: the trainers print a U+2248, and a Windows default of cp1252
    # would decode the log with a different codec than the one it was written in.
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: dict[int, dict] = {}
    kind = None
    for regex, name, metric in ((RE_NF, "flow", "NLL [std]"),
                                (RE_NF4, "flow", "NLL [std]"),
                                (RE_DET_XCO2, "deterministic", "RMSE [std]"),
                                (RE_DET_PROF, "deterministic", "RMSE [std]")):
        hits = list(regex.finditer(text))
        if not hits:
            continue
        kind = kind or name
        for m in hits:
            ep = int(m.group("epoch"))
            r = rows.setdefault(ep, {"epoch": ep, "kind": name, "metric": metric})
            if m.groupdict().get("train") is not None:
                r["train"] = float(m.group("train"))
            r["val"] = float(m.group("val"))
            if m.groupdict().get("val2") is not None:
                r["val_secondary"] = float(m.group("val2"))
        break                       # the first format that matched is the trainer's
    if kind == "deterministic":     # the training loss comes from its own line
        for m in RE_DET_PROF_TR.finditer(text):
            ep = int(m.group("epoch"))
            if ep in rows:
                rows[ep]["train"] = float(m.group("train"))
    if kind == "flow":              # and for nf_last4's published log, from the iterations
        best_it: dict[int, int] = {}
        for m in RE_NF4_IT.finditer(text):
            ep, it = int(m.group("epoch")), int(m.group("it"))
            if ep in rows and rows[ep].get("train") is None and it >= best_it.get(ep, -1):
                best_it[ep] = it
                rows[ep]["train"] = float(m.group("train"))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(sorted(rows.values(), key=lambda r: r["epoch"]))
    df.insert(0, "model", _model_name(path))
    df.insert(1, "source", str(path))
    return df


def parse_history(path: Path) -> pd.DataFrame:
    """Per-epoch rows from a trainer's history.json."""
    blob = json.loads(path.read_text())
    hist = blob.get("history") or []
    if not hist:
        return pd.DataFrame()
    out = []
    for rec in hist:
        train = rec.get("train_nll_std", rec.get("train_rmse_std", rec.get("train_loss")))
        val = rec.get("val_nll_std", rec.get("val_rmse_std", rec.get("rmse_std")))
        # Two extra series the extension runs record and the published ones do not.
        #
        # `lr` says whether a curve that flattens did so because the model converged or
        # because the schedule ran out of learning rate -- the two look identical on a loss
        # axis and are opposite conclusions.
        #
        # `val_rmse_std` is the flows' per-epoch validation RMSE, on when eval.rmse_samples
        # > 0. The flows select their checkpoint on NLL, and RMSE is the metric the paper
        # has to beat ACOS on; if the two curves bottom out at different epochs then the
        # selected checkpoint is not the best point predictor the run produced, and that is
        # a fact worth knowing before it is reported as one.
        second = rec.get("val_rmse_std") if "val_nll_std" in rec else None
        out.append({"epoch": int(rec["epoch"]),
                    "kind": "flow" if blob.get("study") == "nf" else "deterministic",
                    "metric": blob.get("metric", ""),
                    "train": None if train is None else float(train),
                    "val": None if val is None else float(val),
                    "val_secondary": None if second is None else float(second),
                    "lr": None if rec.get("lr") is None else float(rec["lr"])})
    df = pd.DataFrame(out).sort_values("epoch")
    df.insert(0, "model", _model_name(path))
    df.insert(1, "source", str(path))
    return df


def gaps(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model: the selected epoch, its two losses, and the gap between them.

    `selected` is the epoch the checkpoint was taken at -- the minimum of the validation
    curve, which is what all five trainers do. `gap` is validation minus training there:
    in the flows' units it is a difference of log densities per sounding, in the
    deterministic models' a difference of standardized RMSE.
    """
    out = []
    for model, sub in df.groupby("model", sort=False):
        sub = sub.dropna(subset=["val"])
        if sub.empty:
            continue
        k = int(sub["val"].astype(float).idxmin())
        row = df.loc[k]
        last = sub.iloc[-1]
        tr = row.get("train")
        out.append(dict(
            model=model, label=_panel_label(model),
            kind=row.get("kind"), metric=row.get("metric"),
            # A count, not the largest label: train.py numbers epochs from 0 and the flow
            # trainers from 1, so the same 40 epochs would otherwise print as 39 and 40.
            epochs=int(len(sub)), selected_epoch=int(row["epoch"]),
            val_at_selected=float(row["val"]),
            train_at_selected=None if pd.isna(tr) else float(tr),
            gap=None if pd.isna(tr) else float(row["val"]) - float(tr),
            val_at_last=float(last["val"]),
            # How much the validation curve gives back after its minimum. A model that
            # over-fits in the classical sense has a positive number here; one that is
            # still improving at the last epoch has zero and is under-trained instead.
            val_rise_after_selected=float(last["val"]) - float(row["val"]),
            # Under-training, made checkable rather than eyeballed: True when the
            # validation minimum is in the last tenth of the budget, i.e. the run stopped
            # before the curve did. Two of the six published models score True.
            still_improving=bool(int(row["epoch"]) >= sub["epoch"].max()
                                 - max(1, int(0.1 * len(sub)))),
            **_secondary(sub, row),
        ))
    return pd.DataFrame(out)


def _secondary(sub: pd.DataFrame, row: pd.Series) -> dict:
    """Where the SECOND validation curve bottoms out, when a run recorded one.

    For the flows that is validation RMSE against validation NLL, and the question it
    answers is whether the checkpoint the trainer kept -- chosen on NLL -- is also the best
    point predictor it saw. A large gap between the two epochs means the number the paper
    reports as our RMSE is not the best RMSE this configuration reached, which is a
    reporting decision rather than a modelling one and has to be made deliberately.
    """
    if "val_secondary" not in sub or not sub["val_secondary"].notna().any():
        return dict(secondary_metric=None, secondary_selected_epoch=None,
                    secondary_at_its_best=None, secondary_at_selected=None)
    v = sub.dropna(subset=["val_secondary"])
    k = int(v["val_secondary"].astype(float).idxmin())
    at_sel = sub.loc[sub["epoch"] == row["epoch"], "val_secondary"]
    return dict(
        secondary_metric="val RMSE [std]",
        secondary_selected_epoch=int(sub.loc[k, "epoch"]),
        secondary_at_its_best=float(sub.loc[k, "val_secondary"]),
        secondary_at_selected=(None if at_sel.empty or pd.isna(at_sel.iloc[0])
                               else float(at_sel.iloc[0])),
    )


def figure(df: pd.DataFrame, out_png: Path) -> None:
    models = list(dict.fromkeys(df["model"]))
    n = len(models)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.0 * nrow), squeeze=False)
    for ax, model in zip(axes.ravel(), models):
        sub = df[df["model"] == model].sort_values("epoch")
        ep = sub["epoch"].to_numpy()
        if "train" in sub and sub["train"].notna().any():
            ax.plot(ep, sub["train"].to_numpy(dtype=float), lw=2, label="training")
        ax.plot(ep, sub["val"].to_numpy(dtype=float), lw=2, label="validation")
        v = sub.dropna(subset=["val"])
        if not v.empty:
            k = v["val"].astype(float).idxmin()
            sel = float(df.loc[k, "epoch"])
            ax.axvline(sel, color="0.4", ls="--", lw=1.2,
                       label=f"checkpoint (epoch {sel:.0f})")
        # The second validation curve, when there is one, on its own axis: validation RMSE
        # against validation NLL is a comparison of two different quantities, and putting
        # them on one axis would invite reading a crossing that does not exist. What IS
        # meant to be read is where each one bottoms out.
        if "val_secondary" in sub and sub["val_secondary"].notna().any():
            w = sub.dropna(subset=["val_secondary"])
            ax2 = ax.twinx()
            ax2.plot(w["epoch"].to_numpy(), w["val_secondary"].to_numpy(dtype=float),
                     lw=1.6, ls=":", color="tab:green", label="val RMSE [std]")
            kk = w["val_secondary"].astype(float).idxmin()
            ax2.axvline(float(df.loc[kk, "epoch"]), color="tab:green", ls=":", lw=1.2)
            ax2.set_ylabel("val RMSE [std]", color="tab:green", fontsize=9)
            ax2.tick_params(axis="y", labelcolor="tab:green", labelsize=8)
            ax2.legend(frameon=False, loc="center right", fontsize=8)
        # The learning rate, when it was recorded. A flat tail with the rate still high is
        # a converged model; a flat tail because the rate went to zero is a budget that ran
        # out. The loss axis cannot tell them apart and this can.
        elif "lr" in sub and sub["lr"].notna().any():
            w = sub.dropna(subset=["lr"])
            ax2 = ax.twinx()
            ax2.plot(w["epoch"].to_numpy(), w["lr"].to_numpy(dtype=float),
                     lw=1.2, ls=":", color="0.55")
            ax2.set_ylabel("learning rate", color="0.45", fontsize=9)
            ax2.tick_params(axis="y", labelcolor="0.45", labelsize=8)
            ax2.set_yscale("log")
        ax.set_title(_panel_label(model), fontsize=12)
        ax.set_xlabel("epoch")
        ax.set_ylabel(sub["metric"].iloc[0] if "metric" in sub else "loss")
        ax.legend(frameon=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+",
                    help="trainer stdout (.out/.err/.log) and/or history .json files; "
                         "shell wildcards are expanded here too, for PowerShell")
    ap.add_argument("--out", type=Path, default=Path("experiments/scoring/tables"))
    ap.add_argument("--fig", type=Path, default=Path("latex/figures"))
    args = ap.parse_args()

    frames = []
    for p in expand(args.paths):
        if not p.exists():
            print(f"[warn] {p} does not exist")
            continue
        df = parse_history(p) if p.suffix == ".json" else parse_log(p)
        if df.empty:
            print(f"[warn] no per-epoch lines recognised in {p}")
            continue
        frames.append(df)
    if not frames:
        raise SystemExit("nothing to plot: no file carried a recognised per-epoch record.")

    df = pd.concat(frames, ignore_index=True)
    df.insert(1, "label", [_panel_label(m) for m in df["model"]])
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "learning_curves.csv", index=False)
    g = gaps(df)
    g.to_csv(args.out / "learning_curve_gaps.csv", index=False)
    figure(df, args.fig / "learning_curves.png")

    print(g.to_string(index=False))
    print(f"\n-> {args.out / 'learning_curves.csv'}")
    print(f"-> {args.out / 'learning_curve_gaps.csv'}")
    print(f"-> {args.fig / 'learning_curves.png'}")


if __name__ == "__main__":
    main()

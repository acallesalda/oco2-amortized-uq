#!/usr/bin/env python3
"""The temporal-generalization table: one float carrying March and July, both tasks.

The time-disjoint test is nine rows -- three evaluations (reference-disjoint February,
March 2020, July 2020) times three methods -- scored on two tasks. The numbers live in
CSVs *inside* the four month archives rather than in the checkout, and each
archive carries `experiments/scoring/tables/table_{xco2,last4}.csv` at exactly the path
`rebuild.sh` writes February's to, so unpacking one overwrites Tables 1-3 with a
single-month version and re-running `rebuild.sh` silently deletes the months again.

This script ends that. It scores the six artifacts itself, with `score.py`'s scorer, and
writes the table into a directory of its own:

    python3 experiments/scoring/make_temporal_table.py --out experiments/scoring/tables

    temporal/temporal_results.csv    every column score.py computes, plus `experiment`
                                     and the Feb -> month RMSE ratio
    temporal/temporal_results.tex    the float, \\input-able, caption included
    temporal/temporal_paired.csv     the paired method differences on all three
                                     evaluations, not February alone
    temporal/decomposition.tex       the companion float, built from
                                     decompose_error.py's summary rather than pasted

Two things it does that a hand-assembled table cannot.

*It gets ACOS's July near-surface cell.* That cell was missing at first: the month
archives were scored before `metrics.physical_mask` existed, so one failed retrieval
(`clst0002_ref0011_rep0010`, NaN at level 20) made `np.mean` NaN and blanked the control
cell. The current scorer screens and counts it, so the cell is 14.18 ppm and the caption
says what was dropped.

*It refuses to print numbers a known defect voided.* The XCO2 Laplace predictive carried
no observation-noise term in artifacts produced before the `laplace_mode="mc"` fix, and
`lap_cov_std` is then bit-identical to `lap_cov_epi_std` -- a check this makes directly on
each artifact rather than inferring from a variance ratio. While it holds, the Laplace's
XCO2 NLL and coverage cells print as `--` with a comment naming the fix, because those two
cells look exactly like results.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
for p in (str(_ROOT / "code"), str(_ROOT), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evals.artifacts import Artifact                      # noqa: E402
from score import rows_for, paired, METHOD_ORDER          # noqa: E402

# The three evaluations, in the order the argument is made: the model of Section 4 on the
# reference soundings it never saw, then the all-February model on two months it never saw.
FEB = "February (reference-disjoint)"
MONTH_LABEL = {"202003_full": "March 2020", "202007_full": "July 2020",
               "202003": "March 2020", "202007": "July 2020",
               # The extension run (ext.dag) writes its month artifacts under
               # results/month_<month>_ext/, so the tag it is addressed by carries the
               # suffix. Same two months, different February behind them.
               "202003_ext": "March 2020", "202007_ext": "July 2020",
               "202003_extcap": "March 2020", "202007_extcap": "July 2020"}
TASK_LABEL = {"xco2": r"$X\mathrm{CO}_2$", "last4": "Near-surface (levels 17--20)"}
# The caption used to `\ref` the reference-disjoint subsection. It no longer does: the float
# lives inside that subsection in the manuscript, so the reference pointed at the section the
# reader was already in, and in a standalone document that does not define the label it
# compiles to `??`. Both captions now name the experiment in words.
SHORT = {"ACOS L2FP (v10)": "ACOS L2FP", "Laplace": "Laplace",
         "Normalizing flow": "Normalizing flow"}


# `rescore_submit.sh` takes either splits or months, never both in one invocation: a month
# is a different experiment rather than another partition of this one, and it errors out
# instead of guessing. So the hint is built from the artifacts that are actually void.
SPLIT_OF = {"xco2_bo.npz": "group", "xco2_borep.npz": "replicate", "xco2.npz": "published"}


def rescore_commands(void_paths) -> list:
    """The exact command lines that would clear the cells named in `void_paths`."""
    months, splits = [], []
    for path in void_paths:
        path = Path(path)
        tag = path.parent.name                       # month_202003_full, or the task dir
        if tag.startswith("month_"):
            m = tag.split("_")[1]
            if m not in months:
                months.append(m)
        else:
            sp = SPLIT_OF.get(path.name)
            if sp and sp not in splits:
                splits.append(sp)
    out = []
    if months:
        out.append("./chtc/rescore_submit.sh xco2 "
                   + " ".join(f"--month {m}" for m in sorted(months)))
    for sp in splits:
        out.append(f"./chtc/rescore_submit.sh xco2 --split {sp}")
    return out


def epistemic_only(path: Path) -> bool:
    """Is this artifact's Laplace predictive the Note-14 defective one?

    `predict.py` adds the calibrated observation-noise diagonal to the epistemic
    covariance when it assembles `lap_cov_std`. Under the old `marglik` path there was no
    such term and the two arrays are bit-identical. That is an exact test on the artifact,
    which is what we want: a variance-ratio heuristic would also fire on a method that is
    merely badly calibrated, and would not fire at all on a task where the missing term is
    small.
    """
    with np.load(path, allow_pickle=False) as d:
        if "lap_cov_std" not in d.files or "lap_cov_epi_std" not in d.files:
            return False
        return bool(np.array_equal(np.asarray(d["lap_cov_std"], dtype=np.float64),
                                   np.asarray(d["lap_cov_epi_std"], dtype=np.float64)))


def score_one(path: Path, task: str, experiment: str, seed: int = 0,
              with_paired: bool = True):
    """The nine-row table's rows for one artifact, and the paired method differences.

    The paired half is a different question from the temporal one, and it is here because
    it needs the same artifacts: whether the flow "performs worse than the Laplace
    approximation", which is a question about a difference, and the only place that
    difference has ever been given a standard error is February. Two of the three
    orderings change sign on an unseen month, so a February-only paired test answers half
    the question.
    """
    art = Artifact.load(path)
    df = rows_for(art, seed=seed, bootstrap="scene")
    df.insert(0, "experiment", experiment)
    df["task_label"] = TASK_LABEL.get(task, task)
    df["artifact"] = str(path)
    df["lap_epistemic_only"] = epistemic_only(path)
    df["dim"] = int(art.meta.dim)
    if not with_paired:
        return df, None
    p = paired(art, seed=seed)
    p.insert(0, "experiment", experiment)
    p["task"] = task
    return df, p


# ---------------------------------------------------------------------------------
def _bf(s: str, on: bool) -> str:
    return rf"\textbf{{{s}}}" if on else s


def _cell_rmse(r, best=False) -> str:
    if not np.isfinite(r.rmse_ppm):
        return "--"
    se = r.rmse_ppm_se if np.isfinite(r.rmse_ppm_se) else float("nan")
    return (_bf(f"{r.rmse_ppm:.2f}", best)
            + (f" ({se:.2f})" if np.isfinite(se) else ""))


def _cell_ratio(r) -> str:
    """Each method's error on this month over its own February value.

    The column exists because the raw RMSE cannot answer the question on its
    own: the months are intrinsically harder, and the only way to see that is to read a
    learned method's ratio against the ratio of a retrieval that does not train.
    """
    v = getattr(r, "rmse_over_feb", np.nan)
    return "--" if not np.isfinite(v) else f"{v:.2f}"


def _cell_nll(r, void: bool, best=False) -> str:
    if void or not np.isfinite(r.nll_ppm):
        return "--"
    return _bf(f"{r.nll_ppm:.2f}", best)


def _cell_cov(r, void: bool) -> str:
    """Coverage at nominal 95 %. Joint / marginal where they differ, one number where
    they cannot (the scalar task has d = 1, so the two estimate the same quantity)."""
    if void:
        return "--"
    marg, joint = r.cov95_marginal, r.cov95_joint
    if not np.isfinite(marg):
        return "--"
    if int(r.dim) == 1 or not np.isfinite(joint):
        return f"{100 * marg:.1f}"
    return f"{100 * joint:.1f} / {100 * marg:.1f}"


def _thousands(x: int) -> str:
    return format(int(x), ",").replace(",", "{,}")


def _block_tag(sub: pd.DataFrame, tasks) -> str:
    """`$n=44{,}985$`, and how many soundings the point-error columns had to drop."""
    ns, dropped = [], 0
    for t in tasks:
        r = sub[sub.task == t]
        if r.empty:
            continue
        ns.append(int(r["n"].iloc[0]))
        if "n_nonphysical" in r.columns:
            dropped = max(dropped, int(np.nan_to_num(r["n_nonphysical"].max())))
    if not ns:
        return ""
    tag = "$n=" + " / ".join(_thousands(n) for n in sorted(set(ns))) + "$"
    if dropped:
        tag += (f" ({dropped} screened from the point-error columns)" if dropped > 1
                else " (one sounding screened from the point-error columns)")
    return tag


def _void_phrase(voided, experiments, tasks) -> str:
    """`March 2020 and July 2020 $X\\mathrm{CO}_2$` -- the withheld rows, in table order.

    Ordered by `experiments` rather than by sorting the strings, so the sentence reads in
    the order the reader meets the rows, and the task is named once when every withheld
    cell is the same task, which is the only case that has occurred.
    """
    ts = sorted({t for _, t in voided}, key=lambda t: tasks.index(t))
    parts = []
    for t in ts:
        exps = [e for e in experiments if (e, t) in set(voided)]
        label = r"$X\mathrm{CO}_2$" if t == "xco2" else t.replace("_", " ")
        joined = exps[0] if len(exps) == 1 else \
            " and ".join([", ".join(exps[:-1]), exps[-1]]) if len(exps) > 2 else \
            " and ".join(exps)
        parts.append(f"{joined} {label}")
    return " and ".join(parts)


def to_latex(df: pd.DataFrame, experiments, void_note: list,
             fix_commands=None, baseline_shown: bool = True) -> str:
    """One float. Rows are evaluation x method; column groups are the two tasks.

    This shape, rather than a column per month: the question is whether the
    ordering of the three methods survives a change of month, and that is read down a
    column. It also makes the float row-compatible with `decomposition.tex`, so the two
    tables of the temporal subsection have the same nine rows in the same order.
    """
    tasks = [t for t in ("xco2", "last4") if t in set(df["task"])]
    ncol = 1 + 4 * len(tasks)
    voided = []
    L = [
        "% generated by experiments/scoring/make_temporal_table.py -- do not edit by hand",
        "% RMSE and NLL in ppm; parentheses are bootstrap standard errors resampling whole",
        "% reference soundings. Coverage is at nominal 95 %, printed joint / marginal where",
        "% the target is multivariate.",
    ]
    if void_note:
        L += ["%",
              "% !! NOT REPORTABLE -- these cells are a defect, not a result:"]
        L += [f"%      {s}" for s in void_note]
        L += ["%    Fix: ./chtc/make_code_tarball.sh, then"]
        L += [f"%      {c}" for c in (fix_commands or
                                      ["./chtc/rescore_submit.sh xco2"])]
        L += ["%    and re-run this script. Nothing else in the table moves.",
              "%"]
    L += [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{l" + "rrrr" * len(tasks) + "}",
        r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{TASK_LABEL[t]}}}" for t in tasks)
        + r" \\",
        " ".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(tasks))),
        "Method & " + " & ".join([r"RMSE & vs Feb & NLL & Cov$_{95}$"] * len(tasks))
        + r" \\",
        r"\midrule",
    ]
    for exp in experiments:
        sub = df[df["experiment"] == exp]
        if sub.empty:
            continue
        L.append(rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{{exp}}}, {_block_tag(sub, tasks)}}} \\")
        # Best RMSE and best NLL within this block and task, as in Tables 1-3. Coverage is
        # not marked: neither direction of departure from nominal is better, and the void
        # cells would make "best" mean "the only one printed".
        best = {}
        for t in tasks:
            s = sub[sub.task == t]
            for col in ("rmse_ppm", "nll_ppm"):
                live = s[[not (bool(x.lap_epistemic_only) and x.method == "Laplace"
                               and col == "nll_ppm") for x in s.itertuples()]]
                live = live[np.isfinite(live[col])]
                best[(t, col)] = (live.loc[live[col].idxmin(), "method"]
                                  if len(live) else None)
        for m in METHOD_ORDER:
            cells = []
            for t in tasks:
                r = sub[(sub.task == t) & (sub.method == m)]
                if r.empty:
                    cells += ["--", "--", "--", "--"]
                    continue
                r = r.iloc[0]
                void = bool(r.lap_epistemic_only) and m == "Laplace"
                if void:
                    # Collected so the caption can say why a cell is a dash. A printed
                    # table whose dashes are explained only in a comment the reader never
                    # sees is worse than no table: a dash reads as "not measured".
                    voided.append((exp, t))
                cells += [_cell_rmse(r, best.get((t, "rmse_ppm")) == m),
                          _cell_ratio(r),
                          _cell_nll(r, void, best.get((t, "nll_ppm")) == m),
                          _cell_cov(r, void)]
            L.append(rf"\quad {SHORT.get(m, m)} & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule" if exp != experiments[-1] else r"\bottomrule")
    L += [
        r"\end{tabular}",
        r"\caption{\textbf{Temporal generalization.} $X\mathrm{CO}_2$ and near-surface "
        r"retrieval on two months of the same simulation ensemble that the network never "
        r"saw, with no retraining of any kind"
        # With the baseline hidden the caption has to carry what its rows used to show:
        # what the ratio is a ratio to, and that February's model is not this one.
        # No \ref to the reference-disjoint subsection: this float now lives inside it, so
        # the reference would point at the section the reader is already in, and in the
        # letter's variant the label does not exist at all.
        + (r". The February block is the reference-sounding-disjoint model described above, "
           r"on the $45$ reference soundings held out of "
           r"its training; the March and July blocks are the all-February model on every "
           r"sounding of the month. " if baseline_shown else
           r": the network weights, the target standardization, the covariate encodings and "
           r"the spectral channel selection are February's. The corresponding February "
           r"results are those of Table~\ref{tab:splitcompare} and are not repeated here. "
           # No \ref to a manuscript label here: --hide-baseline is the variant
           # meant for a standalone document, in which every label it does not
           # define itself compiles to `??`.
           r"They come from the model of the reference-sounding-disjoint experiment, "
           r"trained on "
           r"$361$ reference soundings and evaluated on the $45$ held out of it, whereas the "
           r"rows below are the all-February model, trained on $406$, evaluated on every "
           r"sounding of the month: the comparison spans a change of training set as well as "
           r"of month. ")
        + r"ACOS L2FP does not train and is the same retrieval in every "
        r"block and in the February comparison, so its rows measure the difficulty of the "
        r"month rather than any model's "
        r"ability to generalize, and every learned ratio should be read against it. The "
        r"\emph{vs Feb} column is each method's RMSE on that month divided by its own "
        r"February value. RMSE "
        r"and NLL in ppm; parentheses are bootstrap standard errors resampling whole "
        r"reference soundings. Coverage is the empirical frequency at a nominal $95\%$ "
        r"level, joint~/~marginal for the four-dimensional target and a single number for "
        r"the column average. Bold marks the best RMSE and the best NLL within a block and "
        r"task; coverage is not marked, since neither direction of departure from nominal "
        r"is better. One July sounding carries a non-physical retrieved state and "
        r"is excluded from the point-error columns of all three methods."
        # Only when there is a dash to explain, and it names the rows rather than
        # gesturing at them: the reader has to be able to tell a withheld cell from an
        # unmeasured one without reading the source.
        + ("" if not voided else
           " A dash marks a cell that is withheld rather than unmeasured: for the "
           + _void_phrase(voided, experiments, tasks)
           + " rows the Laplace predictive was computed by a code path that omitted the "
             "observation-noise term, so its likelihood and coverage are not comparable "
             "with the others. The point estimates, which that defect does not affect, "
             "are reported.")
        + r"}",
        r"\label{tab:months}",
        r"\end{table*}",
    ]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------
DECOMP_ORDER = ["ACOS", "Laplace", "flow"]
DECOMP_NAME = {"ACOS": "ACOS L2FP", "Laplace": "Laplace", "flow": "Normalizing flow"}
# decompose_error.py names its experiments for the console; the float wants the same words
# the table above uses, so that a reader moving between the two sees one set of names.
DECOMP_EXP = {"group split (Feb)": FEB, "March": "March 2020", "July": "July 2020"}


def decomposition_latex(S: pd.DataFrame, task: str = "near surface") -> str:
    """The companion float: the exact split of each method's error, all three evaluations.

    Built from `decompose_error.py`'s summary CSV so that the two tables cannot disagree,
    which has happened once already: the flow's July value was quoted in ACOS's row.
    """
    D = S[(S.task == task) & (S.level == "pooled")]
    if D.empty:
        return ""
    order = [e for e in D.experiment.unique()]
    L = [
        "% generated by experiments/scoring/make_temporal_table.py from",
        "% experiments/scoring/tables/decomposition_summary.csv -- do not edit by hand",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r" & & RMSE & scene & within & bound & within \\",
        r" & & & placement & sounding & & / bound \\",
        r"\midrule",
    ]
    for i, exp in enumerate(order):
        sub = D[D.experiment == exp]
        n = _thousands(sub["n"].iloc[0])
        L.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{DECOMP_EXP.get(exp, exp)}}}, "
                 rf"$n={n}$}} \\")
        for m in DECOMP_ORDER:
            r = sub[sub.method == m]
            if r.empty:
                continue
            r = r.iloc[0]
            # Three decimals on the last column, not two: the finding is that the flow's
            # ratio is 1.001, 1.005, 1.007 -- an equality to within one per cent -- and
            # rounding it to 1.00 / 1.01 / 1.01 throws away exactly that.
            L.append(f"& {DECOMP_NAME[m]:16s} & {r.rmse:5.2f} & {r.placement:4.2f} & "
                     f"{r.within:5.2f} & {r.floor:5.2f} & {r.within_over_floor:.3f} \\\\")
        L.append(r"\midrule" if i < len(order) - 1 else r"\bottomrule")
    what = ("Near-surface error (levels $17$--$20$, ppm)" if task == "near surface"
            else r"Column-average ($X\mathrm{CO}_2$) error (ppm)")
    label = "tab:decomp" if task == "near surface" else "tab:decomp_xco2"
    L += [
        r"\end{tabular}",
        rf"\caption{{{what} split into the part that is "
        r"constant within a reference sounding and the part that varies within it; the two "
        r"add in quadrature to the root mean squared error. The bound is the root mean "
        r"squared error of an estimator that knows each reference sounding's mean true "
        r"state exactly and nothing else about the individual sounding, computed from each "
        r"evaluation set's own truth. A ratio of unity in the last column means the method "
        r"extracts no information about the individual sounding at these levels. Rows and "
        r"blocks match Table~\ref{tab:months}. ACOS L2FP does not train, so its rows "
        r"measure the difficulty of the population.}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="experiments/scoring/tables",
                    help="parent directory; everything is written to <out>/temporal/")
    ap.add_argument("--xco2-group", default="experiments/scoring/xco2_bo.npz")
    ap.add_argument("--last4-group", default="experiments/scoring/last4_bo.npz")
    ap.add_argument("--month-root", default="experiments/scoring/results")
    # For the extension run: --months 202003_ext 202007_ext, with --xco2-group and
    # --last4-group pointed at another split's artifacts and --out at a
    # directory of its own. The February row then comes from the extension models' own
    # held-out test partition, which is group_split_v1's, so the three rows stay one
    # experiment. See docs/hpc.md.
    ap.add_argument("--months", nargs="*", default=["202003_full", "202007_full"])
    ap.add_argument("--decomposition",
                    default="experiments/scoring/tables/decomposition_summary.csv",
                    help="decompose_error.py's output, for the companion float")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hide-baseline", action="store_true",
                    help="print only the two unseen months, into "
                         "temporal_results_months.tex, for a document that already carries "
                         "February in a table of its own; the vs-Feb ratio is "
                         "kept, and the caption then has to say what it is a ratio to -- "
                         "which the full table got for free by showing the rows")
    ap.add_argument("--no-paired", action="store_true",
                    help="skip the paired method differences; they are the slow "
                         "part, 500 scene-bootstrap replicates per artifact")
    args = ap.parse_args(argv)

    out = Path(args.out) / "temporal"
    out.mkdir(parents=True, exist_ok=True)

    cases = [("xco2", FEB, Path(args.xco2_group)),
             ("last4", FEB, Path(args.last4_group))]
    for tag in args.months:
        root = Path(args.month_root) / f"month_{tag}"
        label = MONTH_LABEL.get(tag, tag)
        cases += [("xco2", label, root / "xco2.npz"),
                  ("last4", label, root / "last4.npz")]

    frames, pairs = [], []
    for task, exp, path in cases:
        if not path.exists():
            print(f"!! missing, skipped: {path}")
            continue
        print(f"== {task:6s} {exp:28s} {path}")
        d, p = score_one(path, task, exp, seed=args.seed,
                         with_paired=not args.no_paired)
        frames.append(d)
        if p is not None:
            pairs.append(p)
    if not frames:
        print("nothing to score.")
        return 1
    df = pd.concat(frames, ignore_index=True)

    # Feb -> month ratio, per method and task: the quantity that separates "the month is
    # harder" from "the model did not transfer", and the reason the ACOS row is in the
    # table at all.
    base = (df[df.experiment == FEB].set_index(["task", "method"])["rmse_ppm"]
            if (df.experiment == FEB).any() else None)
    df["rmse_over_feb"] = [
        r.rmse_ppm / base.get((r.task, r.method), np.nan) if base is not None else np.nan
        for r in df.itertuples()]

    experiments = [e for e in [FEB] + [MONTH_LABEL.get(t, t) for t in args.months]
                   if e in set(df["experiment"])]
    void = df[(df.method == "Laplace") & df.lap_epistemic_only & (df.task == "xco2")]
    void_note = [f"{r.experiment} / Laplace / XCO2: lap_cov_std is bit-identical to "
                 f"lap_cov_epi_std, i.e. no observation-noise term "
                 f"({Path(r.artifact).name})" for r in void.itertuples()]

    cols = ["experiment", "task", "task_label", "method", "n", "n_point", "n_nonphysical",
            "rmse_ppm", "rmse_ppm_se", "rmse_over_feb", "mae_ppm", "mae_ppm_se",
            "mae_ppm_median", "nll_ppm", "nll_std", "nll_se", "cov95_joint",
            "cov95_marginal", "cov68_joint", "cov68_marginal", "var_ratio", "dim",
            "lap_epistemic_only", "artifact"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(out / "temporal_results.csv", index=False)

    show = ["experiment", "task", "method", "n", "rmse_ppm", "rmse_over_feb", "nll_ppm",
            "cov95_joint", "cov95_marginal"]
    print("\n### the temporal table\n")
    print(df[show].round(4).to_string(index=False))

    fix = rescore_commands(void["artifact"]) if len(void) else []
    # The letter drops February -- Table 1 is two pages away there and repeating it reads as
    # new results -- while the manuscript keeps it, because its February rows are eight pages
    # back and in different units and the vs-Feb column is unreadable without them.
    shown = [e for e in experiments if not (args.hide_baseline and e == FEB)]
    tex_stem = "temporal_results_months" if args.hide_baseline else "temporal_results"
    (out / f"{tex_stem}.tex").write_text(
        to_latex(df, shown, void_note, fix, baseline_shown=not args.hide_baseline),
        encoding="utf-8")

    if pairs:
        P = pd.concat(pairs, ignore_index=True)
        P.to_csv(out / "temporal_paired.csv", index=False)
        LvF = P[(P.a == "Laplace") & (P.b == "Normalizing flow")]
        print("\n### paired differences, Laplace minus flow "
              "(positive = the Laplace is worse)\n")
        print(LvF[["experiment", "task", "n", "d_rmse_ppm", "d_rmse_se", "d_rmse_z",
                   "d_nll_std", "d_nll_se", "d_nll_z", "p_a_worse_rmse"]]
              .round(4).to_string(index=False))

    dec = Path(args.decomposition)
    if dec.exists():
        S = pd.read_csv(dec)
        for task, stem in (("near surface", "decomposition"),
                           ("XCO2", "decomposition_xco2")):
            tex = decomposition_latex(S, task=task)
            if tex:
                (out / f"{stem}.tex").write_text(tex, encoding="utf-8")
        print(f"\nwrote the companion floats from {dec}")
    else:
        print(f"\n!! {dec} absent: run decompose_error.py for the companion float.")

    if void_note:
        print("\n!! the XCO2 Laplace NLL and coverage cells below are the Note-14 defect")
        print("   and print as '--'. To clear them, from the access point:")
        for c in fix:
            print(f"     {c}")
        if len(fix) > 1:
            print("   -- separately: rescore_submit.sh takes splits or months, not both.")
    print(f"\nwrote {out}/temporal_results.csv and {out}/{tex_stem}.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

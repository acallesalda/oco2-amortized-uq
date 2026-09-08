#!/usr/bin/env python3
"""score.py -- the CPU half of scoring.

Turns the artifacts written by `predict.py` into Tables 1-3: a machine-readable CSV, a
LaTeX body ready to paste, and a per-level breakdown for the figures and the appendix.
No GPU, no checkpoints, no data -- just the `.npz`, so this runs on a laptop in seconds
and can be re-run as often as the wording changes.

    python3 experiments/scoring/score.py results/group_split_v1/*.npz --out tables/

Two artifacts of the same task can be given together to get the side-by-side the revision
needs -- the reference-sounding-disjoint result next to the within-reference one:

    python3 experiments/scoring/score.py \
        results/group_split_v1/profiles.npz results/replicate/profiles.npz \
        --label group_split_v1=group-disjoint --label replicate=within-reference \
        --out tables/

Every column is reported in both standardized and ppm units. The manuscript currently
reports RMSE in ppm and NLL in standardized coordinates, which makes the NLL columns of
Tables 1, 2 and 3 mutually incomparable; printing both makes that
impossible to repeat by accident.
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
from evals.artifacts import Artifact, scene_labels  # noqa: E402

METHOD_ORDER = ["ACOS L2FP (v10)", "Laplace", "Normalizing flow"]


def rows_for(art: Artifact, seed: int = 0, bootstrap: str = "scene") -> pd.DataFrame:
    """One row per method for a single artifact.

    `bootstrap` chooses what the parenthesised standard errors resample. "scene" draws
    whole reference soundings with replacement, which is the honest unit: the test
    partition is 4,489 soundings but only 45 independent scenes, and every metric here is
    strongly correlated among the ~100 replicates of one scene. "sounding" reproduces the
    narrower errors the published tables carry. Falls back to "sounding" if the artifact
    has no parseable IDs, and the choice is recorded in the `bootstrap` column either way.
    """
    y = np.asarray(art["y_std"], dtype=np.float64)
    y_mu, y_sigma = art.y_mu, art.y_sigma
    groups = scene_labels(art) if bootstrap == "scene" else None
    if bootstrap == "scene" and groups is None:
        print(f"note: {art.meta.task}/{art.meta.split_name} has no parseable IDs; "
              f"falling back to the sounding bootstrap.")
    rows = []

    if art.has("acos_mu_std") and art.has("acos_cov_std"):
        valid = art.get("acos_valid")
        valid = None if valid is None else np.asarray(valid, dtype=bool)
        rows.append(M.summarize_gaussian(
            y, np.asarray(art["acos_mu_std"], dtype=np.float64),
            np.asarray(art["acos_cov_std"], dtype=np.float64),
            y_mu, y_sigma, "ACOS L2FP (v10)", valid=valid, seed=seed, groups=groups,
        ))

    if art.has("lap_mu_std") and art.has("lap_cov_std"):
        rows.append(M.summarize_gaussian(
            y, np.asarray(art["lap_mu_std"], dtype=np.float64),
            np.asarray(art["lap_cov_std"], dtype=np.float64),
            y_mu, y_sigma, "Laplace", seed=seed, groups=groups,
        ))

    if art.has("nf_samples_std"):
        rows.append(M.summarize_samples(
            y, np.asarray(art["nf_samples_std"], dtype=np.float64), y_mu, y_sigma,
            "Normalizing flow",
            logprob_std=art.get("nf_logprob_std"),
            latent_std=art.get("nf_latent_std"),
            seed=seed, groups=groups,
        ))

    df = pd.DataFrame(rows)
    df.insert(0, "task", art.meta.task)
    df.insert(1, "split", art.meta.split_name)
    return df


def per_level(art: Artifact) -> pd.DataFrame:
    """Per-pressure-level RMSE and 95% marginal coverage.

    The scalar task has one level and this is redundant; for the 20-level and
    near-surface tasks it is what Figures 4-6 are drawn from, and what shows that the
    aggregate coverage in the tables hides a strong vertical structure.
    """
    y = np.asarray(art["y_std"], dtype=np.float64)
    y_mu, y_sigma = art.y_mu, art.y_sigma
    levels = art.meta.levels or [0]
    out = []

    def add(name, mu, cov, valid=None):
        # Same rule as the summary table: coverage is measured only where the predictive
        # actually has a covariance, so a missing ACOS file does not read as a miss.
        v = np.ones(len(y), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        var = np.diagonal(cov[v], axis1=-2, axis2=-1)
        rmse = M.rmse_ppm_per_level(y, mu, y_mu, y_sigma)
        mae = M.mae_ppm_per_level(y, mu, y_mu, y_sigma)
        cov95 = M.gaussian_marginal_coverage(y[v], mu[v], var, 0.95)
        cov68 = M.gaussian_marginal_coverage(y[v], mu[v], var, 0.68)
        for j, lv in enumerate(levels):
            out.append(dict(task=art.meta.task, split=art.meta.split_name, method=name,
                            level=lv, rmse_ppm=rmse[j], mae_ppm=mae[j], n_cov=int(v.sum()),
                            cov95_marginal=cov95[j], cov68_marginal=cov68[j]))

    if art.has("acos_mu_std"):
        add("ACOS L2FP (v10)",
            np.asarray(art["acos_mu_std"], dtype=np.float64),
            np.asarray(art["acos_cov_std"], dtype=np.float64),
            valid=art.get("acos_valid"))
    if art.has("lap_mu_std"):
        add("Laplace",
            np.asarray(art["lap_mu_std"], dtype=np.float64),
            np.asarray(art["lap_cov_std"], dtype=np.float64))
    if art.has("nf_samples_std"):
        s = np.asarray(art["nf_samples_std"], dtype=np.float64)
        mu = s.mean(axis=1)
        # The per-level MAE of the flow is taken at the predictive median, which is the
        # point estimate that minimizes it -- the same choice the summary table's
        # `mae_ppm_median` column records. `rmse_ppm` stays at the mean for the same
        # reason: each column uses the estimator its own loss is minimized by.
        med = np.median(s, axis=1)
        rmse = M.rmse_ppm_per_level(y, mu, y_mu, y_sigma)
        mae = M.mae_ppm_per_level(y, med, y_mu, y_sigma)
        c95 = M.sample_marginal_coverage(y, s, 0.95)
        c68 = M.sample_marginal_coverage(y, s, 0.68)
        for j, lv in enumerate(levels):
            out.append(dict(task=art.meta.task, split=art.meta.split_name,
                            method="Normalizing flow", level=lv, rmse_ppm=rmse[j],
                            mae_ppm=mae[j], n_cov=len(y),
                            cov95_marginal=c95[j], cov68_marginal=c68[j]))
    return pd.DataFrame(out)


def paired(art: Artifact, seed: int = 0, n_boot: int = 500) -> pd.DataFrame:
    """Paired differences between methods, with a scene bootstrap of the difference.

    Comparing two methods by their separate standard errors is the wrong test and it is
    conservative in the wrong direction here: the two predictives are evaluated on the same
    soundings, so their errors are strongly correlated and the standard error of the
    difference is much smaller than either method's own. The open question is whether the flow "performs
    worse" than the Laplace approximation, and that is a question about a difference, so it
    is the difference that needs a standard error.

    Every bootstrap replicate resamples whole reference soundings once and scores both
    methods on the same resample, which is what makes it paired.
    """
    y = np.asarray(art["y_std"], dtype=np.float64)
    y = y[:, None] if y.ndim == 1 else y
    y_mu, y_sigma = art.y_mu, art.y_sigma
    groups = scene_labels(art)
    ppm = np.asarray(y_sigma, dtype=np.float64) * float(art.meta.ppm_scale)

    per: dict = {}
    # The physical screen of `metrics.physical_mask` is intersected across methods rather
    # than applied per method: a paired difference is only paired if both terms are the
    # same soundings, and dropping one sounding from one method's row would silently make
    # this a comparison of two different test sets.
    fin = np.ones(len(y), dtype=bool)
    if art.has("acos_mu_std") and art.has("acos_cov_std"):
        mu = np.asarray(art["acos_mu_std"], dtype=np.float64)
        cov = np.asarray(art["acos_cov_std"], dtype=np.float64)
        per["ACOS L2FP (v10)"] = (np.mean(((y - mu) * ppm) ** 2, axis=1),
                                  M.mvn_nll(y, mu, cov))
        fin &= M.physical_mask(y, mu, y_mu, y_sigma)
    if art.has("lap_mu_std") and art.has("lap_cov_std"):
        mu = np.asarray(art["lap_mu_std"], dtype=np.float64)
        cov = np.asarray(art["lap_cov_std"], dtype=np.float64)
        per["Laplace"] = (np.mean(((y - mu) * ppm) ** 2, axis=1), M.mvn_nll(y, mu, cov))
        fin &= M.physical_mask(y, mu, y_mu, y_sigma)
    if art.has("nf_samples_std"):
        s = np.asarray(art["nf_samples_std"], dtype=np.float64)
        nll = (-np.asarray(art["nf_logprob_std"], dtype=np.float64)
               if art.has("nf_logprob_std") else M.kde_joint_nll(y, s))
        per["Normalizing flow"] = (np.mean(((y - s.mean(axis=1)) * ppm) ** 2, axis=1), nll)
        fin &= M.physical_mask(y, s.mean(axis=1), y_mu, y_sigma)

    n_dropped = int((~fin).sum())
    per = {k: (mse[fin], nll[fin]) for k, (mse, nll) in per.items()}
    groups = None if groups is None else np.asarray(groups)[fin]
    y = y[fin]

    names = [m for m in METHOD_ORDER if m in per]
    reps = list(M._boot_replicates(len(y), groups, n_boot, seed))
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            mse_a, nll_a = per[a]
            mse_b, nll_b = per[b]
            ok = np.isfinite(nll_a) & np.isfinite(nll_b)
            d_rmse = float(np.sqrt(mse_a.mean()) - np.sqrt(mse_b.mean()))
            d_nll = float(np.mean(nll_a[ok]) - np.mean(nll_b[ok]))
            br = np.array([np.sqrt(mse_a[r].mean()) - np.sqrt(mse_b[r].mean())
                           for r in reps])
            bn = np.array([np.mean(nll_a[r][ok[r]]) - np.mean(nll_b[r][ok[r]])
                           for r in reps])
            rows.append(dict(
                task=art.meta.task, split=art.meta.split_name, a=a, b=b, n=int(ok.sum()),
                n_nonphysical=n_dropped,
                d_rmse_ppm=d_rmse, d_rmse_se=float(np.std(br, ddof=1)),
                d_rmse_z=d_rmse / float(np.std(br, ddof=1)),
                d_nll_std=d_nll, d_nll_se=float(np.std(bn, ddof=1)),
                d_nll_z=d_nll / float(np.std(bn, ddof=1)),
                # Share of resamples in which `a` has the larger RMSE: a sign test that
                # does not assume the bootstrap distribution is symmetric.
                p_a_worse_rmse=float(np.mean(br > 0)),
                p_a_worse_nll=float(np.mean(bn > 0)),
            ))
    return pd.DataFrame(rows)


def _num(row, key: str, default: float = float("nan")) -> float:
    """A column that may be absent, NaN or None, as a float."""
    v = row.get(key, None)
    if v is None:
        return float(default)
    v = float(v)
    return v if np.isfinite(v) else float(default)


def to_latex(df: pd.DataFrame, labels: dict) -> str:
    """The table body, in the manuscript's column order.

    Coverage is printed as joint / marginal rather than as the manuscript's single
    number, because for D>1 those are different quantities and only the first answers
    the question actually being asked.
    """
    lines = [
        "% generated by experiments/scoring/score.py -- do not edit by hand",
        "% columns: method & RMSE [ppm] & MAE [ppm] & NLL [ppm] & NLL [std] "
        "& coverage 95 (joint/marg) & coverage 68 (joint/marg)",
        "% MAE is at each method's own MAE-optimal point estimate: the mean for the two "
        "Gaussians, the predictive median for the flow (they coincide for a Gaussian).",
        "% Parenthesised numbers are bootstrap standard errors; the `bootstrap` column of "
        "the CSV says whether the resampling unit was the reference sounding or the "
        "individual sounding.",
    ]
    # A row whose predictive is grossly mis-scaled produces an NLL and a coverage that look
    # exactly like results, and this file is what gets pasted into the manuscript. So the
    # warning goes here, in the LaTeX, where it cannot be missed -- not only in a CSV column
    # nobody opens. It disappears by itself once the row is re-scored.
    #
    # ACOS is deliberately exempt. Its predictive variance is ~30x smaller than its own
    # squared error on every task, and that is not a defect in our code -- it is the
    # paper's central measurement, the consequence of an optimal-estimation posterior
    # carrying no forward-model-error term. Flagging it would bury the one row where a
    # large ratio is the finding. The ratio is written to the CSV for every method either
    # way, so nothing is hidden.
    ours = [r for _, r in df.iterrows() if "ACOS" not in str(r["method"])]
    bad = [r for r in ours
           if np.isfinite(_num(r, "var_ratio")) and not 0.25 <= _num(r, "var_ratio") <= 4.0]
    if bad:
        lines += ["%",
                  "% !! NOT REPORTABLE -- a predictive fitted here is grossly",
                  "%    inconsistent with its own residuals:"]
        for r in bad:
            lines.append(f"%      {r['split']} / {r['method']}: residual variance is "
                         f"{_num(r, 'var_ratio'):.1f}x the mean predictive variance")
        lines += ["%    Its NLL and coverage columns are not reportable: the usual cause is a",
                  "%    Laplace predictive fitted without its observation-noise term.",
                  "%    Re-score with predict.py --laplace-mode mc.",
                  "%"]
    # The comparator's ratio is the result, so it is stated rather than warned about.
    for _, r in df.iterrows():
        if "ACOS" in str(r["method"]) and np.isfinite(_num(r, "var_ratio")):
            lines.append(f"% {r['split']} / {r['method']}: residual variance is "
                         f"{_num(r, 'var_ratio'):.1f}x its reported predictive variance "
                         f"(this is a finding, not a defect)")

    for split, sub in df.groupby("split", sort=False):
        lines.append(f"% --- {labels.get(split, split)} ---")
        sub = sub.set_index("method").reindex(
            [m for m in METHOD_ORDER if m in set(sub["method"])]
        ).reset_index()
        for _, r in sub.iterrows():
            def pct(x):
                return "--" if not np.isfinite(x) else f"{100 * x:.1f}"
            # A row that came from summarize_gaussian has no median columns at all;
            # concat fills them with NaN, and an older artifact scored before the MAE
            # existed has neither. Both fall back to the mean-based number.
            mae = _num(r, "mae_ppm_median", _num(r, "mae_ppm"))
            mae_se = _num(r, "mae_ppm_median_se", _num(r, "mae_ppm_se"))
            lines.append(
                f"{r['method']} & "
                f"{r['rmse_ppm']:.2f} ({r['rmse_ppm_se']:.2f}) & "
                f"{mae:.2f} ({mae_se:.2f}) & "
                f"{r['nll_ppm']:.2f} ({r['nll_se']:.2f}) & "
                f"{r['nll_std']:.2f} & "
                f"{pct(r['cov95_joint'])} / {pct(r['cov95_marginal'])} & "
                f"{pct(r['cov68_joint'])} / {pct(r['cov68_marginal'])} \\\\"
            )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("artifacts", nargs="+", help=".npz files written by predict.py")
    ap.add_argument("--out", default="tables", help="output directory")
    ap.add_argument("--label", action="append", default=[],
                    metavar="SPLIT=TEXT", help="pretty name for a split, repeatable")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed for the SEs")
    ap.add_argument("--bootstrap", default="scene", choices=["scene", "sounding"],
                    help="what the standard errors resample: whole reference soundings "
                         "(default, and the right unit under a group-disjoint split) or "
                         "individual soundings (what the published tables reported)")
    ap.add_argument("--paired", action="store_true",
                    help="also write paired method-vs-method differences with a scene "
                         "bootstrap of the difference")
    args = ap.parse_args(argv)

    labels = dict(kv.split("=", 1) for kv in args.label)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables, levels, pairs = {}, {}, {}
    for path in args.artifacts:
        art = Artifact.load(path)
        print(f"\n=== {path}\n{art.summary()}")
        if art.meta.notes.get("partial"):
            print("!! this artifact was written with --limit and is a smoke test, "
                  "not a result.")
        tables.setdefault(art.meta.task, []).append(
            rows_for(art, seed=args.seed, bootstrap=args.bootstrap))
        levels.setdefault(art.meta.task, []).append(per_level(art))
        if args.paired:
            pairs.setdefault(art.meta.task, []).append(paired(art, seed=args.seed))

    for task, parts in tables.items():
        df = pd.concat(parts, ignore_index=True)
        csv = out_dir / f"table_{task}.csv"
        df.to_csv(csv, index=False)
        tex = out_dir / f"table_{task}.tex"
        tex.write_text(to_latex(df, labels), encoding="utf-8")

        lv = pd.concat(levels[task], ignore_index=True)
        lv.to_csv(out_dir / f"levels_{task}.csv", index=False)

        print(f"\n=== {task} " + "=" * (60 - len(task)))
        with pd.option_context("display.width", 200, "display.max_columns", 40):
            print(df.drop(columns=["task"]).to_string(index=False))
        print(f"\n{tex.read_text()}")
        print(f"\nwrote {csv}, {tex}, {out_dir / f'levels_{task}.csv'}")

        if args.paired and task in pairs:
            pd_df = pd.concat(pairs[task], ignore_index=True)
            pcsv = out_dir / f"paired_{task}.csv"
            pd_df.to_csv(pcsv, index=False)
            print(f"\n--- paired differences ({task}), a minus b, scene bootstrap")
            with pd.option_context("display.width", 200, "display.max_columns", 40):
                print(pd_df.drop(columns=["task"]).to_string(index=False))
            print(f"wrote {pcsv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

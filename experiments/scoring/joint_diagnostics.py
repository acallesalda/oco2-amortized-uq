#!/usr/bin/env python3
"""joint_diagnostics.py -- joint against marginal calibration, in one table.

Two facts about the metrics in Tables 1-3 constrain how they may be read:

  * the NLLs of the scalar, near-surface and 20-level tasks are reported in standardized
    target coordinates, so they are not on a common physical scale and must not be read
    against one another;
  * the profile coverage is *marginal* coverage averaged over levels, which says nothing
    about whether the joint D-dimensional predictive is calibrated.

`score.py` already answers the first: every table it writes carries `nll_std` and
`nll_ppm`, and the conversion is the constant Jacobian `sum(log(sigma_d * 1e6))`. It also
already carries `cov95_joint` next to `cov95_marginal`. What is missing is the diagnostic
that turns "the joint number differs from the marginal one" into a statement about *why*,
in a single file the appendix can be written from.

This script is that file. For every method on every artifact it maps the truth into the
space where a correctly specified predictive would give standard normal residuals,

    Gaussian predictors (ACOS, Laplace):  z = L^-1 (x - mu),  Sigma = L L^T
    Normalizing flow:                     z = f_x(x),         the flow's own base map

and reports what that space says:

    ||z||^2 mean        should be D.  Above D: the joint predictive is too narrow.
    chi2-PIT KS         KS distance of F_{chi2_D}(||z||^2) from Uniform(0,1). One number
                        for joint calibration; 0 is perfect.
    joint coverage      fraction inside the joint HDR at 68 / 95, i.e. the chi2 ball.
    marginal coverage   the average over levels the manuscript reports, for contrast.
    max per-dim KS      the worst single coordinate, so a joint failure can be attributed
                        to one level rather than to the correlation structure.
    corr(z) off-diag    whitening removes the predicted dependence, so `corr(z)` is the
                        identity under a correct predictive. What is left in it is the
                        between-level dependence the predictive covariance does not carry
                        -- the one thing marginal coverage cannot see.

The z-space diagnostics are invariant to the target units -- standardizing rescales mu and
Sigma consistently, and the flow's latent map is defined on the standardized target -- so
unlike NLL they *are* comparable across the three tasks. That is the honest way to handle
comparability: report NLL in both coordinate systems and stop comparing it
across tasks, and use these for the cross-task statement.

Standard errors resample whole reference soundings, as everywhere else in this project:
the reference-disjoint partition is 4,489 soundings but only 45 independent scenes.

    python3 experiments/scoring/joint_diagnostics.py \
        experiments/scoring/{xco2,profiles,last4}_bo.npz \
        experiments/scoring/results/replicate_split_v1/{xco2,profiles,last4}_borep.npz \
        --label group_split_v1_cov='reference-disjoint' \
        --label replicate_split_v1_cov='within-reference' \
        --out experiments/scoring/tables

Writes `joint_diagnostics.csv` and `joint_diagnostics.tex`. No GPU, no checkpoints, no
data -- only the `.npz` that `predict.py` wrote, so it runs on a laptop in under a minute.

Author(s): Alejandro Calle-Saldarriaga
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
for p in (str(_ROOT / "code"), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evals import metrics as M                      # noqa: E402
from evals.artifacts import Artifact, scene_labels  # noqa: E402

METHOD_ORDER = ["ACOS L2FP (v10)", "Laplace", "Normalizing flow"]
LEVELS = (0.68, 0.95)


# ------------------------------------------------------------------------------------
# the predictives, exactly as figures.py builds them
# ------------------------------------------------------------------------------------
def predictives(art: Artifact) -> dict:
    """`{method -> dict(z, var, mu, valid, nll)}`.

    Deliberately the same construction as `figures.py:predictives` -- same whitening,
    same fallbacks, same `valid` mask -- so this table and the joint panel of Figures
    3, 7 and 9 cannot disagree about what was computed.
    """
    y = np.asarray(art["y_std"], dtype=np.float64)
    n = len(y)
    out = {}

    if art.has("acos_mu_std") and art.has("acos_cov_std"):
        mu = np.asarray(art["acos_mu_std"], dtype=np.float64)
        cov = np.asarray(art["acos_cov_std"], dtype=np.float64)
        valid = art.get("acos_valid")
        valid = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        out["ACOS L2FP (v10)"] = dict(
            valid=valid, z=M.whiten(y[valid], mu[valid], cov[valid]),
            var=np.diagonal(cov[valid], axis1=-2, axis2=-1),
            y=y[valid], mu=mu[valid], nll=M.mvn_nll(y[valid], mu[valid], cov[valid]),
        )

    if art.has("lap_mu_std") and art.has("lap_cov_std"):
        mu = np.asarray(art["lap_mu_std"], dtype=np.float64)
        cov = np.asarray(art["lap_cov_std"], dtype=np.float64)
        out["Laplace"] = dict(
            valid=np.ones(n, dtype=bool), z=M.whiten(y, mu, cov),
            var=np.diagonal(cov, axis1=-2, axis2=-1),
            y=y, mu=mu, nll=M.mvn_nll(y, mu, cov),
        )

    if art.has("nf_samples_std"):
        s = np.asarray(art["nf_samples_std"], dtype=np.float64)
        # The flow's own latent map is the exact analogue of Cholesky whitening and needs
        # no Gaussian assumption. It is absent only for the marginalized artifacts, where
        # the flow's joint density is over the wrong dimension anyway -- there the sample
        # moments are whitened instead, and the row is marked so the two are never read as
        # the same quantity.
        if art.has("nf_latent_std"):
            z, kind = np.asarray(art["nf_latent_std"], dtype=np.float64), "flow latent"
        else:
            z, kind = M.whiten(y, s.mean(axis=1), M.sample_mean_cov(s)[1]), "sample moments"
        out["Normalizing flow"] = dict(
            valid=np.ones(n, dtype=bool), z=z, var=s.var(axis=1, ddof=1),
            y=y, mu=s.mean(axis=1), z_kind=kind,
            nll=(-np.asarray(art["nf_logprob_std"], dtype=np.float64)
                 if art.has("nf_logprob_std") else M.kde_joint_nll(y, s)),
            samples=s,
        )
    return y, out


# ------------------------------------------------------------------------------------
# the statistics
# ------------------------------------------------------------------------------------
def _chi2_pit(z: np.ndarray) -> np.ndarray:
    """`F_{chi2_D}(||z||^2)` over the rows where whitening succeeded."""
    ok = np.all(np.isfinite(z), axis=1)
    return stats.chi2.cdf(np.sum(z[ok] ** 2, axis=1), df=z.shape[1]), ok


def _ks_uniform(u: np.ndarray) -> float:
    return float(stats.kstest(u, "uniform").statistic)


def _boot(values_fn, n: int, groups, n_boot: int, seed: int) -> float:
    """Standard error of a statistic under the reference-sounding bootstrap.

    `values_fn(idx)` recomputes the statistic on a resampled index set. Resampling whole
    scenes rather than soundings is not a refinement here: the ~100 replicates of one
    reference sounding share a geometry and a surface, and a KS statistic computed as if
    they were 100 independent draws has an error bar several times too small.
    """
    if n_boot <= 0:
        return float("nan")
    vals = []
    for idx in M._boot_replicates(n, groups, n_boot, seed):
        v = values_fn(idx)
        if np.isfinite(v):
            vals.append(v)
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")


def _residual_dependence(zf: np.ndarray) -> dict:
    """What is left in `corr(z)` after whitening: identity under a correct predictive.

    A predictive can have every marginal exactly right and still be jointly wrong, and
    that is not a hypothetical -- it is what the 20-level Laplace does. The three numbers
    here separate the two failure modes: per-coordinate scale (`z_sd`, `max_per_dim_ks`)
    against between-level dependence (these). The reference for the largest eigenvalue is
    the Marchenko-Pastur edge `(1 + sqrt(D/n))^2`, which is what an identity correlation
    estimated from `n` finite samples gives -- otherwise D=20 always looks guilty.
    """
    n, d = zf.shape
    if d < 2:
        return {"z_corr_offdiag_absmean": float("nan"),
                "z_corr_offdiag_max": float("nan"),
                "z_corr_eigmax": float("nan"),
                "z_corr_eigmax_null": float("nan")}
    c = np.corrcoef(zf, rowvar=False)
    off = c[~np.eye(d, dtype=bool)]
    return {
        "z_corr_offdiag_absmean": float(np.mean(np.abs(off))),
        "z_corr_offdiag_max": float(off[np.argmax(np.abs(off))]),
        "z_corr_eigmax": float(np.linalg.eigvalsh(c)[-1]),
        "z_corr_eigmax_null": float((1.0 + np.sqrt(d / n)) ** 2),
    }


def row_for(art: Artifact, name: str, p: dict, groups, n_boot: int, seed: int) -> dict:
    z = np.asarray(p["z"], dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    d = z.shape[1]
    pit, ok = _chi2_pit(z)
    zf = z[ok]
    m2 = np.sum(zf ** 2, axis=1)
    g = None if groups is None else np.asarray(groups)[p["valid"]][ok]

    y_v, mu_v, var_v = p["y"][ok], p["mu"][ok], p["var"][ok]
    # Marginal coverage of the flow is read off its samples, not off a Gaussian with its
    # sample variance -- the flow's intervals are the empirical quantiles, and for a skewed
    # predictive the two differ by several points.
    if "samples" in p:
        s_ok = p["samples"][ok]
        marg = {lv: float(np.mean(M.sample_marginal_coverage(y_v, s_ok, lv))) for lv in LEVELS}
    else:
        marg = {lv: float(np.mean(M.gaussian_marginal_coverage(y_v, mu_v, var_v, lv)))
                for lv in LEVELS}

    row = {
        "task": art.meta.task,
        "split": art.meta.split_name,
        "method": name,
        "dim": d,
        "n": int(ok.sum()),
        "n_dropped": int((~ok).sum()),
        "z_kind": p.get("z_kind", "cholesky"),
        # Under calibration: mean 0, sd 1 pooled over coordinates, ||z||^2 mean D.
        "z_mean": float(zf.mean()),
        "z_sd": float(zf.std(ddof=1)),
        "mahalanobis_mean": float(m2.mean()),
        "mahalanobis_expected": float(d),
        "mahalanobis_ratio": float(m2.mean() / d),
        # The mean of ||z||^2 is a second moment of a heavy-tailed quantity: one sounding
        # whose predictive is 30x too narrow moves it by more than the other 4,488 do. The
        # median against the chi2_D median is the same statement about the bulk, and where
        # the two disagree -- ACOS on every task -- the disagreement is itself the finding.
        "mahalanobis_median": float(np.median(m2)),
        "mahalanobis_median_ratio": float(np.median(m2) / stats.chi2.ppf(0.5, df=d)),
        "chi2_pit_ks": _ks_uniform(pit),
        "chi2_pit_ks_se": _boot(lambda i: _ks_uniform(pit[i]), len(pit), g, n_boot, seed),
        "max_per_dim_ks": float(max(stats.kstest(zf[:, j], "norm").statistic
                                    for j in range(d))),
        "var_ratio": M._var_ratio(y_v, mu_v, var_v),
        "nll_std": float(np.nanmean(p["nll"])),
        "nll_ppm": float(M.nll_std_to_ppm(np.nanmean(p["nll"]), art.y_sigma)),
        "nll_ppm_per_dim": float(M.nll_std_to_ppm(np.nanmean(p["nll"]), art.y_sigma)) / d,
    }
    row.update(_residual_dependence(zf))
    for lv in LEVELS:
        tag = f"{int(round(lv * 100))}"
        c = stats.chi2.ppf(lv, df=d)
        hit = (m2 <= c).astype(np.float64)
        row[f"cov{tag}_joint"] = float(hit.mean())
        row[f"cov{tag}_joint_se"] = _boot(lambda i: float(hit[i].mean()),
                                          len(hit), g, n_boot, seed)
        row[f"cov{tag}_marginal"] = marg[lv]
        row[f"cov{tag}_gap"] = marg[lv] - float(hit.mean())
    return row


# ------------------------------------------------------------------------------------
# output
# ------------------------------------------------------------------------------------
def to_latex(df: pd.DataFrame, labels: dict) -> str:
    """The appendix table: one block per task, joint against marginal, with the KS."""
    lines = [
        "% generated by experiments/scoring/joint_diagnostics.py -- do not edit by hand",
        "% columns: method & D & ||z||^2/D & chi2-PIT KS & coverage 68 (joint/marg)",
        "%          & coverage 95 (joint/marg) & NLL [std] & NLL [ppm] & NLL [ppm]/D",
        "% z = L^-1(x-mu) for the Gaussians, the flow's own latent map for the flow; both",
        "% are N(0,I_D) under a correctly calibrated joint predictive, and both are",
        "% invariant to the coordinate system of the target.",
        "% Parenthesised numbers are bootstrap standard errors resampling reference soundings.",
    ]
    ours = [r for _, r in df.iterrows() if "ACOS" not in str(r["method"])]
    bad = [r for r in ours if np.isfinite(r["var_ratio"])
           and not 0.25 <= r["var_ratio"] <= 4.0]
    if bad:
        lines += ["%",
                  "% !! NOT REPORTABLE -- a predictive fitted here is grossly",
                  "%    inconsistent with its own residuals:"]
        for r in bad:
            lines.append(f"%      {r['split']} / {r['task']} / {r['method']}: residual "
                         f"variance is {r['var_ratio']:.1f}x the mean predictive variance")
        lines += ["%    Its whole row is not reportable: the usual cause is a Laplace",
                  "%    predictive fitted without its observation-noise term. Re-score",
                  "%    with predict.py --laplace-mode mc.", "%"]

    lines += [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
              r"Partition & Method & $D$ & $\overline{\|z\|^2}/D$ & KS$_{\chi^2}$ "
              r"& \multicolumn{2}{c}{cov.\ 68 (j/m)} & \multicolumn{2}{c}{cov.\ 95 (j/m)} "
              r"& NLL$_{\rm ppm}$/D \\", r"\midrule"]
    for task, sub_t in df.groupby("task", sort=False):
        lines.append(rf"\multicolumn{{10}}{{l}}{{\emph{{{task}}}}} \\")
        for split, sub in sub_t.groupby("split", sort=False):
            sub = sub.set_index("method").reindex(
                [m for m in METHOD_ORDER if m in set(sub["method"])]).reset_index()
            for _, r in sub.iterrows():
                lines.append(
                    f"{labels.get(split, split)} & {r['method']} & {int(r['dim'])} & "
                    f"{r['mahalanobis_ratio']:.2f} & {r['chi2_pit_ks']:.3f} & "
                    f"{100 * r['cov68_joint']:.1f} & {100 * r['cov68_marginal']:.1f} & "
                    f"{100 * r['cov95_joint']:.1f} & {100 * r['cov95_marginal']:.1f} & "
                    f"{r['nll_ppm_per_dim']:.2f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def by_cluster(art: Artifact, name: str, p: dict, groups) -> list:
    """The same statistics within each geophysical template, plus the tail split.

    The split comparison already shows that the accuracy degradation on the reference-disjoint
    partition is concentrated in the scenes drawn from templates 8 and 9, the two that
    contribute fewest reference soundings to training. The joint diagnostics have to be
    read against that: a joint coverage of 58.7% is a different claim if it is uniform
    over the test partition than if it is nominal on six templates and collapses on two.
    So the rows here are per template, plus two pooled rows -- the tail and everything
    else -- computed on exactly the same z.
    """
    if groups is None:
        return []
    z = np.asarray(p["z"], dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    d = z.shape[1]
    ok = np.all(np.isfinite(z), axis=1)
    g = np.asarray(groups)[p["valid"]][ok]
    m2 = np.sum(z[ok] ** 2, axis=1)
    # "<cluster>/<reference>" -- the template is the part before the slash.
    clust = np.array([s.split("/")[0] for s in g])
    c95, c68 = stats.chi2.ppf(0.95, df=d), stats.chi2.ppf(0.68, df=d)

    def stats_for(mask, label):
        if mask.sum() < 2:
            return None
        mm = m2[mask]
        return {"task": art.meta.task, "split": art.meta.split_name, "method": name,
                "group": label, "dim": d, "n": int(mask.sum()),
                "n_scenes": int(len(np.unique(g[mask]))),
                "mahalanobis_ratio": float(mm.mean() / d),
                "chi2_pit_ks": _ks_uniform(stats.chi2.cdf(mm, df=d)),
                "cov95_joint": float(np.mean(mm <= c95)),
                "cov68_joint": float(np.mean(mm <= c68))}

    rows = [r for c in sorted(np.unique(clust), key=lambda s: int(s))
            if (r := stats_for(clust == c, f"template {c}")) is not None]
    tail = np.isin(clust, ["8", "9"])
    for mask, label in ((tail, "templates 8+9"), (~tail, "templates 1-7 (excl. 8,9)")):
        r = stats_for(mask, label)
        if r is not None:
            rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("artifacts", nargs="+")
    ap.add_argument("--out", default="experiments/scoring/tables")
    ap.add_argument("--label", action="append", default=[],
                    help="SPLIT=NAME, the printed name of a split")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--by-cluster", action="store_true",
                    help="also write the same statistics within each geophysical template")
    args = ap.parse_args()

    labels = {}
    for spec in args.label:
        if "=" not in spec:
            ap.error(f"--label takes SPLIT=NAME, got {spec!r}")
        k, v = spec.split("=", 1)
        labels[k] = v

    rows, cluster_rows = [], []
    for path in args.artifacts:
        art = Artifact.load(path)
        print(f"\n=== {path}\n{art.summary()}")
        groups = scene_labels(art)
        if groups is None:
            print("  note: no parseable IDs; standard errors fall back to the sounding "
                  "bootstrap, which is too narrow. Report them as such.")
        _, preds = predictives(art)
        for name in [m for m in METHOD_ORDER if m in preds]:
            r = row_for(art, name, preds[name], groups, args.n_boot, args.seed)
            rows.append(r)
            print(f"  [{name}] D={r['dim']}  ||z||^2/D={r['mahalanobis_ratio']:.2f}  "
                  f"KS_chi2={r['chi2_pit_ks']:.3f}({r['chi2_pit_ks_se']:.3f})  "
                  f"cov95 joint {100 * r['cov95_joint']:.1f} / marg "
                  f"{100 * r['cov95_marginal']:.1f}  "
                  f"NLL {r['nll_std']:.2f} std, {r['nll_ppm']:.2f} ppm, "
                  f"{r['nll_ppm_per_dim']:.2f} ppm/level")
            if args.by_cluster:
                cluster_rows += by_cluster(art, name, preds[name], groups)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "joint_diagnostics.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (out / "joint_diagnostics.tex").write_text(to_latex(df, labels), encoding="utf-8")
    print(f"\nwrote {out / 'joint_diagnostics.csv'}")
    print(f"wrote {out / 'joint_diagnostics.tex'}")
    if cluster_rows:
        p = out / "joint_diagnostics_by_cluster.csv"
        p.write_text(pd.DataFrame(cluster_rows).to_csv(index=False), encoding="utf-8")
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

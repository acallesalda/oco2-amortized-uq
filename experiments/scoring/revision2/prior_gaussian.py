#!/usr/bin/env python3
"""prior_gaussian.py -- the ACOS prior N(x_a, S_a) as a probabilistic retrieval (second round, R2.2).

Scores the operational a priori as a predictive distribution on the test samples of Tables 6-7,
for all three targets, through the same `metrics.summarize_gaussian` that produces every Gaussian
row of those tables (RMSE, MAE, NLL in standardized coordinates, joint chi^2 and level-averaged
marginal coverage, reference-sounding bootstrap SEs):

  XCO2          N(h'x_a, h'S_a h), with h the reference sounding's pressure-weighting function
  near-surface  N(x_a[17:20], S_a[17:20, 17:20])
  20-level      N(x_a, S_a)

The prior files (one HDF5 per month and cluster: `StateVector/state_vector_apriori`,
`xco2_apriori`, `xco2_pressure_weighting_function`, `apriori_covariance_matrix`, and
`SoundingMetadata/sounding_id`) hold one row per reference sounding, in reference-number order
within each cluster file; the prior is the same for every replicate of a reference sounding.
S_a is identical across soundings and months. As recommended by the data provider, the XCO2
prior is the file's `xco2_apriori`, which differs from the split tables' `XCO2Prior` by at most
0.02 ppm (slightly different pressure weighting).

The script first re-scores ACOS L2FP from each artifact through the same path, as a check that
the conventions match the published rows.

    python3 experiments/scoring/revision2/prior_gaussian.py

Needs the scoring artifacts and the prior files (`./scripts/get_data.sh artifacts apriori`); no
GPU. `--priors` defaults to scratch/simulated/apriori/, where get_data.sh unpacks the
OCO2_RefPrior_YYYYMM/ folders.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
from evals import metrics as M                              # noqa: E402
from evals.artifacts import Artifact, scene_labels          # noqa: E402

S = "experiments/scoring"
# (table, setting, task, artifact relative to the artifacts dir, prior month)
JOBS = [
    ("6", "Feb within-reference", "xco2", "results/replicate_split_v1/xco2_borep.npz", "202002"),
    ("6", "Feb within-reference", "last4", "results/replicate_split_v1/last4_borep.npz", "202002"),
    ("6", "Feb within-reference", "profiles", "results/replicate_split_v1/profiles_borep.npz", "202002"),
    ("6", "Feb reference-disjoint", "xco2", "xco2_bo.npz", "202002"),
    ("6", "Feb reference-disjoint", "last4", "last4_bo.npz", "202002"),
    ("6", "Feb reference-disjoint", "profiles", "profiles_bo.npz", "202002"),
    ("7", "March 2020", "xco2", "results/month_202003_full/xco2.npz", "202003"),
    ("7", "March 2020", "last4", "results/month_202003_full/last4.npz", "202003"),
    ("7", "July 2020", "xco2", "results/month_202007_full/xco2.npz", "202007"),
    ("7", "July 2020", "last4", "results/month_202007_full/last4.npz", "202007"),
]
# Table 7 drops one July sounding (non-physical ACOS state) from the point-error columns of
# every method, and keeps it in the NLL and coverage columns. The prior rows follow the same
# rule. IDs are labels within a month (the same label exists in March), so this is July only.
EXCLUDE = {"July 2020": {"clst0002_ref0011_rep0010"}}
POINT = ["rmse_ppm", "rmse_ppm_se", "mae_ppm", "mae_ppm_se"]
DIST = ["nll_std", "nll_se", "cov95_joint", "cov95_marginal"]


def load_priors(root: Path):
    """{(month, clst, ref): (x_a[20], xco2_apriori, h[20])} and the common S_a (20 x 20)."""
    table, sa = {}, None
    for month in ("202002", "202003", "202007"):
        for c in range(1, 10):
            p = root / f"OCO2_RefPrior_{month}" / f"lnd_nadir_{month}_clst{c:02d}_ref_prior_means.h5"
            with h5py.File(p, "r") as f:
                xa = f["StateVector/state_vector_apriori"][:, :20].astype(np.float64)
                xc = f["StateVector/xco2_apriori"][:].astype(np.float64)
                h = f["StateVector/xco2_pressure_weighting_function"][:].astype(np.float64)
                s = f["StateVector/apriori_covariance_matrix"][:20, :20].astype(np.float64)
            if sa is None:
                sa = s
            elif not np.array_equal(s, sa):
                raise ValueError(f"S_a differs in {p.name}")
            for i in range(len(xa)):
                table[(month, c, i + 1)] = (xa[i], xc[i], h[i])
    return table, sa


def prior_predictive(art, table, sa, month):
    ids = np.asarray(art["ids"]).astype(str)
    keys = [(month, int(i[4:8]), int(i[12:16])) for i in ids]          # clstNNNN_refNNNN_rep...
    xa = np.stack([table[k][0] for k in keys])
    xc = np.array([table[k][1] for k in keys])
    h = np.stack([table[k][2] for k in keys])
    y_mu = np.asarray(art.y_mu, dtype=np.float64)
    y_sig = np.asarray(art.y_sigma, dtype=np.float64)
    if art.meta.task == "xco2":
        mu = xc[:, None]
        var = np.einsum("ni,ij,nj->n", h, sa, h)[:, None, None]
        cov = var
    else:
        lv = np.asarray(art.meta.levels, dtype=int) - 1                   # 1-indexed levels
        mu = xa[:, lv]
        cov = np.broadcast_to(sa[np.ix_(lv, lv)], (len(ids), len(lv), len(lv)))
    mu_std = (mu - y_mu) / y_sig
    cov_std = cov / np.outer(y_sig, y_sig)
    return mu_std, np.array(cov_std)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--priors", default=str(ROOT / "scratch/simulated/apriori"))
    ap.add_argument("--artifacts", default=str(ROOT / S))
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    table, sa = load_priors(Path(args.priors))

    rows = []
    for tab, setting, task, rel, month in JOBS:
        art = Artifact.load(Path(args.artifacts) / rel)
        ids = np.asarray(art["ids"]).astype(str)
        y_all = np.asarray(art["y_std"], dtype=np.float64)
        g_all = scene_labels(art)
        g_all = None if g_all is None else np.asarray(g_all)
        mu, cov = prior_predictive(art, table, sa, month)
        valid = art.get("acos_valid")
        valid = None if valid is None else np.asarray(valid, bool)
        methods = (("ACOS L2FP (check)", np.asarray(art["acos_mu_std"], np.float64),
                    np.asarray(art["acos_cov_std"], np.float64), valid),
                   ("Prior", mu, cov, None))
        point_keep = ~np.isin(ids, list(EXCLUDE.get(setting, ())))
        for name, m, c, v in methods:
            rec = dict(table=tab, setting=setting, task=task, method=name)
            for cols, k in ((POINT, point_keep), (DIST, np.ones(len(ids), bool))):
                r = M.summarize_gaussian(y_all[k], m[k], c[k], art.y_mu, art.y_sigma, name,
                                         valid=None if v is None else v[k], seed=args.seed,
                                         groups=None if g_all is None else g_all[k])
                rec.update({col: r.get(col) for col in cols})
                rec.setdefault("n", r.get("n"))
            if name == "Prior":
                sd = np.sqrt(np.diagonal(c, axis1=-2, axis2=-1)) * np.asarray(art.y_sigma) * 1e6
                rec["prior_sd_ppm"] = float(sd.mean())
            rows.append(rec)

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))
    if args.out:
        out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()

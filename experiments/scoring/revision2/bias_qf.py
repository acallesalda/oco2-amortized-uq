#!/usr/bin/env python3
"""bias_qf.py -- mean error, RMSE with it removed, and quality-flag subsets for every XCO2 method (R2.2).

For the flow, the Laplace approximation, raw ACOS L2FP and the prior XCO2, on the test samples
of Tables 6-7, all soundings and those with a good operational quality flag
(`xco2_quality_flag == 0`):

  mean_error     mean of (estimate - true), ppm
  rmse           RMSE of the estimate, ppm (as in Tables 6-7)
  rmse_debiased  RMSE after subtracting the method's own mean error in that setting (the error
                 SD). The offset is estimated on the same soundings, so this is the smallest RMSE
                 any constant offset can give there. It is not the operational bias correction,
                 which regresses on retrieval features and can also remove scatter.

Point estimates are the predictive mean (the flow's from its 512 draws). Standard errors resample
reference soundings. The July sounding with a non-physical ACOS state is excluded, as in Table 7.

    python3 experiments/scoring/revision2/bias_qf.py

Needs the scoring artifacts and the split tables; no GPU.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SETS = [("Feb within-reference", "results/replicate_split_v1/xco2_borep.npz", "replicate_split_v1_cov.csv"),
        ("Feb reference-disjoint", "xco2_bo.npz", "group_split_v1_cov.csv"),
        ("March 2020", "results/month_202003_full/xco2.npz", "month_202003_full_split.csv"),
        ("July 2020", "results/month_202007_full/xco2.npz", "month_202007_full_split.csv")]
SV = [f"retrieved_state_vector_{i:02d}" for i in range(1, 21)]


def stats(err, groups, rng, n_boot):
    _, inv = np.unique(groups, return_inverse=True)
    G = inv.max() + 1
    n = np.bincount(inv, minlength=G)
    s1, s2 = np.bincount(inv, err, minlength=G), np.bincount(inv, err ** 2, minlength=G)
    idx = rng.integers(0, G, size=(n_boot, G))
    N = n[idx].sum(1)
    me_b, ms_b = s1[idx].sum(1) / N, s2[idx].sum(1) / N
    me, ms = err.mean(), np.mean(err ** 2)
    return dict(mean_error=me, mean_error_se=me_b.std(ddof=1),
                rmse=np.sqrt(ms), rmse_se=np.sqrt(ms_b).std(ddof=1),
                rmse_debiased=np.sqrt(ms - me ** 2),
                rmse_debiased_se=np.sqrt(ms_b - me_b ** 2).std(ddof=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--artifacts", default=str(ROOT / "experiments/scoring"))
    ap.add_argument("--splits-dir", default=str(ROOT / "scratch/simulated/files/splits"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = []
    for name, art, spl in SETS:
        z = np.load(Path(args.artifacts) / art)
        meta = json.loads(bytes(z["__meta__"]).decode())
        sig = float(np.ravel(meta["y_sigma"])[0]) * 1e6
        ids = z["ids"].astype(str)
        df = pd.read_csv(Path(args.splits_dir) / spl, low_memory=False,
                         usecols=["ID", "split", "xco2_quality_flag", "XCO2Prior", "true_xco2"] + SV)
        df = df[df.split == "test"].set_index("ID").reindex(ids)   # IDs are unique within test
        sv = df[SV].to_numpy(float)
        ok = ~((sv < 0).any(1) | np.isnan(sv).any(1))
        qf = df.xco2_quality_flag.to_numpy() == 0
        y = z["y_std"][:, 0].astype(np.float64)
        yppm = df.true_xco2.to_numpy() * 1e6
        g = np.array([i.rsplit("_rep", 1)[0] for i in ids])
        err = {"Flow": (z["nf_samples_std"][:, :, 0].astype(np.float64).mean(1) - y) * sig,
               "Laplace": (z["lap_mu_std"][:, 0] - y) * sig,
               "ACOS L2FP": (z["acos_mu_std"][:, 0] - y) * sig,
               "Prior": df.XCO2Prior.to_numpy() - yppm}
        for subset, m in (("all", ok), ("quality flag good", ok & qf)):
            for meth, e in err.items():
                rec = dict(test_set=name, subset=subset, method=meth, n=int(m.sum()))
                rec.update(stats(e[m], g[m], rng, args.n_boot))
                rows.append(rec)

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(out.round(2).to_string(index=False))
    if args.out:
        out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()

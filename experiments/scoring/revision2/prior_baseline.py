#!/usr/bin/env python3
"""prior_baseline.py -- the a priori XCO2 column as a retrieval (second-round revision, R2.2).

Scores `XCO2Prior` (the ACOS a priori column, one of the network's inputs) against the true
XCO2 on the test samples of Tables 6-7, with no fitting of any kind, next to the raw ACOS L2FP
estimate on the same rows. Standard errors resample reference soundings (cluster/ref), as
everywhere else in the paper.

Also reported, not used in the paper: `prior_recal`, the prior after a linear recalibration
(a + b * prior) fitted on the training rows of the same split. It is kept because the
bootstrap below draws from one generator in a fixed order, and the published standard
errors are reproduced exactly only with that order intact.

One July sounding (clst0002_ref0011_rep0010) has a non-physical ACOS L2FP state (a negative
retrieved profile level); Table 7 drops it from the point-error columns, and so does this
script, by the same rule (any negative or missing retrieved state-vector element). The rule
removes nothing from the other three test sets.

    python3 experiments/scoring/revision2/prior_baseline.py
    python3 experiments/scoring/revision2/prior_baseline.py --splits-dir <dir> --out prior.csv

Needs only the split tables (`./scripts/get_data.sh splits`); no GPU, no checkpoints.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SETS = [("Feb within-reference", "replicate_split_v1_cov.csv"),
        ("Feb reference-disjoint", "group_split_v1_cov.csv"),
        ("March 2020", "month_202003_full_split.csv"),
        ("July 2020", "month_202007_full_split.csv")]
SV = [f"retrieved_state_vector_{i:02d}" for i in range(1, 21)]
COLS = ["clst", "ref", "split", "true_xco2", "retrieved_xco2", "XCO2Prior"] + SV


def boot(err, groups, kind, rng, n_boot):
    """Point value and bootstrap SE of the RMSE or MAE, resampling whole reference soundings."""
    _, inv = np.unique(groups, return_inverse=True)
    G = inv.max() + 1
    n = np.bincount(inv, minlength=G)
    s = np.bincount(inv, err ** 2 if kind == "rmse" else np.abs(err), minlength=G)
    idx = rng.integers(0, G, size=(n_boot, G))
    draws = s[idx].sum(1) / n[idx].sum(1)
    if kind == "rmse":
        return float(np.sqrt(np.mean(err ** 2))), float(np.sqrt(draws).std(ddof=1))
    return float(np.mean(np.abs(err))), float(draws.std(ddof=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--splits-dir", default=str(ROOT / "scratch/simulated/files/splits"))
    ap.add_argument("--out", default=None, help="optional CSV of the results")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)       # one stream, in a fixed order (see above)
    rows = []
    for name, fname in SETS:
        df = pd.read_csv(Path(args.splits_dir) / fname, usecols=COLS, low_memory=False)
        tr, te = df[df.split == "train"], df[df.split == "test"]
        sv = te[SV].to_numpy(dtype=np.float64)
        keep = ~((sv < 0).any(axis=1) | np.isnan(sv).any(axis=1))
        te = te[keep]
        y = te.true_xco2.to_numpy() * 1e6                     # mol/mol -> ppm
        prior = te.XCO2Prior.to_numpy()                       # already in ppm
        groups = (te.clst.astype(str) + "/" + te.ref.astype(str)).to_numpy()
        b, a = np.polyfit(tr.XCO2Prior.to_numpy(), tr.true_xco2.to_numpy() * 1e6, 1)
        rec = dict(test_set=name, n=len(te), excluded=int((~keep).sum()),
                   reference_soundings=len(np.unique(groups)))
        for key, est in (("prior", prior), ("prior_recal", a + b * prior),
                         ("acos", te.retrieved_xco2.to_numpy() * 1e6)):
            for kind in ("rmse", "mae"):
                v, se = boot(est - y, groups, kind, rng, args.n_boot)
                rec[f"{key}_{kind}"], rec[f"{key}_{kind}_se"] = v, se
        rows.append(rec)

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(out.round(2).to_string(index=False))
    if args.out:
        out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()

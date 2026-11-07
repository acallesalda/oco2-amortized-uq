#!/usr/bin/env python3
"""mc_budget.py -- Table 8's shape gap against the Monte Carlo budget (second-round revision, R2.4).

The moment-matched control of Table 8 is the Gaussian whose mean and covariance are estimated
from S flow draws per sounding. A Gaussian fitted to S draws pays an estimation penalty that
shrinks as 1/S, so the shape gap (control NLL minus flow NLL) is overstated at small S. This
reports the gap at every budget available in the published artifacts:

  S = 256, 512   the draws stored in `<task>_bo.npz` (first 256, and all 512)
  S = 512 ... 8,192 (16,384 for XCO2)
                 the streamed controls from fresh draws in `<task>_gauss.npz`
                 (predict.py --gaussianize); Table 8 reports S = 512, 2,048 and 8,192
  S -> infinity  the a + b/S fit through the streamed stages (evals.gaussianize.extrapolate)

Standard errors resample the 45 held-out reference soundings, identified by cluster AND ref
(the `clstNNNN_refNNNN` prefix of the sounding ID): ref numbers restart in every cluster.

    python3 experiments/scoring/revision2/mc_budget.py
    python3 experiments/scoring/revision2/mc_budget.py --artifacts <dir with the .npz files>

Needs only the scoring artifacts (`./scripts/get_data.sh artifacts`); no GPU.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "code"), str(ROOT), str(ROOT / "experiments" / "scoring")]
from evals import metrics as M                         # noqa: E402
from evals.artifacts import Artifact                   # noqa: E402
import nongauss as NG                                  # noqa: E402


def gap_stats(nll_ctrl, nll_flow, refs):
    gap = nll_ctrl - nll_flow
    return np.nanmean(nll_ctrl), np.nanmean(gap), NG._boot_se_grouped(gap, refs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--artifacts", default=str(ROOT / "experiments" / "scoring"))
    args = ap.parse_args()
    art_dir = Path(args.artifacts)

    for task in ("xco2", "last4", "profiles"):
        bo = Artifact.load(art_dir / f"{task}_bo.npz")
        y = np.asarray(bo["y_std"], dtype=np.float64)
        nll_flow = -np.asarray(bo["nf_logprob_std"], dtype=np.float64)
        refs = np.array([i.rsplit("_rep", 1)[0] for i in np.asarray(bo["ids"]).astype(str)])
        s = np.asarray(bo["nf_samples_std"], dtype=np.float64)
        d = s.shape[2]
        print(f"== {task}  D={d}  flow NLL {nll_flow.mean():.3f}  "
              f"({np.unique(refs).size} reference soundings)")
        for S in (256, 512):
            mu, cov, _, _ = NG._moments(s[:, :S])
            c, g, se = gap_stats(M.mvn_nll(y, mu, cov), nll_flow, refs)
            print(f"  S={S:6d} (stored)  moment-matched NLL {c:8.3f}  gap {g:7.3f} ({se:.3f})")

        ga = Artifact.load(art_dir / f"{task}_gauss.npz")
        assert np.array_equal(np.asarray(ga["ids"]), np.asarray(bo["ids"]))
        stages = NG._stages(ga)
        st = np.asarray(ga["nfg_nll_std"], dtype=np.float64)
        nll_flow_g = -np.asarray(ga["nf_logprob_std"], dtype=np.float64)
        for k, S in enumerate(stages):
            c, g, se = gap_stats(st[:, k], nll_flow_g, refs)
            print(f"  S={S:6d} (fresh)   moment-matched NLL {c:8.3f}  gap {g:7.3f} ({se:.3f})")
        fit = NG.gz_extrapolate(stages, np.nanmean(st, axis=0))
        print(f"  S->inf (a + b/S)   moment-matched NLL {fit['a']:8.3f}  "
              f"gap {fit['a'] - nll_flow_g.mean():7.3f}")


if __name__ == "__main__":
    main()

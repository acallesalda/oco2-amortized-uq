"""compute_split_stats.py
Recompute target standardization statistics from the TRAINING rows of a split table.

Both the 20-d profile statistics and the scalar XCO2 statistics must come from training
soundings alone. Under the original replicate-level split this distinction was harmless
because every reference sounding was in the training pool anyway; under a
reference-sounding-disjoint split, reusing the old constants would leak information about
held-out soundings into preprocessing.

Writes, into --out-dir:
    profile_mu.npy      (20,)  mean of true_state_vector_01..20
    profile_sigma.npy   (20,)  std  of true_state_vector_01..20
    xco2_stats.json            mean/std of true_xco2, read by spectral_loader
    provenance.json            what was computed from what

Usage:
    python3 compute_split_stats.py --split-csv /workspace/project/scratch/simulated/files/splits/group_split_v1.csv
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_STATS_ROOT = Path("/workspace/project/experiments/pipelines/stats")

TRUE_COLS = [f"true_state_vector_{i:02d}" for i in range(1, 21)]
XCO2_COL = "true_xco2"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-csv", type=Path, required=True,
                   help="split table written by make_group_split.py")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="destination (default: stats/<split-csv stem>/)")
    p.add_argument("--split", type=str, default="train",
                   help="partition to estimate from (default: train)")
    args = p.parse_args()

    out_dir = args.out_dir or (DEFAULT_STATS_ROOT / args.split_csv.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.split_csv)
    if "split" not in df.columns:
        raise KeyError(f"{args.split_csv} has no 'split' column")
    train = df[df["split"] == args.split]
    if len(train) == 0:
        raise ValueError(f"no rows with split == '{args.split}' in {args.split_csv}")

    missing = [c for c in TRUE_COLS + [XCO2_COL] if c not in train.columns]
    if missing:
        raise KeyError(f"split table is missing target columns: {missing}")

    X = train[TRUE_COLS].astype("float64").to_numpy()
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0.0] = 1.0  # guard

    xco2 = train[XCO2_COL].astype("float64").to_numpy()
    xco2_mu = float(xco2.mean())
    xco2_sd = float(xco2.std(ddof=0)) or 1.0

    np.save(out_dir / "profile_mu.npy", mu)
    np.save(out_dir / "profile_sigma.npy", sd)
    (out_dir / "xco2_stats.json").write_text(
        json.dumps({"mu": xco2_mu, "sigma": xco2_sd}, indent=2)
    )

    n_groups = (
        train.drop_duplicates(subset=["clst", "ref"]).shape[0]
        if {"clst", "ref"}.issubset(train.columns) else None
    )
    (out_dir / "provenance.json").write_text(json.dumps({
        "split_csv": str(args.split_csv),
        "estimated_from_split": args.split,
        "n_rows": int(len(train)),
        "n_reference_soundings": n_groups,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))

    print(f"Estimated from {len(train)} rows"
          + (f" / {n_groups} reference soundings" if n_groups else "")
          + f" (split='{args.split}')")
    print(f"Wrote -> {out_dir}")
    print(f"  profile_mu[:3]    = {np.array2string(mu[:3], precision=10)}")
    print(f"  profile_sigma[:3] = {np.array2string(sd[:3], precision=10)}")
    print(f"  xco2 mu/sigma     = {xco2_mu:.10g} / {xco2_sd:.6g}")
    print("\nPass this directory as `stats_dir` (and the split CSV as `split_csv`) "
          "to the loaders / training configs.")


if __name__ == "__main__":
    main()

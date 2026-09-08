"""make_month_split.py
Build the split table for a temporal hold-out: February trains, another month is the test
partition.

The shape of the problem
------------------------
`predict.py` reads ONE table and takes three partitions from it. The test partition is the
new month; the train and validation partitions have to be February's, because that is what
the model was fitted on and because the Laplace approximation is *refitted* at scoring time
over training soundings -- point it at March and the last-layer covariance would be a
different object from the one the paper describes.

So this concatenates:

    February's group split, rows with split in {train, val}   ->  unchanged
    the new month, every sounding                             ->  split = "test"

and nothing else. February's own test partition is dropped: those rows are `split == test`
already and would silently join the new month in the test set, which would make the
resulting number a mixture of two experiments.

Standardization and covariate encoding
--------------------------------------
Neither is recomputed here, and that is the point of the exercise. `predict.py --stats-dir`
must be pointed at February's `stats/group_split_v1/`, so the targets are standardized with
February's constants and the covariates are encoded -- centred, scaled, level-mapped,
median-imputed -- with February's. A March sounding is therefore presented to the network
in exactly the coordinates it was trained in, which is what makes the result a measure of
transfer rather than of re-normalization.

Two things that follow, and are reported rather than fixed:

* A categorical level that appears in the new month and not in February has no encoding.
  The count is printed per column; the encoder maps it to the unknown/other bucket.
* A continuous covariate whose new-month distribution sits outside February's is *not*
  clipped. Its standardized value simply leaves the range the network saw, which is the
  honest representation of an out-of-distribution scene and is one of the things the
  experiment exists to measure.

Usage:
    python3 make_month_split.py --month 202003 \\
        --february scratch/simulated/files/splits/group_split_v1_cov.csv \\
        --month-aux scratch/simulated/files/202003/aux_data.csv \\
        --month-covariates scratch/simulated/files/covariates_202003.csv.gz \\
        --out scratch/simulated/files/splits/month_202003_split.csv
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns the loader requires of any split table, whatever else it carries.
REQUIRED = ("ID", "split", "source_dir")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--month", required=True, help="e.g. 202003")
    p.add_argument("--february", type=Path, required=True,
                   help="the augmented February split table (group_split_v1_cov.csv)")
    p.add_argument("--month-aux", type=Path, required=True,
                   help="aux_data.csv written by prepare_month.py")
    p.add_argument("--month-covariates", type=Path, required=True,
                   help="covariates .csv.gz written by extract_covariates.py for this month")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    feb = pd.read_csv(args.february, low_memory=False)
    for c in REQUIRED:
        if c not in feb.columns:
            raise SystemExit(f"{args.february} has no '{c}' column")
    pool = feb[feb["split"].isin(("train", "val"))].copy()
    logger.info("February: %d rows, keeping %d train/val (dropping %d test)",
                len(feb), len(pool), len(feb) - len(pool))

    aux = pd.read_csv(args.month_aux, low_memory=False)
    cov = pd.read_csv(args.month_covariates, low_memory=False)
    logger.info("%s: %d aux rows, %d covariate rows", args.month, len(aux), len(cov))

    # An inner join, deliberately. A sounding without covariates cannot be encoded and a
    # covariate row without a sounding has no spectrum; either way it is not scoreable, and
    # a silent NaN row would be counted as a miss in every coverage number.
    dup = [c for c in cov.columns if c in aux.columns and c != "ID"]
    month = aux.merge(cov.drop(columns=dup), on="ID", how="inner", validate="one_to_one")
    if len(month) != len(aux):
        logger.warning("%d of %d soundings have no covariate row and are dropped",
                       len(aux) - len(month), len(aux))
    month["split"] = "test"
    month["source_dir"] = args.month

    missing = [c for c in pool.columns if c not in month.columns]
    extra = [c for c in month.columns if c not in pool.columns]
    if missing:
        logger.info("%d February columns absent from the month (filled with NaN): %s",
                    len(missing), ", ".join(sorted(missing)[:12])
                    + (" ..." if len(missing) > 12 else ""))
    if extra:
        logger.info("%d month columns absent from February (kept, unused): %s",
                    len(extra), ", ".join(sorted(extra)[:12])
                    + (" ..." if len(extra) > 12 else ""))

    out = pd.concat([pool, month], ignore_index=True, sort=False)
    if out["ID"].duplicated().any():
        # Possible in principle: the ID carries cluster/reference/replicate but not the
        # month, so February clst01_ref01_rep01 and March clst01_ref01_rep01 are the same
        # string. They are distinguished by source_dir, which is what resolves the .npy --
        # but a duplicated ID would still break any merge keyed on it, so it is checked.
        n = int(out["ID"].duplicated().sum())
        logger.warning("%d IDs occur in both February and %s. They are distinguished by "
                       "source_dir and the loader resolves them correctly, but do not "
                       "merge this table on ID alone.", n, args.month)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    # --- what the encoder will meet -------------------------------------------------
    report = {"month": args.month, "n_rows": int(len(out)),
              "n_train": int((out.split == "train").sum()),
              "n_val": int((out.split == "val").sum()),
              "n_test": int((out.split == "test").sum()),
              "n_test_reference_soundings": int(month.drop_duplicates(["clst", "ref"]).shape[0])
              if {"clst", "ref"} <= set(month.columns) else None,
              "unseen_categorical_levels": {},
              "out_of_range_continuous": {}}

    for c in ("Footprint_fix", "AerType1_fix", "AerType2_fix"):
        if c in pool.columns and c in month.columns:
            seen = set(pool[c].dropna().unique())
            new = sorted(set(month[c].dropna().unique()) - seen)
            frac = float(month[c].isin(new).mean()) if new else 0.0
            report["unseen_categorical_levels"][c] = {
                "levels": [str(v) for v in new], "fraction_of_soundings": frac}

    for c in ("SlrZen_fix", "sec_sza", "vza_deg", "sec_vza", "airmass", "prior_psfc_pa",
              "XCO2Prior", "AODPrior1", "AODPrior2"):
        if c in pool.columns and c in month.columns:
            lo, hi = pool[c].min(), pool[c].max()
            s = month[c].dropna()
            if len(s) == 0:
                continue
            report["out_of_range_continuous"][c] = {
                "february_min": float(lo), "february_max": float(hi),
                "month_min": float(s.min()), "month_max": float(s.max()),
                "fraction_outside": float(((s < lo) | (s > hi)).mean()),
                # How far outside, in the units the network actually sees.
                "max_abs_z": float(np.nanmax(np.abs((s - pool[c].mean()) / pool[c].std()))),
            }

    args.out.with_suffix(".summary.json").write_text(json.dumps(report, indent=2))
    print()
    print(json.dumps(report, indent=2))
    print()
    print(f"wrote {args.out} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

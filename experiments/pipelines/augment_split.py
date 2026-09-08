"""augment_split.py
Merge the covariate table into a group split table, and fit its normalization on the
TRAINING soundings only.

The split table written by `make_group_split.py` is already a self-contained annotations
file: one row per sounding, every `aux_data.csv` column, plus `source_dir`, `clst`, `ref`
and `split`. This script adds the covariates recovered by `extract_covariates.py` to it
and writes the statistics the data loader needs to encode them.

Everything -- continuous means and standard deviations, categorical vocabularies, and the
median used to impute a missing value -- is estimated from `split == "train"` rows alone.
Under a reference-sounding-disjoint split anything else leaks held-out scenes into
preprocessing, which is exactly the objection the split exists to answer.

Usage:
    python3 experiments/pipelines/augment_split.py \
        --split-csv  /workspace/project/scratch/simulated/files/splits/group_split_v1.csv \
        --covariates /workspace/project/scratch/simulated/files/covariates_v1.csv.gz \
        --out        /workspace/project/scratch/simulated/files/splits/group_split_v1_cov.csv \
        --stats-dir  /workspace/project/experiments/pipelines/stats/group_split_v1
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Never described as covariates, whichever table they come from:
#   ID, split, source_dir, meta_row, sounding_id   bookkeeping
#   clst, ref, rep                                 scene and replicate identifiers. `ref`
#                                                  is a unique key for a scene, so a model
#                                                  given it can memorize the training
#                                                  scenes outright -- exactly what the
#                                                  reference-sounding-disjoint split
#                                                  exists to prevent. They stay in the
#                                                  table (make_group_split writes them and
#                                                  the split logic needs them); they are
#                                                  just never given statistics, so no
#                                                  covariate preset can reach them.
#   pwf_*                                          20 columns constant to 0.02 %;
#                                                  --include-pwf if a model wants the
#                                                  pressure weighting function explicitly
_DROP_PREFIXES = ("pwf_",)
_DROP_EXACT = {"ID", "split", "source_dir", "meta_row", "sounding_id",
               "clst", "ref", "rep",
               # Scene identifiers from the cluster metadata. `Cluster` is the SOM label,
               # which is derived from RETRIEVED states and so leaks the target as well as
               # being unavailable operationally; SoundingID and Orbit are keys. Carried in
               # the covariate table for provenance, never given statistics.
               "SoundingID", "Orbit", "Cluster"}

# The cluster-.csv metadata, kept by extract_covariates.py only so the two pairings can be
# compared. It must never reach a model: it is attached by directory order, which does not
# reproduce. See diagnose_metadata_pairing.py.
_CSVMETA_PREFIX = "csvmeta_"

# Which covariates are categorical is declared, not inferred.
#
# A "fewer than N distinct integral values" rule looked tidier and is wrong: solar zenith
# angle is stored to a few decimals and can be integral, and any covariate is integral on a
# small sample. Misclassifying a continuous covariate produces an embedding table instead
# of a standardized number, which trains and looks fine.
#
# Non-numeric columns (the aerosol type labels) are categorical whatever this set says.
KNOWN_CATEGORICAL = frozenset({
    "Footprint", "AerType1", "AerType2", "Region",
    "Footprint_fix", "AerType1_fix", "AerType2_fix",
    "outcome_flag", "xco2_quality_flag", "mixture_component",
})

# A categorical covariate with more levels than this is almost certainly a mislabelled
# continuous one, and would give the model a near-unique key per sounding.
_MAX_LEVELS = 64


def _is_categorical(s: pd.Series, name: str, extra: frozenset = frozenset()) -> bool:
    if s.dtype.kind not in "ifb":
        return True
    return name in KNOWN_CATEGORICAL or name in extra


def build_stats(train: pd.DataFrame, columns: List[str],
                extra_categorical: frozenset = frozenset()) -> Dict[str, dict]:
    """Per-column encoding parameters, estimated from the training rows only."""
    stats: Dict[str, dict] = {}
    for col in columns:
        s = train[col]
        n_missing = int(s.isna().sum())
        if _is_categorical(s, col, extra_categorical):
            levels = sorted(str(v) for v in s.dropna().unique())
            if len(levels) > _MAX_LEVELS:
                raise ValueError(
                    f"'{col}' has {len(levels)} levels in the training rows. Either it is "
                    f"continuous and should not be in KNOWN_CATEGORICAL, or it identifies "
                    f"individual scenes and must not be a model input.")
            stats[col] = {
                "kind": "categorical",
                "levels": levels,
                "cardinality": len(levels) + 1,      # +1 for an unseen/missing level
                "unknown_index": len(levels),
                "n_missing_train": n_missing,
            }
        else:
            v = s.astype(float)
            mu = float(np.nanmean(v))
            sd = float(np.nanstd(v))
            stats[col] = {
                "kind": "continuous",
                "mean": mu,
                "std": sd if sd > 0 else 1.0,
                "impute": float(np.nanmedian(v)) if n_missing else mu,
                "n_missing_train": n_missing,
                "degenerate": bool(sd == 0.0),
            }
    return stats


def report_presets(available: set, stats: Dict[str, dict]) -> None:
    """Which covariate presets this table can actually serve.

    Worth the twenty lines: a preset that names an absent column fails at dataset
    construction, inside a GPU job, after the data has been unpacked. Saying so here costs
    nothing and turns a wasted job into a line of output.
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
        from data.loading.covariates import PRESETS
    except Exception as exc:            # torch absent, or run outside the checkout
        logger.info("cannot import the covariate presets (%s); skipping the "
                    "satisfiability report", exc)
        return

    usable = {c for c, v in stats.items()
              if not (v["kind"] == "continuous" and v.get("degenerate"))}
    print("\nCovariate presets this split table can serve:")
    for name, spec in sorted(PRESETS.items()):
        if spec.legacy:
            print(f"  {name:16s} OK        (hardcoded encoding, needs no statistics)")
            continue
        absent = [c for c in spec.columns if c not in available]
        degenerate = [c for c in spec.columns if c in stats and c not in usable]
        if absent:
            print(f"  {name:16s} MISSING   {absent}")
        elif degenerate:
            print(f"  {name:16s} OK, but constant in training: {degenerate}")
        else:
            print(f"  {name:16s} OK        ({spec.n_continuous} continuous, "
                  f"{spec.n_categorical} categorical, tier={spec.tier})")
    if any(c not in available for c in ("Latitude", "Longitude")):
        print("\n  Latitude/Longitude are absent. The published aux_data.csv predates the")
        print("  conf/clean_data.yaml change that carries them (chtc/reprocess.sh), so the")
        print("  scene_geo presets cannot run until the processed archives are rebuilt --")
        print("  and that rebuild inherits the same directory-order metadata pairing, so")
        print("  diagnose_metadata_pairing.py has to be settled first either way.")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-csv", type=Path, required=True)
    p.add_argument("--covariates", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stats-dir", type=Path, required=True)
    p.add_argument("--include-pwf", action="store_true",
                   help="also carry the 20 pressure-weighting-function columns")
    p.add_argument("--columns", type=str, default=None,
                   help="comma-separated whitelist; default is every covariate column")
    p.add_argument("--categorical", type=str, default=None,
                   help="comma-separated extra columns to encode as categorical, on top "
                        f"of {sorted(KNOWN_CATEGORICAL)}")
    args = p.parse_args()
    extra_categorical = frozenset(
        c.strip() for c in (args.categorical or "").split(",") if c.strip())

    split = pd.read_csv(args.split_csv)
    cov = pd.read_csv(args.covariates)
    logger.info("split table  %s rows x %s cols", len(split), split.shape[1])
    logger.info("covariates   %s rows x %s cols", len(cov), cov.shape[1])

    if "ID" not in split.columns or "ID" not in cov.columns:
        raise SystemExit("both tables must have an ID column")

    # The published four are the cross-check: if the covariate table's metadata does not
    # reproduce them row for row, the file-to-metadata pairing differs from the one that
    # produced aux_data.csv and every new covariate would be attached to the wrong scene.
    #
    # A row where BOTH tables are missing the value is not a disagreement. That case did
    # not exist until the February extension: every earlier table was built from a
    # delivery with a published `aux_data.csv` behind it, so these four columns were
    # populated everywhere. The extension has no published counterpart and never will --
    # `extract_covariates.py` says so ("no --metadata-from") -- so its rows carry NaN on
    # both sides. `np.isclose(nan, nan)` is False, which turned "nothing to compare" into
    # 23,689 disagreements and killed the job at the last step of the prep.
    #
    # The one-sided case stays a hard failure: a value on one side and not the other means
    # the two tables really are describing different soundings.
    probe = split[["ID", "SlrZen", "Footprint", "AerType1", "AerType2"]].merge(
        cov[["ID", "SlrZen", "Footprint", "AerType1", "AerType2"]],
        on="ID", how="inner", suffixes=("_split", "_cov"))
    if len(probe) == 0:
        raise SystemExit("no IDs in common -- the covariate table is for another dataset")
    disagree, n_checked = {}, {}
    for c in ("SlrZen", "Footprint", "AerType1", "AerType2"):
        a, b = probe[f"{c}_split"], probe[f"{c}_cov"]
        both_missing = (a.isna() & b.isna()).to_numpy()
        if a.dtype.kind in "fc" and b.dtype.kind in "fc":
            with np.errstate(invalid="ignore"):
                same = np.isclose(a.astype(float), b.astype(float),
                                  rtol=1e-5, atol=1e-8, equal_nan=True)
        else:
            # `str(nan)` is "nan" on both sides, so the object branch already treated the
            # both-missing case as equal; `both_missing` below makes that explicit rather
            # than incidental.
            same = a.astype(str).to_numpy() == b.astype(str).to_numpy()
        same = same | both_missing
        n_checked[c] = int((~both_missing).sum())
        if not same.all():
            disagree[c] = int((~same).sum())
    if disagree:
        raise SystemExit(
            "the covariate table disagrees with the split table on covariates both "
            f"contain: {disagree} of {len(probe)} rows. Re-run extract_covariates.py "
            "with --verify-against before going further; a mismatch here means scene "
            "metadata is being attached to the wrong soundings.")

    # How much of the table the check could actually see. A pairing check that verified
    # nothing must not read as a pairing check that passed, so this is a number and not a
    # boolean, and zero is a failure.
    fewest = min(n_checked.values())
    if fewest == 0:
        raise SystemExit(
            "the cross-check verified NOTHING: every shared row is missing the published "
            f"metadata on both sides ({len(probe)} rows). The file-to-metadata pairing is "
            "then untested, so run extract_covariates.py with --metadata-from, or check "
            "the airmass feasibility line in its output, before training on this table.")
    logger.info("cross-check OK on %d shared rows; the published metadata is present on "
                "%d-%d of them (%d carry none on either side, which is expected for a "
                "delivery with no published aux_data.csv -- the `legacy` presets cannot "
                "be trained on those rows)",
                len(probe), fewest, max(n_checked.values()), len(probe) - fewest)

    candidate = [c for c in cov.columns
                 if c not in _DROP_EXACT
                 and not c.startswith(_CSVMETA_PREFIX)
                 and not (c.startswith(_DROP_PREFIXES) and not args.include_pwf)]
    if args.columns:
        want = {c.strip() for c in args.columns.split(",") if c.strip()}
        missing = want - set(cov.columns)
        if missing:
            raise SystemExit(f"requested columns absent from the covariate table: {missing}")
        candidate = [c for c in candidate if c in want]

    # Anything the split table already carries is NOT merged again.
    #
    # The split table holds the targets and the published auxiliary columns, and several
    # of them -- retrieved_xco2, retrieved_xco2_variance, SlrZen, Footprint, the aerosol
    # types -- are also legitimate covariates that the extraction re-derives. Merging both
    # copies makes pandas rename them retrieved_xco2_x / retrieved_xco2_y, after which
    # nothing that asks for `retrieved_xco2` finds it. That is what crashed the first run.
    #
    # They still have to be described in the statistics file, because a preset may name
    # them: `overlap` columns get statistics computed from the split table's own values,
    # which are the published ones and the right ones to use.
    overlap = [c for c in candidate if c in split.columns and c != "ID"]
    merge_cols = [c for c in candidate if c not in split.columns]
    keep = merge_cols + overlap
    logger.info("carrying %d covariate columns (%d merged in, %d already in the split "
                "table)", len(keep), len(merge_cols), len(overlap))
    if overlap:
        logger.info("already present, taken from the split table: %s", sorted(overlap))

    merged = split.merge(cov[["ID"] + merge_cols], on="ID", how="left",
                         validate="one_to_one")
    collided = [c for c in merged.columns if c.endswith(("_x", "_y"))
                and c not in split.columns and c not in cov.columns]
    if collided:
        raise SystemExit(
            f"the merge renamed {collided} -- a column exists in both tables and was not "
            f"caught by the overlap check. Fix the split above rather than the symptom.")
    n_unmatched = int(merged[merge_cols[0]].isna().sum()) if merge_cols else 0
    if n_unmatched:
        logger.warning("%d split rows had no covariate row (left as NaN)", n_unmatched)

    train = merged[merged["split"] == "train"]
    if len(train) == 0:
        raise SystemExit("the split table has no training rows")
    stats = build_stats(train, keep, extra_categorical)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    args.stats_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "columns": keep,
        "stats": stats,
        "provenance": {
            "split_csv": str(args.split_csv),
            "covariates": str(args.covariates),
            "out": str(args.out),
            "estimated_from": "split == 'train'",
            "n_train_rows": int(len(train)),
            "n_train_reference_soundings": int(
                train.drop_duplicates(["clst", "ref"]).shape[0]),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    stats_path = args.stats_dir / "covariate_stats.json"
    stats_path.write_text(json.dumps(payload, indent=2))

    n_cat = sum(1 for v in stats.values() if v["kind"] == "categorical")
    print(f"\nWrote {args.out}  ({len(merged)} rows, {merged.shape[1]} columns)")
    print(f"Wrote {stats_path}  ({len(keep)} covariates: "
          f"{len(keep) - n_cat} continuous, {n_cat} categorical)")
    degenerate = [c for c, v in stats.items()
                  if v["kind"] == "continuous" and v["degenerate"]]
    if degenerate:
        print(f"\nConstant in the training rows (carry no information, excluded from any "
              f"preset): {degenerate}")
    missing = {c: v["n_missing_train"] for c, v in stats.items() if v["n_missing_train"]}
    if missing:
        print(f"\nMissing values in training rows (imputed at the training median): {missing}")

    report_presets(set(merged.columns), stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

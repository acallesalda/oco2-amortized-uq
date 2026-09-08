"""make_full_feb_split.py
Build the ALL-FEBRUARY training table: February's test scenes are folded back into
training, and the validation scenes are left exactly where they were.

Why this table exists
---------------------
`group_split_v1` holds 10 % of February's reference soundings out as a test partition,
because February was the only month there was and a held-out score had to come from
somewhere. Once March and July are the test set that reservation costs accuracy and buys
nothing: those 45 scenes are never scored again, and the model scored on another month should be
the best February model we can fit rather than the one that was fit around a February
hold-out.

So this relabels `test` -> `train` and leaves `val` alone:

| | group_split_v1 | full_feb_v1 |
|---|---|---|
| train | 361 reference soundings, 36,003 spectra | **406, 40,492** |
| val | 45, 4,490 | 45, 4,490 (the same scenes) |
| test | 45, 4,489 | none |

Why the validation partition STAYS
----------------------------------
Two reasons, and the second is the one that settles it.

1. **The searched hyperparameters were ranked on those 45 scenes.** The winners in
   `chtc/bo/results/bo_<study>_best.json` came out of 150 trials scored on this exact
   validation set. Retraining them against a different one would mean selecting a
   configuration on one criterion and its checkpoint on another.

2. **The Laplace approximation needs it at scoring time.** `predict.py` calibrates the
   aleatoric diagonal of the Laplace predictive on the validation partition for the profile
   and near-surface tasks (`mc` mode; the scalar tasks use `marglik`, which estimates the
   same quantity from the training set). Calibrating it in-sample would make it the smaller
   of the two estimates and the Laplace's intervals correspondingly narrower -- an
   optimistic baseline, in the one place where our argument is that the flow's calibration
   beats the Laplace's. The validation partition is what keeps that number honest, and it
   is why "train on everything" stops at the test partition rather than going further.

So 45 of 451 reference soundings are still not fitted. They are not wasted: they are what
selects the checkpoint and what calibrates the baseline's noise model.

What is NOT done here
---------------------
No statistics are computed. Target standardization and covariate encoding are estimated
from this table's *training* rows by `compute_split_stats.py` and `augment_split.py`, which
run next -- and because the training rows have changed, both must be re-run rather than
reused from `group_split_v1`. Pointing a full-February model at February's old constants
would standardize with a mean estimated from 90 % of the data the model was fitted on:
not wrong so much as untraceable.

Usage:
    python3 make_full_feb_split.py \
        --group-split scratch/simulated/files/splits/group_split_v1.csv \
        --out         scratch/simulated/files/splits/full_feb_v1.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_FILES_ROOT = Path("/workspace/project/scratch/simulated/files")
DEFAULT_IN = DEFAULT_FILES_ROOT / "splits" / "group_split_v1.csv"
DEFAULT_OUT = DEFAULT_FILES_ROOT / "splits" / "full_feb_v1.csv"

REQUIRED = ("ID", "split", "source_dir", "clst", "ref")


def absorb_test(df: pd.DataFrame) -> pd.DataFrame:
    """`test` -> `train`; `val` untouched. No rows are added, dropped or reordered."""
    out = df.copy()
    n = int((out["split"] == "test").sum())
    out.loc[out["split"] == "test", "split"] = "train"
    logger.info("moved %d test rows into train; val untouched", n)
    return out


def n_groups_per_split(df: pd.DataFrame) -> dict:
    """Reference soundings, not rows, in each partition."""
    g = df.drop_duplicates(subset=["clst", "ref", "split"])
    return {k: int(v) for k, v in g.groupby("split").size().to_dict().items()}


def validate(out: pd.DataFrame, src: pd.DataFrame) -> None:
    """The properties every downstream job assumes, checked against the table itself.

    None of them is visible in the numbers that come out later: a table whose validation
    scenes are not the ones the search ranked on trains perfectly well, selects a
    checkpoint, calibrates a Laplace, and produces a plausible March RMSE with nothing
    anywhere to say the experiment was not the one described.
    """
    if (out["split"] == "test").any():
        raise AssertionError(
            "this table still has a test partition. It is the ALL-February training "
            "table; the test set is another month, and predict.py takes its test rows "
            "from month_<month>_split.csv, not from here.")
    for split in ("train", "val"):
        if (out["split"] == split).sum() == 0:
            raise AssertionError(
                f"partition '{split}' is empty. The trainers select their checkpoint on "
                f"val and predict.py calibrates the Laplace's aleatoric diagonal on it; "
                f"neither has anything to work with.")
    # Duplicates first: a duplicated ID also changes the row count, and "44983 rows in,
    # 44982 out" is a much worse diagnosis of it than naming the offending IDs.
    if out["ID"].duplicated().any():
        dupes = out.loc[out["ID"].duplicated(), "ID"].head().tolist()
        raise AssertionError(f"duplicate sounding IDs, e.g. {dupes}")
    if len(out) != len(src):
        raise AssertionError(f"{len(src)} rows in, {len(out)} out. Nothing here drops rows.")

    def scenes(frame, split):
        s = frame[frame["split"] == split]
        return set(map(tuple, s[["clst", "ref"]].drop_duplicates().to_numpy()))

    train_refs, val_refs = scenes(out, "train"), scenes(out, "val")
    if train_refs & val_refs:
        raise AssertionError(
            f"{len(train_refs & val_refs)} reference soundings are in both train and val. "
            f"The two must stay scene-disjoint -- that is the property inherited from "
            f"group_split_v1, and it is what makes the validation partition a held-out "
            f"set rather than a subsample of the training data.")
    if val_refs != scenes(src, "val"):
        raise AssertionError(
            "the validation scenes are not group_split_v1's. The searched configurations "
            "were ranked on those scenes; selecting their checkpoints on a different set "
            "is not the same experiment.")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group-split", type=Path, default=DEFAULT_IN,
                   help="the reference-sounding-disjoint table to derive from")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    src = pd.read_csv(args.group_split, low_memory=False)
    for c in REQUIRED:
        if c not in src.columns:
            raise SystemExit(f"{args.group_split} has no '{c}' column")

    before_rows = {k: int(v) for k, v in src["split"].value_counts().to_dict().items()}
    before_groups = n_groups_per_split(src)
    logger.info("%s: %d rows, %s", args.group_split, len(src),
                ", ".join(f"{k}={v}" for k, v in sorted(before_rows.items())))

    out = absorb_test(src)
    validate(out, src)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    groups = out.drop_duplicates(subset=["clst", "ref", "split"])
    after_groups = n_groups_per_split(out)
    after_rows = {k: int(v) for k, v in out.groupby("split").size().to_dict().items()}

    print()
    print(f"wrote {args.out}  ({len(out)} rows)")
    print()
    print("reference soundings per cluster x partition:")
    print(pd.crosstab(groups["clst"], groups["split"]).to_string())
    print()
    for split in ("train", "val"):
        print(f"  {split:<5s} {after_groups.get(split, 0):>4d} reference soundings"
              f"   {after_rows.get(split, 0):>6d} spectra")
    print( "  test     none   (the test set is another month; see make_month_split.py)")
    print()

    meta = {
        "out": str(args.out),
        "derived_from": str(args.group_split),
        "n_reference_soundings": int(out.drop_duplicates(["clst", "ref"]).shape[0]),
        "n_spectra": int(len(out)),
        "groups_per_split": after_groups,
        "rows_per_split": after_rows,
        "source_groups_per_split": before_groups,
        "source_rows_per_split": before_rows,
        # The two numbers worth quoting: how much more of February this model
        # saw than the one the published tables report.
        "training_scenes_gained": int(after_groups.get("train", 0)
                                      - before_groups.get("train", 0)),
        "training_rows_gained": int(after_rows.get("train", 0)
                                    - before_rows.get("train", 0)),
        "validation": ("group_split_v1's scenes, unchanged -- checkpoint selection and the "
                       "Laplace's aleatoric calibration both read it"),
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

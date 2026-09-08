"""make_replicate_split.py
Write the ORIGINAL replicate-level partition out as a split table.

Why this exists
---------------
The revision needs two columns that differ in one thing. We have the reference-disjoint
column (`group_split_v1`), trained with the corrected scene metadata, the `scene_prior`
covariates and a 25-trial search. The within-reference column in the submitted manuscript
was trained with none of those. Comparing them attributes to "the split" a difference that
is partly the covariates and partly the search, and the question -- what does an
unseen scene cost? -- is not answerable from that pair.

So the old partition has to be trainable by the same pipeline as the new one. Everything
downstream (`spectral_loader.build_dataset_*_grouped`, `augment_split.py`,
`compute_split_stats.py`, `predict.py --split-csv`) is driven by a split table with a
`split` column, and the only reason the replicate partition could not use that path is that
it was never written down as a table: it lives in the *directory* a sounding sits in, plus
a `random_split` inside the training scripts. This script materializes it.

What the partition is
---------------------
`clean_data.py` wrote each sounding into `files/train/` or `files/test/` -- roughly 80 of a
reference sounding's ~100 replicates to train and ~20 to test, independently per reference,
so every reference sounding appears in both. That assignment is preserved here exactly:

    source_dir == "test"   ->  split = "test"

The published training scripts then drew validation from the training pool with
`torch.utils.data.random_split(..., generator=manual_seed(42))`, which is a permutation of
the *dataset's* enumeration order and therefore not reproducible from the annotations alone.
Rather than pretend otherwise, validation is drawn here from the training pool at the
replicate level with a recorded numpy seed, and the provenance JSON says so. That is a
different draw from the published one and it does not matter: the test partition -- the only
thing any reported number is computed on -- is identical, and no test sounding is used for
fitting or for model selection either way.

What this deliberately is NOT
-----------------------------
A group-disjoint split. Every reference sounding appears in all three partitions here, on
purpose: this table *is* the within-reference interpolation experiment. `validate()` below
asserts the opposite of `make_group_split.py`'s check -- that references do span partitions
-- so that the two tables can never be confused for one another downstream.

Usage:
    python3 make_replicate_split.py                    # -> files/splits/replicate_split_v1.csv
    python3 make_replicate_split.py --val-frac 0.10 --seed 42
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Keep in sync with data.out_root in code/shared_conf/data/local_simulated.yaml.
DEFAULT_FILES_ROOT = Path("/workspace/project/scratch/simulated/files")
DEFAULT_OUT = DEFAULT_FILES_ROOT / "splits" / "replicate_split_v1.csv"

# IDs are written by oco2_utils.add_uniq_id as clst%04d_ref%04d_rep%04d
_ID_RE = re.compile(r"clst(\d+)_ref(\d+)_rep(\d+)")

SOURCE_DIRS = ("train", "test")


def load_all_annotations(files_root: Path) -> pd.DataFrame:
    """Concatenate the preprocessed partitions, tagging each row with its subdirectory.

    Identical to `make_group_split.py`'s loader, and deliberately so: the two tables must
    contain the same 44,982 rows with the same columns, or a difference between the two
    experiments could be a difference in which soundings they saw.
    """
    frames: List[pd.DataFrame] = []
    for sub in SOURCE_DIRS:
        path = files_root / sub / "aux_data.csv"
        if not path.exists():
            raise FileNotFoundError(f"Expected annotations at {path}")
        df = pd.read_csv(path)
        df["source_dir"] = sub
        frames.append(df)
        logger.info("loaded %s rows from %s", len(df), path)
    return pd.concat(frames, ignore_index=True)


def add_group_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Parse cluster, reference and replicate indices out of the ID column."""
    parsed = df["ID"].str.extract(_ID_RE)
    if parsed.isna().any().any():
        bad = df.loc[parsed.isna().any(axis=1), "ID"].head().tolist()
        raise ValueError(f"Could not parse cluster/ref from IDs, e.g. {bad}")
    df = df.copy()
    df["clst"] = parsed[0].astype(int)
    df["ref"] = parsed[1].astype(int)
    df["rep"] = parsed[2].astype(int)
    return df


def assign_splits(df: pd.DataFrame, val_frac: float, seed: int) -> pd.Series:
    """test = the published test directory; val = a seeded draw from the training pool.

    The draw is stratified by reference sounding rather than taken over the pool as a
    whole. An unstratified draw of 10 % from 36,024 soundings would leave a handful of
    references with no validation replicate and a handful over-represented, which is a
    needless source of variance in the checkpoint selection -- and the published protocol,
    a uniform random draw over the pool, is not being reproduced exactly in any case.
    """
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=df.index, dtype=object)
    split[df["source_dir"] == "test"] = "test"

    pool = df.index[df["source_dir"] == "train"]
    for _, idx in df.loc[pool].groupby(["clst", "ref"]).groups.items():
        idx = np.asarray(idx)
        # At least one, so no reference sounding is unrepresented in validation.
        k = max(1, int(round(val_frac * len(idx))))
        split[rng.choice(idx, size=k, replace=False)] = "val"
    return split


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster reference and row counts for each partition."""
    grp = (
        df.drop_duplicates(subset=["clst", "ref", "split"])
        .groupby(["clst", "split"]).size().unstack(fill_value=0)
    )
    row = df.groupby(["clst", "split"]).size().unstack(fill_value=0)
    grp.columns = [f"refs_{c}" for c in grp.columns]
    row.columns = [f"rows_{c}" for c in row.columns]
    return pd.concat([grp, row], axis=1)


def validate(df: pd.DataFrame) -> None:
    """Fail loudly if this is not the table it claims to be.

    The interesting assertion is the second one, and it is the inverse of the check in
    `make_group_split.py`: this table is *supposed* to leak reference soundings across
    partitions. If it did not, someone has pointed this script at a group split, and the
    experiment it feeds would silently stop being the control it exists to be.
    """
    if df["ID"].duplicated().any():
        dupes = df.loc[df["ID"].duplicated(), "ID"].head().tolist()
        raise AssertionError(f"duplicate sounding IDs in split table, e.g. {dupes}")

    per_ref = df.groupby(["clst", "ref"])["split"].nunique()
    if (per_ref == 1).all():
        raise AssertionError(
            "no reference sounding spans more than one partition -- this is a "
            "group-disjoint table, not the replicate-level one. Check --files-root."
        )
    shared = int((per_ref > 1).sum())
    logger.info("%d of %d reference soundings appear in more than one partition "
                "(this is the point)", shared, len(per_ref))

    for split in ("train", "val", "test"):
        if (df["split"] == split).sum() == 0:
            raise AssertionError(f"partition '{split}' is empty")

    # The test partition must be exactly the published one, or the whole comparison moves.
    mismatch = int(((df["split"] == "test") != (df["source_dir"] == "test")).sum())
    if mismatch:
        raise AssertionError(
            f"{mismatch} sounding(s) disagree between `split == test` and the published "
            "test directory. The test partition must be the published one exactly."
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--files-root", type=Path, default=DEFAULT_FILES_ROOT,
                   help="root containing train/ and test/ from clean_data.py")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="destination CSV for the split table")
    p.add_argument("--val-frac", type=float, default=0.10,
                   help="fraction of each reference sounding's training replicates held "
                        "out for validation")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not 0.0 < args.val_frac < 1.0:
        p.error("--val-frac must be in (0, 1)")

    df = add_group_keys(load_all_annotations(args.files_root))
    if "split" in df.columns:
        logger.warning("dropping pre-existing 'split' column from annotations")
        df = df.drop(columns=["split"])

    n_refs = df.drop_duplicates(subset=["clst", "ref"]).shape[0]
    logger.info("found %d reference soundings across %d clusters (%d spectra)",
                n_refs, df["clst"].nunique(), len(df))

    df["split"] = assign_splits(df, args.val_frac, args.seed)
    validate(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    logger.info("wrote %s (%d rows)", args.out, len(df))

    summary = summarize(df)
    print()
    print(summary.to_string())
    print()
    print(df["split"].value_counts().to_string())

    meta = {
        "kind": "replicate-level (within-reference interpolation)",
        "source": "source_dir from clean_data.py; validation drawn per reference sounding",
        "n_rows": int(len(df)),
        "n_reference_soundings": int(n_refs),
        "n_clusters": int(df["clst"].nunique()),
        "rows_per_split": df["split"].value_counts().to_dict(),
        "references_per_split": {
            s: int(sub.drop_duplicates(subset=["clst", "ref"]).shape[0])
            for s, sub in df.groupby("split")
        },
        "val_frac": args.val_frac,
        "seed": args.seed,
        "validation_note": (
            "The published runs drew validation with torch.utils.data.random_split over "
            "the dataset's enumeration order, which is not reproducible from the "
            "annotations. This draw is a stratified numpy draw with the seed above. The "
            "test partition is the published one exactly."
        ),
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(meta, indent=2))
    logger.info("wrote %s", summary_path)


if __name__ == "__main__":
    main()

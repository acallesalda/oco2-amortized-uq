"""make_group_split.py
Build a reference-sounding-disjoint (group) train/val/test split.

The original pipeline (extract_data_tree.py) assigns "train"/"test" independently to each
of the 100 stochastic replicates of a reference sounding, so all 479 reference soundings
appear in both partitions. Evaluating on that split measures within-reference
interpolation, not retrieval of an unseen scene.

This script instead assigns *whole reference soundings* to train/val/test, stratified so
that the nine geophysical clusters keep approximately their original proportions in every
partition. It reads only the already-preprocessed aux_data.csv files, so no HDF5
re-extraction is needed; the .npy spectra stay where clean_data.py put them and each row
records which subdirectory it lives in.

Output: one self-contained annotations table with every original aux_data column plus
`source_dir`, `clst`, `ref` and `split`.

Usage (defaults reproduce the 383/48/48 split used in the revision):
    python3 make_group_split.py
    python3 make_group_split.py --val-frac 0.10 --test-frac 0.10 --seed 42
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Keep in sync with data.out_root in code/shared_conf/data/local_simulated.yaml:
# the source volume under /workspace/data is read-only for us, so everything the
# pipeline writes lives under the scratch root instead.
DEFAULT_FILES_ROOT = Path("/workspace/project/scratch/simulated/files")
DEFAULT_OUT = DEFAULT_FILES_ROOT / "splits" / "group_split_v1.csv"

# IDs are written by oco2_utils.add_uniq_id as clst%04d_ref%04d_rep%04d
_ID_RE = re.compile(r"clst(\d+)_ref(\d+)_rep(\d+)")

SOURCE_DIRS = ("train", "test")


def load_all_annotations(files_root: Path) -> pd.DataFrame:
    """Concatenate the preprocessed partitions, tagging each row with its subdirectory."""
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
    """Parse cluster and reference-sounding indices out of the ID column."""
    parsed = df["ID"].str.extract(_ID_RE)
    if parsed.isna().any().any():
        bad = df.loc[parsed.isna().any(axis=1), "ID"].head().tolist()
        raise ValueError(f"Could not parse cluster/ref from IDs, e.g. {bad}")
    df = df.copy()
    df["clst"] = parsed[0].astype(int)
    df["ref"] = parsed[1].astype(int)
    df["rep"] = parsed[2].astype(int)
    return df


def apportion(sizes: Dict[int, int], total: int) -> Dict[int, int]:
    """Largest-remainder (Hamilton) apportionment of `total` across clusters.

    Allocates proportionally to cluster size, giving the leftover units to the clusters
    with the largest fractional parts. Ties break on cluster id so the result is
    deterministic.
    """
    grand = sum(sizes.values())
    if grand == 0:
        return {c: 0 for c in sizes}
    quotas = {c: n * total / grand for c, n in sizes.items()}
    base = {c: int(np.floor(q)) for c, q in quotas.items()}
    remainder = total - sum(base.values())
    order = sorted(sizes, key=lambda c: (-(quotas[c] - base[c]), c))
    for c in order[:remainder]:
        base[c] += 1
    return base


def cap_and_rebalance(
    alloc: Dict[int, int], caps: Dict[int, int], total: int
) -> Dict[int, int]:
    """Clip an allocation to per-cluster caps and re-place the freed units elsewhere.

    Only binds when a cluster has very few reference soundings; with 479 groups over nine
    clusters it is normally a no-op, but it guarantees every cluster retains at least one
    training group.
    """
    alloc = {c: min(k, caps[c]) for c, k in alloc.items()}
    deficit = total - sum(alloc.values())
    while deficit > 0:
        # give to the cluster with the most unused capacity, largest cluster first
        candidates = [c for c in alloc if alloc[c] < caps[c]]
        if not candidates:
            logger.warning(
                "could not place %d group(s): every cluster is at capacity", deficit
            )
            break
        c = max(candidates, key=lambda c: (caps[c] - alloc[c], caps[c], -c))
        alloc[c] += 1
        deficit -= 1
    return alloc


def assign_splits(
    groups: pd.DataFrame, val_frac: float, test_frac: float, seed: int
) -> pd.DataFrame:
    """Assign each (clst, ref) group to train/val/test, stratified by cluster."""
    rng = np.random.default_rng(seed)
    sizes = groups.groupby("clst").size().to_dict()
    n_groups = len(groups)

    n_val = int(round(val_frac * n_groups))
    n_test = int(round(test_frac * n_groups))

    # Reserve at least one training group per cluster.
    caps = {c: max(n - 1, 0) for c, n in sizes.items()}
    test_alloc = cap_and_rebalance(apportion(sizes, n_test), caps, n_test)
    val_caps = {c: caps[c] - test_alloc[c] for c in sizes}
    val_alloc = cap_and_rebalance(apportion(sizes, n_val), val_caps, n_val)

    assignments = {}
    for clst, sub in groups.groupby("clst"):
        refs = np.sort(sub["ref"].to_numpy())
        rng.shuffle(refs)
        n_t, n_v = test_alloc[clst], val_alloc[clst]
        for r in refs[:n_t]:
            assignments[(clst, int(r))] = "test"
        for r in refs[n_t:n_t + n_v]:
            assignments[(clst, int(r))] = "val"
        for r in refs[n_t + n_v:]:
            assignments[(clst, int(r))] = "train"

    out = groups.copy()
    out["split"] = [assignments[(int(c), int(r))] for c, r in zip(out["clst"], out["ref"])]
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster group and row counts for each partition."""
    grp = (
        df.drop_duplicates(subset=["clst", "ref"])
        .groupby(["clst", "split"]).size().unstack(fill_value=0)
    )
    row = df.groupby(["clst", "split"]).size().unstack(fill_value=0)
    grp.columns = [f"groups_{c}" for c in grp.columns]
    row.columns = [f"rows_{c}" for c in row.columns]
    return pd.concat([grp, row], axis=1)


def validate(df: pd.DataFrame) -> None:
    """Fail loudly if the split is not actually group-disjoint."""
    per_group = df.groupby(["clst", "ref"])["split"].nunique()
    leaked = per_group[per_group > 1]
    if len(leaked) > 0:
        raise AssertionError(
            f"{len(leaked)} reference sounding(s) span multiple partitions: "
            f"{leaked.index.tolist()[:5]}"
        )
    if df["ID"].duplicated().any():
        dupes = df.loc[df["ID"].duplicated(), "ID"].head().tolist()
        raise AssertionError(f"duplicate sounding IDs in split table, e.g. {dupes}")
    for split in ("train", "val", "test"):
        if (df["split"] == split).sum() == 0:
            raise AssertionError(f"partition '{split}' is empty")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--files-root", type=Path, default=DEFAULT_FILES_ROOT,
                   help="root containing train/ and test/ from clean_data.py")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="destination CSV for the split table")
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.val_frac + args.test_frac >= 1.0:
        p.error("--val-frac + --test-frac must be < 1.0")

    df = add_group_keys(load_all_annotations(args.files_root))
    # aux_data.csv should not carry a 'split' column (clean_data.py does not save one),
    # but drop it defensively so the merge below cannot produce split_x / split_y.
    if "split" in df.columns:
        logger.warning("dropping pre-existing 'split' column from annotations")
        df = df.drop(columns=["split"])

    groups = df.drop_duplicates(subset=["clst", "ref"])[["clst", "ref"]].reset_index(drop=True)
    logger.info(
        "found %d reference soundings across %d clusters (%d spectra)",
        len(groups), groups["clst"].nunique(), len(df),
    )

    groups = assign_splits(groups, args.val_frac, args.test_frac, args.seed)
    df = df.merge(groups, on=["clst", "ref"], how="left", validate="many_to_one")
    validate(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    table = summarize(df)
    n_groups = groups.groupby("split").size().to_dict()
    n_rows = df.groupby("split").size().to_dict()

    print(f"\nWrote {args.out}  ({len(df)} rows)\n")
    print(table.to_string())
    print("\nTotals:")
    for split in ("train", "val", "test"):
        print(f"  {split:<5s} {n_groups.get(split, 0):>4d} reference soundings"
              f"   {n_rows.get(split, 0):>6d} spectra")

    meta = {
        "out": str(args.out),
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "n_reference_soundings": int(len(groups)),
        "n_spectra": int(len(df)),
        "groups_per_split": {k: int(v) for k, v in n_groups.items()},
        "rows_per_split": {k: int(v) for k, v in n_rows.items()},
        "per_cluster": json.loads(table.to_json(orient="index")),
    }
    meta_path = args.out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nSummary written to {meta_path}")
    print(
        "\nNext: recompute standardization statistics from the TRAINING soundings only:\n"
        f"  python3 compute_split_stats.py --split-csv {args.out}"
    )


if __name__ == "__main__":
    main()

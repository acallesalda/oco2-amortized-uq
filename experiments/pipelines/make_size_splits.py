"""make_size_splits.py
Build NESTED SUBSETS of `group_split_v1`'s training partition, so the same models can be
retrained on 90, 180 and 270 reference soundings and the learning curve in the *number of
scenes* can be measured.

Why this table exists
---------------------
Three of the paper's claims turn on one unmeasured quantity: the slope of test error
against the number of distinct reference soundings.

* the tail of Figure~\\ref{fig:scenes} concentrates on the two clusters
  that contribute fewest training scenes, and hypothesizes that more scenes would help.
* the same is true of the March and July degradation.
* it is open whether 47,900 spectra is enough at all, next to the millions of soundings
  OCO-2's bias correction is tuned on.
* a diagnostic that speaks to generalization is wanted.

A curve of held-out error against training scenes answers all four in the same picture,
and it distinguishes the two explanations that the pooled numbers cannot: a model that is
*data-limited* is still improving at 361 scenes, and a model that is *capacity-* or
*conditioning-limited* has already flattened, in which case simulating more February
scenes is not the fix. The cluster-balanced experiment already showed that
re-weighting the scenes we have buys nothing for a regime the network cannot identify from
its inputs; this measures the other axis.

What it does
------------
For each requested size `k`, the training reference soundings of `group_split_v1` are
subsampled to `k` -- stratified across geophysical clusters by the same largest-remainder
apportionment `make_group_split.py` uses, and NESTED, so the 90-scene training set is a
subset of the 180-scene one and so on. Nesting matters: it makes the difference between two
points on the curve the scenes that were added, not a different draw.

The validation and test partitions are left exactly as they are -- the same 45 + 45
reference soundings, in every table -- because the point of the exercise is a curve in
which only the training population moves. Training rows that are not drawn are relabelled
`unused` rather than deleted, so the table still documents the whole population and
`compute_split_stats.py` / `augment_split.py`, which select `split == "train"`, ignore them.

| | group_split_v1 | size_90_v1 | size_180_v1 | size_270_v1 |
|---|---|---|---|---|
| train | 361 | 90 | 180 | 270 |
| unused | 0 | 271 | 181 | 91 |
| val | 45 | 45 | 45 | 45 |
| test | 45 | 45 | 45 | 45 |

What is NOT done here
---------------------
No statistics, no covariate merge: `compute_split_stats.py` and `augment_split.py` run
next, per size, because both estimate from the training rows and the training rows are the
thing that changed. `chtc/prep_size.sh` runs the three in order.

Nor is the hyperparameter search redone. Each size is trained at the configuration that won
the search at 361 scenes, which is the honest comparison for the question being asked --
the same model, less data -- and biases the small-data points *pessimistically*, since a
smaller training set would have selected a smaller model. Say so when the curve is quoted.

Usage:
    python3 experiments/pipelines/make_size_splits.py \
        --group-split scratch/simulated/files/splits/group_split_v1.csv \
        --sizes 90 180 270 --out-dir scratch/simulated/files/splits
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("make_size_splits")

# Reuse the apportionment of the split this subsets, so the two agree on what
# "stratified by cluster" means.
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from make_group_split import apportion, cap_and_rebalance  # noqa: E402


def nested_order(train_groups: pd.DataFrame, sizes: list[int], seed: int) -> list[tuple]:
    """A single ordering of the training reference soundings that every size cuts.

    Built size by size: the k-scene set is the (k-1)-th prefix plus the scenes the
    apportionment for k adds, so the sets are nested by construction and each is
    stratified by cluster on its own terms rather than inheriting the largest set's
    proportions.
    """
    rng = np.random.default_rng(seed)
    per_cluster = {}
    for clst, sub in train_groups.groupby("clst"):
        refs = np.sort(sub["ref"].to_numpy())
        rng.shuffle(refs)
        per_cluster[int(clst)] = [int(r) for r in refs]

    sizes_total = {c: len(v) for c, v in per_cluster.items()}
    taken = {c: 0 for c in per_cluster}
    order: list[tuple] = []
    for k in sorted(sizes):
        caps = dict(sizes_total)
        alloc = cap_and_rebalance(apportion(sizes_total, k), caps, k)
        for c in sorted(alloc):
            want = alloc[c]
            if want < taken[c]:
                # A smaller allocation for a cluster than the previous size already took.
                # Rounding can do this by one scene; the nesting takes precedence, so the
                # scene stays and another cluster gives one up instead.
                continue
            for r in per_cluster[c][taken[c]:want]:
                order.append((c, r))
            taken[c] = max(taken[c], want)
        # Rounding can leave the prefix a scene or two short of k; fill from the clusters
        # with the most unused scenes, largest first, which is cap_and_rebalance's rule.
        while len(order) < k:
            c = max((c for c in per_cluster if taken[c] < sizes_total[c]),
                    key=lambda c: (sizes_total[c] - taken[c], sizes_total[c], -c),
                    default=None)
            if c is None:
                break
            order.append((c, per_cluster[c][taken[c]]))
            taken[c] += 1
    # Everything not placed by the largest size, so the ordering covers the partition.
    for c in sorted(per_cluster):
        for r in per_cluster[c][taken[c]:]:
            order.append((c, r))
    return order


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.drop_duplicates(subset=["clst", "ref"])
    groups = g.pivot_table(index="clst", columns="split", values="ref",
                           aggfunc="count", fill_value=0)
    rows = df.pivot_table(index="clst", columns="split", values="ID",
                          aggfunc="count", fill_value=0)
    out = groups.join(rows, lsuffix="_groups", rsuffix="_rows")
    return out.sort_index()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group-split", type=Path, required=True,
                   help="group_split_v1.csv, the table this subsets")
    p.add_argument("--sizes", type=int, nargs="+", default=[90, 180, 270],
                   help="training reference soundings per output table")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default="size_{k}_v1",
                   help="output table name, {k} replaced by the size")
    args = p.parse_args()

    df = pd.read_csv(args.group_split, low_memory=False)
    for col in ("clst", "ref", "split", "ID"):
        if col not in df.columns:
            raise KeyError(f"{args.group_split} has no '{col}' column")

    train = df[df["split"] == "train"]
    groups = train.drop_duplicates(subset=["clst", "ref"])[["clst", "ref"]]
    n_train_groups = len(groups)
    logger.info("%s: %d training reference soundings (%d rows), %d val, %d test",
                args.group_split.name, n_train_groups, len(train),
                df[df["split"] == "val"].drop_duplicates(subset=["clst", "ref"]).shape[0],
                df[df["split"] == "test"].drop_duplicates(subset=["clst", "ref"]).shape[0])

    bad = [k for k in args.sizes if k < 1 or k > n_train_groups]
    if bad:
        raise SystemExit(f"--sizes {bad} outside 1..{n_train_groups}")

    order = nested_order(groups, list(args.sizes), args.seed)
    if len(order) != n_train_groups or len(set(order)) != n_train_groups:
        raise SystemExit(f"the ordering covers {len(set(order))} of {n_train_groups} "
                         "training reference soundings; this is a bug, not a config error")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    key = list(zip(df["clst"].astype(int), df["ref"].astype(int)))
    prev: set | None = None
    for k in sorted(args.sizes):
        keep = set(order[:k])
        if prev is not None and not prev.issubset(keep):
            raise SystemExit("nesting broken: a smaller size is not a subset of a larger")
        prev = keep
        out = df.copy()
        is_train = (out["split"] == "train").to_numpy()
        drawn = np.array([kk in keep for kk in key])
        out.loc[is_train & ~drawn, "split"] = "unused"
        name = args.name.format(k=k)
        path = args.out_dir / f"{name}.csv"
        out.to_csv(path, index=False)

        table = summarize(out)
        counts = out["split"].value_counts().to_dict()
        n_groups = {s: int(out[out["split"] == s]
                           .drop_duplicates(subset=["clst", "ref"]).shape[0])
                    for s in counts}
        (args.out_dir / f"{name}.summary.json").write_text(json.dumps({
            "source": str(args.group_split),
            "size_reference_soundings": int(k),
            "seed": args.seed,
            "nested_in": [args.name.format(k=j) for j in sorted(args.sizes) if j > k],
            "rows": {s: int(v) for s, v in counts.items()},
            "reference_soundings": n_groups,
            "per_cluster": json.loads(table.to_json(orient="index")),
            "drawn": [[int(c), int(r)] for c, r in order[:k]],
        }, indent=2))
        logger.info("%s: train %d refs / %d rows, unused %d refs, val %d, test %d -> %s",
                    name, n_groups.get("train", 0), counts.get("train", 0),
                    n_groups.get("unused", 0), n_groups.get("val", 0),
                    n_groups.get("test", 0), path.name)
        print(table.to_string())


if __name__ == "__main__":
    main()

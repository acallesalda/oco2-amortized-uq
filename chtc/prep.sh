#!/bin/bash
# Root node of oco2.dag: build the reference-sounding-disjoint split and the statistics
# estimated from its training soundings only.
#
# CPU only, ~10 minutes, almost all of it untarring 43k .npy files. It runs once and every
# training job then consumes its output, which is the whole point of doing it here rather
# than six times over: all six models must see the *same* split, and compute_split_stats
# must see only its training rows or the standardization leaks held-out soundings back in
# (which is exactly the leakage this split exists to avoid).
#
# Produces: oco2_split_v1.tar.gz
#     scratch/simulated/files/splits/group_split_v1.csv
#     scratch/simulated/files/splits/group_split_v1.summary.json
#     experiments/pipelines/stats/group_split_v1/{profile_mu,profile_sigma}.npy
#     experiments/pipelines/stats/group_split_v1/{xco2_stats,provenance}.json
set -euo pipefail

JOB="oco2_split_v1"

# shellcheck source=job_common.sh
source ./job_common.sh

oco2_banner "$JOB (group split + statistics)"
oco2_trap_pack "$JOB"      # before anything that can fail; see the note in train.sh
oco2_setup
oco2_unpack_data

SPLIT_CSV="$FILES_DIR/splits/group_split_v1.csv"

echo "=============================================================="
echo " 1/2  make_group_split.py  (383/48/48 reference soundings)"
echo "=============================================================="
python3 experiments/pipelines/make_group_split.py \
    --files-root "$FILES_DIR" \
    --out        "$SPLIT_CSV" \
    --val-frac   0.10 \
    --test-frac  0.10 \
    --seed       42

echo
echo "=============================================================="
echo " 2/2  compute_split_stats.py  (training rows only)"
echo "=============================================================="
python3 experiments/pipelines/compute_split_stats.py \
    --split-csv "$SPLIT_CSV" \
    --out-dir   "$STATS_DIR/group_split_v1" \
    --split     train

echo
echo "=============================================================="
echo " verification"
echo "=============================================================="
python3 - "$SPLIT_CSV" <<'PY'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
print(f"rows: {len(df)}")
print()
print("replicates per partition:")
print(df["split"].value_counts().to_string())
print()

groups = df.drop_duplicates(subset=["clst", "ref"])
print("reference soundings per partition:")
print(groups["split"].value_counts().to_string())
print()
print("reference soundings per cluster x partition:")
print(pd.crosstab(groups["clst"], groups["split"]).to_string())
print()

# The one invariant the whole revision rests on: no reference sounding may appear in more
# than one partition. If this ever trips, every downstream number is meaningless.
spans = df.groupby(["clst", "ref"])["split"].nunique()
bad = spans[spans > 1]
if len(bad):
    print(f"FATAL: {len(bad)} reference soundings span partitions:")
    print(bad.head(20).to_string())
    sys.exit(1)
print("OK: every reference sounding lives in exactly one partition.")
PY

echo
echo "--- stats written ---"
ls -l "$STATS_DIR/group_split_v1"
cat "$STATS_DIR/group_split_v1/xco2_stats.json"
echo
cat "$STATS_DIR/group_split_v1/provenance.json"
echo
echo "prep complete."

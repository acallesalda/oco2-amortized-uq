#!/bin/bash
# Build ONE learning-curve subset of group_split_v1's training partition, with statistics
# and covariates, so the same pipeline that trained the full split can train it.
#
#     ./prep_size.sh <k>          # k = training reference soundings, e.g. 90
#
# CPU only, a couple of minutes, nearly all of it file transfer. It opens no spectrum: the
# group split table already carries every aux_data column, the statistics come out of that
# table's own true_state_vector columns, and the covariate merge is two CSVs.
#
# Why this table exists
# ---------------------
# A diagnostic of generalization, and of whether 47,900 spectra is
# enough. Both are answered by the same picture: held-out error against the number of
# distinct reference soundings trained on. If it is still falling at 361 scenes the models
# are data-limited and simulating more scenes is the fix; if it has flattened, it is not,
# and the limit is the conditioning or the architecture. Every table in the family holds
# the SAME 45 validation and 45 test scenes, so a curve over k moves one thing.
#
# See experiments/pipelines/make_size_splits.py for the subsampling and why the sets are
# nested.
#
# Produces: oco2_size_<k>_v1.tar.gz
#     scratch/simulated/files/splits/size_<k>_v1.csv
#     scratch/simulated/files/splits/size_<k>_v1.summary.json
#     scratch/simulated/files/splits/size_<k>_v1_cov.csv
#     experiments/pipelines/stats/size_<k>_v1/{profile_mu,profile_sigma}.npy
#     experiments/pipelines/stats/size_<k>_v1/{xco2_stats,provenance}.json
#     experiments/pipelines/stats/size_<k>_v1/covariate_stats.json
set -euo pipefail

K="${1:?usage: prep_size.sh <k>   (training reference soundings, e.g. 90)}"
case "$K" in
    ''|*[!0-9]*) echo "ERROR: k must be a positive integer, got '$K'." >&2; exit 2 ;;
esac

JOB="oco2_size_${K}_v1"

# The split this job builds. Set here rather than in the submit file's `environment`
# because it is derived from the one argument the job takes.
export OCO2_SPLIT="size_${K}_v1"

# shellcheck source=job_common.sh
source ./job_common.sh

export OCO2_PACK_PATHS="scratch/simulated/files/splits/${OCO2_SPLIT}.csv scratch/simulated/files/splits/${OCO2_SPLIT}.summary.json scratch/simulated/files/splits/${OCO2_SPLIT}_cov.csv experiments/pipelines/stats/${OCO2_SPLIT} chtc/GIT_PROVENANCE.txt"

oco2_banner "$JOB ($K training reference soundings + statistics + covariates)"
oco2_trap_pack "$JOB"
oco2_setup

# NOT oco2_unpack_split: that would look for the archive this job produces. The group
# split's two archives are taken apart by hand instead, and only the two files that are
# actually inputs are extracted -- pulling the rest would put group-split statistics into
# this job's output directory, where the next job to unpack them would read them as this
# split's.
for tarball in oco2_split_v1.tar.gz oco2_covariates_v1.tar.gz; do
    if [ ! -f "$tarball" ]; then
        echo "ERROR: $tarball was not transferred. It is an input to this job:"
        echo "         oco2_split_v1.tar.gz      the group split table (prep.sub)"
        echo "         oco2_covariates_v1.tar.gz covariates_v1.csv.gz (covariates.sub)"
        exit 1
    fi
done

echo "--- taking group_split_v1.csv out of the split archive ---"
tar -xzf oco2_split_v1.tar.gz -C "$PROJECT" \
    scratch/simulated/files/splits/group_split_v1.csv
rm -f oco2_split_v1.tar.gz

echo "--- taking covariates_v1.csv.gz out of the covariate archive ---"
tar -xzf oco2_covariates_v1.tar.gz -C "$PROJECT" \
    scratch/simulated/files/covariates_v1.csv.gz
rm -f oco2_covariates_v1.tar.gz

GROUP_SPLIT="$FILES_DIR/splits/group_split_v1.csv"
COV="$FILES_DIR/covariates_v1.csv.gz"
RAW_SPLIT="$FILES_DIR/splits/${OCO2_SPLIT}.csv"
for f in "$GROUP_SPLIT" "$COV"; do
    if [ ! -s "$f" ]; then
        echo "ERROR: $f is missing or empty after extraction."
        exit 1
    fi
    echo "    $(basename "$f"): $(stat -c %s "$f") bytes"
done
echo

echo "=============================================================="
echo " 1/3  make_size_splits.py  ($K training scenes; val/test untouched)"
echo "=============================================================="
python3 experiments/pipelines/make_size_splits.py \
    --group-split "$GROUP_SPLIT" \
    --sizes       "$K" \
    --out-dir     "$FILES_DIR/splits"

echo
echo "=============================================================="
echo " 2/3  compute_split_stats.py  (training rows only)"
echo "=============================================================="
# Re-estimated, not copied. These models are fitted on a fraction of the scenes
# group_split_v1's constants were computed from; reusing them would standardize with a
# mean estimated from data this model never sees -- exactly the leakage to avoid,
# in the direction that flatters the small-data end of the curve.
python3 experiments/pipelines/compute_split_stats.py \
    --split-csv "$RAW_SPLIT" \
    --out-dir   "$SPLIT_STATS" \
    --split     train

echo
echo "=============================================================="
echo " 3/3  augment_split.py  (merge covariates, encode from train)"
echo "=============================================================="
python3 experiments/pipelines/augment_split.py \
    --split-csv  "$RAW_SPLIT" \
    --covariates "$COV" \
    --out        "$SPLIT_CSV" \
    --stats-dir  "$SPLIT_STATS"

echo
echo "=============================================================="
echo " verification"
echo "=============================================================="
# Four things, none of them visible in the numbers that come out later. Validation or test
# scenes that are not group_split_v1's would make this point of the curve incomparable
# with the others; a training scene that is also a test scene would make the whole curve
# meaningless; and a training partition of the wrong size would put the point at the wrong
# place on the x axis. All four train fine and produce plausible numbers.
python3 - "$SPLIT_CSV" "$GROUP_SPLIT" "$K" <<'PY'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1], low_memory=False)
old = pd.read_csv(sys.argv[2], low_memory=False)
k = int(sys.argv[3])
print(f"rows: {len(df)}")
print()
print(df["split"].value_counts().to_string())
print()


def scenes(frame, split):
    s = frame[frame["split"] == split]
    return set(map(tuple, s[["clst", "ref"]].drop_duplicates().to_numpy()))


n_train = len(scenes(df, "train"))
if n_train != k:
    raise SystemExit(f"FATAL: {n_train} training reference soundings, expected {k}.")
print(f"OK: {n_train} training reference soundings, "
      f"{len(scenes(df, 'unused'))} held out of training as 'unused'.")

for part in ("val", "test"):
    if scenes(df, part) != scenes(old, part):
        raise SystemExit(f"FATAL: the {part} scenes are not group_split_v1's. Every point "
                         "of the learning curve has to be scored on the same soundings.")
print("OK: validation and test are group_split_v1's 45 + 45 scenes, unchanged.")

if scenes(df, "train") & (scenes(df, "val") | scenes(df, "test")):
    raise SystemExit("FATAL: a reference sounding is in training and in val/test.")
if len(df) != len(old):
    raise SystemExit(f"FATAL: {len(old)} rows in, {len(df)} out. Nothing here drops rows.")
if df["ID"].duplicated().any():
    raise SystemExit("FATAL: duplicate sounding IDs in the table.")
print(f"OK: all {len(df)} February soundings accounted for.")

if "XCO2Prior" not in df.columns:
    raise SystemExit("FATAL: XCO2Prior is not in the augmented table, so the scene_prior "
                     "preset would train on something else.")
cov = [c for c in df.columns if c.endswith("_fix")] + [c for c in ("XCO2Prior",
       "AODPrior1", "AODPrior2", "Latitude", "Longitude") if c in df.columns]
print(f"covariate columns present: {len(cov)}  {sorted(cov)}")
PY

echo
echo "--- stats written ---"
ls -l "$SPLIT_STATS"
cat "$SPLIT_STATS/xco2_stats.json"
echo
cat "$SPLIT_STATS/provenance.json"
echo

echo "=============================================================="
echo " $JOB finished"
echo "=============================================================="
ls -lh "$FILES_DIR/splits" "$SPLIT_STATS"

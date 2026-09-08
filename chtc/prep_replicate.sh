#!/bin/bash
# Build the ORIGINAL replicate-level partition as a split table, with statistics and
# covariates, so the same pipeline that trained the group split can train it.
#
#     condor_submit prep_replicate.sub
#
# CPU only, a couple of minutes, most of it file transfer: it takes the annotations out
# of the two dataset archives and never extracts a spectrum.
#
# Why the rerun needs this
# ------------------------
# The revision's main table has two columns: reference-disjoint and within-reference. As
# submitted they differ in three things at once -- the split, the auxiliary covariates
# and the hyperparameter
# search. A reader cannot attribute the difference to any of them. Training the old split
# with the new recipe closes that: the two columns then differ in the partition alone.
#
# The expensive half of the covariate work does not repeat. extract_covariates.py reads 451
# HDF5 files and writes covariates_v1.csv.gz keyed by sounding ID, which is split-
# independent -- so this job takes that file from the existing covariates archive and only
# re-runs the merge, which is seconds. That is also why this produces ONE archive where the
# group split has two: there is no second job to separate out.
#
# Produces: oco2_replicate_split_v1.tar.gz
#     scratch/simulated/files/splits/replicate_split_v1.csv
#     scratch/simulated/files/splits/replicate_split_v1.summary.json
#     scratch/simulated/files/splits/replicate_split_v1_cov.csv
#     experiments/pipelines/stats/replicate_split_v1/{profile_mu,profile_sigma}.npy
#     experiments/pipelines/stats/replicate_split_v1/{xco2_stats,provenance}.json
#     experiments/pipelines/stats/replicate_split_v1/covariate_stats.json
set -euo pipefail

JOB="oco2_replicate_split_v1"

# The split this job builds. Set here rather than in the submit file's `environment`
# because this is the one job whose split is not a choice -- it exists to make this table.
export OCO2_SPLIT=replicate_split_v1

# shellcheck source=job_common.sh
source ./job_common.sh

export OCO2_PACK_PATHS="scratch/simulated/files/splits/replicate_split_v1.csv scratch/simulated/files/splits/replicate_split_v1.summary.json scratch/simulated/files/splits/replicate_split_v1_cov.csv experiments/pipelines/stats/replicate_split_v1 chtc/GIT_PROVENANCE.txt"

oco2_banner "$JOB (replicate split + statistics + covariates)"
oco2_trap_pack "$JOB"
oco2_setup
# Annotations only. Every step below is CSV work -- the split table comes out of
# aux_data.csv, the statistics out of that table's own true_state_vector columns, the
# covariate merge out of two CSVs -- so extracting 43k spectra would be ten minutes and
# 40 GB of scratch spent on files nothing here opens.
oco2_unpack_annotations

# The covariate table, from the group split's archive. Only covariates_v1.csv.gz is wanted
# -- the rest of that archive is the group split's own augmented table and statistics, and
# extracting them here would put a group-split file in this job's output.
if [ ! -f oco2_covariates_v1.tar.gz ]; then
    echo "ERROR: oco2_covariates_v1.tar.gz was not transferred. It is the output of"
    echo "       covariates.sub and holds covariates_v1.csv.gz, which is keyed by"
    echo "       sounding ID and does not depend on the split."
    exit 1
fi
echo "--- taking covariates_v1.csv.gz from the covariate archive ---"
tar -xzf oco2_covariates_v1.tar.gz -C "$PROJECT" \
    scratch/simulated/files/covariates_v1.csv.gz
rm -f oco2_covariates_v1.tar.gz

COV="$FILES_DIR/covariates_v1.csv.gz"
RAW_SPLIT="$FILES_DIR/splits/replicate_split_v1.csv"
if [ ! -s "$COV" ]; then
    echo "ERROR: $COV is missing or empty after extraction."
    exit 1
fi
echo "    $(stat -c %s "$COV") bytes"
echo

echo "=============================================================="
echo " 1/3  make_replicate_split.py"
echo "=============================================================="
# No --val-frac or --seed on the command line: the defaults are the experiment, and a
# value typed here would be a second place for it to live.
python3 experiments/pipelines/make_replicate_split.py \
    --files-root "$FILES_DIR" \
    --out        "$RAW_SPLIT"

echo
echo "=============================================================="
echo " 2/3  compute_split_stats.py  (training rows only)"
echo "=============================================================="
# Under this split every reference sounding is in training anyway, so restricting the
# statistics to training rows is not what stops a leak -- it is what stops the two splits
# from differing in one more thing than the partition.
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
# Three things, and the second is the one that matters: this table must NOT be
# group-disjoint. make_replicate_split.py asserts it internally, but the check is repeated
# here against the file that was actually written, because what the training jobs read is
# the file and not the script.
python3 - "$SPLIT_CSV" <<'PY'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1], low_memory=False)
print(f"rows: {len(df)}")
print()
print(df["split"].value_counts().to_string())
print()

per_ref = df.groupby(["clst", "ref"])["split"].nunique()
shared = int((per_ref > 1).sum())
print(f"reference soundings: {len(per_ref)}")
print(f"  spanning more than one partition: {shared}")
if shared == 0:
    raise SystemExit("FATAL: this table is group-disjoint. It is supposed to be the "
                     "replicate-level split, in which every reference sounding appears "
                     "in all three partitions.")

print()
print("test partition == the published test directory: "
      f"{bool(((df['split'] == 'test') == (df['source_dir'] == 'test')).all())}")

cov = [c for c in df.columns if c.endswith("_fix")] + [c for c in ("XCO2Prior",
       "AODPrior1", "AODPrior2", "Latitude", "Longitude") if c in df.columns]
print(f"covariate columns present: {len(cov)}  {sorted(cov)}")
if "XCO2Prior" not in df.columns:
    raise SystemExit("FATAL: XCO2Prior is not in the augmented table, so the scene_prior "
                     "preset would train on something else.")
PY

echo
echo "=============================================================="
echo " $JOB finished"
echo "=============================================================="
ls -lh "$FILES_DIR/splits" "$SPLIT_STATS"

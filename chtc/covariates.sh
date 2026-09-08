#!/bin/bash
# Recover the auxiliary covariates from the raw soundings and merge them into the group
# split table. CPU only, ~20 minutes, dominated by reading 479 HDF5 files.
#
#     condor_submit covariates.sub
#
# Produces oco2_covariates_v1.tar.gz in /staging:
#     scratch/simulated/files/covariates_v1.csv.gz
#     scratch/simulated/files/covariates_v1.inventory.csv
#     scratch/simulated/files/covariates_v1.summary.json
#     scratch/simulated/files/splits/group_split_v1_cov.csv
#     experiments/pipelines/stats/group_split_v1/covariate_stats.json
#
# It does NOT rewrite the .npy spectra and does NOT change the split: the same 383/48/48
# reference soundings stay in the same partitions, so a covariate run and the baseline
# differ in exactly one thing. That is the whole reason this is a separate small job
# rather than a re-run of reprocess.sh.
#
# Depends on prep.sub having produced oco2_split_v1.tar.gz, and on
# `./stage_data.sh raw` having put the 2.2 GB of .h5 files in /staging.
set -euo pipefail

JOB="oco2_covariates_v1"

# shellcheck source=job_common.sh
source ./job_common.sh

oco2_banner "$JOB (covariate recovery + augmented split)"
# The trap builds oco2_covariates_v1.tar.gz, which is also the archive the submit file
# declares as its output -- so OCO2_PACK_PATHS has to name exactly what belongs in it, and
# building a second tarball by hand at the end would be overwritten by the trap on exit.
# The 2.2 GB of raw .h5 and the 43k .npy stay on the execute node.
export OCO2_PACK_PATHS="scratch/simulated/files/metadata_pairing.json scratch/simulated/files/covariates_v1.csv.gz scratch/simulated/files/covariates_v1.inventory.csv scratch/simulated/files/covariates_v1.summary.json scratch/simulated/files/splits/group_split_v1_cov.csv experiments/pipelines/stats/group_split_v1/covariate_stats.json chtc/GIT_PROVENANCE.txt"
oco2_trap_pack "$JOB"
oco2_setup
oco2_unpack_data
oco2_unpack_split

echo "--- unpacking raw soundings (2.2 GB) ---"
RAW_ROOT="$PROJECT/data/oco2/simulated"
mkdir -p "$RAW_ROOT"
tar -xzf oco2_unprocessed_og_data.tar.gz --strip-components=3 -C "$RAW_ROOT"
rm -f oco2_unprocessed_og_data.tar.gz
echo "    clusters : $(find "$RAW_ROOT/og_data" -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "    .h5 files: $(find "$RAW_ROOT/og_data" -name '*.h5' | wc -l)"
echo

COV="$FILES_DIR/covariates_v1.csv.gz"
SPLIT_CSV="$FILES_DIR/splits/group_split_v1.csv"
SPLIT_COV="$FILES_DIR/splits/group_split_v1_cov.csv"
STATS="$STATS_DIR/group_split_v1"

echo "=============================================================="
echo " 1/2  extract_covariates.py   (.h5 -> covariates_v1.csv.gz)"
echo "=============================================================="
# --metadata-from is load-bearing, not bookkeeping. oco2_utils.get_cluster pairs the i-th
# .h5 file in *directory order* with the i-th row of the cluster's metadata .csv, and
# directory order is neither sorted nor stable across filesystems: re-running the
# extraction from the archive reproduced a different pairing from the published one, on
# 97.6 % of SlrZen values. So the scene metadata is taken from the published aux_data.csv
# by ID, which is by construction what the models were trained with. Everything read out
# of the .h5 -- airmass, the filter variables, the retrieved state, the radiances -- is
# keyed by the filename and was never in question.
#
# The cluster-.csv pairing is still read and reported against the published one, because
# the disagreement is itself a finding. diagnose_metadata_pairing.py, below, is what
# decides which of the two is physically consistent.
python3 experiments/pipelines/extract_covariates.py \
    --raw-root "$RAW_ROOT/og_data" \
    --out      "$COV" \
    --metadata-from "$FILES_DIR/train/aux_data.csv" \
    --metadata-from "$FILES_DIR/test/aux_data.csv"

echo
echo "=============================================================="
echo " 1b/2  diagnose_metadata_pairing.py   (which pairing is real?)"
echo "=============================================================="
# Never fatal: it is a diagnostic and the covariate table does not depend on its answer.
# But it is the thing to read first in this log.
python3 experiments/pipelines/diagnose_metadata_pairing.py \
    --raw-root "$RAW_ROOT/og_data" \
    --aux "$FILES_DIR/train/aux_data.csv" \
    --aux "$FILES_DIR/test/aux_data.csv" \
    --out "$FILES_DIR/metadata_pairing.json" || \
    echo "WARNING: the pairing diagnostic failed; the covariate table is unaffected."

echo
echo "=============================================================="
echo " 2/2  augment_split.py        (merge + training-only statistics)"
echo "=============================================================="
python3 experiments/pipelines/augment_split.py \
    --split-csv  "$SPLIT_CSV" \
    --covariates "$COV" \
    --out        "$SPLIT_COV" \
    --stats-dir  "$STATS"

echo
echo "=============================================================="
echo " verification"
echo "=============================================================="
python3 - "$SPLIT_CSV" "$SPLIT_COV" "$STATS/covariate_stats.json" <<'PY'
import json, sys
import pandas as pd

base = pd.read_csv(sys.argv[1])
aug = pd.read_csv(sys.argv[2])
stats = json.loads(open(sys.argv[3]).read())

ok = True
if len(base) != len(aug):
    print(f"FATAL: augmented split has {len(aug)} rows, base has {len(base)}")
    ok = False
if not base["ID"].equals(aug["ID"]):
    print("FATAL: row order or ID set changed")
    ok = False
if not base["split"].equals(aug["split"]):
    print("FATAL: partition assignment changed -- this must be the SAME split")
    ok = False

print(f"rows: {len(aug)}   columns: {base.shape[1]} -> {aug.shape[1]}")
print(f"covariates carried: {len(stats['columns'])}")
n_missing = aug[stats["columns"]].isna().sum()
bad = n_missing[n_missing > 0]
if len(bad):
    print("columns with missing values (imputed at the training median at load time):")
    print(bad.to_string())

# The recovered quantities that the manuscript claims as inputs. If either of these is
# constant or absent the recovery did not work, and the covariate runs would silently be
# the baseline with extra zeros.
for col in ("vza_deg", "prior_psfc_pa", "airmass"):
    if col not in aug.columns:
        print(f"FATAL: {col} is missing from the augmented split")
        ok = False
        continue
    v = aug[col].dropna()
    print(f"  {col:15s} n={len(v)}  min={v.min():.4f}  max={v.max():.4f}  sd={v.std():.4f}")
    if len(v) == 0 or v.std() == 0:
        print(f"FATAL: {col} carries no information")
        ok = False

# A per-scene covariate must be constant within a reference sounding. If one is not, it is
# a retrieval output that has been mislabelled as prior information.
for col in ("vza_deg", "prior_psfc_pa", "airmass", "Latitude", "Longitude"):
    if col in aug.columns:
        n = aug.groupby(["clst", "ref"])[col].nunique().max()
        print(f"  {col:15s} distinct values within one reference sounding: {n}")

sys.exit(0 if ok else 1)
PY

echo
echo "--- inventory (which covariates are available when) ---"
column -s, -t < "$FILES_DIR/covariates_v1.inventory.csv" 2>/dev/null \
    || cat "$FILES_DIR/covariates_v1.inventory.csv"

echo
echo "covariates complete. The exit trap writes oco2_covariates_v1.tar.gz from"
echo "OCO2_PACK_PATHS and lists its contents below."

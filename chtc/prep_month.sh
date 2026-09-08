#!/bin/bash
# Turn one raw simulation month into everything predict.py needs to score it.
#
#     ./prep_month.sh <month>          # 202003 | 202007
#
# CPU only, roughly 40 minutes: one pass over ~450 HDF5 files for the spectra and targets,
# a second for the covariates. It produces the test half of the temporal generalization
# experiment; no model is trained and none is loaded.
#
# Input, from /staging:
#   OCO2SimUQExpt_<month>.zip     the raw month, as delivered
#   oco2_processed_train.tar.gz   one February .npy, to check the channel layout against
#   and February's split, which is one archive or two depending on OCO2_SPLIT:
#     group_split_v1  oco2_split_v1.tar.gz + oco2_covariates_v1.tar.gz
#     full_feb_v1     oco2_full_feb_v1.tar.gz
#
# Output: oco2_month_<month><suffix>.tar.gz
#   scratch/simulated/files/<month>/{<ID>.npy, <ID>_cov.npy, aux_data.csv}
#   scratch/simulated/files/covariates_<month>.csv.gz
#   scratch/simulated/files/splits/month_<month><suffix>_split.csv  (+ .summary.json)
#
# The split table is February's train/val rows with this month appended as `test`. That is
# what keeps the Laplace refit on February -- see make_month_split.py, which explains why
# scoring a March sounding against a Laplace fitted on March would not be the same model.
# WHICH February's train/val rows follows OCO2_SPLIT, and that is the whole difference
# between prep_month.sub and prep_month_full.sub: the spectra are identical, the table is
# not. Under full_feb_v1 the Laplace is refitted over 406 training scenes instead of 361,
# which is the same set the model was fitted on -- as it is for the group split. Both
# tables also carry February's 45 validation scenes, which is what predict.py
# calibrates the Laplace's aleatoric diagonal on.
set -euo pipefail

MONTH="${1:?usage: prep_month.sh <month>   e.g. 202003}"
case "$MONTH" in
    2020[0-9][0-9]) ;;
    *) echo "ERROR: '$MONTH' does not look like a YYYYMM month."; exit 2 ;;
esac

# shellcheck source=job_common.sh
source ./job_common.sh

# Which February the month is appended to. OCO2_SPLIT comes from the submit file's
# `environment`, and everything split-dependent follows it:
#
#   ""       group_split_v1  February's train/val rows, 361 + 45 reference soundings
#   "_full"  full_feb_v1     406 + the same 45 -- prep_full_feb.sub
#
# The suffix is on the archive and the table because the two are different objects that
# would otherwise share one name, and OSDF refuses to overwrite an object: a collision is
# a held job with a transfer error that says nothing about the cause.
SUFFIX="$(oco2_split_suffix)"
JOB="oco2_month_${MONTH}${SUFFIX}"
MONTH_SPLIT_NAME="month_${MONTH}${SUFFIX}_split"

export OCO2_PACK_PATHS="scratch/simulated/files/${MONTH} scratch/simulated/files/covariates_${MONTH}.csv.gz scratch/simulated/files/covariates_${MONTH}.summary.json scratch/simulated/files/splits/${MONTH_SPLIT_NAME}.csv scratch/simulated/files/splits/${MONTH_SPLIT_NAME}.summary.json chtc/GIT_PROVENANCE.txt"

oco2_banner "$JOB (raw month -> spectra, covariates, split table)"
oco2_trap_pack "$JOB"
oco2_setup
oco2_unpack_data
oco2_unpack_split
oco2_unpack_covariates

RAW="$PROJECT/data/oco2/simulated/${MONTH}"
ZIP="OCO2SimUQExpt_${MONTH}.zip"
if [ ! -f "$ZIP" ]; then
    echo "ERROR: $ZIP was not transferred. Put it in /staging first:"
    echo "         cp OCO2SimUQExpt_${MONTH}.zip \$(OSDF-equivalent staging path)"
    exit 1
fi
echo "--- unpacking $ZIP ($(stat -c %s "$ZIP") bytes) ---"
mkdir -p "$RAW"
# `unzip` is not in the image: oco2-gpu.def installs python3, git, curl, procps and two
# graphics libraries, and the CUDA base does not carry one. Without this fallback the
# job dies here, two minutes after transferring 2.1 GB, and leaves a failure archive on
# OSDF to clean before the retry. Rebuilding the image to add one binary is 30 minutes
# and a re-staged SIF that every running job would pick up; zipfile ships with python3.
if command -v unzip >/dev/null 2>&1; then
    unzip -q "$ZIP" -d "$RAW"
else
    echo "    no unzip in the image; extracting with python3 -m zipfile"
    python3 -m zipfile -e "$ZIP" "$RAW"
fi
rm -f "$ZIP"

# The archive is flat: ClusterNN/ directories and the per-cluster metadata .csv at top
# level. prepare_month.py and extract_covariates.py both want the .csv *inside* the
# cluster directory it belongs to, which is how the February tree is laid out, so the
# loose ones are moved in.
moved=0
for csv in "$RAW"/*clst*_refsdgs_*.csv; do
    [ -e "$csv" ] || continue
    n=$(basename "$csv" | sed -n 's/.*clst0*\([0-9][0-9]*\)_refsdgs.*/\1/p')
    dest=$(printf "%s/Cluster%02d" "$RAW" "$n")
    if [ -d "$dest" ]; then mv "$csv" "$dest/"; moved=$((moved + 1)); fi
done
echo "    clusters : $(find "$RAW" -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "    .h5 files: $(find "$RAW" -name '*.h5' | wc -l)"
echo "    metadata .csv moved into their cluster directories: $moved"
if [ "$moved" -eq 0 ]; then
    echo "ERROR: no cluster metadata .csv was placed. Without it the covariates cannot be"
    echo "       recovered and the scene_prior preset would train on nothing."
    exit 2
fi
echo

# Any February spectrum. Its channel count is what prepare_month.py checks this month's
# against -- a different input width would not crash, it would produce a number.
#
# `-print -quit`, NOT `| head -1`. head closes the pipe after the first line, find takes
# SIGPIPE, `set -o pipefail` turns that into 141 and `set -e` kills the job -- which is
# exactly how both month preps died on 2026-08-25, 36 seconds in, with an empty .err and a
# 512-byte archive. There are 72,048 files under train/, so find is still writing long
# after head has gone; the race is not close. oco2_setup carries the same scar (see the
# note on `grep | head` there).
#
# -quit also stops at the first match instead of walking all 72k entries, so the correct
# version is the fast one.
REF_NPY=$(find "$FILES_DIR/train" -name '*.npy' ! -name '*_cov.npy' -print -quit)
if [ -z "$REF_NPY" ]; then
    echo "ERROR: no February .npy under $FILES_DIR/train to check the channel layout."
    exit 2
fi

echo "=============================================================="
echo " 1/3  prepare_month.py   (.h5 -> .npy spectra, covariances, aux_data.csv)"
echo "=============================================================="
python3 experiments/pipelines/prepare_month.py \
    --raw-root      "$RAW" \
    --month         "$MONTH" \
    --files-root    "$FILES_DIR" \
    --reference-npy "$REF_NPY"

echo
echo "=============================================================="
echo " 2/3  extract_covariates.py   (.h5 -> covariates_${MONTH}.csv.gz)"
echo "=============================================================="
# No --metadata-from: there is no published aux_data.csv for this month, so the corrected
# sorted-reference pairing is the only source. The airmass feasibility line in the output
# is the check that it is the right one -- it must read 100% feasible, as it does for
# February. If it does not, stop and do not score this month.
python3 experiments/pipelines/extract_covariates.py \
    --raw-root "$RAW" \
    --out      "$FILES_DIR/covariates_${MONTH}.csv.gz" \
    --pairing  sorted_ref

echo
echo "=============================================================="
echo " 3/3  make_month_split.py   (February train/val + this month as test)"
echo "=============================================================="
python3 experiments/pipelines/make_month_split.py \
    --month            "$MONTH" \
    --february         "$SPLIT_CSV" \
    --month-aux        "$FILES_DIR/${MONTH}/aux_data.csv" \
    --month-covariates "$FILES_DIR/covariates_${MONTH}.csv.gz" \
    --out              "$FILES_DIR/splits/${MONTH_SPLIT_NAME}.csv"

echo
echo "=============================================================="
echo " $JOB finished"
echo "=============================================================="
echo "spectra: $(find "$FILES_DIR/$MONTH" -name '*.npy' ! -name '*_cov.npy' | wc -l)"
echo "covariances: $(find "$FILES_DIR/$MONTH" -name '*_cov.npy' | wc -l)"
ls -lh "$FILES_DIR/splits/${MONTH_SPLIT_NAME}.csv"

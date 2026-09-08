#!/bin/bash
# Score a February-trained model on a DIFFERENT MONTH. No training of any kind.
#
#     ./eval_month.sh <task> <month> <covariates> [tag]
#
#   task        xco2 | profiles | last4 | last4_marginal
#   month       202003 | 202007
#   covariates  the preset both checkpoints were trained with (scene_prior)
#   tag         names the artifact and the archive; default `<month><split suffix>`.
#
# This is the temporal generalization experiment. The test partition is
# every sounding of another month, and the statistics -- target standardization, covariate
# centring and scaling, categorical level maps, imputation medians -- are February's,
# unchanged. A March sounding is therefore presented in exactly the coordinates the network
# was trained in, which is what makes the number a measure of transfer rather than of
# re-normalization.
#
# WHICH February model is scored follows OCO2_SPLIT, and that is the only difference
# between eval_month.sub and eval_month_full.sub:
#
#   group_split_v1  the search winners trained on 361 reference soundings, the models
#                   the published tables report. February's own test scenes were held out.
#   full_feb_v1     the same searched configurations retrained on 406 -- February's
#                   test scenes folded back into training, since March and July are
#                   the test set now. The 45 validation scenes stay held out, which is
#                   what keeps the Laplace's aleatoric calibration out of sample.
#                   train_full.sub.
#
# The Laplace approximation is refitted at scoring time, as it always is, over the TRAINING
# partition of the split table -- which month_<month><suffix>_split.csv defines as that
# split's February training soundings, the same rows the network was fitted on. Refitting
# it on the new month would be fitting to the test set.
#
# Two rows to read before anything else, both in the output of score.py at the end:
#
#   ACOS   does not train, so its row measures how hard this month is, not how well
#          anything generalized. February: 2.91 ppm XCO2 RMSE. March and July are 3.08 and
#          3.52 in the raw files, so expect the learned models to lose *something* even if
#          they transfer perfectly.
#   n      must be ~45,000, not ~4,500. If it is the latter the split table's test
#          partition is February's, and the job scored the wrong thing.
set -euo pipefail

TASK="${1:?usage: eval_month.sh <task> <month> <covariates> [tag]}"
MONTH="${2:?missing month}"
COVARIATES="${3:?missing covariate preset}"

# shellcheck source=job_common.sh
source ./job_common.sh
# shellcheck source=bo_studies.sh
source ./bo_studies.sh

# Which February the models were trained on. OCO2_SPLIT comes from the submit file's
# `environment`; every split-dependent name below derives from it rather than being typed,
# so a job cannot unpack one split's checkpoints and score them against another's table.
#
#   ""       group_split_v1  the search winners, trained on 361 reference soundings
#   "_full"  full_feb_v1     the same configurations on 406 -- train_full.sub
SUFFIX="$(oco2_split_suffix)"
TAG="${4:-${MONTH}${SUFFIX}}"

JOB="eval_${TASK}_${TAG}"

export OCO2_PACK_PATHS="experiments/scoring/results experiments/scoring/tables chtc/GIT_PROVENANCE.txt"

oco2_banner "$JOB (February model, ${MONTH} data)"
oco2_trap_pack "$JOB"
oco2_gpu_check

case "$TASK" in
    xco2|profiles|last4|last4_marginal) ;;
    *) echo "ERROR: unknown task '$TASK'. Valid: xco2 profiles last4 last4_marginal"
       exit 2 ;;
esac
case "$MONTH" in
    2020[0-9][0-9]) ;;
    *) echo "ERROR: '$MONTH' does not look like a YYYYMM month."; exit 2 ;;
esac

read -r DET_STUDY FLOW_STUDY <<EOF
$(bo_task_studies "$TASK")
EOF

oco2_setup
oco2_unpack_data
oco2_unpack_split
oco2_unpack_covariates
oco2_unpack_results "${DET_STUDY}_final${SUFFIX}.tar.gz" "${FLOW_STUDY}_final${SUFFIX}.tar.gz"

# The month's spectra and its split table, from prep_month.sub.
#
# OCO2_MONTH_TAR overrides the name. It exists because OSDF caches by URL and treats
# objects as immutable: a month archive that was deleted and rebuilt at the same path
# is served stale to execute nodes -- as a checksum mismatch if you are lucky, and as
# the previous run's 512-byte failure stub if you are not, which tar reports as
# "Unexpected EOF in archive" while gzip -t at the origin says the file is fine.
# The fix is a name that has never been requested; the submit file passes it in here.
MONTH_TAR="${OCO2_MONTH_TAR:-oco2_month_${MONTH}${SUFFIX}.tar.gz}"
if [ ! -f "$MONTH_TAR" ]; then
    echo "ERROR: $MONTH_TAR was not transferred. Run prep_month.sub for $MONTH first."
    exit 1
fi
echo "--- unpacking $MONTH_TAR ---"
tar -xzf "$MONTH_TAR" -C "$PROJECT"
rm -f "$MONTH_TAR"

MONTH_SPLIT="$FILES_DIR/splits/month_${MONTH}${SUFFIX}_split.csv"
for f in "$MONTH_SPLIT" "$FILES_DIR/$MONTH/aux_data.csv"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f is missing after extraction."
        exit 2
    fi
done
echo "    split table: $(wc -l < "$MONTH_SPLIT") rows"
echo "    spectra    : $(find "$FILES_DIR/$MONTH" -name '*.npy' ! -name '*_cov.npy' | wc -l)"
echo

# The checkpoints, from the same table that trained them. February's weights tree, because
# February is what these models were trained on -- nothing about this job changes that.
bo_study_spec "$DET_STUDY"
DET_CKPT="$WEIGHTS_ROOT/$(bo_final_dir "$DET_STUDY")/$BO_CKPT"
DET_FALLBACK="$WEIGHTS_ROOT/$(bo_final_dir "$DET_STUDY")/$(basename "$BO_CKPT")"
bo_study_spec "$FLOW_STUDY"
FLOW_CKPT="$WEIGHTS_ROOT/$(bo_final_dir "$FLOW_STUDY")/$BO_CKPT"
if [ ! -s "$DET_CKPT" ] && [ -s "$DET_FALLBACK" ]; then
    DET_CKPT="$DET_FALLBACK"
fi
for pair in "$DET_STUDY:$DET_CKPT" "$FLOW_STUDY:$FLOW_CKPT"; do
    who="${pair%%:*}"; path="${pair#*:}"
    if [ ! -s "$path" ]; then
        echo "ERROR: no checkpoint for $who at"
        echo "         $path"
        find "$WEIGHTS_ROOT" -name '*.pt' -printf '         %p\n' 2>/dev/null | sort
        exit 2
    fi
done

OUT_DIR="$PROJECT/experiments/scoring/results/month_${TAG}"
mkdir -p "$OUT_DIR"

echo "=============================================================="
echo " temporal hold-out: $TASK on $MONTH"
echo "   trained on   : February ($OCO2_SPLIT), covariates=$COVARIATES"
echo "   deterministic: $DET_CKPT"
echo "   flow         : $FLOW_CKPT"
echo "   split table  : $MONTH_SPLIT"
echo "   statistics   : $SPLIT_STATS   (February's -- NOT recomputed)"
echo "=============================================================="

python3 experiments/scoring/predict.py \
    --task          "$TASK" \
    --files-root    "$FILES_DIR" \
    --split-csv     "$MONTH_SPLIT" \
    --stats-dir     "$SPLIT_STATS" \
    --weights-root  "$WEIGHTS_ROOT" \
    --covariates    "$COVARIATES" \
    --det-ckpt      "$DET_CKPT" \
    --flow-ckpt     "$FLOW_CKPT" \
    --nf-samples    "${NF_SAMPLES:-512}" \
    --out           "$OUT_DIR/${TASK}.npz"

python3 experiments/scoring/score.py "$OUT_DIR/${TASK}.npz" \
    --out "$PROJECT/experiments/scoring/tables" || \
    echo "WARNING: score.py failed; the artifact is still packed and can be scored locally."

echo
echo "=============================================================="
echo " $JOB finished"
echo "=============================================================="
ls -lh "$OUT_DIR"

#!/bin/bash
# Spectral attribution for one task: which wavelengths the retrieval actually reads.
#
#     ./attribution.sh <task> <covariates> [tag]
#
#   task        xco2 | profiles | last4
#   covariates  the preset both checkpoints were trained with, exactly as bo_eval.sh
#   tag         names the archive; default `attr`.
#
# Optional environment variables, set from the submit file:
#
#   ATTR_WINDOW      channels per occlusion window          (default 32)
#   ATTR_STRIDE      window stride                          (default = window)
#   ATTR_NF_SAMPLES  flow draws for the band-level point estimate (default 256, 0 skips)
#   ATTR_SKIP_SWEEP  1 to run only the band-level occlusion and the saliency
#   ATTR_EXTRA       anything else to pass through to the script
#
# Why this is a separate job from bo_eval.sh
# ------------------------------------------
# It reads the same checkpoints and the same held-out partition, but it writes no
# predictive artifact -- only the tables in experiments/scoring/tables -- and it makes tens
# of forward passes rather than one, because every occlusion window is a pass. Folding it
# into bo_eval.sh would make the job that produces Tables 1-3 several times longer for a
# diagnostic that has nothing to do with them.
#
# It reads the TEST partition, like bo_eval.sh, and for the same reason: the question is
# what the network reads on soundings it has never seen. Nothing is selected on it here --
# the script fits nothing at all, it only re-scores fixed checkpoints under substitution.
set -euo pipefail

TASK="${1:?usage: attribution.sh <task> <covariates> [tag]}"
COVARIATES="${2:?missing covariate preset}"
TAG="${3:-attr}"

JOB="attr_${TASK}_${TAG}"

# shellcheck source=job_common.sh
source ./job_common.sh
# shellcheck source=bo_studies.sh
source ./bo_studies.sh

export OCO2_PACK_PATHS="experiments/scoring/tables latex/figures chtc/GIT_PROVENANCE.txt"

oco2_banner "$JOB"
oco2_trap_pack "$JOB"
oco2_gpu_check

case "$TASK" in
    xco2|profiles|last4) ;;
    *) echo "ERROR: unknown task '$TASK'. Valid: xco2 profiles last4"
       exit 2 ;;
esac

read -r DET_STUDY FLOW_STUDY <<EOF
$(bo_task_studies "$TASK")
EOF
SFX="$(oco2_split_suffix)"

BAL=""
if [ "${OCO2_BALANCED:-0}" = "1" ]; then
    BAL="_bal"
    echo "--- attributing the CLUSTER-BALANCED retrains (OCO2_BALANCED=1) ---"
fi
DET_TAR="${DET_STUDY}_final${BAL}${SFX}.tar.gz"
FLOW_TAR="${FLOW_STUDY}_final${BAL}${SFX}.tar.gz"

oco2_setup
oco2_unpack_data
oco2_unpack_split
oco2_unpack_covariates
oco2_unpack_results "$DET_TAR" "$FLOW_TAR"

# The same two paths bo_eval.sh computes, from the same table, for the same reason: a wrong
# weights directory here produces a plausible curve instead of an error.
bo_study_spec "$DET_STUDY"
DET_CKPT="$WEIGHTS_ROOT/$(bo_final_dir "$DET_STUDY")$BAL/$BO_CKPT"
DET_FALLBACK="$WEIGHTS_ROOT/$(bo_final_dir "$DET_STUDY")$BAL/$(basename "$BO_CKPT")"
bo_study_spec "$FLOW_STUDY"
FLOW_CKPT="$WEIGHTS_ROOT/$(bo_final_dir "$FLOW_STUDY")$BAL/$BO_CKPT"

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

STATS="$SPLIT_STATS"

SWEEP_ARGS=()
if [ "${ATTR_SKIP_SWEEP:-0}" = "1" ]; then
    SWEEP_ARGS=(--skip-sweep)
    echo "--- ATTR_SKIP_SWEEP=1: band-level occlusion and saliency only ---"
fi
STRIDE_ARGS=()
if [ -n "${ATTR_STRIDE:-}" ]; then
    STRIDE_ARGS=(--stride "$ATTR_STRIDE")
fi

echo "=============================================================="
echo " spectral attribution: $TASK  (split=$OCO2_SPLIT, covariates=$COVARIATES,"
echo "                               tag=$TAG)"
echo "   deterministic : $DET_STUDY"
echo "                   $DET_CKPT"
echo "   flow          : $FLOW_STUDY"
echo "                   $FLOW_CKPT"
echo "   split         : $SPLIT_CSV"
echo "   window        : ${ATTR_WINDOW:-32} channels"
echo "=============================================================="

# shellcheck disable=SC2086  # ATTR_EXTRA is deliberately word-split
python3 experiments/scoring/spectral_attribution.py \
    --task          "$TASK" \
    --files-root    "$FILES_DIR" \
    --split-csv     "$SPLIT_CSV" \
    --stats-dir     "$STATS" \
    --weights-root  "$WEIGHTS_ROOT" \
    --covariates    "$COVARIATES" \
    --det-ckpt      "$DET_CKPT" \
    --flow-ckpt     "$FLOW_CKPT" \
    --window        "${ATTR_WINDOW:-32}" \
    --nf-samples    "${ATTR_NF_SAMPLES:-256}" \
    ${STRIDE_ARGS[@]+"${STRIDE_ARGS[@]}"} \
    ${SWEEP_ARGS[@]+"${SWEEP_ARGS[@]}"} \
    ${ATTR_EXTRA:-} \
    --out           "$PROJECT/experiments/scoring/tables" \
    --fig           "$PROJECT/latex/figures"

echo
echo "=============================================================="
echo " $JOB finished"
echo "=============================================================="
find "$PROJECT/experiments/scoring/tables" -name 'attribution_*' -print 2>/dev/null | sort || true
find "$PROJECT/latex/figures" -name 'spectral_attribution_*' -print 2>/dev/null | sort || true

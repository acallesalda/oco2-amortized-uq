#!/bin/bash
# Score one covariate-trained model on the held-out partition.
#
#     ./eval_cov.sh <task> <det_archive> <flow_archive|none> <covariates> <det_dir> [tag]
#
# The sibling of eval.sh, and a separate file for the same reason train_cov.sub is
# separate from train.sub: eval.sh scores the six published-architecture models and must
# stay runnable exactly as it is. This one scores a model whose weights directory is named
# after something chosen at run time -- a covariate preset, and for the Bayesian
# optimization's winner a `_best` suffix on top of it -- which is not a path the layout
# table in code/evals/checkpoints.py can know. So the caller names it.
#
#   task        xco2 | profiles | last4 | last4_marginal   (as in eval.sh)
#   det_dir     the weights subdirectory the training job wrote, relative to
#               experiments/weights/group_split_v1/. For a covariate-comparison
#               model that is the preset-named directory train.sh wrote:
#                 xco2      full_regressor_<preset>_best
#                 profiles  full_regressor_profiles_<preset>_ppm_best
#                 last4     full_regressor_profiles_last4_<preset>_ppm_best
#   covariates  the preset the checkpoint was trained with. It must match, or the
#               auxiliary encoder is built with the wrong shapes; predict.py takes the
#               name recorded in the checkpoint's own best_hparams.yaml over this one and
#               warns if they disagree.
#   flow        `none` scores the Laplace and ACOS rows only. The flows are separate
#               models and are not what the deterministic search produced, so this is the
#               ordinary case rather than a degraded one.
#   tag         names the artifact and the archive; default `best`. Two covariate models
#               of the same task differ only by this, so it is what keeps their .npz
#               files apart.
#
# The split table is the AUGMENTED one -- group_split_v1_cov.csv. Same rows as the base
# table, extra columns; the loader needs those columns to build the covariates at all.
set -euo pipefail

TASK="${1:?usage: eval_cov.sh <task> <det_tar> <flow_tar|none> <covariates> <det_dir> [tag]}"
DET_TAR="${2:?missing det archive}"
FLOW_TAR="${3:-none}"
COVARIATES="${4:?missing covariate preset}"
DET_DIR="${5:?missing weights subdirectory}"
TAG="${6:-best}"

# shellcheck source=job_common.sh
source ./job_common.sh

JOB="eval_${TASK}_${TAG}"

export OCO2_PACK_PATHS="experiments/scoring/results experiments/scoring/tables chtc/GIT_PROVENANCE.txt"

oco2_banner "$JOB"
oco2_trap_pack "$JOB"
oco2_gpu_check
oco2_setup
oco2_unpack_data
oco2_unpack_split
oco2_unpack_covariates

case "$TASK" in
    xco2|profiles|last4|last4_marginal) ;;
    *) echo "ERROR: unknown task '$TASK'. Valid: xco2 profiles last4 last4_marginal"
       exit 2 ;;
esac

# A plain string, not an array: an empty array expanded under `set -u` is an error on
# bash before 4.4, and one fixed flag needs no quoting to survive word splitting.
SKIP_FLOW=""
if [ "$FLOW_TAR" = "none" ] || [ -z "$FLOW_TAR" ]; then
    SKIP_FLOW="--skip-flow"
    oco2_unpack_results "$DET_TAR"
else
    oco2_unpack_results "$DET_TAR" "$FLOW_TAR"
fi

SPLIT_CSV="$FILES_DIR/splits/group_split_v1_cov.csv"
STATS="$STATS_DIR/group_split_v1"
OUT_DIR="$PROJECT/experiments/scoring/results/group_split_v1"
mkdir -p "$OUT_DIR"

# Find the checkpoint inside the directory we were given. train.py writes into
# global_best/; train_profiles.py writes there AND leaves a copy one level up, so both
# layouts are checked rather than assumed -- and a missing file stops the job here, before
# the Laplace refit, rather than a hundred lines later.
DET_CKPT=""
for cand in "$WEIGHTS_ROOT/$DET_DIR/global_best/net.pt" \
            "$WEIGHTS_ROOT/$DET_DIR/net.pt" \
            "$WEIGHTS_ROOT/$DET_DIR/global_best/net_profiles.pt" \
            "$WEIGHTS_ROOT/$DET_DIR/net_profiles.pt"; do
    if [ -s "$cand" ]; then DET_CKPT="$cand"; break; fi
done
if [ -z "$DET_CKPT" ]; then
    echo "ERROR: no checkpoint under $WEIGHTS_ROOT/$DET_DIR."
    echo "       The archive $DET_TAR unpacked, but not to the directory this job was"
    echo "       told to look in. What it did contain:"
    find "$WEIGHTS_ROOT" -name '*.pt' -printf '         %p\n' 2>/dev/null | sort
    echo
    echo "       det_dir must be the weights subdirectory the training job wrote --"
    echo "       A search winner is not scored here: chtc/bo_eval.sh does that,"
    echo "       and computes both paths from chtc/bo_studies.sh."
    exit 2
fi

echo "=============================================================="
echo " scoring: $TASK  (covariates=$COVARIATES, tag=$TAG)"
echo "   checkpoint : $DET_CKPT"
echo "   split      : $SPLIT_CSV"
if [ -n "$SKIP_FLOW" ]; then
    echo "   flow       : skipped -- Laplace and ACOS rows only"
else
    echo "   flow       : $FLOW_TAR"
fi
echo "=============================================================="

python3 experiments/scoring/predict.py \
    --task          "$TASK" \
    --files-root    "$FILES_DIR" \
    --split-csv     "$SPLIT_CSV" \
    --stats-dir     "$STATS" \
    --weights-root  "$WEIGHTS_ROOT" \
    --covariates    "$COVARIATES" \
    --det-ckpt      "$DET_CKPT" \
    --nf-samples    "${NF_SAMPLES:-512}" \
    ${SKIP_FLOW} \
    --out           "$OUT_DIR/${TASK}_${TAG}.npz"

python3 experiments/scoring/score.py "$OUT_DIR/${TASK}_${TAG}.npz" \
    --out "$PROJECT/experiments/scoring/tables" || \
    echo "WARNING: score.py failed; the artifact is still packed and can be scored locally."

echo
echo "=============================================================="
echo " $JOB finished"
echo "=============================================================="
ls -lh "$OUT_DIR"

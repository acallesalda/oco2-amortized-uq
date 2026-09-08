#!/bin/bash
# Score the PUBLISHED checkpoints on the PUBLISHED replicate-level split.
#
#     ./eval_pub.sh <task>
#
# task is xco2, profiles, last4, last4_marginal, last1 or last1_marginal -- the same set
# eval.sh takes, and the same code path, which is the entire point of this job.
#
# Why it exists
# -------------
# The principal results are on reference-sounding-disjoint test data. We have
# those (eval.sh / bo_eval.sh). What we do not have is a within-reference number produced
# by the same scorer: Tables 1-3 of the submitted manuscript came from four notebooks with
# their own conventions -- NLL in standardized coordinates, coverage averaged over levels,
# validation drawn from the training pool by a generator that lives in one of them. Put
# those numbers next to `group_split_v1` and the difference between the two splits is
# confounded with the difference between the notebooks and `experiments/scoring/`.
#
# So: same checkpoints as the paper, same test soundings as the paper, but predict.py and
# score.py rather than the notebooks. The artifact lands in results/replicate/ and
# score.py merges it with the group-split one:
#
#     python3 experiments/scoring/score.py \
#         experiments/scoring/results/replicate/xco2.npz \
#         experiments/scoring/results/group_split_v1/xco2_bo.npz \
#         --label replicate='within-reference' \
#         --label group_split_v1_cov='reference-disjoint' --out tables/
#
# What it needs that eval.sh does not
# -----------------------------------
# The published checkpoints are not in oco2_code.tar.gz (make_code_tarball.sh drops every
# .pt) and they are not a training result either. They come from the Zenodo weights
# archive, which has to be staged once:
#
#     ./stage_data.sh weights          # ~320 MB, on the transfer node
#
# and they sit at experiments/weights/, one level above the group_split_v1/ tree every
# other scoring job points at -- hence the WEIGHTS_ROOT override below.
#
# Standardization constants are deliberately the published ones: no --stats-dir, no
# --split-csv. Under the replicate split every reference sounding is in the training pool
# anyway, so the published constants *are* that split's training statistics; recomputing
# them would change the numbers we are trying to reproduce.
set -euo pipefail

TASK="${1:?usage: eval_pub.sh <task>}"

# shellcheck source=job_common.sh
source ./job_common.sh

JOB="eval_${TASK}_replicate"

export OCO2_PACK_PATHS="experiments/scoring/results experiments/scoring/tables chtc/GIT_PROVENANCE.txt"

oco2_banner "$JOB"
oco2_trap_pack "$JOB"
oco2_gpu_check
oco2_setup
oco2_unpack_data
oco2_unpack_weights

# The published tree, not the group-split one. oco2_setup pointed WEIGHTS_ROOT at
# experiments/weights/group_split_v1/; oco2_unpack_weights writes one level up.
WEIGHTS_ROOT="$PROJECT/experiments/weights"
export WEIGHTS_ROOT

OUT_DIR="$PROJECT/experiments/scoring/results/replicate"
NF_ARCH_DIR="$PROJECT/experiments/scoring/nf_arch"
mkdir -p "$OUT_DIR"

# Per-task extras. Two published checkpoints cannot describe themselves and the loader
# refuses to guess, so the architecture is named here:
#
#   net_profiles_old.pt  is the legacy deterministic profile architecture
#   the flows            record layers/hidden/bins but not the spline clamps, and the
#                        scalar one records nothing at all
#
# Anything not listed is left to the checkpoint's own record. If a load fails with a shape
# error, that is the honest outcome -- do not add a guessed override to make it pass.
EXTRA=()
DET_CKPT=""
FLOW_CKPT=""
case "$TASK" in
    xco2)
        DET_CKPT="$WEIGHTS_ROOT/full_regressor/net.pt"
        FLOW_CKPT="$WEIGHTS_ROOT/flow_regressor_nflow/best_flow_nflows.pt"
        # !! The values in this file are transcribed from xco2.ipynb and are not
        # independently verified; the file says so at the top. Confirm against the state
        # dict with experiments/dump_checkpoint_config.py before quoting the row.
        EXTRA+=(--nf-arch "$NF_ARCH_DIR/published_nf_xco2.yaml") ;;
    profiles|last4_marginal|last1_marginal)
        DET_CKPT="$WEIGHTS_ROOT/full_regressor_profiles/global_best/net_profiles.pt"
        FLOW_CKPT="$WEIGHTS_ROOT/flow_regressor_profiles/best_flow_profiles.pt"
        EXTRA+=(--nf-arch "$NF_ARCH_DIR/published_nf_profiles.yaml") ;;
    last4)
        DET_CKPT="$WEIGHTS_ROOT/full_regressor_profiles_last4_bo/net_profiles.pt"
        FLOW_CKPT="$WEIGHTS_ROOT/flow_regressor_profiles_nf4/best_nf4.pt" ;;
    last1)
        DET_CKPT="$WEIGHTS_ROOT/full_regressor_profiles_last1/net_profiles.pt"
        FLOW_CKPT="$WEIGHTS_ROOT/flow_regressor_nflow_surface/best_flow_nflows.pt" ;;
    *)  echo "ERROR: unknown task '$TASK'."
        echo "       Valid: xco2 profiles last4 last4_marginal last1 last1_marginal"
        exit 2 ;;
esac

# The archive's layout is not the repository's, and three of the seven paths differ from
# the obvious guess. Fall back rather than fail: LAYOUT in
# code/evals/checkpoints.py knows the alternatives, so an unresolved path here just means
# "let the loader look", not "the checkpoint is missing".
if [ ! -s "$DET_CKPT" ]; then
    echo "note: $DET_CKPT is not in the archive; leaving the path to checkpoints.LAYOUT."
    case "$TASK" in
        profiles|last4_marginal|last1_marginal)
            alt="$WEIGHTS_ROOT/full_regressor_profiles/global_best/net_profiles_old.pt"
            if [ -s "$alt" ]; then
                # The *old* file is the legacy architecture. It is what full_profiles.ipynb
                # loaded, so it is the checkpoint behind the published Table 2, and it will
                # not load into the current class without this flag.
                echo "      using $alt (legacy architecture)"
                DET_CKPT="$alt"
                EXTRA+=(--legacy-backbone)
            else
                DET_CKPT=""
            fi ;;
        xco2)
            alt="$WEIGHTS_ROOT/full_regressor/global_best/net.pt"
            [ -s "$alt" ] && DET_CKPT="$alt" || DET_CKPT="" ;;
        *)  DET_CKPT="" ;;
    esac
fi
[ -n "$DET_CKPT" ] && EXTRA+=(--det-ckpt "$DET_CKPT")
if [ -s "$FLOW_CKPT" ]; then
    EXTRA+=(--flow-ckpt "$FLOW_CKPT")
else
    echo "note: $FLOW_CKPT is not in the archive; leaving the path to checkpoints.LAYOUT."
fi

echo "=============================================================="
echo " scoring: $TASK  (published checkpoints, replicate-level split)"
echo " weights-root: $WEIGHTS_ROOT"
echo " extras:       ${EXTRA[*]:-none}"
echo "=============================================================="

# No --split-csv: predict.py then rebuilds the published partition, including the
# validation draw (torch.Generator seeded with --seed) the notebooks used, and stamps the
# artifact `split=replicate`. No --stats-dir: the published constants, for the reason in
# the header. --covariates legacy: the four auxiliary inputs the published models were
# trained with, in the published (mis-paired) assignment. Changing it here
# would score a checkpoint against inputs it never saw.
python3 experiments/scoring/predict.py \
    --task          "$TASK" \
    --files-root    "$FILES_DIR" \
    --weights-root  "$WEIGHTS_ROOT" \
    --covariates    legacy \
    --nf-samples    "${NF_SAMPLES:-512}" \
    "${EXTRA[@]}" \
    --out           "$OUT_DIR/${TASK}.npz"

python3 experiments/scoring/score.py "$OUT_DIR/${TASK}.npz" \
    --label replicate="within-reference (published models)" \
    --out "$PROJECT/experiments/scoring/tables" || \
    echo "WARNING: score.py failed; the artifact is still packed and can be scored locally."

echo
echo "=============================================================="
echo " $JOB finished"
echo "=============================================================="
ls -lh "$OUT_DIR"

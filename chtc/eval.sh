#!/bin/bash
# Score one task on the held-out partition.
#
#     ./eval.sh <task> <det_archive> <flow_archive>
#
# task is xco2, profiles, last4, last4_marginal, last1 or last1_marginal; the two
# archives are the training results the task needs, transferred in from /staging by
# eval.sub.
#
# last1 is the scalar surface task: level 20 alone, one number per sounding, against
# ACOS's own level-20 posterior variance. last1_marginal is its marginalized twin, the
# 20-level models sliced at level 20.
#
# last4_marginal re-scores the *20-level* models on the near-surface levels, which is the
# "marginalized" half of the published Table 3. It therefore takes the det_profiles and
# nf_profiles archives, not the last4 ones, and it writes last4_marginal.npz -- score.py
# merges its rows into table_last4 alongside the native ones.
#
# What comes back is one .npz per task -- every predictive quantity for every test
# sounding -- and nothing else. The tables and the figures are made from it afterwards,
# on a laptop, by experiments/scoring/score.py and figures.py. Splitting the two is the
# whole reason this job exists: the expensive part (a Laplace refit and 512 flow draws
# per sounding) runs once on a GPU, and the part that gets re-run twenty times while a
# caption is being argued about runs offline in seconds.
set -euo pipefail

TASK="${1:?usage: eval.sh <task> <det_archive> <flow_archive>}"
DET_TAR="${2:-}"
FLOW_TAR="${3:-}"

# shellcheck source=job_common.sh
source ./job_common.sh

JOB="eval_${TASK}"

# The result archive is a couple of hundred MB of samples, not 235 MB of checkpoints we
# were handed as input. See the note on OCO2_PACK_PATHS in job_common.sh.
export OCO2_PACK_PATHS="experiments/scoring/results experiments/scoring/tables chtc/GIT_PROVENANCE.txt"

oco2_banner "$JOB"
oco2_trap_pack "$JOB"
oco2_gpu_check
oco2_setup
oco2_unpack_data
oco2_unpack_split
oco2_unpack_results "$DET_TAR" "$FLOW_TAR"

SPLIT_CSV="$FILES_DIR/splits/group_split_v1.csv"
STATS="$STATS_DIR/group_split_v1"
OUT_DIR="$PROJECT/experiments/scoring/results/group_split_v1"
mkdir -p "$OUT_DIR"

# Both models must be present. predict.py can skip either one (--skip-laplace /
# --skip-flow) but an archive that silently contains only two of the three methods is
# indistinguishable from a complete one once it is off the cluster, and would quietly
# become a table with a missing row.
# The candidate lists mirror code/evals/checkpoints.py, which is what actually loads
# these files. Where it accepts two layouts, so does this gate.
case "$TASK" in
    xco2)     oco2_require_ckpt_any "$WEIGHTS_ROOT/full_regressor/global_best/net.pt" \
                                    "$WEIGHTS_ROOT/full_regressor/net.pt"
              oco2_require_ckpt "$WEIGHTS_ROOT/flow_regressor_nflow/best_flow_nflows.pt" ;;
    profiles) oco2_require_ckpt_any "$WEIGHTS_ROOT/full_regressor_profiles/net_profiles.pt" \
                                    "$WEIGHTS_ROOT/full_regressor_profiles/global_best/net_profiles.pt"
              oco2_require_ckpt "$WEIGHTS_ROOT/flow_regressor_profiles/best_flow_profiles.pt" ;;
    last4)    oco2_require_ckpt_any "$WEIGHTS_ROOT/full_regressor_profiles_last4/net_profiles.pt" \
                                    "$WEIGHTS_ROOT/full_regressor_profiles_last4/global_best/net_profiles.pt"
              oco2_require_ckpt "$WEIGHTS_ROOT/flow_regressor_profiles_nf4/best_nf4.pt" ;;
    last4_marginal)
              # The 20-level models, deliberately: this task marginalizes them.
              oco2_require_ckpt_any "$WEIGHTS_ROOT/full_regressor_profiles/net_profiles.pt" \
                                    "$WEIGHTS_ROOT/full_regressor_profiles/global_best/net_profiles.pt"
              oco2_require_ckpt "$WEIGHTS_ROOT/flow_regressor_profiles/best_flow_profiles.pt" ;;
    last1)    oco2_require_ckpt_any "$WEIGHTS_ROOT/full_regressor_profiles_last1/net_profiles.pt" \
                                    "$WEIGHTS_ROOT/full_regressor_profiles_last1/global_best/net_profiles.pt"
              oco2_require_ckpt "$WEIGHTS_ROOT/flow_regressor_nflow_surface/best_flow_nflows.pt" ;;
    last1_marginal)
              # As last4_marginal, one level further in: the 20-level models scored on
              # level 20 alone, so it takes the profiles archives.
              oco2_require_ckpt_any "$WEIGHTS_ROOT/full_regressor_profiles/net_profiles.pt" \
                                    "$WEIGHTS_ROOT/full_regressor_profiles/global_best/net_profiles.pt"
              oco2_require_ckpt "$WEIGHTS_ROOT/flow_regressor_profiles/best_flow_profiles.pt" ;;
    *)        echo "ERROR: unknown task '$TASK'."
              echo "       Valid: xco2 profiles last4 last4_marginal last1 last1_marginal"
              exit 2 ;;
esac

echo "=============================================================="
echo " scoring: $TASK  (reference-sounding-disjoint split)"
echo "=============================================================="

python3 experiments/scoring/predict.py \
    --task          "$TASK" \
    --files-root    "$FILES_DIR" \
    --split-csv     "$SPLIT_CSV" \
    --stats-dir     "$STATS" \
    --weights-root  "$WEIGHTS_ROOT" \
    --nf-samples    "${NF_SAMPLES:-512}" \
    --out           "$OUT_DIR/${TASK}.npz"

# The tables are cheap and are regenerated locally anyway, but producing them here means
# the .err file carries the headline numbers. That is often all you have when a run
# finishes at three in the morning and you want to know whether it is worth fetching.
python3 experiments/scoring/score.py "$OUT_DIR/${TASK}.npz" \
    --out "$PROJECT/experiments/scoring/tables" || \
    echo "WARNING: score.py failed; the artifact is still packed and can be scored locally."

echo
echo "=============================================================="
echo " $JOB finished"
echo "=============================================================="
ls -lh "$OUT_DIR"

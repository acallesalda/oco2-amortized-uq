#!/bin/bash
# Submit the moment-matched Gaussian control for one task, or for all four.
#
#     ./chtc/gauss_submit.sh                 # xco2 profiles last4 last4_marginal
#     ./chtc/gauss_submit.sh profiles
#     ./chtc/gauss_submit.sh --dry-run       # print the condor_submit lines and stop
#
# Runs on the ACCESS POINT, not in a job. It exists so that the two study names and the
# sample budget are derived rather than typed: chtc/bo_studies.sh already knows which two
# checkpoints belong to a task, and a mismatched pair passed by hand does not fail -- it
# scores a different model and reports a plausible number.
#
# What comes back, per task, in eval_<task>_gauss.tar.gz:
#
#   experiments/scoring/results/group_split_v1/<task>_gauss.npz
#       everything eval_<task>_bo.npz holds, plus nfg_* : the flow's own mean, covariance,
#       marginal PIT, skewness and excess kurtosis, and the NLL of the truth under the
#       Gaussian matching its first two moments, at each sample budget in the schedule.
#
# Then, locally and in seconds:
#
#   python3 experiments/scoring/nongauss.py \
#       experiments/scoring/results/group_split_v1/*_gauss.npz \
#       --split-csv group_split_v1_cov.csv --out experiments/scoring/tables/
set -euo pipefail

cd "$(dirname "$0")"

# shellcheck source=bo_studies.sh
source ./bo_studies.sh

DRY=0
TASKS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY=1 ;;
        -*) echo "ERROR: unknown flag '$arg'. Only --dry-run." >&2; exit 2 ;;
        *)  TASKS+=("$arg") ;;
    esac
done
if [ "${#TASKS[@]}" -eq 0 ]; then
    TASKS=(xco2 profiles last4 last4_marginal)
fi

# Streamed samples per sounding, per task. The three that matter are set by the dimension:
# the plug-in penalty a moment-matched Gaussian pays for estimating its own covariance is
# d(d+3)/(4S) nats, which at S=512 is 0.002 at d=1 and 0.22 at d=20 -- the latter is the
# same size as the effect being measured, so the 20-level task is the one that needs the
# samples. xco2 gets more of them because at d=1 they are nearly free, and a longer lever
# arm makes the 1/S extrapolation better conditioned.
#
# These are budgets, not requirements. The number that goes in the paper is the intercept
# of the fitted a + b/S, so halving them costs precision in the fit and nothing else.
gauss_samples() {
    case "$1" in
        xco2)                     echo 16384 ;;
        last4|last4_marginal)     echo  8192 ;;
        profiles)                 echo  8192 ;;
        *) echo "ERROR: unknown task '$1'." >&2; return 2 ;;
    esac
}

# Samples per draw. The context/backbone forward is recomputed on every call to
# flow.sample, so this is the knob that decides how many times the encoder runs: at
# chunk=1024 and 8192 samples it runs eight times per batch instead of once, which is a
# few percent. At chunk=64 it would run 128 times and dominate the job.
CHUNK="${GAUSS_CHUNK:-1024}"

mkdir -p logs

for task in "${TASKS[@]}"; do
    read -r det_study flow_study <<EOF
$(bo_task_studies "$task")
EOF
    samples=$(gauss_samples "$task")

    cmd=(condor_submit gauss.sub
         -a "task = $task"
         -a "det_study = $det_study"
         -a "flow_study = $flow_study"
         -a "covariates = $BO_COVARIATES_DEFAULT"
         -a "samples = $samples"
         -a "chunk = $CHUNK")

    printf '%s\n' "--- $task: $det_study + $flow_study, $samples samples/sounding"
    if [ "$DRY" -eq 1 ]; then
        printf '    %q ' "${cmd[@]}"; echo
    else
        "${cmd[@]}"
    fi
done

if [ "$DRY" -eq 0 ]; then
    echo
    echo "watch with:  ./chtc/status.py   or   condor_q"
    echo "collect  :  ./chtc/fetch_results.sh --gauss"
fi

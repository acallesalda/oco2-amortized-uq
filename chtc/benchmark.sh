#!/bin/bash
# Test-time cost of all three tasks, on ONE GPU, in one job.
#
#     ./benchmark.sh [covariates] [tasks...]
#
#   covariates  the preset the checkpoints were trained with (default scene_prior)
#   tasks       default: xco2 profiles last4
#
# The timing table has to state the hardware, batch size, precision, warm-up and repetitions behind
# Figure 8, and for the timings not to be presented as a same-hardware comparison against
# the operational retrieval. `experiments/scoring/benchmark.py` records all of the first
# and refuses to write an ACOS row at all; this script's job is to make sure every number
# in the table comes off the same card in the same job.
#
# That is the whole reason the three tasks are looped here rather than submitted as three
# jobs. On a shared pool three jobs are three different GPUs, and a table whose rows were
# measured on an A100, an L40 and a V100 is not a table -- the columns would be comparing
# hardware. One job, six checkpoints, nine rows, one `benchmark_provenance.json` naming
# the device they all came from.
#
# Not timed, and recorded as such: reading the simulated files, building the covariate
# table, loading checkpoints, and fitting the Laplace approximation (which is offline and
# happens once, before any sounding is retrieved).
set -euo pipefail

COVARIATES="${1:-scene_prior}"
shift || true
TASKS=("$@")
if [ "${#TASKS[@]}" -eq 0 ]; then
    TASKS=(xco2 profiles last4)
fi

JOB="benchmark"

# shellcheck source=job_common.sh
source ./job_common.sh
# shellcheck source=bo_studies.sh
source ./bo_studies.sh

# The tables directory is the whole output: three CSV rows per task per batch size, one
# provenance JSON, one .tex. Kilobytes, not the hundreds of megabytes an eval job returns.
export OCO2_PACK_PATHS="experiments/scoring/tables chtc/GIT_PROVENANCE.txt"

oco2_banner "$JOB"
oco2_trap_pack "$JOB"
oco2_gpu_check
oco2_setup
oco2_unpack_data
oco2_unpack_split
oco2_unpack_covariates

SFX="$(oco2_split_suffix)"

# Every archive the requested tasks need, unpacked in one go. Duplicates are harmless --
# last4_marginal and profiles share a pair -- but the list is deduplicated anyway so the
# log says once what arrived.
ARCHIVES=()
for task in "${TASKS[@]}"; do
    read -r DET_STUDY FLOW_STUDY <<EOF
$(bo_task_studies "$task")
EOF
    for study in "$DET_STUDY" "$FLOW_STUDY"; do
        tar_name="${study}_final${SFX}.tar.gz"
        case " ${ARCHIVES[*]-} " in
            *" $tar_name "*) ;;
            *) ARCHIVES+=("$tar_name") ;;
        esac
    done
done
oco2_unpack_results "${ARCHIVES[@]}"

OUT_DIR="$PROJECT/experiments/scoring/tables"
mkdir -p "$OUT_DIR"

# benchmark.py appends, so a stale CSV from a previous attempt in the same sandbox would
# silently mix two devices into one table -- exactly the failure the single-job design
# exists to prevent.
rm -f "$OUT_DIR/benchmark.csv"

echo "=============================================================="
echo " test-time benchmark: ${TASKS[*]}"
echo "   split      : $OCO2_SPLIT"
echo "   covariates : $COVARIATES"
echo "   batch sizes: ${BATCH_SIZES:-1 32 256}"
echo "=============================================================="

for task in "${TASKS[@]}"; do
    read -r DET_STUDY FLOW_STUDY <<EOF
$(bo_task_studies "$task")
EOF
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

    echo
    echo "--- $task ---"
    echo "    deterministic : $DET_CKPT"
    echo "    flow          : $FLOW_CKPT"

    # shellcheck disable=SC2086
    python3 experiments/scoring/benchmark.py \
        --task          "$task" \
        --files-root    "$FILES_DIR" \
        --split-csv     "$SPLIT_CSV" \
        --stats-dir     "$SPLIT_STATS" \
        --weights-root  "$WEIGHTS_ROOT" \
        --covariates    "$COVARIATES" \
        --det-ckpt      "$DET_CKPT" \
        --flow-ckpt     "$FLOW_CKPT" \
        --batch-sizes   ${BATCH_SIZES:-1 32 256} \
        --batches       "${BENCH_BATCHES:-8}" \
        --warmup        "${BENCH_WARMUP:-1}" \
        --repeats       "${BENCH_REPEATS:-5}" \
        --nf-samples    "${NF_SAMPLES:-512}" \
        --tf32          "${BENCH_TF32:-off}" \
        --out           "$OUT_DIR"
done

echo
echo "=============================================================="
echo " $JOB finished"
echo "=============================================================="
cat "$OUT_DIR/benchmark_provenance.json"
echo
column -t -s, "$OUT_DIR/benchmark.csv" 2>/dev/null || cat "$OUT_DIR/benchmark.csv"

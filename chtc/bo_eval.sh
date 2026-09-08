#!/bin/bash
# Score one task on the HELD-OUT TEST PARTITION, from the two retrained search winners.
#
#     ./bo_eval.sh <task> <covariates> [tag]
#
#   task        xco2 | profiles | last4 | last4_marginal
#   covariates  the preset both checkpoints were trained with. It must match, or the
#               auxiliary encoder is built with the wrong shapes; predict.py prefers the
#               name recorded in the checkpoint's own config and warns if they disagree.
#   tag         names the artifact and the archive; default `bo`.
#
# Three optional environment variables, set from the submit file rather than passed as
# arguments because they change how much work the job does and not what it computes:
#
#   EVAL_PARTITION     which partition to score: test (default), train or val. Anything
#                      but test is a diagnostic -- see the block that reads it below
#   EVAL_REFS          with EVAL_PARTITION, score this many whole reference soundings of
#                      it rather than all of them (default all)
#   NF_SAMPLES         flow samples stored per sounding      (default 512)
#   GAUSSIANIZE        flow samples STREAMED per sounding for the moment-matched Gaussian
#                      control, reduced online and not stored (default 0, i.e. off)
#   GAUSSIANIZE_CHUNK  samples per draw during that stream   (default 1024)
#
# With EVAL_PARTITION unset this is the only place in the whole pipeline where the test
# set is read. The search ranked configurations on validation, the retrain used validation
# to pick its checkpoint, and neither ever saw these soundings -- which is the entire
# reason the number this job produces can go in a table.
#
# Both checkpoint paths are computed from chtc/bo_studies.sh, the same table bo_final.sh
# trained them with, rather than being passed in by hand. A wrong weights directory is the
# one mistake here that produces a plausible number instead of an error.
set -euo pipefail

TASK="${1:?usage: bo_eval.sh <task> <covariates> [tag]}"
COVARIATES="${2:?missing covariate preset}"
TAG="${3:-bo}"

JOB="eval_${TASK}_${TAG}"

# shellcheck source=job_common.sh
source ./job_common.sh
# shellcheck source=bo_studies.sh
source ./bo_studies.sh

export OCO2_PACK_PATHS="experiments/scoring/results experiments/scoring/tables chtc/GIT_PROVENANCE.txt"

oco2_banner "$JOB"
oco2_trap_pack "$JOB"
oco2_gpu_check

case "$TASK" in
    xco2|profiles|last4|last4_marginal) ;;
    *) echo "ERROR: unknown task '$TASK'. Valid: xco2 profiles last4 last4_marginal"
       exit 2 ;;
esac

read -r DET_STUDY FLOW_STUDY <<EOF
$(bo_task_studies "$TASK")
EOF
SFX="$(oco2_split_suffix)"

# OCO2_BALANCED=1 scores the cluster-balanced retrains (train_balanced.sub) instead of the
# published winners. It moves both the archive names and the weights directory together --
# scoring a balanced archive out of an unbalanced weights path would not fail, it would
# report the wrong model's numbers under the balanced label.
BAL=""
if [ "${OCO2_BALANCED:-0}" = "1" ]; then
    BAL="_bal"
    echo "--- scoring the CLUSTER-BALANCED retrains (OCO2_BALANCED=1) ---"
fi
DET_TAR="${DET_STUDY}_final${BAL}${SFX}.tar.gz"
FLOW_TAR="${FLOW_STUDY}_final${BAL}${SFX}.tar.gz"

oco2_setup
oco2_unpack_data
oco2_unpack_split
oco2_unpack_covariates
oco2_unpack_results "$DET_TAR" "$FLOW_TAR"

# Where each of the two models put its checkpoint, from the table that trained them.
bo_study_spec "$DET_STUDY"
DET_CKPT="$WEIGHTS_ROOT/$(bo_final_dir "$DET_STUDY")$BAL/$BO_CKPT"
DET_FALLBACK="$WEIGHTS_ROOT/$(bo_final_dir "$DET_STUDY")$BAL/$(basename "$BO_CKPT")"
bo_study_spec "$FLOW_STUDY"
FLOW_CKPT="$WEIGHTS_ROOT/$(bo_final_dir "$FLOW_STUDY")$BAL/$BO_CKPT"

# train_profiles.py writes global_best/net_profiles.pt AND leaves a copy one level up, so
# both layouts are checked rather than assumed.
if [ ! -s "$DET_CKPT" ] && [ -s "$DET_FALLBACK" ]; then
    DET_CKPT="$DET_FALLBACK"
fi
for pair in "$DET_STUDY:$DET_CKPT" "$FLOW_STUDY:$FLOW_CKPT"; do
    who="${pair%%:*}"; path="${pair#*:}"
    if [ ! -s "$path" ]; then
        echo "ERROR: no checkpoint for $who at"
        echo "         $path"
        echo "       Its archive unpacked, but not to the directory bo_studies.sh says"
        echo "       it writes to. What did arrive:"
        find "$WEIGHTS_ROOT" -name '*.pt' -printf '         %p\n' 2>/dev/null | sort
        exit 2
    fi
done

# All three follow OCO2_SPLIT (oco2_setup exported the first two). The artifact directory
# carrying the split name is what lets score.py print both partitions side by side without
# either run having to know the other exists.
STATS="$SPLIT_STATS"
OUT_DIR="$PROJECT/experiments/scoring/results/$OCO2_SPLIT"
mkdir -p "$OUT_DIR"

# The moment-matched Gaussian control, off unless GAUSSIANIZE is set in the submit file's
# `environment`. It streams far more flow samples than the artifact stores -- the samples
# are reduced online and thrown away -- so it costs GPU time and no disk. Everything else
# about the job is unchanged, which is the point: the Laplace and ACOS rows of a
# GAUSSIANIZE run must reproduce those of the plain one, and if they do not, something
# other than the control has moved.
GAUSS_ARGS=()
if [ "${GAUSSIANIZE:-0}" -gt 0 ]; then
    GAUSS_ARGS=(--gaussianize        "$GAUSSIANIZE"
                --gaussianize-chunk  "${GAUSSIANIZE_CHUNK:-1024}"
                --gaussianize-first  "${GAUSSIANIZE_FIRST:-512}")
fi

# EVAL_PARTITION: which partition to score. Default `test`, i.e. everything above.
#
# `train` scores the rows the two checkpoints were fit on and `val` the rows the search
# ranked configurations on. Neither is a result -- the point of the pair is the difference,
# which is the generalization gap -- and predict.py marks the artifact
# with meta.notes.eval_partition and suffixes its split name (`..._cov+train`) so a table
# cannot mix the two rows up. EVAL_REFS=45 draws 45 whole reference soundings of that
# partition instead of all 361, which gives the in-sample estimate the same number of
# bootstrap groups as the held-out one.
#
# A tag of its own is required for the same reason PANEL_IDS needs one: the artifact would
# otherwise overwrite the run whose number goes in the table.
EVAL_ARGS=()
if [ "${EVAL_PARTITION:-test}" != "test" ]; then
    EVAL_ARGS=(--eval-partition "$EVAL_PARTITION")
    if [ -n "${EVAL_REFS:-}" ]; then
        EVAL_ARGS+=(--eval-subsample-refs "$EVAL_REFS")
    fi
    if [ "$TAG" = "bo" ]; then
        echo "ERROR: EVAL_PARTITION=$EVAL_PARTITION with the default tag 'bo'; the"
        echo "       in-sample artifact would overwrite the held-out run's. Pass a"
        echo "       distinct tag, e.g. 'gap_${EVAL_PARTITION}'." >&2
        exit 2
    fi
    case "$EVAL_PARTITION" in
        resim_*)
            echo "--- scoring $EVAL_PARTITION: the February extension's FRESH realisations"
            echo "    of scenes group_split_v1 already holds out. Held out of training, not"
            echo "    part of the default test partition. Four reference soundings, so it is"
            echo "    comparable with the same four scenes' rows in the test artifact and"
            echo "    NOT with the pooled test number. ---" ;;
        *)
            echo "--- scoring the ${EVAL_PARTITION^^} partition (EVAL_PARTITION=$EVAL_PARTITION,"
            echo "    refs=${EVAL_REFS:-all}): a DIAGNOSTIC, not a result ---" ;;
    esac
fi

# PANEL_IDS: score only the named soundings, for Figure 7's bivariate panel.
#
# The figure needs thousands of flow draws per sounding and the artifact stores 512. At
# NF_SAMPLES=20000 the full profile artifact would be about three gigabytes to serve six
# panels; twelve soundings at 20000 is ten megabytes. Only the flow is run -- the Laplace
# and ACOS moments for those soundings are already in the full artifact on the laptop, and
# figures.py takes them from there and the draws from here.
#
# TAG must differ from the full run's, or the panel artifact lands on its filename. And
# score.py is SKIPPED: pointed at twelve soundings it would happily write a twelve-sounding
# table_profiles.tex over the real one, which is the worst failure available here because
# the file would look entirely normal.

PANEL_ARGS=()
if [ -n "${PANEL_IDS:-}" ]; then
    # Comma-separated, because `environment` in a submit file delimits variables with
    # spaces: a space-separated PANEL_IDS is rejected by condor_submit before the job
    # exists (`Missing '=' after environment variable`). Whitespace is stripped first so a
    # list written with ", " separators works too, and the split is into an array rather
    # than by word splitting so `set -u` and an empty element cannot bite.
    IFS=',' read -r -a PANEL_ID_LIST <<< "${PANEL_IDS//[[:space:]]/}"
    PANEL_ARGS=(--ids "${PANEL_ID_LIST[@]}" --skip-laplace --skip-acos)
    echo "PANEL_IDS: ${#PANEL_ID_LIST[@]} soundings, ${NF_SAMPLES:-512} draws each"
    printf '    %s\n' "${PANEL_ID_LIST[@]}"
    if [ "$TAG" = "bo" ]; then
        echo "ERROR: PANEL_IDS is set with the default tag 'bo'; the panel artifact would"
        echo "       overwrite the full run's. Pass a distinct tag, e.g. 'panel'." >&2
        exit 2
    fi
fi

echo "=============================================================="
echo " scoring on the ${EVAL_PARTITION:-test} partition: $TASK  (split=$OCO2_SPLIT, covariates=$COVARIATES,"
echo "                                 tag=$TAG)"
echo "   deterministic : $DET_STUDY"
echo "                   $DET_CKPT"
echo "   flow          : $FLOW_STUDY"
echo "                   $FLOW_CKPT"
echo "   split         : $SPLIT_CSV"
if [ "${GAUSSIANIZE:-0}" -gt 0 ]; then
echo "   gaussianize   : $GAUSSIANIZE samples/sounding, chunk ${GAUSSIANIZE_CHUNK:-1024}"
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
    --flow-ckpt     "$FLOW_CKPT" \
    --nf-samples    "${NF_SAMPLES:-512}" \
    ${GAUSS_ARGS[@]+"${GAUSS_ARGS[@]}"} \
    ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} \
    ${PANEL_ARGS[@]+"${PANEL_ARGS[@]}"} \
    --out           "$OUT_DIR/${TASK}_${TAG}.npz"

# Where the in-job tables go.
#
# `tables/` is the paper's table directory, and OCO2_PACK_PATHS ships it, so whatever this
# job writes there lands on the checkout when the archive is fetched. That is right for a
# canonical run -- one row per method for the partition the paper reports -- and wrong for
# every diagnostic run: score.py called on a single artifact writes a table_<task>.csv
# containing ONLY that artifact's rows, so fetching a gap or a learning-curve archive would
# quietly replace the two-partition table with a one-partition one. rebuild.sh regenerates
# it, but the window between the fetch and the rebuild is exactly when someone reads a
# number. So a run that is not the canonical one writes into its own subdirectory.
TABLES_OUT="$PROJECT/experiments/scoring/tables"
case "$OCO2_SPLIT" in
    size_*_v1) TABLES_OUT="$TABLES_OUT/$OCO2_SPLIT" ;;
esac
case "${EVAL_PARTITION:-test}" in
    test) ;;
    # `gap_train` / `gap_val` are the two halves of the generalization gap. A resim_*
    # partition is not a gap measurement -- it is the same held-out scenes on a second
    # set of discrepancy draws -- so it gets its own directory rather than a name that
    # would be read as one.
    resim_*) TABLES_OUT="$PROJECT/experiments/scoring/tables/${EVAL_PARTITION}" ;;
    *)       TABLES_OUT="$PROJECT/experiments/scoring/tables/gap_${EVAL_PARTITION}" ;;
esac
mkdir -p "$TABLES_OUT"

# score.py failing is not worth losing the artifact over: the .npz is the expensive thing
# and every table can be rebuilt from it on a laptop in seconds.
if [ -n "${PANEL_IDS:-}" ]; then
    echo "PANEL_IDS set: skipping score.py. A twelve-sounding table is not a table."
else
    python3 experiments/scoring/score.py "$OUT_DIR/${TASK}_${TAG}.npz" \
        --out "$TABLES_OUT" || \
        echo "WARNING: score.py failed; the artifact is still packed and can be scored locally."
fi

echo
echo "=============================================================="
echo " $JOB finished"
echo "=============================================================="
ls -lh "$OUT_DIR"
# `|| true`: this is the last command in the script, and `find` on a directory score.py
# never created returns 1 -- which under `set -e` would fail a scoring job that had
# actually succeeded, and take the DAG node down with it.
find "$TABLES_OUT" -type f -print 2>/dev/null | sort || true

#!/bin/bash
# Retrain one study's winning configuration at the full epoch count.
#
#     ./bo_final.sh <study> <covariates> <epochs> [balance_alpha]
#
# The trials keep no checkpoint -- bo_trial.sh deletes them, because 150 twenty-epoch
# checkpoints nobody will score is four GB on /staging -- so this is the first time the
# searched configuration exists as a model that can be evaluated. It is also the only
# model of the six that the test set ever sees.
#
# The configuration is NOT typed out here. It is read from bo_<study>_best.json, which
# the study's select node wrote to /staging from the 25 trial records; the overrides in
# that file are the same strings the search produced, so the model trained here and the
# configuration searched are the same object by construction rather than by care. What
# this script adds is exactly three things, and they are the three the search deliberately
# held out:
#
#   epochs        2x the trial count. The trials ran short to rank configurations, not to
#                 produce models; the ranking is not the model.
#   weights dir   its own, under bo_final/<study>, so a winner trained with scene_prior
#                 cannot land on top of the fixed-hyperparameter covariate run that shares
#                 the preset.
#   from scratch  for the flows, the switch that keeps the backbone untrained -- repeated
#                 here because the config is shared with the warm-started runs.
#
# A FOURTH, OPTIONAL ARGUMENT turns on cluster-balanced sampling: training draws are
# weighted by n_cluster ** -alpha, so that the two smallest SOM clusters -- 12 and 14
# training reference soundings against 86 in the largest -- are not drowned out by the
# majority regimes. Per-cluster test error of the learned retrievals falls off with a
# cluster's training size (Spearman -0.86 to -0.91) while the operational retrieval's does
# not, which is what makes the sampling, rather than the scenes, the thing to change.
#
# It changes nothing else: same winner, same epochs, same covariates, same validation
# scenes. Its archive and its weights carry a `_bal` suffix, because OSDF refuses to
# overwrite an object and because two models at one path is a failure with no trace in any
# log. alpha=0.5 (square root) is the recommended setting; alpha=1 is full inverse
# frequency and makes 12 reference soundings carry as much gradient as 86, which risks
# memorising those 12. Balancing cannot manufacture regime diversity -- the number of
# distinct scenes in a rare cluster is what it is -- so it corrects the loss weighting
# only.
#
# THE SCHEDULE, from the environment rather than from an argument, because it is a property
# of the run and not of the study. Every one of these is unset by default and an unset one
# changes nothing -- a job that exports none of them trains exactly the model this script
# trained before.
#
#   OCO2_WARMUP_EPOCHS  linear warm-up, in epochs, before the cosine starts (default 0)
#   OCO2_ETA_MIN_FRAC   the cosine's floor as a fraction of the peak lr (default 0.1 for
#                       the deterministic trainers and train_nf*, 0.0 for the 4-D flow)
#   OCO2_GRAD_CLIP      gradient-norm clip (default: none for the deterministic trainers,
#                       the config's value for the flows)
#   OCO2_NF_SCHED       "cosine" turns annealing on for train_nf.py and
#                       train_nf_profiles.py, which have none (default "none")
#   OCO2_NF_DROPOUT     dropout in the 20-D flow's coupling nets (default 0.0)
#   OCO2_NF_LAYERS      coupling layers in the 20-D flow (default: the searched 12). The
#                       one knob here that changes the searched ARCHITECTURE, so a run
#                       using it is a different model for that task and must say so
#   OCO2_NF_WD          weight decay for whichever flow is being trained (default: the
#                       searched value; the 4-D trainer spells the key differently)
#   OCO2_RMSE_SAMPLES   posterior samples per validation sounding for the flows' per-epoch
#                       val RMSE (default 0, i.e. the curve is NLL only)
#
# The published learning curves are why these exist. They show the three deterministic regressors still
# descending at epoch 40, two of them checkpointing at 38 and 39 of 40; the scalar flow's
# validation NLL ending ABOVE its epoch-1 value with no schedule to settle it; and the 20-D
# flow best after 7 of 40 epochs and then diverging by 18 nats. The first two are schedule
# problems and the third is a capacity problem. None of them is a reason to re-run the
# search: the PEAK learning rate, the architecture and the weight decay stay exactly as the
# winner record has them, because those are what the search ranked.
set -euo pipefail

STUDY="${1:?usage: bo_final.sh <study> <covariates> <epochs> [balance_alpha]}"
COVARIATES="${2:?missing covariate preset}"
EPOCHS="${3:?missing epoch count}"
BALANCE_ALPHA="${4:-}"

# shellcheck source=job_common.sh
source ./job_common.sh

# Validated here rather than inside the trainer, so a typo costs a second and not a GPU
# slot plus the forty minutes of unpacking that precede the first line of training.
BALANCE_HYDRA=()
BALANCE_SET=()
BAL_SUFFIX=""
if [ -n "$BALANCE_ALPHA" ]; then
    if ! echo "$BALANCE_ALPHA" | grep -Eq '^(0|1|0\.[0-9]+|1\.0+)$'; then
        echo "ERROR: balance_alpha must be a number in [0, 1], got '$BALANCE_ALPHA'."
        exit 2
    fi
    # Two spellings of the same override, because the two trainer families parse
    # differently and the first balanced run died on exactly this. The deterministic
    # trainers are @hydra.main, which composes in STRUCT mode: `data.balance_by=clst`
    # against a config that does not declare the key is a hard error, and `++` is the
    # add-or-override form that works whether or not it is declared. The flow trainers
    # take `--set k=v` into a plain dict (code/utils/cfg.py:dotted_overrides), which
    # creates missing keys and would read a leading `++` as part of the key name.
    BALANCE_HYDRA=("++data.balance_by=clst" "++data.balance_alpha=$BALANCE_ALPHA")
    BALANCE_SET=("data.balance_by=clst" "data.balance_alpha=$BALANCE_ALPHA")
    BAL_SUFFIX="_bal"
fi

JOB="${STUDY}_final${BAL_SUFFIX}$(oco2_split_suffix)"
WINNER="bo_${STUDY}_best.json"

# --- the schedule overrides, in the two dialects ---------------------------------------
# Built here, before anything expensive, so a malformed value costs a second rather than a
# GPU slot plus forty minutes of unpacking. The deterministic trainers are @hydra.main and
# compose in STRUCT mode, where an override of an undeclared key is a hard error and `++`
# is the add-or-override form; the flow trainers take `--set k=v` into a plain dict
# (code/utils/cfg.py:dotted_overrides), which creates missing keys and would read a leading
# `++` as part of the key name. Same split as the balancing overrides above.
SCHED_HYDRA=()
SCHED_SET=()
oco2_num() {   # name value -> fail loudly on anything that is not a number
    if ! echo "$2" | grep -Eq '^[0-9]+([.][0-9]+)?$'; then
        echo "ERROR: $1 must be a non-negative number, got '$2'." >&2
        exit 2
    fi
}
if [ -n "${OCO2_WARMUP_EPOCHS:-}" ]; then
    oco2_num OCO2_WARMUP_EPOCHS "$OCO2_WARMUP_EPOCHS"
    SCHED_HYDRA+=("++hyps.warmup_epochs=$OCO2_WARMUP_EPOCHS")
    SCHED_SET+=("optim.warmup_epochs=$OCO2_WARMUP_EPOCHS")
fi
if [ -n "${OCO2_ETA_MIN_FRAC:-}" ]; then
    oco2_num OCO2_ETA_MIN_FRAC "$OCO2_ETA_MIN_FRAC"
    SCHED_HYDRA+=("++hyps.eta_min_frac=$OCO2_ETA_MIN_FRAC")
    SCHED_SET+=("optim.eta_min_frac=$OCO2_ETA_MIN_FRAC")
fi
if [ -n "${OCO2_GRAD_CLIP:-}" ]; then
    oco2_num OCO2_GRAD_CLIP "$OCO2_GRAD_CLIP"
    SCHED_HYDRA+=("++hyps.grad_clip=$OCO2_GRAD_CLIP")
    SCHED_SET+=("optim.grad_clip=$OCO2_GRAD_CLIP")
fi
if [ -n "${OCO2_NF_SCHED:-}" ]; then
    case "$OCO2_NF_SCHED" in
        cosine|cosine_warmup|none|constant) ;;
        *) echo "ERROR: OCO2_NF_SCHED must be cosine or none, got '$OCO2_NF_SCHED'." >&2
           exit 2 ;;
    esac
    SCHED_SET+=("optim.sched=$OCO2_NF_SCHED")
fi
if [ -n "${OCO2_NF_DROPOUT:-}" ]; then
    oco2_num OCO2_NF_DROPOUT "$OCO2_NF_DROPOUT"
    # Only train_nf_profiles.py builds an NFProfiles from cfg["nflow"], so this is the one
    # override that is study-specific. Passing it to the others would create a key their
    # builders never read -- harmless, but it would appear in their checkpoints' stored cfg
    # as a setting that did nothing, which is worse than not passing it.
    if [ "$STUDY" = nf_profiles ]; then SCHED_SET+=("nflow.dropout=$OCO2_NF_DROPOUT"); fi
fi
# nf_profiles only, and the one override here that changes the ARCHITECTURE the search
# selected rather than the schedule around it. At 120 epochs the
# 20-D flow's train/validation gap doubled to 17.4 nats and its joint 95 % coverage fell to
# 27.9 %, so dropout at 0.1 did not touch the memorisation. Halving the
# coupling stack is the cheapest of the three remedies that report listed, and it must be
# reported as a different model for this task rather than "the same model, more data".
if [ -n "${OCO2_NF_LAYERS:-}" ]; then
    oco2_num OCO2_NF_LAYERS "$OCO2_NF_LAYERS"
    if [ "$STUDY" = nf_profiles ]; then
        SCHED_SET+=("nflow.layers=$OCO2_NF_LAYERS")
    fi
fi
if [ -n "${OCO2_NF_WD:-}" ]; then
    oco2_num OCO2_NF_WD "$OCO2_NF_WD"
    case "$STUDY" in
        nf_profiles) SCHED_SET+=("optim.wd=$OCO2_NF_WD") ;;
        nf_xco2)     SCHED_SET+=("optim.wd=$OCO2_NF_WD") ;;
        nf_last4)    SCHED_SET+=("optim.weight_decay=$OCO2_NF_WD") ;;
    esac
fi
if [ -n "${OCO2_RMSE_SAMPLES:-}" ]; then
    oco2_num OCO2_RMSE_SAMPLES "$OCO2_RMSE_SAMPLES"
    # nf_profiles only, for the same reason: it is the trainer with eval.rmse_samples.
    if [ "$STUDY" = nf_profiles ]; then
        SCHED_SET+=("eval.rmse_samples=$OCO2_RMSE_SAMPLES")
    fi
fi

# shellcheck source=bo_studies.sh
source ./bo_studies.sh

oco2_banner "$JOB"
oco2_trap_pack "$JOB"
oco2_gpu_check

if [ ! -f "$WINNER" ]; then
    echo "ERROR: $WINNER was not transferred."
    echo "       The study's select node writes it from the 25 trial records; if that"
    echo "       node did not run, there is no winner to retrain."
    exit 2
fi
echo "--- the winning configuration ---"
cat "$WINNER"
echo

oco2_setup

WD="$WEIGHTS_ROOT/$(bo_final_dir "$STUDY")$BAL_SUFFIX"
mkdir -p "$WD"
bo_study_spec "$STUDY" "$WD"

if [ ! -f "$BO_CONF" ]; then
    echo "ERROR: $BO_CONF is not in this code tarball."
    echo "       cd ~/OCO-2 && git pull && ./chtc/make_code_tarball.sh"
    exit 2
fi

# The winner's overrides, and a check that the record is the one this job was told to
# retrain. A select node that wrote the wrong study's winner would otherwise train a
# perfectly good model of the wrong thing, and nothing downstream would notice.
mapfile -t OVERRIDES < <(python3 - "$WINNER" "$STUDY" <<'PY'
import json, sys
rec = json.load(open(sys.argv[1]))
if rec.get("study") != sys.argv[2]:
    sys.exit("ERROR: {} is the winner of study {!r}, not {!r}.".format(
        sys.argv[1], rec.get("study"), sys.argv[2]))
ov = rec.get("overrides") or []
if not ov:
    sys.exit("ERROR: {} carries no overrides; it cannot be turned back into a run."
             .format(sys.argv[1]))
print("\n".join(ov))
PY
)
if [ "${#OVERRIDES[@]}" -eq 0 ]; then
    echo "ERROR: could not read the overrides out of $WINNER (see above)."
    exit 2
fi

oco2_unpack_data
oco2_unpack_split
oco2_unpack_covariates

# The split table and its statistics, from OCO2_SPLIT rather than from the experiment
# config.
#
# Every `*_cov_*.yaml` hardcodes `data.split_csv` and `data.stats_dir`, and
# bo_config_suffix maps only three splits to a config; anything else falls through to
# `cov_grouped`. So a job launched with OCO2_SPLIT=size_90_v1 would unpack the 90-scene
# table, name its archive `_s90`, write its weights into weights/size_90_v1/ -- and
# train on group_split_v1's 361 scenes, because that is what the config it composed
# says. It would not fail; it would report the wrong experiment.
#
# Passing both explicitly makes the trainer read the table this job unpacked. For the
# three configured splits the values are identical to the config's, so it is a no-op
# there, and the paths are printed below either way. They go LAST on the command line
# so that nothing in the winner's own override list can shadow them.
BO_SPLIT_OVERRIDE=("data.split_csv=$SPLIT_CSV" "data.stats_dir=$SPLIT_STATS")
for f in "$SPLIT_CSV" "$SPLIT_STATS/xco2_stats.json"; do
    if [ ! -s "$f" ]; then
        echo "ERROR: $f is missing after unpacking $OCO2_SPLIT. The trainer would fall"
        echo "       back to whatever the experiment config names, which is a"
        echo "       different partition."
        exit 2
    fi
done

echo "=============================================================="
echo " final retrain: $STUDY"
echo " covariates  : $COVARIATES"
echo " epochs      : $EPOCHS  (the trials ran at half this)"
echo " weights     : $WD"
echo " checkpoint  : $WD/$BO_CKPT"
if [ "$BO_ROLE" = flow ]; then
    echo " warm start  : off (${BO_SCRATCH[*]})"
fi
echo " split table : $SPLIT_CSV"
echo " statistics  : $SPLIT_STATS"
echo " overrides   : ${OVERRIDES[*]}"
if [ -n "$BAL_SUFFIX" ]; then
    echo " balancing   : ${BALANCE_SET[*]}  (alpha=$BALANCE_ALPHA)"
else
    echo " balancing   : off (uniform shuffling, as published)"
fi
if [ "$BO_ROLE" = det ]; then
    echo " schedule    : ${SCHED_HYDRA[*]:-published defaults (cosine to lr/10, no warmup, no clip)}"
else
    echo " schedule    : ${SCHED_SET[*]:-published defaults (constant lr for train_nf*, cosine to 0 for the 4-D flow)}"
fi
echo "=============================================================="

if [ "$BO_ROLE" = det ]; then
    python3 "$BO_SCRIPT" \
        "+experiment=$BO_EXP" \
        "covariates=$COVARIATES" \
        "select_metric=$BO_SELECT" \
        "$BO_EPOCH_KEY=$EPOCHS" \
        "${BO_DIR_OVERRIDE[@]}" \
        "${BALANCE_HYDRA[@]}" \
        "${OVERRIDES[@]}" \
        "${SCHED_HYDRA[@]}" \
        "${BO_SPLIT_OVERRIDE[@]}"
else
    python3 "$BO_SCRIPT" --exp_cfg "$BO_CONF" --set \
        "covariates=$COVARIATES" \
        "$BO_EPOCH_KEY=$EPOCHS" \
        "${BO_DIR_OVERRIDE[@]}" \
        "${BO_SCRATCH[@]}" \
        "${BALANCE_SET[@]}" \
        "${OVERRIDES[@]}" \
        "${SCHED_SET[@]}" \
        "${BO_SPLIT_OVERRIDE[@]}"
fi

# The scoring job looks for exactly one path, computed from the same table this job used.
# If it is not there the run "succeeded" and produced nothing scoreable, and the failure
# belongs here rather than forty minutes later in a job that unpacks 40 GB of data first.
echo
echo "--- what the run produced ---"
find "$WD" -name '*.pt' -printf '    %p  %s bytes\n' | sort
if [ ! -s "$WD/$BO_CKPT" ]; then
    echo
    echo "ERROR: the checkpoint bo_eval.sh will look for is missing or empty:"
    echo "         $WD/$BO_CKPT"
    echo "       Training exited 0, so this is a path disagreement between the trainer"
    echo "       and BO_CKPT in chtc/bo_studies.sh, not a training failure."
    exit 3
fi

# The record travels with the model. Six months from now the archive is the only thing
# left, and "which configuration is this?" has to be answerable from inside it.
cp "$WINNER" "$WD/"
for f in "$WD/global_best/best_metric.txt" "$WD/global_best/metrics.json" \
         "$WD/${BO_CKPT%.pt}.metrics.json"; do
    if [ -f "$f" ]; then echo "--- $f"; cat "$f"; echo; fi
done

echo "=============================================================="
echo " $JOB finished"
echo "=============================================================="

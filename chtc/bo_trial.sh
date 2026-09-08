#!/bin/bash
# One Bayesian-optimization trial.
#
#     ./bo_trial.sh <study> <round> <trial> <covariates> <epochs>
#
# The hyperparameters are NOT passed on the command line. They are read from
# bo_<study>_r<round>_t<trial>.params.json, which the round's propose job wrote to
# /staging and the submit file transferred in. That is what lets a static DAG run an
# adaptive search: the submit files are fixed, and only the contents of the parameter
# files change from round to round.
#
# Trials run at <epochs> (20) rather than the full run's 40. Both profile models reach
# their best validation RMSE by epoch 8-15 and then oscillate without trend, so the back
# half does not change the ranking. The winner is retrained at 2x by bo_final.sh -- which
# is the same number in both places only because make_bo_dag.py writes both.
#
# The result record is written twice: once before training starts, so the declared output
# file exists no matter how the job dies, and once at the end with the value. A record
# with "value": null is a trial that did not finish; bo_propose.py treats it as pruned
# rather than as a bad score, because a job that died in setup says nothing about its
# hyperparameters.
set -euo pipefail

STUDY="${1:?usage: bo_trial.sh <study> <round> <trial> <covariates> <epochs>}"
ROUND="${2:?missing round}"
TRIAL="${3:?missing trial}"
COVARIATES="${4:?missing covariate preset}"
EPOCHS="${5:?missing epoch count}"

NAME="bo_${STUDY}_r${ROUND}_t${TRIAL}"
PARAMS="${NAME}.params.json"
RESULT="${NAME}.json"

# shellcheck source=job_common.sh
source ./job_common.sh
# shellcheck source=bo_studies.sh
source ./bo_studies.sh

# Write a placeholder record immediately, before anything that can fail. HTCondor puts a
# job on hold when a declared output file is missing, and a hold message says nothing
# about why the run failed.
cat > "$RESULT" <<EOF
{"study": "$STUDY", "round": $ROUND, "trial": $TRIAL, "value": null,
 "status": "started", "host": "$(hostname)", "covariates": "$COVARIATES",
 "epochs": $EPOCHS}
EOF

oco2_banner "$NAME"
oco2_trap_pack "$NAME"
export OCO2_PACK_PATHS="experiments/weights/bo chtc/GIT_PROVENANCE.txt"
oco2_gpu_check

if [ ! -f "$PARAMS" ]; then
    echo "ERROR: $PARAMS was not transferred."
    echo "       The round's propose job must have completed and left the parameter"
    echo "       files in /staging/<l>/<netid>/oco2/results/<run>/bo/."
    exit 2
fi
echo "--- parameters ---"
cat "$PARAMS"
echo

oco2_setup

# Every trial gets its own weights directory. The trainers promote a checkpoint into
# <weights_dir>/global_best whenever it beats what is already there, and a shared
# directory would mean trial 4's score being compared against trial 1's -- which is how a
# sweep silently reports the best of a different configuration.
WD="$PROJECT/experiments/weights/bo/$STUDY/r${ROUND}_t${TRIAL}"
mkdir -p "$WD"

# What this study runs, in which dialect, and what switches its warm start off. One table,
# shared with bo_final.sh so that the trial and the retrained winner cannot drift apart.
bo_study_spec "$STUDY" "$WD"

if [ ! -f "$BO_CONF" ]; then
    echo "ERROR: $BO_CONF is not in this code tarball."
    echo "       cd ~/OCO-2 && git pull && ./chtc/make_code_tarball.sh"
    exit 2
fi
if [ ! -f "$BO_SCRIPT" ]; then
    echo "ERROR: $BO_SCRIPT is not in this code tarball."
    exit 2
fi

# Hydra overrides, one per line in the parameter file's "overrides" array.
mapfile -t OVERRIDES < <(python3 -c '
import json, sys
print("\n".join(json.load(open(sys.argv[1]))["overrides"]))' "$PARAMS")
if [ "${#OVERRIDES[@]}" -eq 0 ]; then
    echo "ERROR: $PARAMS carries no overrides."
    exit 2
fi

oco2_unpack_data
oco2_unpack_split
oco2_unpack_covariates

if [ "$BO_ROLE" = det ]; then
    OBJECTIVE="validation RMSE ($BO_SELECT)"
else
    OBJECTIVE="validation NLL, backbone from scratch (${BO_SCRATCH[*]})"
fi

echo "=============================================================="
echo " BO trial: $STUDY round $ROUND trial $TRIAL"
echo " covariates : $COVARIATES"
echo " epochs     : $EPOCHS"
echo " objective  : $OBJECTIVE"
echo " overrides  : ${OVERRIDES[*]}"
echo "=============================================================="

set +e
if [ "$BO_ROLE" = det ]; then
    python3 "$BO_SCRIPT" \
        "+experiment=$BO_EXP" \
        "covariates=$COVARIATES" \
        "select_metric=$BO_SELECT" \
        "$BO_EPOCH_KEY=$EPOCHS" \
        "${BO_DIR_OVERRIDE[@]}" \
        "${OVERRIDES[@]}"
else
    python3 "$BO_SCRIPT" --exp_cfg "$BO_CONF" --set \
        "covariates=$COVARIATES" \
        "$BO_EPOCH_KEY=$EPOCHS" \
        "${BO_DIR_OVERRIDE[@]}" \
        "${BO_SCRATCH[@]}" \
        "${OVERRIDES[@]}"
fi
STATUS=$?
set -e

# --- collect ----------------------------------------------------------------------
# Four dialects, because four trainers wrote them:
#   train_profiles.py   global_best/best_metric.txt   validation RMSE
#   train.py            global_best/metrics.json      "val_rmse"
#   the three flows     *.metrics.json beside the checkpoint, "best_val_nll"
# All four are minimized, so the search does not care which it got -- but the record says
# which metric it is, because a leaderboard that mixes RMSE and NLL rows is nonsense.
python3 - "$RESULT" "$PARAMS" "$WD" "$STATUS" "$COVARIATES" "$EPOCHS" <<'PY'
import json, sys
from pathlib import Path

result, params, wd, status, covariates, epochs = sys.argv[1:7]
rec = json.loads(Path(params).read_text())
wd = Path(wd)

value = None
source = None
metric = None
txt = wd / "global_best" / "best_metric.txt"
js = wd / "global_best" / "metrics.json"
if txt.exists():
    try:
        value, source, metric = float(txt.read_text().strip()), str(txt), "val_rmse"
    except ValueError:
        pass
if value is None and js.exists():
    try:
        value = float(json.loads(js.read_text())["val_rmse"])
        source, metric = str(js), "val_rmse"
    except Exception:
        pass
if value is None:
    # The flows. One checkpoint per run, so at most one of these exists; sorted() only
    # makes the choice deterministic if a stale one ever survives.
    for cand in sorted(wd.rglob("*.metrics.json")):
        try:
            value = float(json.loads(cand.read_text())["best_val_nll"])
            source, metric = str(cand), "val_nll"
            break
        except Exception:
            continue

rec.update({
    "value": value,
    "metric": metric,
    "status": "completed" if (value is not None and int(status) == 0) else "failed",
    "exit_status": int(status),
    "metric_source": source,
    "covariates": covariates,
    "epochs": int(epochs),
    "weights_dir": str(wd),
})
Path(result).write_text(json.dumps(rec, indent=2))
print(json.dumps(rec, indent=2))
PY

# Drop the checkpoints before the exit trap packs the archive.
#
# OCO2_PACK_PATHS names experiments/weights/bo, and the trainers write net.pt /
# net_profiles.pt / best_flow_*.pt into it -- round 0 of det_profiles alone put 290 MB on
# /staging, so a 150-trial run would put four or five GB there for files nothing reads.
# oco2_pack_results has no exclude mechanism, so they are removed rather than skipped.
#
# Nothing needs them. The metric is already in the result record, and the winner is
# retrained at 2x epochs by bo_final.sh -- a 20-epoch checkpoint is not the model that
# gets reported.
echo "--- removing trial checkpoints (the winner is retrained; see bo_final.sh) ---"
du -sh "$WD" 2>/dev/null | sed "s/^/    before: /" || true
find "$WD" -name '*.pt' -print -delete 2>/dev/null || true
du -sh "$WD" 2>/dev/null | sed "s/^/    after:  /" || true

echo
echo "=============================================================="
echo " $NAME finished (training exit status $STATUS)"
echo "=============================================================="
# A trial that failed must not fail the DAG: bo_propose.py records it as pruned and the
# search continues. Only a missing result record is fatal, and that cannot happen.
exit 0

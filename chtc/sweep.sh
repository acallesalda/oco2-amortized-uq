#!/bin/bash
# One hyperparameter-search trial on the reference-sounding-disjoint split.
#
#     ./sweep.sh <base_job> <trial_id> <hydra overrides...>
#
# base_job is det_profiles or det_last4; the grids in sweeps/*.txt supply the rest, and
# sweep.sub queues one job per line.
#
# Runs at SWEEP_EPOCHS (20 by default) rather than the full 40. Both models reach their
# best validation RMSE by epoch 8-15 and then oscillate without trend, so the back half of
# a full run contributes nothing to ranking configurations -- it doubles the cost of the
# search to re-confirm a plateau. Retrain the winner at full epochs afterwards.
#
# The archive carries metrics and hydra configs only, never the .pt: sixteen checkpoints we
# will not use is 300 MB of transfer for nothing. The winner gets retrained.
set -euo pipefail

BASE="${1:?usage: sweep.sh <base_job> <trial_id> [overrides...]}"
TRIAL="${2:?missing trial id}"
shift 2
OVERRIDES=("$@")

# shellcheck source=job_common.sh
source ./job_common.sh

JOB="sweep_${BASE}_${TRIAL}"
export OCO2_PACK_PATHS="experiments/weights/sweep chtc/GIT_PROVENANCE.txt"

oco2_banner "$JOB"
oco2_trap_pack "$JOB"
oco2_gpu_check
oco2_setup
oco2_unpack_data
oco2_unpack_split

case "$BASE" in
    det_profiles) SCRIPT=train_profiles.py; EXP=full_regressor_profiles_grouped ;;
    det_last4)    SCRIPT=train_profiles.py; EXP=full_regressor_profiles_last4_grouped ;;
    *) echo "ERROR: unknown base job '$BASE'. Valid: det_profiles det_last4"; exit 2 ;;
esac

WD="$PROJECT/experiments/weights/sweep/$BASE/$TRIAL"
mkdir -p "$WD"

echo "=============================================================="
echo " sweep trial: $BASE / $TRIAL"
echo " overrides  : ${OVERRIDES[*]}"
echo " epochs     : ${SWEEP_EPOCHS:-20}"
echo "=============================================================="

python3 "experiments/train/$SCRIPT" \
    "+experiment=$EXP" \
    "weights_dir=$WD" \
    "hyps.epochs=${SWEEP_EPOCHS:-20}" \
    "${OVERRIDES[@]}"

# One machine-readable line per trial, so collect_sweep.py does not have to parse logs.
BEST="$WD/global_best/best_metric.txt"
python3 - "$BASE" "$TRIAL" "$BEST" "$WD" "${OVERRIDES[@]}" <<'PY'
import json, sys
from pathlib import Path

base, trial, best_path, wd, *ov = sys.argv[1:]
p = Path(best_path)
rec = {
    "base": base,
    "trial": trial,
    "best_val_rmse": float(p.read_text().strip()) if p.exists() else None,
    "overrides": {k: v for k, v in (o.split("=", 1) for o in ov)},
}
Path(wd, "trial.json").write_text(json.dumps(rec, indent=2))
print(json.dumps(rec))
PY

echo
echo "=============================================================="
echo " $JOB finished"
echo "=============================================================="

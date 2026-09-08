#!/bin/bash
# Propose one round of Bayesian-optimization trials. CPU only, under a minute.
#
#     ./bo_round.sh <study> <round> [batch]
#
# Reads every bo_<study>_r*_t*.json the earlier rounds left in the job directory (the
# submit file names them explicitly in transfer_input_files -- /staging cannot be globbed
# at submit time, but the names are deterministic, so the DAG can list them) and writes
# the next batch of bo_<study>_r<round>_t*.params.json.
#
# Round 0 reads nothing and always proposes the current configuration as trial 0, so every
# study contains its own baseline.
set -euo pipefail

STUDY="${1:?usage: bo_round.sh <study> <round> [batch]}"
ROUND="${2:?missing round}"
BATCH="${3:-5}"

# shellcheck source=job_common.sh
source ./job_common.sh

oco2_banner "bo_propose_${STUDY}_r${ROUND}"
oco2_trap_pack "bo_propose_${STUDY}_r${ROUND}"
export OCO2_PACK_PATHS="chtc/GIT_PROVENANCE.txt"
oco2_setup

# The earlier rounds' records arrive flat in the job directory; bo_propose.py wants them
# in one place it can glob.
mkdir -p bo_results bo_params
found=0
for f in bo_${STUDY}_r*_t*.json; do
    # A literal, unmatched glob is the round-0 case.
    case "$f" in *'*'*) continue ;; esac
    case "$f" in *.params.json) continue ;; esac
    cp "$f" bo_results/
    found=$((found + 1))
done
echo "--- $found trial record(s) from earlier rounds ---"
if [ "$found" -gt 0 ]; then
    # A quoted heredoc, not `python3 -c`, and no backslashes anywhere.
    #
    # This block previously used  print(f"...{r.get(\"value\")}...")  inside a
    # single-quoted -c string. Backslashes are not allowed inside an f-string
    # expression until Python 3.12, and the container is on 3.10, so it was a
    # SyntaxError -- which under `set -e` killed the script before bo_propose.py ran,
    # left the declared .params.json outputs unwritten, and held the node with a
    # transfer error naming five missing files and not the cause.
    #
    # Round 0 never reached it: `found` is 0 there, so the guard skipped the block and
    # the bug waited for round 1, after five GPU trials had already been spent.
    python3 <<'PY'
import json
from pathlib import Path
for p in sorted(Path("bo_results").glob("*.json")):
    r = json.loads(p.read_text())
    print("  {}: value={} status={}".format(p.name, r.get("value"), r.get("status")))
PY
fi
echo

# The other split's winner, if propose_r0.sub transferred it. Round 0 only -- later
# rounds have real history and bo_propose.py ignores the flag there anyway; passing it
# only where it can act keeps the log honest about what was enqueued.
EXTRA=()
if [ -f "bo_${STUDY}_best.json" ]; then
    echo "--- extra baseline: bo_${STUDY}_best.json (another run's winner) ---"
    EXTRA=(--extra-baseline "bo_${STUDY}_best.json")
fi

python3 chtc/bo_propose.py \
    "${EXTRA[@]}" \
    --study "$STUDY" \
    --round "$ROUND" \
    --batch "$BATCH" \
    --results-dir bo_results \
    --out-dir bo_params \
    --seed "${BO_SEED:-20260818}" \
    --n-startup "${BO_STARTUP:-8}"

# Flatten back out: the submit file declares the parameter files as outputs by name.
cp bo_params/*.params.json .
ls -l bo_${STUDY}_r${ROUND}_t*.params.json

echo
echo "propose complete: round $ROUND of $STUDY, $BATCH trials."

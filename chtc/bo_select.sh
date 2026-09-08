#!/bin/bash
# Pick one study's winner out of its finished trial records. CPU only, under a minute.
#
#     ./bo_select.sh <study>
#
# The sibling of bo_round.sh, and it works the same way: the trial records arrive flat in
# the job directory (the submit file names all 25 explicitly -- /staging cannot be globbed
# at submit time, but the names are deterministic, so the DAG can list them), this puts
# them where bo_select.py can glob them, and the two output files go back to /staging flat.
#
# It is a node of its own rather than the first ten lines of bo_final.sh because it is a
# one-minute CPU job that can say "every trial failed" before a two-hour GPU job unpacks
# 40 GB of data to discover the same thing -- and because the leaderboard it writes is
# what gets read into the paper, whether or not the retrain that follows succeeds.
set -euo pipefail

STUDY="${1:?usage: bo_select.sh <study>}"

# shellcheck source=job_common.sh
source ./job_common.sh

oco2_banner "bo_select_${STUDY}"
oco2_trap_pack "bo_select_${STUDY}"
export OCO2_PACK_PATHS="chtc/GIT_PROVENANCE.txt"
oco2_setup

mkdir -p bo_results
found=0
for f in bo_${STUDY}_r*_t*.json; do
    # A literal, unmatched glob means nothing was transferred.
    case "$f" in *'*'*) continue ;; esac
    case "$f" in *.params.json) continue ;; esac
    cp "$f" bo_results/
    found=$((found + 1))
done
echo "--- $found trial record(s) ---"
if [ "$found" -eq 0 ]; then
    echo "ERROR: no trial records were transferred for $STUDY."
    echo "       Every trial node of this study must have completed. A trial that failed"
    echo "       still writes its record and still exits 0, so an absent record means the"
    echo "       job was held or evicted, not that the configuration was bad."
    exit 2
fi

python3 chtc/bo_select.py --study "$STUDY" --results-dir bo_results --out-dir .

ls -l "bo_${STUDY}_best.json" "bo_${STUDY}_leaderboard.md"
echo
echo "select complete: $STUDY"

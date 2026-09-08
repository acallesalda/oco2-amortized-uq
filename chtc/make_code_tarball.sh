#!/bin/bash
# Package the project source for shipping to execute nodes.
#
# Run this on the CHTC access point, from anywhere inside the git checkout, every time
# you change code or a config:
#
#     cd ~/OCO-2
#     ./chtc/make_code_tarball.sh
#
# Produces chtc/oco2_code.tar.gz (~15 MB), which the submit files transfer from /home.
# Execute nodes have no outbound network, so cloning from GitHub inside a job is not an
# option -- the code has to travel with the job.
#
# Every .pt and .pkl is excluded -- 245 MB of them, and no training job reads one. The
# two flows that warm-start get their backbone from the DAG parent's result archive, not
# from the repository; the published checkpoints are only needed for scoring, and those
# come from the Zenodo weights archive on /staging. The .yaml/.json next to them are kept,
# because they are the hyperparameter record the grouped configs refer to.
#
# Every .npz is excluded too, and this one bites in a way the .pt exclusion does not.
# `fetch_results.sh` unpacks result archives back INTO the checkout, so scoring artifacts
# accumulate in experiments/scoring/results/ -- profiles.npz alone is 223 MB. Nothing in a
# job reads them (predict.py writes them; score.py reads them locally, off CHTC), but they
# are under experiments/, so they were being packed and shipped to every execute node. The
# tarball went from 15 MB to 236 MB after one `fetch_results.sh --cov`, which is 6 GB of
# transfer across a 25-trial search, from /home, which CHTC asks us not to do.
#
# chtc/bo_propose.py and chtc/bo_select.py travel too: bo_round.sh and bo_select.sh run
# them inside the job, so unlike the rest of chtc/ (which runs on the access point or is
# transferred as the executable) they have to be in the tarball. If a propose or select
# node ever dies with "No such file or directory: chtc/bo_propose.py", these lines are
# what is missing.
#
# code/test/data/*.h5 (7 MB) IS kept for `pytest code/test`. Note that pytest was NOT in
# requirements.txt when the current image was built, so that does not work inside it yet
# -- it is added for the next rebuild. Until then the tests run anywhere with numpy,
# scipy and pandas, and the container-side check is `predict.py --limit 2`.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO=$PWD
OUT="$REPO/chtc/oco2_code.tar.gz"

echo "packaging $REPO -> $OUT"

if [ ! -d "$REPO/code" ] || [ ! -d "$REPO/experiments" ]; then
    echo "ERROR: run this from inside the OCO-2 checkout (code/ and experiments/ not found)."
    exit 1
fi

# Record the commit so a result tarball can always be traced back to its source.
if git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
    {
        echo "commit:   $(git -C "$REPO" rev-parse HEAD)"
        echo "branch:   $(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
        echo "dirty:    $(test -n "$(git -C "$REPO" status --porcelain)" && echo yes || echo no)"
        echo "packaged: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "$REPO/chtc/GIT_PROVENANCE.txt"
    cat "$REPO/chtc/GIT_PROVENANCE.txt"
else
    echo "packaged: $(date -u +%Y-%m-%dT%H:%M:%SZ) (not a git checkout)" \
        > "$REPO/chtc/GIT_PROVENANCE.txt"
fi

tar -czf "$OUT" \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='*.pyc' \
    --exclude='*.pt' \
    --exclude='*.pkl' \
    --exclude='*/runs/*' \
    --exclude='*/outputs/*' \
    --exclude='*/multirun/*' \
    --exclude='mlruns' \
    --exclude='*.npz' \
    --exclude='experiments/scoring/results' \
    --exclude='experiments/scoring/tables' \
    --exclude='experiments/scoring/figures' \
    code \
    experiments \
    chtc/GIT_PROVENANCE.txt \
    chtc/bo_propose.py \
    chtc/bo_select.py

echo
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo "contents (top level):"
tar -tzf "$OUT" | awk -F/ '{print $1"/"$2}' | sort -u | head -20 || true

# The tarball is transferred once per job -- 26 times for one BO study -- so the threshold
# that matters is far below the 1 GB at which CHTC asks for /staging. 50 MB is already
# large for source code, and the usual cause is an artifact that fetch_results.sh unpacked
# back into the checkout. Say what, not just how big.
SIZE=$(stat -c %s "$OUT")
if [ "$SIZE" -gt 52428800 ]; then
    echo
    echo "WARNING: $OUT is $(du -h "$OUT" | cut -f1) -- large for source code, and it is"
    echo "         transferred once per job. The ten biggest things in it:"
    tar -tzvf "$OUT" | sort -k3 -n -r | head -10 | awk '{printf "           %10.1f MB  %s' + BS + 'n", $3/1048576, $NF}'
    echo "         If those are results rather than code, add them to the exclude list."
fi

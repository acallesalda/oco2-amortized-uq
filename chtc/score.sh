#!/bin/bash
# Build the final tables and figures from the eval artifacts, inside the container.
#
#     condor_submit score.sub
#
# Exists because the two natural places to run score.py both have a catch. The CHTC access
# point may not have numpy/scipy/pandas/matplotlib -- the container does, but nothing
# guarantees the login node does -- and copying 250 MB of .npz to a laptop is fine but not
# something to discover you need at midnight. This is a two-minute CPU job that needs
# neither.
#
# It is NOT the intended way to iterate on figures. That is still: fetch the .npz once and
# run score.py/figures.py wherever you like. This is the path that always works.
set -euo pipefail

JOB="score"

# shellcheck source=job_common.sh
source ./job_common.sh

export OCO2_PACK_PATHS="experiments/scoring/tables experiments/scoring/figures chtc/GIT_PROVENANCE.txt"

oco2_banner "$JOB"
oco2_trap_pack "$JOB"
oco2_setup                      # no GPU, no dataset: the artifacts carry everything

# The eval archives hold experiments/scoring/results/<split>/<task>.npz
#
# Only the archives that were actually transferred are unpacked. score.sub names the ones
# it expects, but an eval that has not been run yet has nothing in /staging, and the
# surface pair (eval_last1, eval_last1_marginal) may well not exist when this is first
# submitted. oco2_unpack_results treats a named-but-absent archive as an error -- correct
# for a training job's own dependency, wrong here, where a missing task should cost its
# rows in the tables and nothing else.
ARCHIVES=""
# Three families, and which of them you transfer decides what the tables contain:
#   eval_<task>            the pre-search models, group split
#   eval_<task>_bo         the search winners, group split
#   eval_<task>_borep      the search winners, replicate split (the rerun)
#   eval_<task>_replicate  the published checkpoints, replicate split
# score.py keys its blocks by split name, so two artifacts of the SAME task and
# split -- eval_xco2 and eval_xco2_bo, say -- print two sets of rows under one
# heading with nothing to tell them apart. Ask for one per (task, split).
for a in eval_xco2_bo.tar.gz eval_profiles_bo.tar.gz eval_last4_bo.tar.gz \
         eval_last4_marginal_bo.tar.gz \
         eval_xco2_202003_full.tar.gz eval_profiles_202003_full.tar.gz \
         eval_last4_202003_full.tar.gz eval_last4_marginal_202003_full.tar.gz \
         eval_xco2_202007_full.tar.gz eval_profiles_202007_full.tar.gz \
         eval_last4_202007_full.tar.gz eval_last4_marginal_202007_full.tar.gz \
         eval_xco2_borep.tar.gz eval_profiles_borep.tar.gz \
         eval_last4_borep.tar.gz eval_last4_marginal_borep.tar.gz \
         eval_xco2_replicate.tar.gz eval_profiles_replicate.tar.gz \
         eval_last4_replicate.tar.gz eval_last4_marginal_replicate.tar.gz \
         eval_last1.tar.gz eval_last1_marginal.tar.gz; do
    if [ -f "$a" ]; then
        ARCHIVES="$ARCHIVES $a"
    else
        echo "note: $a was not transferred; its rows will be missing from the tables."
    fi
done
if [ -z "$ARCHIVES" ]; then
    echo "ERROR: none of the eval archives were transferred."
    exit 2
fi
# shellcheck disable=SC2086  -- deliberate word splitting: one argument per archive.
oco2_unpack_results $ARCHIVES

# Every split directory, not just group_split_v1.
#
# This used to be one hardcoded path, which was right while there was one split.
# The rerun (chtc/bo_replicate/) writes results/replicate_split_v1/*_borep.npz, and
# the whole point of it is a table whose two halves were produced by one scorer --
# so a scoring job that cannot see the second directory cannot build the table the
# experiment exists for. score.py takes several artifacts of the same task and
# prints them as separate blocks keyed by split name, so it needs the files.
RESULTS="$PROJECT/experiments/scoring/results"
mapfile -t NPZ < <(find "$RESULTS" -name '*.npz' 2>/dev/null | sort)
echo "--- artifacts ---"
find "$RESULTS" -name '*.npz' -printf '    %P  %s bytes\n' | sort
if [ "${#NPZ[@]}" -eq 0 ]; then
    echo "ERROR: no .npz under $RESULTS. Did the eval jobs finish?"
    exit 2
fi
echo "    ${#NPZ[@]} artifact(s), over $(find "$RESULTS" -mindepth 1 -maxdepth 1 -type d | wc -l) split(s)"
echo

# Pretty names for the splits that exist, so the .tex header says which column is
# which. An artifact carries the stem of its split table as `split_name`, hence the
# _cov suffixes: the augmented tables are what every covariate-aware job reads.
LABELS=(--label "group_split_v1_cov=reference-disjoint"
        --label "replicate_split_v1_cov=within-reference"
        --label "replicate=within-reference (published checkpoints)"
        # The temporal hold-out. These artifacts come from the ALL-February
        # models, so they sit beside -- not inside -- the reference-disjoint block:
        # a different training set as well as a different month.
        --label "month_202003_full_split=March 2020 (all-February model)"
        --label "month_202007_full_split=July 2020 (all-February model)")

python3 experiments/scoring/score.py "${NPZ[@]}" \
    "${LABELS[@]}" \
    --out "$PROJECT/experiments/scoring/tables"

python3 experiments/scoring/figures.py "${NPZ[@]}" \
    --out "$PROJECT/experiments/scoring/figures" --cases both || \
    echo "WARNING: figures.py failed; the tables above are still good."

echo
echo "=============================================================="
echo " score finished"
echo "=============================================================="
ls -R "$PROJECT/experiments/scoring/tables" "$PROJECT/experiments/scoring/figures" 2>/dev/null || true

#!/bin/bash
# Submit score.sub with exactly the eval artifacts that exist.
#
#     ./chtc/score_submit.sh              # every eval_*.tar.gz currently in /staging
#     ./chtc/score_submit.sh --dry-run    # print the condor_submit line and stop
#
# Runs on the ACCESS POINT, not in a job.
#
# Why this exists. score.sub can only ask for artifacts it names, and naming one that is
# not in /staging does not degrade gracefully -- HTCondor cannot stage the input, so it
# HOLDS the job before score.sh starts, with a Pelican 404 as the only diagnostic. Naming
# too few is the opposite failure and quieter: the tables come out missing rows and
# nothing says so.
#
# Both of those are questions about what is on disk, which is a thing to look up rather
# than remember. So: look it up.
set -euo pipefail

cd "$(dirname "$0")"

NETID=$(whoami)
RESULTS="/staging/${NETID:0:1}/${NETID}/oco2/results"

# The three that score.sub requires outright, and the ones it takes only if present. Keep
# in step with the case statement in eval.sh -- if a task can be scored, it belongs here.
# Nothing is required any more. The three originals were a safe floor while they
# were the only artifacts; now the tables the paper wants are built from the search
# winners (_bo), the rerun (_borep) and the published checkpoints (_replicate), and
# requiring the pre-search three would put two blocks under one split heading.
REQUIRED=()
OPTIONAL=(eval_xco2_bo eval_profiles_bo eval_last4_bo eval_last4_marginal_bo
          eval_xco2_borep eval_profiles_borep eval_last4_borep
          eval_last4_marginal_borep
          eval_xco2_replicate eval_profiles_replicate eval_last4_replicate
          eval_last4_marginal_replicate
          eval_last1 eval_last1_marginal
          # The temporal hold-out (months_full.dag). Optional like the rest: a month
          # whose eval has not finished costs its rows and nothing else.
          eval_xco2_202003_full eval_profiles_202003_full eval_last4_202003_full
          eval_last4_marginal_202003_full
          eval_xco2_202007_full eval_profiles_202007_full eval_last4_202007_full
          eval_last4_marginal_202007_full)

missing=()
for job in "${REQUIRED[@]}"; do
    if [ ! -f "$RESULTS/$job.tar.gz" ]; then missing+=("$job"); fi
done
if [ ${#missing[@]} -gt 0 ]; then
    echo "ERROR: score.sub requires these and they are not in $RESULTS:"
    printf '    %s.tar.gz\n' "${missing[@]}"
    echo
    echo "Run the eval job first, or edit REQUIRED here if the task has been retired."
    echo "    ./fetch_results.sh --status"
    exit 2
fi

extra=""
echo "--- artifacts in $RESULTS ---"
if [ ${#REQUIRED[@]} -gt 0 ]; then
    printf '    %-24s required, present\n' "${REQUIRED[@]/%/.tar.gz}"
fi
for job in "${OPTIONAL[@]}"; do
    if [ -f "$RESULTS/$job.tar.gz" ]; then
        extra="$extra, \$(OSDF)/results/$job.tar.gz"
        printf '    %-24s optional, present\n' "$job.tar.gz"
    else
        printf '    %-24s optional, ABSENT -- its rows will be missing from the tables\n' \
            "$job.tar.gz"
    fi
done
echo

# With no artifact at all there is nothing to score, and score.sh would exit 2 after
# a queue wait and a container pull. Say so here instead.
if [ -z "$extra" ]; then
    echo "ERROR: none of the scoreable artifacts are in $RESULTS."
    echo "       Run an eval job first: ./fetch_results.sh --status"
    exit 2
fi

# The value is passed through unexpanded: $(OSDF) is defined once, in site.conf, and
# resolving it here would put a second copy of the staging path in a second file.
cmd=(condor_submit score.sub -a "EXTRA_EVALS = $extra")

if [ "${1:-}" = "--dry-run" ]; then
    printf '%q ' "${cmd[@]}"; echo
    exit 0
fi

"${cmd[@]}"

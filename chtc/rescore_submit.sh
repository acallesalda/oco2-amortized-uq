#!/bin/bash
# Re-score a task on the test partition, on either or both splits, after a change to the
# SCORING code that leaves the checkpoints alone.
#
#     ./chtc/rescore_submit.sh xco2                # all three splits
#     ./chtc/rescore_submit.sh xco2 --split group  # just the reference-disjoint one
#     ./chtc/rescore_submit.sh xco2 --month 202003 --month 202007   # the temporal hold-out
#     ./chtc/rescore_submit.sh --dry-run xco2
#
# Runs on the ACCESS POINT. It exists for one reason: re-scoring is not re-training, but
# the only path to it was a DAG node, and re-submitting a node by hand means retyping the
# two study names and remembering that OSDF refuses to overwrite an archive. Both of those
# fail quietly -- a mismatched study pair scores a different model and reports a plausible
# number, and a stale archive leaves the job held with a transfer error that says nothing.
#
# What it does NOT do: retrain. `bo/eval.sub` transfers `<study>_final.tar.gz` from
# /staging and scores the checkpoints already in it, so this is ~40 minutes of GPU per
# task per split and nothing that was searched or fitted changes.
#
# Why this exists
# -----------------------------
# `predict.py`'s TASKS table gave the scalar tasks `laplace_mode="marglik"`, and that path
# never adds an observation-noise term: the XCO2 Laplace predictive was epistemic-only and
# 5.7x too narrow, which is the whole of its 38.4 % coverage at nominal 95 %. The table now
# says `mc`, like every other task. `bo_eval.sh` passes no --laplace-mode, so the fix
# travels in oco2_code.tar.gz and nothing else needs editing.
#
# EVERY xco2 artifact ever produced carries the defect -- `lap_cov_std` is bit-identical to
# `lap_cov_epi_std` in xco2.npz, xco2_bo.npz and xco2_borep.npz alike -- so the default is
# all three splits rather than the two the search wrote:
#
#   group      bo/eval.sub            -> xco2_bo.npz      Table 1, reference-disjoint block
#   replicate  bo_replicate/eval.sub  -> xco2_borep.npz   Table 1, within-reference block
#   published  eval.sub               -> xco2.npz         the published-inputs column of the
#                                                         input comparison, and the
#                                                         "new split, old recipe" cell of the
#                                                         input/split 2x2.
#
# and `--month` re-scores the temporal hold-out (eval_month_full.sub), whose Laplace NLL and
# coverage columns in the temporal table are the same epistemic-only predictive.
#
# The three splits do not share a submit file or even its macro names: `eval.sub` names the
# checkpoint archives directly (det_tar/flow_tar) because it predates the search, while both
# searched ones name studies (det_study/flow_study) and append `_final.tar.gz` themselves.
# That difference is the reason this script builds the argument list per split instead of
# passing one through.
#
# THE TARBALL IS NOT OPTIONAL. `./make_code_tarball.sh` first, or the job re-runs the same
# code and returns the same wrong numbers, with no error anywhere.
set -euo pipefail

cd "$(dirname "$0")"

# shellcheck source=bo_studies.sh
source ./bo_studies.sh

DRY=0
SPLITS=()
MONTHS=()
TASKS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1 ;;
        --split)   SPLITS+=("${2:?--split needs group, replicate or published}"); shift ;;
        --month)   MONTHS+=("${2:?--month needs a YYYYMM, e.g. 202003}"); shift ;;
        -*) echo "ERROR: unknown flag '$1'. Valid: --dry-run, --split, --month." >&2; exit 2 ;;
        *)  TASKS+=("$1") ;;
    esac
    shift
done
if [ "${#TASKS[@]}" -eq 0 ]; then
    echo "usage: $0 <task> [<task>...] [--split group|replicate|published]" >&2
    echo "                  [--month YYYYMM] [--dry-run]" >&2
    echo "       tasks: xco2 profiles last4 last4_marginal" >&2
    exit 2
fi
# `--month` selects a different experiment, not a different partition of this one, so it
# replaces the split list rather than adding to it: asking for both in one invocation is
# almost always a typo for two invocations, and silently doing both would submit five jobs
# where two were meant.
if [ "${#MONTHS[@]}" -gt 0 ] && [ "${#SPLITS[@]}" -gt 0 ]; then
    echo "ERROR: --split and --month are different experiments. Run them separately." >&2
    exit 2
fi
if [ "${#SPLITS[@]}" -eq 0 ] && [ "${#MONTHS[@]}" -eq 0 ]; then
    SPLITS=(group replicate published)
fi

# Where each split's eval node lives, and what its archive is called. The two differ in
# the directory and in the artifact tag only; the task and the study pair are the same.
#
# The DIRECTORY is the load-bearing part. Every path in `bo/eval.sub` is written `../` --
# `include : ../site.conf`, `executable = ../bo_eval.sh`, the three `../logs/` lines and
# every entry of transfer_input_files -- and HTCondor resolves all of them against the
# working directory of `condor_submit`, not against the submit file. The DAG is always
# submitted from inside `bo/` (`cd chtc/bo && condor_submit_dag bo_all.dag`), so those
# paths are correct there and nowhere else. Submitting `bo/eval.sub` from `chtc/` fails
# immediately with
#
#   Error "../site.conf", Line 0, Include Depth 1: can't open file ../site.conf
#
# which is the good case; a submit file whose only relative path were the executable would
# have been accepted and then held on file transfer half an hour later. So the submit runs
# in a subshell that cd's into the directory first.
#
# `published` is the exception that proves the rule: `eval.sub` lives in chtc/ itself and
# every path in it is written without a prefix, so its working directory is `.` and the
# same argument holds.
split_subdir() {
    case "$1" in
        group)     echo "bo" ;;
        replicate) echo "bo_replicate" ;;
        published) echo "." ;;
        *) echo "ERROR: unknown split '$1'. Valid: group, replicate, published." >&2
           return 2 ;;
    esac
}
# The artifact tag, which is also the archive name's suffix. `published` has none: its
# job is `eval_xco2` and its artifact is `xco2.npz`, from before there was anything to
# distinguish it from.
split_tag() {
    case "$1" in
        group)     echo "bo" ;;
        replicate) echo "borep" ;;
        published) echo "" ;;
    esac
}

# Both checks are warnings under --dry-run so that the plan can be inspected from a laptop,
# where the tarball is gitignored and never exists, and hard errors otherwise.
fail() {
    if [ "$DRY" -eq 1 ]; then echo "WARNING (--dry-run): $*"; else echo "ERROR: $*" >&2; exit 1; fi
}
if [ ! -f oco2_code.tar.gz ]; then
    fail "chtc/oco2_code.tar.gz is absent. Run ./chtc/make_code_tarball.sh -- without it
       the jobs cannot start at all."
# A tarball older than the file whose change is being shipped is the failure mode this
# whole script exists to avoid, and it produces no error at any later stage.
elif [ ../experiments/scoring/predict.py -nt oco2_code.tar.gz ]; then
    fail "experiments/scoring/predict.py is NEWER than oco2_code.tar.gz. The change you
       are re-scoring for would not travel. Run ./chtc/make_code_tarball.sh"
fi

# The mtime check above compares the working tree against the tarball. It cannot tell
# whether the working tree itself has the fix -- a `git pull` that did not bring it, or a
# tarball built on a different branch, both pass. So the content is checked directly, out
# of the archive that will actually be transferred. This is specific to the Laplace fix by
# design: it is the reason this script exists, and a generic "is the code current" check
# is not a thing a shell script can honestly claim.
if [ -f oco2_code.tar.gz ] && printf '%s\n' "${TASKS[@]}" | grep -qx "xco2"; then
    if tar -xzOf oco2_code.tar.gz experiments/scoring/predict.py 2>/dev/null \
         | grep -A1 '"xco2":' | grep -q 'laplace_mode="mc"'; then
        echo "checked: the tarball's predict.py scores xco2 with laplace_mode=mc."
    else
        fail "the predict.py INSIDE oco2_code.tar.gz still scores xco2 with
       laplace_mode=\"marglik\". That is the defect being re-scored for -- the job would
       return the same epistemic-only Laplace. Check that the fix is on this branch:
           grep -A1 '\"xco2\":' experiments/scoring/predict.py
       then ./chtc/make_code_tarball.sh and try again."
    fi
fi

mkdir -p logs

# One submission: clean the archive the previous SUCCESS left behind, then submit from the
# directory the submit file's relative paths were written against.
submit_one() {
    local job=$1 subdir=$2 subfile=$3; shift 3
    local cmd=(condor_submit "$subfile" "$@")

    # OSDF refuses to overwrite an object, and ON_EXIT uploaded one on the previous
    # SUCCESS, so every re-score needs the old archive removed first or the job is
    # held on file transfer with an error about the object rather than about the run.
    if [ "$DRY" -eq 1 ]; then
        echo "    ./fetch_results.sh --clean $job"
        printf '    (cd %s && ' "$subdir"; printf '%s ' "${cmd[@]}"; echo ')'
        return 0
    fi
    ./fetch_results.sh --clean "$job" || {
        echo "    (nothing to clean, or clean failed -- continuing)"
    }
    # Subshell: the cd must not leak into the next iteration, which submits from a
    # different directory, or into the fetch_results.sh call above it.
    ( cd "$subdir" && "${cmd[@]}" )
}

for task in "${TASKS[@]}"; do
    read -r det_study flow_study <<EOF
$(bo_task_studies "$task")
EOF
    for split in ${SPLITS[@]+"${SPLITS[@]}"}; do
        subdir=$(split_subdir "$split")
        tag=$(split_tag "$split")
        job="eval_${task}${tag:+_$tag}"

        # `eval.sub` names the checkpoint ARCHIVES; the two searched submit files name the
        # STUDIES and append `_final.tar.gz` themselves. Same two models either way.
        if [ "$split" = published ]; then
            args=(-a "task = $task"
                  -a "det_tar = ${det_study}.tar.gz"
                  -a "flow_tar = ${flow_study}.tar.gz")
        else
            args=(-a "task = $task"
                  -a "det_study = $det_study"
                  -a "flow_study = $flow_study")
        fi

        printf '%s\n' "--- $job: $det_study + $flow_study  ($subdir/eval.sub)"
        submit_one "$job" "$subdir" eval.sub "${args[@]}"
    done

    for month in ${MONTHS[@]+"${MONTHS[@]}"}; do
        job="eval_${task}_${month}_full"
        printf '%s\n' "--- $job: $det_study + $flow_study  (eval_month_full.sub)"
        submit_one "$job" "." eval_month_full.sub \
            -a "task = $task" \
            -a "month = $month" \
            -a "det_study = $det_study" \
            -a "flow_study = $flow_study"
    done
done

if [ "$DRY" -eq 0 ]; then
    cat <<'EOF'

watch with:  ./chtc/status.py   or   condor_q

The line to read in each job's .out, printed by predict.py where the covariance is
assembled:

    Laplace predictive variance vs residual variance: 1.0x (... aleatoric term present)

`aleatoric term ABSENT`, or a ratio far from 1, means the fix did not travel: the code
tarball is stale. Anything else is a real result.

collect  :  ./chtc/fetch_results.sh --best         # group      -> <task>_bo.npz
            ./chtc/fetch_results.sh --rerun        # replicate  -> <task>_borep.npz
            ./chtc/fetch_results.sh --eval         # published  -> <task>.npz
            ./chtc/fetch_results.sh --months-full  # months     -> the month artifacts

then, on a machine with numpy:  ./experiments/scoring/rebuild.sh
EOF
fi

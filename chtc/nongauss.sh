#!/bin/bash
# Build the non-Gaussianity tables from the gaussianize artifacts, inside the container.
#
#     condor_submit nongauss.sub
#
# The sibling of score.sh, and it exists for the same reason: the access point may not have
# numpy/pandas, and the artifacts are ~120 MB each once the 512 stored samples are counted,
# which is not something to copy to a laptop to answer a six-row question. Two CPU-minutes.
#
# What it produces, in experiments/scoring/tables/:
#
#   nongauss_shape.csv     the flow against its own moment-matched Gaussian -- the number
#                          the non-Gaussianity claim needs, with the 1/S intercept beside it
#   nongauss_budgets.csv   the same control as a function of sample budget
#   nongauss_levels.csv    per-level posterior shape, and the state ensemble beside it
#   nongauss_clusters.csv  every headline number by geophysical template
#   nongauss_mixture.csv   what the state-generating mechanism actually is
#   nongauss.md            all five rendered, ready to paste
set -euo pipefail

JOB="nongauss"

# shellcheck source=job_common.sh
source ./job_common.sh

export OCO2_PACK_PATHS="experiments/scoring/tables chtc/GIT_PROVENANCE.txt"

oco2_banner "$JOB"
oco2_trap_pack "$JOB"
oco2_setup                      # no GPU, no spectra: the artifacts carry everything

# The split table, for the cluster labels, `mixture_component` and `dof_co2`. Only the
# covariate archive has the augmented table -- oco2_unpack_split alone gives
# group_split_v1.csv, which has none of those columns -- and oco2_unpack_covariates
# insists on running after it.
oco2_unpack_split
oco2_unpack_covariates

# Only what was actually transferred is unpacked. nongauss.sub names the archives it
# expects, but an eval that has not been run has nothing in /staging, and
# oco2_unpack_results treats a named-but-absent archive as an error -- correct for a
# training job's own dependency, wrong here, where a missing task should cost its rows and
# nothing else.
#
# Both suffixes are accepted. `_gauss` carries the streamed control (thousands of samples,
# the plug-in penalty extrapolated away); `_bo` carries only the 512 stored samples and
# nongauss.py falls back to a control fitted from those, labelling the row
# `control = stored samples`. Do not transfer both for one task: two rows differing only in
# how well the control was estimated is an invitation to quote the wrong one, and the
# guard below refuses rather than letting it happen.
ARCHIVES=""
declare -A SEEN=()
for task in xco2 profiles last4 last4_marginal last1 last1_marginal; do
    picked=""
    for suffix in gauss bo; do
        a="eval_${task}_${suffix}.tar.gz"
        if [ -f "$a" ]; then
            if [ -n "$picked" ]; then
                echo "ERROR: both eval_${task}_gauss.tar.gz and eval_${task}_bo.tar.gz were"
                echo "       transferred. They hold two artifacts of one task whose only"
                echo "       difference is the control, and scoring both would put two rows"
                echo "       for $task in nongauss_shape.csv. Name one in nongauss.sub."
                exit 2
            fi
            picked="$suffix"
            ARCHIVES="$ARCHIVES $a"
            SEEN[$task]="$suffix"
        fi
    done
    if [ -z "$picked" ]; then
        echo "note: no archive for '$task'; it will have no rows in the tables."
    fi
done
if [ -z "$ARCHIVES" ]; then
    echo "ERROR: none of the eval archives were transferred. Check transfer_input_files"
    echo "       in nongauss.sub against ./fetch_results.sh --status."
    exit 2
fi
# shellcheck disable=SC2086  -- deliberate word splitting: one argument per archive.
oco2_unpack_results $ARCHIVES

RESULTS="$PROJECT/experiments/scoring/results/group_split_v1"
SPLIT_CSV="$FILES_DIR/splits/group_split_v1_cov.csv"

# The exact file list, not a glob. An earlier archive may have left a sibling .npz of the
# same task in that directory -- unpack order decides which -- and `*.npz` would score
# both. Naming them is the difference between a table and a coin flip.
NPZ=()
for task in "${!SEEN[@]}"; do
    f="$RESULTS/${task}_${SEEN[$task]}.npz"
    if [ ! -s "$f" ]; then
        echo "ERROR: eval_${task}_${SEEN[$task]}.tar.gz unpacked but $f is not there."
        echo "       What did arrive:"
        find "$RESULTS" -name '*.npz' -printf '         %f  %s bytes\n' 2>/dev/null | sort
        exit 2
    fi
    NPZ+=("$f")
done

echo "--- artifacts ---"
for f in "${NPZ[@]}"; do
    printf '    %-28s %s bytes\n' "$(basename "$f")" "$(stat -c %s "$f")"
    # The stage schedule is what makes the nfg_* columns interpretable, and its absence is
    # what distinguishes a streamed control from the 512-sample fallback. Print it here so
    # the .out file says which of the two each row came from, without opening the CSV.
    python3 - "$f" <<'PY'
import json, sys
import numpy as np
meta = json.loads(bytes(np.load(sys.argv[1], allow_pickle=False)["__meta__"]).decode())
nf = meta.get("nf") or {}
stages = nf.get("gaussianize_stages") or []
print("        control: streamed, stages {}".format(stages) if stages
      else "        control: none stored -- falls back to the {} stored samples".format(
          meta.get("n_nf_samples", "?")))
PY
done
echo

python3 experiments/scoring/nongauss.py "${NPZ[@]}" \
    --split-csv "$SPLIT_CSV" \
    --out "$PROJECT/experiments/scoring/tables"

echo
echo "=============================================================="
echo " $JOB finished"
echo "=============================================================="
# The whole point of the job is these six files; printing the headline one into the log
# means the answer is in `condor_tail`/the .out file before the archive is even fetched.
ls -1 "$PROJECT/experiments/scoring/tables" | sed 's/^/    /'
echo
echo "--- nongauss.md ---"
cat "$PROJECT/experiments/scoring/tables/nongauss.md" 2>/dev/null || \
    echo "(nongauss.py wrote no markdown; check the error above)"

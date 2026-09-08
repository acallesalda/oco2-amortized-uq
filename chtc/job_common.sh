#!/bin/bash
# Shared setup for every OCO-2 job on CHTC. Sourced by prep.sh and train.sh; not meant to
# be run on its own.
#
# The job runs inside the Apptainer image, with the HTCondor scratch directory as CWD.
# The submit files set `container_target_dir = /workspace/project`; on CHTC as of
# 2026-08-14 the observed CWD is `/workspace/project/scratch`, one level below that --
# HTCondor maps the sandbox root to the target and works inside its `scratch` subdirectory.
# So the project's hardcoded absolute paths do NOT resolve as written, and oco2_setup's
# rewrite is load-bearing rather than a safety net. Nothing here assumes either layout:
# PROJECT is whatever $PWD turns out to be, and every path derives from it.

set -euo pipefail

# ---------------------------------------------------------------------------------
# Environment report. Worth the twenty lines: nearly every remote failure is explained
# by something in here, and the .err file is all you get after the fact.
# ---------------------------------------------------------------------------------
oco2_banner() {
    echo "=============================================================="
    echo " OCO-2 job: ${1:-unnamed}"
    echo " date      : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo " host      : $(hostname)"
    echo " cwd       : $PWD"
    echo " condor    : cluster=${_CONDOR_JOB_AD:+set} slot=${_CONDOR_SLOT:-?}"
    echo " cpus      : ${_CONDOR_ASSIGNED_CPUS:-$(nproc)}"
    echo " memory    : $(free -g 2>/dev/null | awk 'NR==2 {print $2" GB"}' || echo '?')"
    echo " disk free : $(df -h . | awk 'NR==2 {print $4}')"
    echo " python    : $(python3 --version 2>&1)"
    echo "=============================================================="
    echo
}

oco2_gpu_check() {
    echo "--- GPU ---"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version \
                   --format=csv,noheader || true
    else
        echo "nvidia-smi not found"
    fi
    python3 - <<'PY'
import sys, torch
ok = torch.cuda.is_available()
print(f"torch {torch.__version__} | cuda available: {ok}")
if ok:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  sm_{p.major}{p.minor}  {p.total_memory/2**30:.1f} GiB")
else:
    # Falling through to CPU would silently turn a 4-hour job into a 4-day one, so stop.
    print("ERROR: no usable GPU. Check request_gpus and the capability bounds in "
          "site.conf -- the cu121 wheels in this image do not support sm_100+.",
          file=sys.stderr)
    sys.exit(1)
PY
    echo
}

# What to append to an archive name so that the same study, run on two splits, does not
# produce two archives with one name.
#
# OSDF refuses to overwrite an object, so a collision is not a silently replaced result --
# it is a held job with a transfer error that says nothing about why. Cheap to prevent,
# expensive to diagnose. Deliberately usable before oco2_setup: bo_final.sh names its JOB
# (and therefore its archive) before it lays out the project.
oco2_split_suffix() {
    case "${OCO2_SPLIT:-group_split_v1}" in
        replicate_split_v1) echo "_rep" ;;
        full_feb_v1)        echo "_full" ;;
        ext_v1)             echo "_ext" ;;
        ext_cap_v1)         echo "_extcap" ;;
        # size_90_v1 -> _s90. The learning-curve subsets differ from each other only in
        # how many training scenes they keep, so the size has to be in the archive name.
        size_*_v1)          local k="${OCO2_SPLIT#size_}"; echo "_s${k%_v1}" ;;
        *)                  echo "" ;;
    esac
}

# ---------------------------------------------------------------------------------
# Lay out the project inside the scratch directory.
# ---------------------------------------------------------------------------------
# Sets: PROJECT, FILES_DIR, STATS_DIR, WEIGHTS_ROOT, OCO2_SPLIT, SPLIT_CSV, SPLIT_STATS
#
# OCO2_SPLIT names which partition this job trains and scores on, and everything that
# depends on the partition -- the table, the statistics, the weights tree, the scoring
# artifacts -- derives from it. Two values are supported:
#
#   group_split_v1      the reference-sounding-disjoint split          (default)
#   replicate_split_v1  the original replicate-level split, written out as a table by
#                       experiments/pipelines/make_replicate_split.py
#   size_<k>_v1         group_split_v1 with its training partition subsampled to k whole
#                       reference soundings, stratified by cluster and nested across k
#                       (experiments/pipelines/make_size_splits.py). Val and test are
#                       untouched, so a curve over k is a curve in training scenes and
#                       nothing else -- the model half of the learning-curve diagnostic.
#   ext_cap_v1          ext_v1 with the extension capped at 100 replicates per scene, so
#                       the training rows are scene-proportional. THE ONE TO USE: uncapped,
#                       clusters 8 and 9 take 42.5 % of the training soundings while holding
#                       25.1 % of the scenes, and the capped table is what corrects that
#                       cost -- XCO2 gained 16 % and both profile tasks regressed.
#   ext_v1              group_split_v1 with the February EXTENSION delivery folded into
#                       its TRAINING partition: 86 new reference soundings in clusters 8
#                       and 9 plus 200 fresh replicates of the 26 shared training scenes,
#                       so 447 training scenes and 58,115 spectra against 361 and 36,003.
#                       Validation and test are byte-for-byte group_split_v1's, on
#                       February's own realisations, which is what keeps every number
#                       comparable with the published ones and with the size curve
#                       (experiments/pipelines/make_ext_split.py).
#   full_feb_v1         group_split_v1 with its TEST scenes folded back into training and
#                       its validation scenes untouched: 406 + 45, no test partition
#                       (experiments/pipelines/make_full_feb_split.py). The test set for
#                       these models is another month -- see months_full.dag -- so a
#                       February hold-out would only be 45 scenes not learned from. The
#                       validation partition stays because the checkpoint is selected on
#                       it and the Laplace's aleatoric diagonal is calibrated on it.
#
# It is set from a submit file's `environment` rather than passed as an argument because
# every script here already takes its arguments from the DAG, and a split that travelled
# as argument $4 of five scripts would be four more places to get it wrong. The weights
# root carrying the split name is what stops two runs of the same study, on two
# partitions, from writing to one directory -- which would leave no trace anywhere.
oco2_setup() {
    PROJECT="$PWD"
    FILES_DIR="$PROJECT/scratch/simulated/files"
    STATS_DIR="$PROJECT/experiments/pipelines/stats"
    OCO2_SPLIT="${OCO2_SPLIT:-group_split_v1}"
    case "$OCO2_SPLIT" in
        group_split_v1|replicate_split_v1|full_feb_v1|ext_v1|ext_cap_v1|size_*_v1) ;;
        *)  echo "ERROR: OCO2_SPLIT='$OCO2_SPLIT' is not a split this pipeline knows."
            echo "       Valid: group_split_v1, replicate_split_v1, full_feb_v1, ext_v1,"
            echo "              ext_cap_v1,"
            echo "              size_<k>_v1 (k training reference soundings)"
            return 1 ;;
    esac
    WEIGHTS_ROOT="$PROJECT/experiments/weights/$OCO2_SPLIT"
    SPLIT_CSV="$FILES_DIR/splits/${OCO2_SPLIT}_cov.csv"
    SPLIT_STATS="$STATS_DIR/$OCO2_SPLIT"
    export PROJECT FILES_DIR STATS_DIR WEIGHTS_ROOT OCO2_SPLIT SPLIT_CSV SPLIT_STATS
    echo "--- split: $OCO2_SPLIT ---"

    echo "--- unpacking code ---"
    if [ ! -f oco2_code.tar.gz ]; then
        echo "ERROR: oco2_code.tar.gz was not transferred. Run make_code_tarball.sh on"
        echo "       the access point before submitting."
        return 1
    fi
    tar -xzf oco2_code.tar.gz
    if [ -f chtc/GIT_PROVENANCE.txt ]; then cat chtc/GIT_PROVENANCE.txt; fi

    mkdir -p "$FILES_DIR" "$STATS_DIR" "$WEIGHTS_ROOT"

    # The configs and a handful of module-level defaults hardcode absolute /workspace
    # paths (the layout of the development machine). Three different roots appear:
    #     /workspace/project   -- code, scratch data, weights
    #     /workspace/data      -- the read-only source volume
    #     /workspace/outputs   -- conf/config_profiles.yaml's default hydra.run.dir
    # Only the first is mapped by container_target_dir, and only /workspace/project is
    # writable, so the rewrite runs unconditionally: even in the happy path it is what
    # keeps a config that forgot to override hydra.run.dir from trying to mkdir into a
    # read-only image.
    #
    # @@P@@ is a placeholder so the expressions cannot cascade into each other -- with a
    # literal substitution, rule 3 would re-match rule 1's output and produce
    # /workspace/project/project when PROJECT really is /workspace/project.
    echo "--- rewriting absolute /workspace paths -> $PROJECT ---"
    local n=0
    while IFS= read -r f; do
        sed -i -e 's|/workspace/project|@@P@@|g' \
               -e 's|/workspace/data|@@P@@/data|g' \
               -e 's|/workspace/|@@P@@/|g' \
               -e "s|@@P@@|$PROJECT|g" "$f"
        n=$((n + 1))
    done < <(grep -rl "/workspace/" code experiments --include='*.yaml' \
                                                      --include='*.yml' \
                                                      --include='*.py' 2>/dev/null || true)
    echo "    rewrote $n files"

    # Check for paths the rewrite missed -- and exclude the ones it got right.
    #
    # This check used to grep for "/workspace/" and flag every hit. That is wrong in the
    # happy path: when container_target_dir works, $PROJECT *is* /workspace/project/...,
    # so every correctly rewritten path still contains the string, and the warning fired
    # on all of them. Filtering by $PROJECT leaves only genuine leftovers -- something
    # still pointing at /workspace/outputs, say, that no rule reached.
    #
    # The `|| true` on the second line is not decoration. `grep | head` under
    # `set -o pipefail` returns 141 when head closes the pipe and grep takes SIGPIPE, and
    # this was the last command of the function, so oco2_setup returned 141 and `set -e`
    # killed the job before it did any work. Every job in both DAGs would have died here.
    # Note the globs: the rewrite above touches *.yaml/*.yml/*.py, but this scans every
    # text file (-I skips binaries). Scanning the same set would make the check dead code
    # -- rule 3 is a catch-all, so nothing it looks at can survive it. The leftovers that
    # can actually exist are in file types the rewrite never opens: a /workspace path in a
    # .json, a .sh, a .cfg. That is what this is for.
    local leftovers
    leftovers=$(grep -rInH "/workspace/" code experiments 2>/dev/null \
                | grep -v -F "$PROJECT" || true)
    if [ -n "$leftovers" ]; then
        echo "WARNING: /workspace paths the rewrite did not reach."
        echo "         These are in file types oco2_setup does not rewrite, so they will"
        echo "         resolve against the read-only image and fail at runtime."
        echo "         ($(printf '%s\n' "$leftovers" | wc -l) lines; first 5 shown)"
        printf '%s\n' "$leftovers" | head -5 || true
    fi

    export PYTHONPATH="$PROJECT/code:$PROJECT:${PYTHONPATH:-}"
    export MPLBACKEND=Agg
    export HDF5_USE_FILE_LOCKING=FALSE

    # train.py and train_profiles.py log to a local `mlruns` directory. requirements.txt
    # does not pin mlflow, so the image resolved a version that refuses the filesystem
    # tracking backend outright:
    #
    #   MlflowException: The filesystem tracking backend (e.g., './mlruns') is in
    #   maintenance mode ... set MLFLOW_ALLOW_FILE_STORE=true to opt out of this exception.
    #
    # It raises at `mlflow.start_run()`, twenty seconds in, before a single batch. The
    # variable name is taken from that message, not guessed.
    #
    # Set here rather than in the image so it takes effect on the next submit: job_common.sh
    # travels as a transfer_input_files entry, so no rebuild is needed. mlflow is also
    # pinned in requirements.txt now, which is the real fix and costs a rebuild -- this is
    # the one that gets the DAG running today. Exactly the drift the reproducibility note
    # in requirements.txt warned about, arriving on schedule.
    export MLFLOW_ALLOW_FILE_STORE=true

    # Stream the training logs instead of delivering them in 8 KB bursts.
    #
    # Python block-buffers stdout when it is a file rather than a terminal, which is what
    # HTCondor gives it. `logging` is unaffected -- StreamHandler flushes on every record,
    # so the hydra-based trainers (train.py, train_profiles.py) tick along fine -- but the
    # three flow trainers report progress with bare print(), e.g.
    #     [ep 01] val_nll=-38.4213 | best=-38.4213
    # and those sit in the buffer. On a 35-epoch run that can mean an .out file that looks
    # frozen for an hour and then arrives all at once, which is indistinguishable from a
    # hung job at exactly the moment you want to tell the difference.
    export PYTHONUNBUFFERED=1
    # Match the thread pools to what HTCondor actually allocated. Left at the default,
    # OpenMP spawns one thread per physical core on the machine and thrashes the slot.
    local ncpu="${_CONDOR_ASSIGNED_CPUS:-$(nproc)}"
    export OMP_NUM_THREADS="$ncpu"
    export MKL_NUM_THREADS="$ncpu"
    echo "PYTHONPATH=$PYTHONPATH"
    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
    echo
}

# ---------------------------------------------------------------------------------
# Data.
# ---------------------------------------------------------------------------------
# The Zenodo archives are rooted at the development-machine layout:
#   oco2_processed_{train,test}.tar.gz  ->  data/oco2/simulated/files/{train,test}/...
#   oco2_profile_stats.tar.gz           ->  project/experiments/pipelines/stats/...
#   oco2_unprocessed_og_data.tar.gz     ->  data/oco2/simulated/og_data/...
# so the strip-components counts below are fixed by the archives, not by choice.
oco2_unpack_data() {
    local t0=$SECONDS

    echo "--- unpacking processed spectra ---"
    for part in train test; do
        local tarball="oco2_processed_${part}.tar.gz"
        if [ ! -f "$tarball" ]; then
            echo "ERROR: $tarball was not transferred. Check transfer_input_files."
            return 1
        fi
        mkdir -p "$FILES_DIR"
        tar -xzf "$tarball" --strip-components=4 -C "$FILES_DIR"
        local n
        n=$(find "$FILES_DIR/$part" -name '*.npy' 2>/dev/null | wc -l)
        echo "    $part: $n .npy files"
        if [ ! -f "$FILES_DIR/$part/aux_data.csv" ]; then
            echo "ERROR: $FILES_DIR/$part/aux_data.csv missing after extraction."
            return 1
        fi
        # Free the scratch disk as we go; ~780 MB of tarball we will not read again.
        rm -f "$tarball"
    done

    if [ -f oco2_profile_stats.tar.gz ]; then
        echo "--- unpacking profile stats ---"
        tar -xzf oco2_profile_stats.tar.gz --strip-components=1 -C "$PROJECT"
        ls -1 "$STATS_DIR"
        rm -f oco2_profile_stats.tar.gz
    fi

    echo "    unpacked in $((SECONDS - t0))s"
    echo
}

# Just the annotations, for a job that never opens a spectrum.
#
# prep_replicate.sub is the only caller today. Its three steps are all CSV work:
# make_replicate_split.py reads files/{train,test}/aux_data.csv, compute_split_stats.py
# reads the split table that writes, and augment_split.py reads that table and the
# covariate CSV. None of them opens a .npy -- so oco2_unpack_data's 43k-file extraction
# was the job's entire runtime, and the 40 GB of scratch it needs to hold the result was
# why the job sat idle waiting for a slot big enough to do nothing with.
#
# gzip has no index, so both archives are still read end to end -- what goes away is
# writing 43k small files and the disk to keep them on. Tens of seconds, not ten minutes.
oco2_unpack_annotations() {
    local t0=$SECONDS

    echo "--- unpacking annotations only (no spectra) ---"
    for part in train test; do
        local tarball="oco2_processed_${part}.tar.gz"
        if [ ! -f "$tarball" ]; then
            echo "ERROR: $tarball was not transferred. Check transfer_input_files."
            return 1
        fi
        mkdir -p "$FILES_DIR"
        # Same --strip-components as oco2_unpack_data, so the file lands in the same
        # place; tar exits nonzero if the pattern matches nothing, which is the failure
        # we want rather than an empty directory three steps later.
        tar -xzf "$tarball" --strip-components=4 -C "$FILES_DIR" --wildcards '*aux_data.csv'
        if [ ! -f "$FILES_DIR/$part/aux_data.csv" ]; then
            echo "ERROR: $FILES_DIR/$part/aux_data.csv missing after extraction."
            return 1
        fi
        echo "    $part: $(($(wc -l < "$FILES_DIR/$part/aux_data.csv") - 1)) annotation rows"
        rm -f "$tarball"
    done

    echo "    unpacked in $((SECONDS - t0))s"
    echo
}

# Unpack the group split table + its statistics, produced by the prep job.
oco2_unpack_split() {
    # One archive per split. The group split's is built by prep.sub and carries the table
    # and its statistics; the replicate split's is built by prep_replicate.sub and carries
    # the augmented table as well, because the expensive half of the covariate work --
    # reading 451 HDF5 files -- is split-independent and has already run.
    local tarball root
    case "$OCO2_SPLIT" in
        replicate_split_v1) tarball="oco2_replicate_split_v1.tar.gz"
                            root="prep_replicate.sub" ;;
        # As the replicate split: one archive with the table, its statistics AND the
        # augmented table, because the covariate extraction is split-independent and
        # covariates.sub has already paid for it.
        full_feb_v1)        tarball="oco2_full_feb_v1.tar.gz"
                            root="prep_full_feb.sub" ;;
        # As full_feb_v1's, plus the extension's own .npy spectra and ACOS covariances
        # under files/202002ext/ -- roughly 0.5 GB more. They travel in the split archive
        # rather than in a fourth processed-data tarball because they are only ever read
        # by a job that has already decided to train on this split.
        ext_v1)             tarball="oco2_ext_v1.tar.gz"
                            root="prep_ext.sub" ;;
        # ext_v1 with the extension capped at 100 replicates per scene and the shared
        # scenes' extra replicates dropped, so every training scene carries the same
        # replicate count and clusters 8 and 9 take 25.0 % of the rows against 25.1 % of
        # the scenes. Uncapped they take 42.5 % of the rows
        # and the two profile tasks paid for it.
        ext_cap_v1)         tarball="oco2_ext_cap_v1.tar.gz"
                            root="prep_ext_cap.sub" ;;
        # The learning-curve subsets, one archive each, same layout as full_feb_v1's.
        size_*_v1)          tarball="oco2_${OCO2_SPLIT}.tar.gz"
                            root="prep_size.sub -a 'size=${OCO2_SPLIT#size_}'" ;;
        *)                  tarball="oco2_split_v1.tar.gz"
                            root="the root node of oco2.dag" ;;
    esac
    if [ ! -f "$tarball" ]; then
        echo "ERROR: $tarball was not transferred. Run $root first."
        return 1
    fi
    echo "--- unpacking $OCO2_SPLIT ---"
    tar -xzf "$tarball" -C "$PROJECT"
    local csv="$FILES_DIR/splits/${OCO2_SPLIT}.csv"
    if [ ! -f "$csv" ]; then
        echo "ERROR: $csv missing after extraction."
        return 1
    fi
    echo "    split table: $(wc -l < "$csv") rows"
    echo "    stats:"
    ls -1 "$SPLIT_STATS"
    rm -f "$tarball"
    echo
}

# Unpack the recovered covariates: the augmented split table and the covariate statistics
# the loader encodes with. Must run AFTER oco2_unpack_split, because both write into
# experiments/pipelines/stats/group_split_v1 and this one only adds a file to it.
#
# Optional by design: a job that trains with covariates=legacy does not need it, and
# passing --optional lets one script serve both.
oco2_unpack_covariates() {
    local tarball="oco2_covariates_v1.tar.gz"

    # Every split but the group one carries its augmented table inside its own split
    # archive, which oco2_unpack_split has already extracted -- so there is nothing to
    # unpack here, only something to check. Verifying rather than returning silently
    # matters: a job that trains with covariates=scene_prior against a table that has none
    # does not fail, it trains a different model.
    if [ "$OCO2_SPLIT" != "group_split_v1" ]; then
        local f builder
        case "$OCO2_SPLIT" in
            full_feb_v1) builder="oco2_full_feb_v1.tar.gz; rebuild it with prep_full_feb.sub" ;;
            ext_v1)      builder="oco2_ext_v1.tar.gz; rebuild it with prep_ext.sub" ;;
            ext_cap_v1)  builder="oco2_ext_cap_v1.tar.gz; rebuild it with prep_ext_cap.sub" ;;
            size_*_v1)   builder="oco2_${OCO2_SPLIT}.tar.gz; rebuild it with prep_size.sub" ;;
            *)           builder="oco2_replicate_split_v1.tar.gz; rebuild it with prep_replicate.sub" ;;
        esac
        for f in "$SPLIT_CSV" "$SPLIT_STATS/covariate_stats.json"; do
            if [ ! -f "$f" ]; then
                echo "ERROR: $f is missing. For $OCO2_SPLIT the covariates travel inside"
                echo "       $builder."
                return 1
            fi
        done
        echo "--- covariates for $OCO2_SPLIT: already in the split archive ---"
        echo "    augmented split: $(wc -l < "$SPLIT_CSV") rows"
        echo
        return 0
    fi

    if [ ! -f "$tarball" ]; then
        if [ "${1:-}" = "--optional" ]; then
            echo "--- no $tarball; covariate presets other than 'legacy' will fail ---"
            echo
            return 0
        fi
        echo "ERROR: $tarball was not transferred. Run covariates.sub first"
        echo "       (it needs prep.sub's split and the raw .h5 archive in /staging)."
        return 1
    fi
    echo "--- unpacking covariates ---"
    tar -xzf "$tarball" -C "$PROJECT"
    local csv="$SPLIT_CSV"
    local stats="$SPLIT_STATS/covariate_stats.json"
    for f in "$csv" "$stats"; do
        if [ ! -f "$f" ]; then
            echo "ERROR: $f missing after extraction. The archive layout is not what was"
            echo "       expected:"
            tar -tzf "$tarball" | head -10
            return 1
        fi
    done
    echo "    augmented split: $(wc -l < "$csv") rows"
    echo "    covariates: $(python3 -c '
import json, sys
print(len(json.load(open(sys.argv[1]))["columns"]))' "$stats")"
    rm -f "$tarball"
    echo
}

# Unpack a deterministic model's results so a flow can warm-start from its backbone.
oco2_unpack_warmstart() {
    local tarball="${1:-}"
    # Note the `if` rather than `[ ... ] && return 0`: under `set -e` a bare test-and-act
    # whose test is FALSE returns 1 and kills the script. Same reason everywhere below.
    if [ -z "$tarball" ]; then return 0; fi
    if [ ! -f "$tarball" ]; then
        echo "ERROR: warm-start archive $tarball was not transferred."
        echo "       The DAG parent that produces it must have completed."
        return 1
    fi
    echo "--- unpacking warm-start checkpoint from $tarball ---"
    tar -xzf "$tarball" -C "$PROJECT"
    find "$WEIGHTS_ROOT" -name '*.pt' -printf '    %p (%s bytes)\n'
    rm -f "$tarball"
    echo
}

# Unpack the PUBLISHED checkpoints from the Zenodo weights archive into
# experiments/weights/, next to (never on top of) anything under group_split_v1/.
#
# These are not in oco2_code.tar.gz: make_code_tarball.sh excludes every .pt, because 245
# MB of checkpoints no training job reads has no business travelling with the source. So
# without this the interactive session has experiments/weights/*/ containing only the
# .yaml and .json files, and dump_checkpoint_config.py reports every path MISSING -- which
# looks like a broken script rather than an absent input.
#
# Stage it first, once:  ./stage_data.sh weights
oco2_unpack_weights() {
    local tarball="${1:-oco2_model_weights.tar.gz}"
    if [ ! -f "$tarball" ]; then
        echo "ERROR: $tarball was not transferred."
        echo "       Add \$(OSDF)/$tarball to transfer_input_files, and make sure it is"
        echo "       in /staging -- './stage_data.sh --list' will say. It is not staged"
        echo "       by 'stage_data.sh train test stats'."
        return 1
    fi

    # The Zenodo archives are rooted at the development-machine layout and each needs its own
    # strip-components count. Rather than hardcode a fourth guess, find the depth at
    # which experiments/weights/ appears and strip exactly that much.
    local first strip
    first=$(tar -tzf "$tarball" | grep -m1 'experiments/weights/' || true)
    if [ -z "$first" ]; then
        echo "ERROR: $tarball contains no experiments/weights/ path. First entries:"
        tar -tzf "$tarball" | head -5 || true
        return 1
    fi
    strip=$(echo "${first%%experiments/weights/*}" | tr -cd '/' | wc -c)
    echo "--- unpacking published weights ($tarball, strip-components=$strip) ---"
    tar -xzf "$tarball" --strip-components="$strip" -C "$PROJECT"

    local n
    n=$(find "$PROJECT/experiments/weights" -name '*.pt' | wc -l)
    echo "    $n .pt files under experiments/weights/"
    if [ "$n" -eq 0 ]; then
        echo "ERROR: no .pt files after extraction. Layout is not what was expected:"
        tar -tzf "$tarball" | head -10 || true
        return 1
    fi
    find "$PROJECT/experiments/weights" -name '*.pt' -printf '      %p  %s bytes\n' | sort
    echo
}

# Unpack any number of finished training archives, so an evaluation job can see the
# checkpoints of several models at once. They all extract into the same
# experiments/weights/group_split_v1/ tree, which is what predict.py expects as its
# --weights-root; the archives are disjoint by construction (one subdirectory each).
oco2_unpack_results() {
    local tarball
    for tarball in "$@"; do
        if [ -z "$tarball" ]; then continue; fi
        if [ ! -f "$tarball" ]; then
            echo "ERROR: result archive $tarball was not transferred."
            echo "       The training job that produces it must have completed and left"
            echo "       its archive in /staging/<l>/<netid>/oco2/results/."
            return 1
        fi
        echo "--- unpacking $tarball ---"
        # Only the weights matter here; skipping mlruns/outputs keeps scratch small.
        tar -xzf "$tarball" -C "$PROJECT" experiments/weights 2>/dev/null || \
            tar -xzf "$tarball" -C "$PROJECT"
        rm -f "$tarball"
    done
    if [ -d "$WEIGHTS_ROOT" ]; then
        find "$WEIGHTS_ROOT" -name '*.pt' -printf '    %p (%s bytes)\n' | sort
    fi
    echo
}

# Fail loudly if a warm-start checkpoint is missing.
#
# This exists because the training scripts do the opposite. train_nf_profiles_4d.py prints
#   "[warmstart] checkpoint not found: ... -> training from scratch"
# and carries on, which in a batch job means a flow that was supposed to warm-start from
# the deterministic backbone quietly trains cold, finishes, and reports a plausible-looking
# number that is not the experiment we said we ran. Better to lose the job than the result.
# As oco2_require_ckpt, but for a checkpoint that may sit in more than one place.
#
# train_profiles.py writes weights_dir/global_best/net_profiles.pt unconditionally,
# and weights_dir/net_profiles.pt only when the run beats the val_rmse recorded in
# log/best_runs/best_ckpt/<seed>/BEST.json. Which of the two exists is therefore a
# property of the run history, not of the task -- and code/evals/checkpoints.py has
# always tried both, so predict.py never cared. eval.sh gated on the top-level path
# alone, which is why eval_last1 exited 2 before predict.py started and returned a
# 512-byte archive twice: an empty tar is what oco2_trap_pack produces when the job
# dies before writing anything.
#
# Takes candidates in the order checkpoints.py lists them and accepts the first that
# exists. A gate that is stricter than the loader it guards is not a safety check.
oco2_require_ckpt_any() {
    local c
    for c in "$@"; do
        if [ -s "$c" ]; then
            echo "checkpoint: $c ($(stat -c %s "$c") bytes)"
            return 0
        fi
    done
    echo "ERROR: none of the expected checkpoint paths exist:"
    printf '         %s\n' "$@"
    echo "       The training archive unpacked, but not to a layout this job knows."
    echo "       What did arrive:"
    if [ -d "$WEIGHTS_ROOT" ]; then
        find "$WEIGHTS_ROOT" -type f -printf '         %p\n'
    fi
    return 1
}

oco2_require_ckpt() {
    local ckpt="$1"
    if [ ! -s "$ckpt" ]; then
        echo "ERROR: required warm-start checkpoint is missing or empty:"
        echo "         $ckpt"
        echo "       The training script would silently fall back to training from"
        echo "       scratch, so this job is stopping instead. Check that the DAG parent"
        echo "       node completed and that its archive was transferred."
        if [ -d "$WEIGHTS_ROOT" ]; then
            find "$WEIGHTS_ROOT" -type f -printf '         %p\n'
        fi
        return 1
    fi
    echo "warm start: $ckpt ($(stat -c %s "$ckpt") bytes)"
    echo
}

# ---------------------------------------------------------------------------------
# Results.
# ---------------------------------------------------------------------------------
# Always called, success or failure, via a trap. A failed run's partial checkpoints and
# hydra logs are usually the only way to work out what went wrong on a machine you can
# no longer reach, so they come home too.
oco2_pack_results() {
    local job="$1" status="$2"
    # May fire before oco2_setup has run -- the trap is installed first on purpose, so
    # that a failure during setup still produces the archive HTCondor was promised. A
    # missing output file puts the job on hold with a transfer error that says nothing
    # about the actual cause.
    local root="${PROJECT:-$PWD}"
    local out="$root/${job}.tar.gz"
    cd "$root"

    echo
    echo "--- packing results ($job, exit status $status) ---"

    cat > "$root/run_meta_${job}.json" <<EOF
{
  "job": "$job",
  "exit_status": $status,
  "host": "$(hostname)",
  "finished_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "elapsed_seconds": $SECONDS,
  "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
}
EOF

    # Collect whatever exists. `for ... if` rather than `[ -d x ] && paths+=(x)`, which
    # under `set -e` would abort the trap at the first path that happens to be absent --
    # producing a silently truncated archive, which is the worst possible failure here.
    #
    # OCO2_PACK_PATHS overrides the default set. Evaluation jobs use it: they *unpack*
    # 235 MB of checkpoints as input, and packing those straight back into the result
    # archive would triple the transfer for nothing.
    local -a paths=()
    local p
    # The two split-dependent entries follow OCO2_SPLIT. A replicate-split training
    # job writes to experiments/weights/replicate_split_v1/, and a default naming only
    # the group split would pack an archive with no checkpoints in it -- quietly,
    # because "collect whatever exists" cannot tell an absent path from an empty run.
    for p in ${OCO2_PACK_PATHS:-"experiments/weights/${OCO2_SPLIT:-group_split_v1}" \
             "experiments/weights/replicate_control" \
             "scratch/simulated/files/splits" \
             "experiments/pipelines/stats/${OCO2_SPLIT:-group_split_v1}" \
             "experiments/scoring/results" \
             "experiments/scoring/tables" \
             "mlruns" \
             "outputs" \
             "chtc/GIT_PROVENANCE.txt"}; do
        if [ -e "$p" ]; then paths+=("$p"); fi
    done
    paths+=("run_meta_${job}.json")

    tar -czf "$out" -C "$root" "${paths[@]}" 2>/dev/null || {
        echo "WARNING: tar reported errors; the archive may be incomplete."
    }

    echo "    wrote $(basename "$out") ($(du -h "$out" | cut -f1))"
    tar -tzf "$out" | grep -E '\.(pt|json|csv|yaml|npy)$' | head -30 || true
    echo
}

# Wire oco2_pack_results to run on every exit path.
oco2_trap_pack() {
    local job="$1"
    # shellcheck disable=SC2064  # $job must expand now, not at trap time.
    # `|| true` so a packing hiccup cannot mask the real exit status of the run.
    trap "status=\$?; oco2_pack_results '$job' \$status || true; exit \$status" EXIT
}

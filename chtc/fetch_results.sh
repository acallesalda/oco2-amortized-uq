#!/bin/bash
# Collect the trained models out of /staging and unpack them into the checkout, ready for
# the scoring notebooks.
#
#     ssh <netid>@transfer.chtc.wisc.edu
#     cd ~/OCO-2 && ./chtc/fetch_results.sh
#
# Everything lands under experiments/weights/group_split_v1/, which is where the grouped
# configs point and which the published checkpoints in experiments/weights/ never touch.
#
#     ./chtc/fetch_results.sh              # all six models
#     ./chtc/fetch_results.sh det_xco2     # just one
#     ./chtc/fetch_results.sh --status     # what has finished so far
#     ./chtc/fetch_results.sh --cov        # the covariate comparison (covariates.dag)
#     ./chtc/fetch_results.sh --bo         # the Bayesian-optimization trial records
#     ./chtc/fetch_results.sh --best       # the search winners, retrained and scored
#     ./chtc/fetch_results.sh --gauss      # the moment-matched Gaussian control
#     ./chtc/fetch_results.sh --last1      # the scalar surface task (level 20)
#     ./chtc/fetch_results.sh --replicate  # published models, replicate-level split
#     ./chtc/fetch_results.sh --rerun      # the new recipe on the replicate-level split
#     ./chtc/fetch_results.sh --months     # February's models on March and July
#     ./chtc/fetch_results.sh --months-full  # ALL-February models on March and July
#     ./chtc/fetch_results.sh --ext        # the extension run: February test AND both months
#     ./chtc/fetch_results.sh --ext-cap    # the same, with the replicates capped
#     ./chtc/fetch_results.sh --dens       # the covariate-density weighting arm (dens.dag)
#     ./chtc/fetch_results.sh --balanced     # the cluster-balanced retrains + scores
#     ./chtc/fetch_results.sh --discrepancy  # delta over all 451 reference soundings
#     ./chtc/fetch_results.sh --gap          # the in-sample side of the generalization gap
#     ./chtc/fetch_results.sh --size         # the learning curve in training scenes
#     ./chtc/fetch_results.sh --size-tasks   # the same curve for last4 and profiles
#     ./chtc/fetch_results.sh --attr         # spectral attribution (attribution.dag)
set -euo pipefail

cd "$(dirname "$0")/.."
REPO=$PWD

NETID=$(whoami)
STAGING="/staging/${NETID:0:1}/${NETID}/oco2"
RESULTS="$STAGING/results"

# Where a job's archive actually lands.
#
# Most jobs write to results/. The ones that build INPUTS for other jobs write to the
# staging root instead, because that is where their consumers' transfer_input_files look
# -- prep_month.sub remaps to $(OSDF)/oco2_month_<month>.tar.gz, not to results/. Listing
# those under results/ is how eight month jobs read as "not finished" for a week: the
# archive was where its consumer expects it, and only the status printer was looking in
# the wrong place.
job_archive() {
    case "$1" in
        oco2_month_*|oco2_full_feb_v1|oco2_ext_v1|oco2_ext_cap_v1|oco2_split_v1|oco2_covariates_v1|oco2_replicate_split_v1|oco2_size_*_v1)
            echo "$STAGING/$1.tar.gz" ;;
        *)  echo "$RESULTS/$1.tar.gz" ;;
    esac
}

# A finished archive is not a successful job. Every job packs and uploads its results even
# when it dies -- that is deliberate, so a crash still ships its logs -- so an entry with a
# size and a timestamp means "the job ended", not "the job worked". Two runs were read as
# successful for days because of that: the balanced deterministic trainers, which died on a
# Hydra struct-mode override before their first epoch, and nf_profiles on the all-February
# split, which has been shipping a checkpoint-free archive since 25 Aug.
#
# So --status now says what run_meta recorded, and flags a training archive that carries no
# model at all. Both checks read the tarball's index only; neither unpacks anything.
job_verdict() {
    local job="$1" f="$2" status verdict=""
    status=$(tar -xzOf "$f" "run_meta_${job}.json" 2>/dev/null \
             | grep -o '"exit_status": *[0-9-]*' | grep -o '[0-9-]*$' || echo "")
    if [ -n "$status" ] && [ "$status" != "0" ]; then
        verdict="  ** FAILED, exit $status -- see chtc/logs/$job.err **"
    fi
    case "$job" in
        det_*|nf_*)
            # grep -c, not grep -q, and the count captured rather than the exit status.
            # `tar | grep -q` is wrong HERE, under `set -o pipefail`: grep -q exits the
            # instant it matches, tar dies of SIGPIPE with 141, pipefail makes the whole
            # pipeline non-zero, and `if !` then reports "no checkpoint" for every archive
            # that HAS one early enough to trigger the early exit. It fired on all six
            # balanced retrains on 31 Aug, including two known-good 300 MB flows. The
            # synthetic test that passed was small enough that tar finished
            # before grep exited, which is exactly the case the bug does not affect.
            # grep -c reads to EOF, so there is no early exit and no SIGPIPE.
            local n_ckpt
            n_ckpt=$(tar -tzf "$f" 2>/dev/null | grep -c '[.]pt$' || true)
            if [ "${n_ckpt:-0}" -eq 0 ]; then
                verdict="$verdict  ** no checkpoint in the archive **"
            fi ;;
    esac
    echo "$verdict"
}

# The code tarball is what the jobs actually RUN. It is built by hand, and a submission
# made against a stale one is the failure that looks most like success: the job exits 0 and
# silently does the old thing. On 31 Aug all six cluster-balanced retrains ran the 27 Aug
# tarball, which contained no balanced sampler at all -- `++data.balance_by=clst` was
# accepted by Hydra, read by nobody, and three of the six exited 0 having trained the
# published model a second time. Nothing in the pipeline said so. This does.
code_tarball_banner() {
    local t="$REPO/chtc/oco2_code.tar.gz" packed head
    if [ ! -f "$t" ]; then
        echo "!! chtc/oco2_code.tar.gz does not exist -- run ./chtc/make_code_tarball.sh"
        return 0
    fi
    packed=$(tar -xzOf "$t" chtc/GIT_PROVENANCE.txt 2>/dev/null \
             | awk '/^commit:/{print $2}' || true)
    head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)
    echo "code tarball: $(date -r "$t" '+%Y-%m-%d %H:%M')  built from ${packed:0:7}"
    if [ -n "$head" ] && [ -n "$packed" ] && [ "$head" != "$packed" ]; then
        echo "  ** STALE: HEAD is ${head:0:7}, the tarball is ${packed:0:7}."
        echo "  ** Anything submitted now runs ${packed:0:7}. Rebuild before submitting:"
        echo "  **     ./chtc/make_code_tarball.sh"
    fi
}

TRAIN_JOBS=(det_xco2 det_profiles det_last4 nf_xco2 nf_profiles nf_last4)
EVAL_JOBS=(eval_xco2 eval_profiles eval_last4 eval_last4_marginal)
# The scalar surface task (last1.dag, eval_last1.dag): CO2 at pressure level 20 alone.
# Its own list, fetched by --last1 and listed by --status, for the same reason the
# covariate family has one -- a bare ./fetch_results.sh should keep unpacking the six
# models the published tables are built from and nothing else. The archives are named
# like any other job's, so nothing here is special except that this list has to exist:
# --status only prints jobs it is told about, and an archive nobody names looks exactly
# like a job that never ran.
SURFACE_JOBS=(det_last1 nf_last1 eval_last1 eval_last1_marginal)
# Not in ALL, and so not fetched by a bare `./fetch_results.sh`: the replicate-split
# controls are a secondary result and `score` is regenerated whenever a table changes.
# They ARE listed by --status, because an archive that exists and is never mentioned
# reads as an experiment that was never run -- which is exactly what happened after the
# 14 Aug batch, where all five sat on /staging unlisted for a weekend.
EXTRA_JOBS=(det_xco2_replicate det_profiles_replicate det_last4_replicate
            nf_xco2_replicate nongauss diagnose_tail)
# `score` is no longer here. Its archive is a few kB of tables and figures and it is
# re-run constantly, so score.sub now returns it to chtc/ directly instead of to
# /staging, where OSDF's refusal to overwrite held the job after every re-run. It is
#     tar -xzf chtc/score.tar.gz -C ~/OCO-2
# and there is nothing to fetch or clean.
# The covariate comparison (covariates.dag). Fetched by --cov, listed by --status, and
# deliberately not in ALL: they are a second family of models, not a replacement for the
# six, and unpacking them by accident on top of a scoring run is not something to make easy.
COV_JOBS=(det_xco2_base2 det_xco2_covctl det_xco2_covfix det_xco2_cov det_xco2_covprior
          det_xco2_covgeo det_xco2_covacos det_profiles_base2 det_profiles_ppm
          det_profiles_covctl det_profiles_covfix det_profiles_cov det_profiles_covprior
          det_profiles_covgeo det_last4_base2 det_last4_ppm det_last4_covctl
          det_last4_covfix det_last4_cov det_last4_covprior det_last4_covgeo)
# All six studies are searched now, deterministic and flow, each from scratch. The trial
# records live in results/$BO_RUN rather than results/bo so that a re-run cannot collide
# with an earlier search -- OSDF refuses to overwrite an object, so a collision is a held
# job rather than a warning. Keep BO_RUN in step with --run in chtc/make_bo_dag.py.
BO_STUDIES=(det_xco2 det_profiles det_last4 nf_xco2 nf_profiles nf_last4)
BO_RUN="${OCO2_BO_RUN:-bo25}"
# The search winners, retrained at twice the trial epochs, and the four jobs that score
# them on the test partition. Fetched by --best, listed by --status, and not in ALL for
# the same reason the covariate comparison is not: they are a second family of models and
# unpacking them on top of a scoring run by accident is not something to make easy.
BEST_JOBS=(det_xco2_final det_profiles_final det_last4_final
           nf_xco2_final nf_profiles_final nf_last4_final
           eval_xco2_bo eval_profiles_bo eval_last4_bo eval_last4_marginal_bo)
# The moment-matched Gaussian control (chtc/gauss_submit.sh): the same four scoring jobs
# with the streamed control switched on. Separate from BEST_JOBS because they are a
# follow-up on the same checkpoints rather than part of the search, and because they write
# <task>_gauss.npz beside <task>_bo.npz -- unpacking one over the other is harmless, but
# fetching them together makes it far too easy to score the wrong file into a table.
GAUSS_JOBS=(eval_xco2_gauss eval_profiles_gauss eval_last4_gauss eval_last4_marginal_gauss)
# The published checkpoints re-scored on the published replicate-level split
# (chtc/eval_pub.sub). Separate from EVAL_JOBS because they are the *other* split: they
# write experiments/scoring/results/replicate/<task>.npz, which score.py prints beside the
# group-split rows as the within-reference comparison. Not in ALL --
# they need the Zenodo weights archive staged, so a bare fetch should not imply they exist.
PUB_JOBS=(eval_xco2_replicate eval_profiles_replicate eval_last4_replicate
          eval_last4_marginal_replicate)
# The rerun: the same six studies and four scoring jobs as BEST_JOBS/GAUSS_JOBS, but on
# the ORIGINAL replicate-level split with the corrected covariates and the same search
# protocol (chtc/bo_replicate/bo_all.dag). It is what makes the two columns of the
# revision's main table differ in the partition alone. Archive names carry `_rep` and the
# artifacts carry the `borep` tag, because OSDF refuses to overwrite an object and these
# would otherwise be the group-split run's names exactly.
RERUN_JOBS=(det_xco2_final_rep det_profiles_final_rep det_last4_final_rep
            nf_xco2_final_rep nf_profiles_final_rep nf_last4_final_rep
            eval_xco2_borep eval_profiles_borep eval_last4_borep
            eval_last4_marginal_borep)
# The search records for the rerun live under their own staging namespace, for the same
# reason. Keep in step with --run in the make_bo_dag.py invocation that generated
# chtc/bo_replicate/.
BO_RUN_RERUN="${OCO2_BO_RUN_RERUN:-bo25rep}"
# The temporal generalization experiment (months.dag): February's models scored on March
# and July. Two CPU jobs build the months, six GPU jobs score them. Not in ALL -- they need
# the raw month archives in /staging, which are not Zenodo items.
MONTH_JOBS=(oco2_month_202003 oco2_month_202007
            eval_xco2_202003 eval_profiles_202003 eval_last4_202003
            eval_xco2_202007 eval_profiles_202007 eval_last4_202007)
# The definitive version of the same experiment (months_full.dag): the six searched
# configurations RETRAINED on 406 of February's reference soundings, then scored on the
# two months. Its own list because it is a different set of models -- same architectures,
# 12.5 % more training data, no February test partition at all -- and because everything it
# writes carries `_full` so that it cannot land on top of the group-split run's archives.
# The `oco2_full_feb_v1` row is the split table itself: the root of that DAG, and the one
# input whose absence stops every other node.
MONTH_FULL_JOBS=(oco2_full_feb_v1
                 det_xco2_final_full det_profiles_final_full det_last4_final_full
                 nf_xco2_final_full nf_profiles_final_full nf_last4_final_full
                 oco2_month_202003_full oco2_month_202007_full
                 eval_xco2_202003_full eval_profiles_202003_full eval_last4_202003_full
                 eval_xco2_202007_full eval_profiles_202007_full eval_last4_202007_full
                 eval_last4_marginal_202003_full eval_last4_marginal_202007_full)
# The February EXTENSION run (ext.dag): the same six searched configurations retrained on
# 447 reference soundings and 58,115 spectra -- 86 new scenes in clusters 8 and 9 plus 200
# fresh replicates of the 26 shared training scenes -- at three times the epoch budget and
# with the schedule the learning curves called for. Scored on group_split_v1's OWN 45 test
# scenes (so the numbers sit beside Table 1's) and on both months. Its own list because it
# is the largest single graph here -- one prep, six retrains, four February evals, two month
# preps, eight month evals -- and because everything it writes carries `_ext`.
EXT_JOBS=(oco2_ext_v1
          det_xco2_final_ext det_profiles_final_ext det_last4_final_ext
          nf_xco2_final_ext nf_profiles_final_ext nf_last4_final_ext
          eval_xco2_ext eval_profiles_ext eval_last4_ext eval_last4_marginal_ext
          eval_xco2_resim eval_profiles_resim eval_last4_resim
          oco2_month_202003_ext oco2_month_202007_ext
          eval_xco2_202003_ext eval_profiles_202003_ext eval_last4_202003_ext
          eval_last4_marginal_202003_ext
          eval_xco2_202007_ext eval_profiles_202007_ext eval_last4_202007_ext
          eval_last4_marginal_202007_ext)
# The CORRECTED extension run (ext_cap.dag): the same 447 training scenes with the
# extension capped at 100 replicates per scene, so clusters 8 and 9 take 25.0 % of the
# training rows against 25.1 % of the scenes rather than 42.5 %.
# Plus the three per-study fixes its learning curves asked for: nf_last4 at 200 epochs,
# nf_profiles at 6 coupling layers and dropout 0.2, the rest unchanged at 120.
EXT_CAP_JOBS=(oco2_ext_cap_v1
              det_xco2_final_extcap det_profiles_final_extcap det_last4_final_extcap
              nf_xco2_final_extcap nf_profiles_final_extcap nf_last4_final_extcap
              eval_xco2_extcap eval_profiles_extcap eval_last4_extcap
              eval_last4_marginal_extcap
              eval_xco2_resimcap eval_profiles_resimcap eval_last4_resimcap
              oco2_month_202003_extcap oco2_month_202007_extcap
              eval_xco2_202003_extcap eval_profiles_202003_extcap
              eval_last4_202003_extcap eval_last4_marginal_202003_extcap
              eval_xco2_202007_extcap eval_profiles_202007_extcap
              eval_last4_202007_extcap eval_last4_marginal_202007_extcap)
# The covariate-density weighting arm (dens.dag): the two PROFILE studies only, retrained
# on ext_v1 with the sampler swapped for a density over the `scene_prior` covariates in
# scene space, then rescored on the same 4,489 February test soundings. Four jobs and no
# prep, because oco2_ext_v1.tar.gz is already in /staging from prep_ext -- which is what
# makes this the cheapest arm in the project. Everything carries `_dens` so it cannot land
# on `_ext` or `_bal`.
DENS_JOBS=(det_profiles_final_dens_ext nf_profiles_final_dens_ext
           eval_profiles_dens eval_last4_marginal_dens)
# The discrepancy characterization (discrepancy.sub): one small CPU job that recovers
# delta + epsilon from every raw file and reports the block structure, the effective
# rank of Sigma_delta and the share of the ACOS error delta drives. Its own list
# because it produces tables rather than checkpoints, and because it is the evidence
# under the discrepancy characterization -- which until it runs rests on a single file.
DISCREPANCY_JOBS=(oco2_discrepancy_v1)
# Cluster-balanced training (balanced.dag): the same six searched configurations
# retrained with training draws weighted by n_cluster ** -0.5, then rescored on the same
# test partition. Its own list because it is a second family of models rather than a
# replacement for the six, and because everything it writes carries `_bal` so that it
# cannot land on top of the published run's archives.
BALANCED_JOBS=(det_xco2_final_bal det_profiles_final_bal det_last4_final_bal
               nf_xco2_final_bal nf_profiles_final_bal nf_last4_final_bal
               eval_xco2_bo_bal eval_profiles_bo_bal eval_last4_bo_bal
               eval_last4_marginal_bo_bal)
# The generalization gap (gap.dag): the SAME published checkpoints, scored on 45 reference
# soundings of the TRAINING partition instead of the held-out one. No training, so there is
# no model to fetch -- only the artifacts, whose split name carries `+train` and whose meta
# records eval_partition.
GAP_JOBS=(eval_xco2_gap_train eval_last4_gap_train eval_profiles_gap_train)
# The learning curve in the number of training reference soundings (size.dag): three nested
# subsets of the training partition, each with its own table, statistics, pair of retrained
# models and score on the SAME held-out scenes. Its own list because it is twelve jobs of
# three different kinds, and because everything it writes carries `_s<k>` so that it cannot
# land on top of the published run's archives.
SIZE_JOBS=(oco2_size_90_v1 oco2_size_180_v1 oco2_size_270_v1
           det_xco2_final_s90 det_xco2_final_s180 det_xco2_final_s270
           nf_xco2_final_s90 nf_xco2_final_s180 nf_xco2_final_s270
           eval_xco2_s90 eval_xco2_s180 eval_xco2_s270)
# The same curve on the two profile tasks (size_tasks.dag). Its own list because it reuses
# size.dag's subset tables rather than rebuilding them -- there are no prep jobs here -- and
# because the profile artifacts are ~224 MB each, so it is not something to fetch by
# accident along with the XCO2 ones.
SIZE_TASK_JOBS=(det_last4_final_s90 det_last4_final_s180 det_last4_final_s270
                nf_last4_final_s90 nf_last4_final_s180 nf_last4_final_s270
                eval_last4_s90 eval_last4_s180 eval_last4_s270
                det_profiles_final_s90 det_profiles_final_s180 det_profiles_final_s270
                nf_profiles_final_s90 nf_profiles_final_s180 nf_profiles_final_s270
                eval_profiles_s90 eval_profiles_s180 eval_profiles_s270)
# Spectral attribution (attribution.dag): the SAME published checkpoints, re-scored on the
# held-out partition with one band -- or one sliding window of channels -- replaced by the
# training mean spectrum. No training and no fitting of any kind, so the archives carry
# tables and one figure and nothing else.
ATTR_JOBS=(attr_xco2_attr attr_last4_attr attr_profiles_attr)
ALL=("${TRAIN_JOBS[@]}" "${EVAL_JOBS[@]}")

if [ "${1:-}" = "--status" ]; then
    code_tarball_banner
    echo
    echo "results in $RESULTS:"
    printf "\n%-30s %10s  %s\n" JOB SIZE FINISHED
    for job in "${ALL[@]}" "${EXTRA_JOBS[@]}" "${PUB_JOBS[@]}" "${RERUN_JOBS[@]}" \
               "${MONTH_JOBS[@]}" "${BALANCED_JOBS[@]}" "${DENS_JOBS[@]}"; do
        f=$(job_archive "$job")
        if [ -f "$f" ]; then
            printf "%-30s %10s  %s%s\n" "$job" \
                "$(du -h "$f" | cut -f1)" \
                "$(date -r "$f" '+%Y-%m-%d %H:%M')" \
                "$(job_verdict "$job" "$f")"
        else
            printf "%-30s %10s  %s\n" "$job" "-" "not finished"
        fi
    done
    printf "\n--- the scalar surface task (last1.dag, eval_last1.dag) ---\n"
    for job in "${SURFACE_JOBS[@]}"; do
        f=$(job_archive "$job")
        if [ -f "$f" ]; then
            printf "%-30s %10s  %s\n" "$job" \
                "$(du -h "$f" | cut -f1)" "$(date -r "$f" '+%Y-%m-%d %H:%M')"
        else
            printf "%-30s %10s  %s\n" "$job" "-" "not finished"
        fi
    done

    printf "\n--- the generalization gap (gap.dag) ---\n"
    for job in "${GAP_JOBS[@]}"; do
        f=$(job_archive "$job")
        if [ -f "$f" ]; then
            printf "%-30s %10s  %s%s\n" "$job" \
                "$(du -h "$f" | cut -f1)" "$(date -r "$f" '+%Y-%m-%d %H:%M')" \
                "$(job_verdict "$job" "$f")"
        else
            printf "%-30s %10s  %s\n" "$job" "-" "not finished"
        fi
    done

    printf "\n--- the learning curve in training scenes (size.dag, size_tasks.dag) ---\n"
    for job in "${SIZE_JOBS[@]}" "${SIZE_TASK_JOBS[@]}"; do
        f=$(job_archive "$job")
        if [ -f "$f" ]; then
            printf "%-30s %10s  %s%s\n" "$job" \
                "$(du -h "$f" | cut -f1)" "$(date -r "$f" '+%Y-%m-%d %H:%M')" \
                "$(job_verdict "$job" "$f")"
        else
            printf "%-30s %10s  %s\n" "$job" "-" "not finished"
        fi
    done

    printf "\n--- spectral attribution (attribution.dag) ---\n"
    for job in "${ATTR_JOBS[@]}"; do
        f=$(job_archive "$job")
        if [ -f "$f" ]; then
            printf "%-30s %10s  %s%s\n" "$job" \
                "$(du -h "$f" | cut -f1)" "$(date -r "$f" '+%Y-%m-%d %H:%M')" \
                "$(job_verdict "$job" "$f")"
        else
            printf "%-30s %10s  %s\n" "$job" "-" "not finished"
        fi
    done

    printf "\n--- delta characterization (discrepancy.sub) ---\n"
    for job in "${DISCREPANCY_JOBS[@]}"; do
        f=$(job_archive "$job")
        if [ -f "$f" ]; then
            printf "%-30s %10s  %s\n" "$job" \
                "$(du -h "$f" | cut -f1)" "$(date -r "$f" '+%Y-%m-%d %H:%M')"
        else
            printf "%-30s %10s  %s\n" "$job" "-" "not finished"
        fi
    done

    printf "\n--- covariate comparison (covariates.dag) ---\n"
    for job in "${COV_JOBS[@]}"; do
        f=$(job_archive "$job")
        if [ -f "$f" ]; then
            printf "%-30s %10s  %s\n" "$job" \
                "$(du -h "$f" | cut -f1)" "$(date -r "$f" '+%Y-%m-%d %H:%M')"
        else
            printf "%-30s %10s  %s\n" "$job" "-" "not finished"
        fi
    done

    printf "\n--- the search winners, retrained at 2x epochs and scored on the test set ---\n"
    for job in "${BEST_JOBS[@]}"; do
        f=$(job_archive "$job")
        if [ -f "$f" ]; then
            printf "%-30s %10s  %s\n" "$job" \
                "$(du -h "$f" | cut -f1)" "$(date -r "$f" '+%Y-%m-%d %H:%M')"
        else
            printf "%-30s %10s  %s\n" "$job" "-" "not finished"
        fi
    done

    printf "\n--- ALL-February models on March and July (months_full.dag) ---\n"
    for job in "${MONTH_FULL_JOBS[@]}"; do
        f=$(job_archive "$job")
        if [ -f "$f" ]; then
            printf "%-30s %10s  %s\n" "$job" \
                "$(du -h "$f" | cut -f1)" "$(date -r "$f" '+%Y-%m-%d %H:%M')"
        else
            printf "%-30s %10s  %s\n" "$job" "-" "not finished"
        fi
    done

    printf "\n--- the February extension (ext.dag) ---\n"
    for job in "${EXT_JOBS[@]}"; do
        f=$(job_archive "$job")
        if [ -f "$f" ]; then
            printf "%-30s %10s  %s\n" "$job" \
                "$(du -h "$f" | cut -f1)" "$(date -r "$f" '+%Y-%m-%d %H:%M')"
        else
            printf "%-30s %10s  %s\n" "$job" "-" "not finished"
        fi
    done

    printf "\n--- the CAPPED extension (ext_cap.dag) ---\n"
    for job in "${EXT_CAP_JOBS[@]}"; do
        f=$(job_archive "$job")
        if [ -f "$f" ]; then
            printf "%-30s %10s  %s\n" "$job" \
                "$(du -h "$f" | cut -f1)" "$(date -r "$f" '+%Y-%m-%d %H:%M')"
        else
            printf "%-30s %10s  %s\n" "$job" "-" "not finished"
        fi
    done

    printf "\n--- the moment-matched Gaussian control (gauss_submit.sh) ---\n"
    for job in "${GAUSS_JOBS[@]}"; do
        f=$(job_archive "$job")
        if [ -f "$f" ]; then
            printf "%-30s %10s  %s\n" "$job" \
                "$(du -h "$f" | cut -f1)" "$(date -r "$f" '+%Y-%m-%d %H:%M')"
        else
            printf "%-30s %10s  %s\n" "$job" "-" "not finished"
        fi
    done

    # The Zenodo items and the job-built tables are listed apart because the fix differs:
    # one is ./stage_data.sh, the other is a submit file. Neither group was listed at all
    # until the rerun's preflight went looking for the processed archives and found no row
    # for them -- an input nobody prints reads as an input that is missing.
    printf "\n--- the dataset (./stage_data.sh, from Zenodo) ---\n"
    for f in "$STAGING/oco2_processed_train.tar.gz" "$STAGING/oco2_processed_test.tar.gz" \
             "$STAGING/oco2_profile_stats.tar.gz" "$STAGING/oco2_model_weights.tar.gz"; do
        if [ -f "$f" ]; then
            printf "%-34s %10s  %s\n" "$(basename "$f")" \
                "$(du -h "$f" | cut -f1)" "$(date -r "$f" '+%Y-%m-%d %H:%M')"
        else
            printf "%-34s %10s  %s\n" "$(basename "$f")" "-" "not staged"
        fi
    done
    printf "    the weights are for eval_pub.sub alone; the other three every GPU job reads.\n"

    printf "\n--- the split tables (prep.sub, covariates.sub, prep_replicate.sub, prep_full_feb.sub) ---\n"
    for f in "$STAGING/oco2_split_v1.tar.gz" "$STAGING/oco2_covariates_v1.tar.gz" \
             "$STAGING/oco2_replicate_split_v1.tar.gz" \
             "$STAGING/oco2_full_feb_v1.tar.gz"; do
        if [ -f "$f" ]; then
            printf "%-34s %10s  %s\n" "$(basename "$f")" \
                "$(du -h "$f" | cut -f1)" "$(date -r "$f" '+%Y-%m-%d %H:%M')"
        else
            printf "%-34s %10s  %s\n" "$(basename "$f")" "-" "not built"
        fi
    done
    # Neither a Zenodo item nor a job output: the two extra months were delivered
    # directly, so months.dag simply expects them on /staging and says nothing useful
    # if they are absent -- the prep job is accepted and then fails file transfer.
    printf "\n--- the raw months (delivered; upload to /staging yourself -- see docs/hpc.md) ---\n"
    for f in "$STAGING/OCO2SimUQExpt_202003.zip" "$STAGING/OCO2SimUQExpt_202007.zip"; do
        if [ -f "$f" ]; then
            printf "%-34s %10s  %s\n" "$(basename "$f")" \
                "$(du -h "$f" | cut -f1)" "$(date -r "$f" '+%Y-%m-%d %H:%M')"
        else
            printf "%-34s %10s  %s\n" "$(basename "$f")" "-" "not staged"
        fi
    done

    n_sweep=$({ find "$RESULTS/sweeps" -name 'sweep_*.tar.gz' 2>/dev/null || true; } | wc -l)
    printf "\n%-30s %10s  %s\n" "sweep trials" "$n_sweep" \
        "./fetch_results.sh --sweep"
    for run in "$BO_RUN" "$BO_RUN_RERUN"; do
        printf "\n--- the search itself (results/%s) ---\n" "$run"
        for study in "${BO_STUDIES[@]}"; do
            n_done=$({ find "$RESULTS/$run" -name "bo_${study}_r*_t*.json" \
                       ! -name '*.params.json' 2>/dev/null || true; } | wc -l)
            winner="-"
            if [ -f "$RESULTS/$run/bo_${study}_best.json" ]; then winner="winner picked"; fi
            printf "%-30s %10s  %s\n" "BO $study" "$n_done/25" "$winner"
        done
    done
    echo
    echo "A bare ./fetch_results.sh takes only the first ${#ALL[@]}. Name the others explicitly:"
    echo "    ./fetch_results.sh ${EXTRA_JOBS[*]}"
    exit 0
fi

# Remove result archives from /staging so a job can be resubmitted.
#
# OSDF will not overwrite an existing object -- a second upload to the same URL fails with
# "remote object already exists, upload aborted" and the job is held. Since ON_EXIT
# transfers the archive even when the job fails, ANY re-run of a job that has already run
# once needs its old archive removed first. This is why max_retries is 0 everywhere: an
# automatic retry would hit exactly this and replace a real error message with a confusing
# transfer one.
if [ "${1:-}" = "--clean" ]; then
    shift
    todo=("$@")
    if [ ${#todo[@]} -eq 0 ]; then
        echo "usage: ./fetch_results.sh --clean <job> [job...]"
        echo "       jobs: ${ALL[*]} prep prep_replicate prep_full_feb prep_ext prep_ext_cap resim covariates audit"
        echo "             ${COV_JOBS[*]}"
        echo "             ${BEST_JOBS[*]}"
        echo "             ${SURFACE_JOBS[*]}"
        echo "             ${RERUN_JOBS[*]}"
        echo "             ${MONTH_FULL_JOBS[*]}"
        echo "             ${GAP_JOBS[*]}"
        echo "             ${SIZE_JOBS[*]}"
        echo "             ${SIZE_TASK_JOBS[*]}"
        echo "             ${ATTR_JOBS[*]}"
        echo "             bo        (removes every BO record so a study can restart)"
        echo "             bo_rerun  (the same, for the replicate-split search)"
        echo
        echo "Removes the named archives from $RESULTS so the job can be resubmitted."
        echo "Nothing else reads them, and fetch_results.sh has already unpacked anything"
        echo "you wanted -- but they are the only copy, so check before removing."
        exit 2
    fi
    for job in "${todo[@]}"; do
        # Two jobs whose output is not named after them.
        if [ "$job" = "prep" ]; then
            f="$STAGING/oco2_split_v1.tar.gz"
        elif [ "$job" = "covariates" ]; then
            f="$STAGING/oco2_covariates_v1.tar.gz"
        elif [ "$job" = "prep_replicate" ]; then
            f="$STAGING/oco2_replicate_split_v1.tar.gz"
        elif [ "$job" = "prep_full_feb" ]; then
            f="$STAGING/oco2_full_feb_v1.tar.gz"
        elif [ "$job" = "prep_ext" ]; then
            f="$STAGING/oco2_ext_v1.tar.gz"
        elif [ "$job" = "prep_ext_cap" ]; then
            f="$STAGING/oco2_ext_cap_v1.tar.gz"
        elif [ "$job" = "resim" ]; then
            # The three tail-scene evals as one unit: they are re-run together or not at
            # all, and the first attempt at them was held on an OSDF collision (a submit
            # file shadowing the .dag's VARS -- see eval_resim.sub's header).
            for t in xco2 profiles last4; do
                g="$STAGING/results/eval_${t}_resim.tar.gz"
                if [ -f "$g" ]; then rm -f "$g"; echo "removed $g"; fi
            done
            continue
        elif [ "$job" = "bo_rerun" ]; then
            # As `bo`, for the rerun's namespace. Whole or not at all, same reasoning.
            rm -rf "${RESULTS:?}/$BO_RUN_RERUN"
            echo "removed $RESULTS/$BO_RUN_RERUN"
            continue
        elif [ "$job" = "bo" ]; then
            # The whole search: every record, parameter file and winner, so bo_all.dag can
            # be re-run from round 0. There is no partial clean -- a study is only
            # meaningful whole, and a half-cleaned one silently mixes two searches.
            rm -rf "${RESULTS:?}/$BO_RUN"
            echo "removed $RESULTS/$BO_RUN"
            continue
        else
            f=$(job_archive "$job")
        fi
        if [ -f "$f" ]; then
            rm -f "$f"
            echo "removed $f"
        else
            echo "not present: $f"
        fi
    done
    exit 0
fi

# Sweep archives are one per trial and named by trial, so there is no fixed list to walk.
if [ "${1:-}" = "--sweep" ]; then
    n=0
    for f in "$RESULTS"/sweeps/sweep_*.tar.gz; do
        if [ ! -f "$f" ]; then echo "no sweep archives in $RESULTS/sweeps"; exit 0; fi
        tar -xzf "$f" -C "$REPO"
        n=$((n + 1))
    done
    echo "unpacked $n sweep archives"
    echo "now: python3 chtc/collect_sweep.py"
    exit 0
fi

# The search's own output is small JSON and markdown rather than archives, so it is
# copied, not unpacked. bo_select.py reads the records from chtc/bo/results; the winners
# and leaderboards it already wrote inside the jobs come down beside them, and those are
# what gets read into the paper.
if [ "${1:-}" = "--bo" ]; then
    # The destination follows the run. It used to be chtc/bo/results always, so
    #     OCO2_BO_RUN=bo25rep ./fetch_results.sh --bo
    # copied the rerun's records, winners and leaderboards on top of the group
    # split's -- identical filenames, no warning, two searches mixed in one
    # directory. bo_select.py reads that directory, so the next leaderboard built
    # from it would have ranked trials from both partitions as though they were one
    # study.
    if [ "$BO_RUN" = "$BO_RUN_RERUN" ]; then
        dest="$REPO/chtc/bo_replicate/results"
    else
        dest="$REPO/chtc/bo/results"
    fi
    mkdir -p "$dest"
    n=0
    for f in "$RESULTS/$BO_RUN"/bo_*_r*_t*.json "$RESULTS/$BO_RUN"/bo_*_best.json \
             "$RESULTS/$BO_RUN"/bo_*_leaderboard.md; do
        if [ -f "$f" ]; then
            cp "$f" "$dest/"
            n=$((n + 1))
        fi
    done
    if [ "$n" -eq 0 ]; then
        echo "no BO records in $RESULTS/$BO_RUN"
        exit 0
    fi
    echo "copied $n files to $dest"
    echo
    echo "the leaderboards, as the jobs wrote them:"
    ls -1 "$dest"/bo_*_leaderboard.md 2>/dev/null | sed 's/^/    /' || echo "    (none yet)"
    echo
    echo "to rebuild them from the records, or to see a study the DAG has not selected"
    echo "yet:  python3 chtc/bo_select.py --results-dir ${dest#$REPO/} --out-dir $(dirname "${dest#$REPO/}")"
    exit 0
fi

jobs=("$@")
case "${1:-}" in
    "")       jobs=("${ALL[@]}") ;;
    --train)  jobs=("${TRAIN_JOBS[@]}") ;;
    --eval)   jobs=("${EVAL_JOBS[@]}") ;;
    --cov)    jobs=("${COV_JOBS[@]}") ;;
    --best)   jobs=("${BEST_JOBS[@]}") ;;
    --gauss)  jobs=("${GAUSS_JOBS[@]}") ;;
    --replicate) jobs=("${PUB_JOBS[@]}") ;;
    --rerun)  jobs=("${RERUN_JOBS[@]}") ;;
    --months) jobs=("${MONTH_JOBS[@]}") ;;
    --months-full) jobs=("${MONTH_FULL_JOBS[@]}") ;;
    --ext)    jobs=("${EXT_JOBS[@]}") ;;
    --ext-cap) jobs=("${EXT_CAP_JOBS[@]}") ;;
    --dens)   jobs=("${DENS_JOBS[@]}") ;;
    --balanced) jobs=("${BALANCED_JOBS[@]}") ;;
    --last1)  jobs=("${SURFACE_JOBS[@]}") ;;
    --discrepancy) jobs=("${DISCREPANCY_JOBS[@]}") ;;
    --gap)    jobs=("${GAP_JOBS[@]}") ;;
    --size)   jobs=("${SIZE_JOBS[@]}") ;;
    --size-tasks) jobs=("${SIZE_TASK_JOBS[@]}") ;;
    --attr)   jobs=("${ATTR_JOBS[@]}") ;;
esac

if [ ! -d "$RESULTS" ]; then
    echo "ERROR: $RESULTS does not exist. Has anything finished yet?"
    exit 1
fi

n_ok=0
n_missing=0
n_bad=0
for job in "${jobs[@]}"; do
    tarball=$(job_archive "$job")
    echo
    echo "--- $job"
    if [ ! -f "$tarball" ]; then
        echo "    not present in $RESULTS -- still running, or the node failed."
        n_missing=$((n_missing + 1))
        continue
    fi

    # Is the archive even whole? A job that is evicted part-way through UPLOADING leaves a
    # truncated object on OSDF, and the retry is then refused with "remote object already
    # exists" and held -- so a held job can mean either "ran twice" or "uploaded half
    # once", and only this tells them apart. `gzip -t` reads the whole stream and is the
    # cheapest complete check there is.
    #
    # Tested BEFORE run_meta is read, because a truncated archive makes that read return
    # "?" and the warning about it blames the job for what the transfer did. And tested at
    # all because `tar -xzf` on a truncated archive exits non-zero under `set -e`, which
    # aborted this whole loop on 2026-09-08 and left twenty other archives unfetched.
    if ! gzip -t "$tarball" 2>/dev/null; then
        echo "    TRUNCATED: $(du -h "$tarball" | cut -f1) and not a complete gzip stream."
        echo "               The upload was interrupted; the compute may well have"
        echo "               succeeded but this archive cannot be recovered. Delete it and"
        echo "               re-run the node:"
        echo "                 ./fetch_results.sh --clean $job"
        n_bad=$((n_bad + 1))
        continue
    fi

    # Every archive carries run_meta_<job>.json with the exit status the job really had.
    # A job that crashed still ships its partial results, so check before trusting them.
    status=$(tar -xzOf "$tarball" "run_meta_${job}.json" 2>/dev/null \
             | grep -o '"exit_status": *[0-9-]*' | grep -o '[0-9-]*$' || echo "?")
    if [ "$status" != "0" ]; then
        echo "    WARNING: this run exited with status $status. Unpacking anyway, but the"
        echo "             checkpoints are partial. Look at chtc/logs/$job.err."
    fi

    # `|| { ...; continue; }` rather than a bare call: one unreadable archive must not stop
    # the other twenty from being fetched.
    if ! tar -xzf "$tarball" -C "$REPO"; then
        echo "    EXTRACTION FAILED even though the gzip stream is whole. Left in place;"
        echo "    look at it by hand before deleting it."
        n_bad=$((n_bad + 1))
        continue
    fi
    echo "    unpacked:"
    tar -tzf "$tarball" | grep -E '\.pt$' | sed 's/^/      /' || echo "      (no .pt files)"
    n_ok=$((n_ok + 1))
done

echo
echo "=============================================================="
if [ "$n_bad" -gt 0 ]; then
    echo " $n_ok unpacked, $n_missing not available yet, $n_bad UNUSABLE."
    echo
    echo " An unusable archive is almost always an interrupted upload: the job's compute"
    echo " may have finished, but what reached OSDF is a partial object, and the transfer"
    echo " retry was then refused with 'remote object already exists' and held. There is"
    echo " nothing to salvage in the archive. Delete it and re-run the node:"
    echo "     ./fetch_results.sh --clean <job>"
    echo "     condor_rm <the held job>       # it can only collide again"
    echo " then resubmit that node. Its children's inputs are the archive, so they cannot"
    echo " run until it exists."
else
    echo " $n_ok unpacked, $n_missing not available yet."
fi
echo
# Every split's weights tree, not just the group split's. A model that unpacked into
# full_feb_v1/ and was never printed here is a model nobody knows arrived -- which is the
# same failure mode as an archive nobody lists.
for split in group_split_v1 replicate_split_v1 full_feb_v1; do
    if [ -d "$REPO/experiments/weights/$split" ]; then
        echo " experiments/weights/$split:"
        find "$REPO/experiments/weights/$split" -name '*.pt' -printf '   %p  %s bytes\n' \
            | sort
    fi
done
echo
echo " Split tables and their statistics:"
ls -1 "$REPO/scratch/simulated/files/splits" 2>/dev/null | sed 's/^/   /' || echo "   (none)"
for split in group_split_v1 replicate_split_v1 full_feb_v1; do
    if [ -d "$REPO/experiments/pipelines/stats/$split" ]; then
        echo "   stats/$split:"
        ls -1 "$REPO/experiments/pipelines/stats/$split" | sed 's/^/     /'
    fi
done
echo
if [ -d "$REPO/experiments/scoring/results" ]; then
    echo " Scoring artifacts:"
    find "$REPO/experiments/scoring/results" -name '*.npz' -printf '   %p  %s bytes\n' | sort
    echo
    echo " Tables and figures are made from those, locally, with no GPU:"
    echo "   python3 experiments/scoring/score.py   experiments/scoring/results/*/*.npz --out tables/"
    echo "   python3 experiments/scoring/figures.py experiments/scoring/results/*/*.npz --out figures/"
fi
echo "=============================================================="

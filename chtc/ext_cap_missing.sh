#!/bin/bash
# Which ext_cap.dag nodes have no archive, and the exact command to submit each.
#
#     ./ext_cap_missing.sh              # list what is missing, with its submit line
#     ./ext_cap_missing.sh --run        # submit them
#
# Why this exists
# ---------------
# DAGMan stops at a failed node and exits, and anything downstream of it never runs. The
# ext_cap graph lost `det_profiles_final_extcap` to a truncated upload, so seven of its
# children were never submitted -- and after DAGMan exits there is no record of which.
#
# Retyping seven `condor_submit ... -a task=... -a det_study=... -a flow_study=...` lines by
# hand is exactly the operation that has already cost this project three GPU rounds: a
# wrong VARS value does not fail, it scores a different model and produces a plausible
# number. So the pairs live here once, taken from bo_studies.sh's own table, and the
# dependency order is enforced rather than remembered.
#
# It never re-submits a node whose archive already exists, so it is safe to run repeatedly.
set -euo pipefail

RUN=0
[ "${1:-}" = "--run" ] && RUN=1

# shellcheck source=site.conf
NETID=$(whoami)
STAGING="/staging/${NETID:0:1}/${NETID}/oco2"
RESULTS="$STAGING/results"

# job name : submit file : VARS.  The pairs are bo_studies.sh:bo_task_studies's.
NODES=(
  "det_xco2_final_extcap|train_ext_cap.sub|study=det_xco2 epochs=120 nf_dropout=0.0 nf_layers=0"
  "det_profiles_final_extcap|train_ext_cap.sub|study=det_profiles epochs=120 nf_dropout=0.0 nf_layers=0"
  "det_last4_final_extcap|train_ext_cap.sub|study=det_last4 epochs=120 nf_dropout=0.0 nf_layers=0"
  "nf_xco2_final_extcap|train_ext_cap.sub|study=nf_xco2 epochs=120 nf_dropout=0.0 nf_layers=0"
  "nf_last4_final_extcap|train_ext_cap.sub|study=nf_last4 epochs=200 nf_dropout=0.0 nf_layers=0"
  "nf_profiles_final_extcap|train_ext_cap.sub|study=nf_profiles epochs=80 nf_dropout=0.2 nf_layers=6"
  "eval_xco2_extcap|eval_ext_cap.sub|task=xco2 det_study=det_xco2 flow_study=nf_xco2"
  "eval_profiles_extcap|eval_ext_cap.sub|task=profiles det_study=det_profiles flow_study=nf_profiles"
  "eval_last4_extcap|eval_ext_cap.sub|task=last4 det_study=det_last4 flow_study=nf_last4"
  "eval_last4_marginal_extcap|eval_ext_cap.sub|task=last4_marginal det_study=det_profiles flow_study=nf_profiles"
  "eval_xco2_resimcap|eval_resim_cap.sub|task=xco2 det_study=det_xco2 flow_study=nf_xco2"
  "eval_profiles_resimcap|eval_resim_cap.sub|task=profiles det_study=det_profiles flow_study=nf_profiles"
  "eval_last4_resimcap|eval_resim_cap.sub|task=last4 det_study=det_last4 flow_study=nf_last4"
  "oco2_month_202003_extcap|prep_month_ext_cap.sub|month=202003"
  "oco2_month_202007_extcap|prep_month_ext_cap.sub|month=202007"
  "eval_xco2_202003_extcap|eval_month_ext_cap.sub|task=xco2 month=202003 det_study=det_xco2 flow_study=nf_xco2"
  "eval_profiles_202003_extcap|eval_month_ext_cap.sub|task=profiles month=202003 det_study=det_profiles flow_study=nf_profiles"
  "eval_last4_202003_extcap|eval_month_ext_cap.sub|task=last4 month=202003 det_study=det_last4 flow_study=nf_last4"
  "eval_last4_marginal_202003_extcap|eval_month_ext_cap.sub|task=last4_marginal month=202003 det_study=det_profiles flow_study=nf_profiles"
  "eval_xco2_202007_extcap|eval_month_ext_cap.sub|task=xco2 month=202007 det_study=det_xco2 flow_study=nf_xco2"
  "eval_profiles_202007_extcap|eval_month_ext_cap.sub|task=profiles month=202007 det_study=det_profiles flow_study=nf_profiles"
  "eval_last4_202007_extcap|eval_month_ext_cap.sub|task=last4 month=202007 det_study=det_last4 flow_study=nf_last4"
  "eval_last4_marginal_202007_extcap|eval_month_ext_cap.sub|task=last4_marginal month=202007 det_study=det_profiles flow_study=nf_profiles"
)

# What each node needs before it can be submitted, as archive basenames. A node whose
# parents are not on /staging is listed as BLOCKED rather than submitted: its inputs are
# `transfer_input_files` entries, so submitting it early is a held job, not a queued one.
parents_of() {
    case "$1" in
        *_final_extcap)                 echo "oco2_ext_cap_v1" ;;
        eval_xco2_extcap|eval_xco2_resimcap)
                                        echo "det_xco2_final_extcap nf_xco2_final_extcap" ;;
        eval_profiles_extcap|eval_profiles_resimcap|eval_last4_marginal_extcap)
                                        echo "det_profiles_final_extcap nf_profiles_final_extcap" ;;
        eval_last4_extcap|eval_last4_resimcap)
                                        echo "det_last4_final_extcap nf_last4_final_extcap" ;;
        oco2_month_*_extcap)            echo "oco2_ext_cap_v1" ;;
        # The month is matched as the six-digit field. `${1//[!0-9]/}` strips every
        # non-digit from the whole name, so `eval_last4_202003_extcap` came out as
        # "4202003" -- the 4 from last4 -- and every month eval reported itself blocked
        # on an archive that could never exist.
        eval_xco2_2020*)                m=$(echo "$1" | grep -oE '20[0-9]{4}' | head -1)
                                        echo "oco2_month_${m}_extcap det_xco2_final_extcap nf_xco2_final_extcap" ;;
        eval_profiles_2020*|eval_last4_marginal_2020*)
                                        m=$(echo "$1" | grep -oE '20[0-9]{4}' | head -1)
                                        echo "oco2_month_${m}_extcap det_profiles_final_extcap nf_profiles_final_extcap" ;;
        eval_last4_2020*)               m=$(echo "$1" | grep -oE '20[0-9]{4}' | head -1)
                                        echo "oco2_month_${m}_extcap det_last4_final_extcap nf_last4_final_extcap" ;;
        *)                              echo "" ;;
    esac
}

archive_of() {
    case "$1" in
        oco2_*) echo "$STAGING/$1.tar.gz" ;;
        *)      echo "$RESULTS/$1.tar.gz" ;;
    esac
}

n_done=0 n_ready=0 n_blocked=0
echo "=============================================================="
echo " ext_cap.dag: what is on /staging, and what still has to run"
echo "=============================================================="
for entry in "${NODES[@]}"; do
    IFS='|' read -r node sub vars <<<"$entry"
    arch=$(archive_of "$node")
    note=""
    if [ -f "$arch" ]; then
        if gzip -t "$arch" 2>/dev/null; then
            printf "  %-36s done\n" "$node"; n_done=$((n_done + 1)); continue
        fi
        # Present but not a whole gzip stream: an interrupted upload. It has to be deleted
        # before the node can run again, because OSDF will not overwrite it -- so the clean
        # is part of the recipe rather than a separate thing to discover.
        note="   <-- TRUNCATED: ./fetch_results.sh --clean $node   FIRST"
    fi
    missing=""
    for pnode in $(parents_of "$node"); do
        p=$(archive_of "$pnode")
        if [ ! -f "$p" ] || ! gzip -t "$p" 2>/dev/null; then missing="$missing $pnode"; fi
    done
    if [ -n "$missing" ]; then
        printf "  %-36s BLOCKED on:%s\n" "$node" "$missing"
        n_blocked=$((n_blocked + 1)); continue
    fi
    args=""
    for kv in $vars; do args="$args -a $kv"; done
    printf "  %-36s READY%s\n" "$node" "$note"
    printf "        condor_submit %s%s\n" "$sub" "$args"
    n_ready=$((n_ready + 1))
    if [ "$RUN" -eq 1 ]; then
        # shellcheck disable=SC2086
        condor_submit "$sub" $args
    fi
done
echo "--------------------------------------------------------------"
echo " $n_done done, $n_ready ready to submit, $n_blocked blocked on a parent"
if [ "$RUN" -eq 0 ] && [ "$n_ready" -gt 0 ]; then
    echo " Re-run with --run to submit the READY ones."
fi
if [ "$n_blocked" -gt 0 ]; then
    echo " Blocked nodes need their parent's archive first. Run this again when it lands."
fi

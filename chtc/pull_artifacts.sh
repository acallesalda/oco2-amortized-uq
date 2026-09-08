#!/bin/bash
# Copy scored artifacts from the CHTC transfer node into this checkout, in the places the
# local scripts actually read.
#
#     ./chtc/pull_artifacts.sh                 # the xco2 re-score: 6 artifacts, ~60 MB
#     ./chtc/pull_artifacts.sh --months        # only the two months' xco2, ~210 MB
#     ./chtc/pull_artifacts.sh --all           # every task, every split
#     ./chtc/pull_artifacts.sh --panel         # Figure 7's dense-draw artifact
#     ./chtc/pull_artifacts.sh --dry-run       # print the scp commands and stop
#
# This is the rung between `fetch_results.sh` and `rebuild.sh`. `fetch_results.sh` runs ON
# the access point and unpacks each archive into the checkout THERE; `rebuild.sh` runs here
# and reads `experiments/scoring/*.npz`. The step between them has been done by hand every
# time, and it has the same failure twice over:
#
#   * `fetch_results.sh` unpacks the group-split artifacts to
#     experiments/scoring/results/group_split_v1/, while `rebuild.sh` reads
#     experiments/scoring/xco2_bo.npz -- so copying only into results/ leaves rebuild.sh
#     reading last week's file and reporting last week's numbers. rebuild.sh's shadow check
#     warns about it; this script just does the copy.
#   * a partial scp leaves a truncated .npz that numpy opens and reads as far as it goes.
#     Every transfer here is verified against the remote byte count.
#
# NetID comes from chtc/site.conf (the same file the submit files include), or $NETID, or
# --netid. Nothing else is read from the network.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO=$PWD

HOST="${OCO2_XFER_HOST:-transfer.chtc.wisc.edu}"
REMOTE="${OCO2_REMOTE_DIR:-OCO-2}"
DRY=0
WHAT=xco2

while [ $# -gt 0 ]; do
    case "$1" in
        --all)     WHAT=all ;;
        --months)  WHAT=months ;;
        --panel)   WHAT=panel ;;
        --xco2)    WHAT=xco2 ;;
        --dry-run) DRY=1 ;;
        --netid)   NETID="${2:?--netid needs a value}"; shift ;;
        --host)    HOST="${2:?--host needs a value}"; shift ;;
        --remote)  REMOTE="${2:?--remote needs a value}"; shift ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

# site.conf is `NETID = bucky` in condor macro syntax, and is gitignored precisely so that
# a pull cannot reset it.
if [ -z "${NETID:-}" ] && [ -f chtc/site.conf ]; then
    NETID=$(sed -n 's/^[[:space:]]*NETID[[:space:]]*=[[:space:]]*\([^[:space:]]*\).*/\1/p' \
            chtc/site.conf | head -1)
fi
if [ -z "${NETID:-}" ] || [ "$NETID" = "CHANGEME" ]; then
    echo "ERROR: no NetID. Set it in chtc/site.conf, or pass --netid <netid>." >&2
    exit 2
fi

# Which remote directory holds what, and where it has to end up locally.
#
#   <remote subdirectory under experiments/scoring>|<glob>|<extra local destination>
#
# The third field is the copy rebuild.sh depends on: it reads the group-split artifacts
# from experiments/scoring/ directly and the within-reference ones from
# results/replicate_split_v1/, so only the first list gets promoted.
case "$WHAT" in
    xco2)  JOBS=(
        "results/group_split_v1|xco2*.npz|experiments/scoring"
        "results/replicate_split_v1|xco2_borep.npz|"
        "results/month_202003_full|xco2.npz|"
        "results/month_202007_full|xco2.npz|"
    ) ;;
    all)   JOBS=(
        "results/group_split_v1|*.npz|experiments/scoring"
        "results/replicate_split_v1|*.npz|"
        "results/month_202003_full|*.npz|"
        "results/month_202007_full|*.npz|"
    ) ;;
    months) JOBS=(
        "results/month_202003_full|xco2.npz|"
        "results/month_202007_full|xco2.npz|"
    ) ;;
    panel) JOBS=(
        "results/group_split_v1|*_panel.npz|"
    ) ;;
esac

echo "=============================================================="
echo " pulling $WHAT from $NETID@$HOST:~/$REMOTE"
echo "=============================================================="

pulled=()
for job in "${JOBS[@]}"; do
    IFS='|' read -r sub glob promote <<< "$job"
    dest="$REPO/experiments/scoring/$sub"
    src="$NETID@$HOST:~/$REMOTE/experiments/scoring/$sub/$glob"
    echo
    echo "--- $sub/$glob"
    echo "    -> ${dest#$REPO/}"
    if [ "$DRY" -eq 1 ]; then
        echo "    scp -p \"$src\" \"$dest/\""
        [ -n "$promote" ] && echo "    cp $sub/$glob -> $promote/"
        continue
    fi
    mkdir -p "$dest"
    # -p keeps mtimes, which is what rebuild.sh's shadow check compares. Without it every
    # freshly copied file looks newer than everything and the check cries wolf.
    if ! scp -p "$src" "$dest/"; then
        echo "    scp failed. Nothing local was changed for this entry." >&2
        continue
    fi
    # The sidecar meta is small and its absence is survivable (artifact_status.py reads
    # __meta__ from inside the npz), so a failure here is a note rather than an error.
    scp -p "${src%.npz}.meta.json" "$dest/" 2>/dev/null || \
        echo "    (no .meta.json alongside; the npz carries its own __meta__)"
    for f in "$dest"/$glob; do
        [ -f "$f" ] || continue
        pulled+=("$f")
        if [ -n "$promote" ]; then
            cp -p "$f" "$REPO/$promote/"
            side="${f%.npz}.meta.json"
            [ -f "$side" ] && cp -p "$side" "$REPO/$promote/"
            echo "    promoted $(basename "$f") -> $promote/  (rebuild.sh reads this one)"
        fi
    done
done

if [ "$DRY" -eq 1 ]; then
    echo
    echo "dry run; nothing copied."
    exit 0
fi

# A truncated .npz opens fine and reads as far as it goes, so the byte count is checked
# against the remote rather than trusted. One ssh call for the whole list.
echo
echo "--- verifying sizes against the remote"
if command -v ssh >/dev/null 2>&1 && [ "${#pulled[@]}" -gt 0 ]; then
    remote_list=""
    for f in "${pulled[@]}"; do
        remote_list+=" ~/$REMOTE/${f#$REPO/}"
    done
    # shellcheck disable=SC2029  # the path list is meant to expand locally
    if remote_sizes=$(ssh "$NETID@$HOST" "stat -c '%s %n' $remote_list 2>/dev/null"); then
        bad=0
        while read -r size name; do
            # `stat %n` prints the expanded remote path (/home/<netid>/<dir>/...), so the
            # mapping back is anchored on the one component both sides share rather than on
            # the checkout's name, which --remote can change.
            local_f="$REPO/experiments/scoring/${name#*experiments/scoring/}"
            [ -f "$local_f" ] || { echo "    MISSING locally: $name"; bad=1; continue; }
            have=$(stat -c '%s' "$local_f")
            if [ "$have" != "$size" ]; then
                echo "    TRUNCATED: ${local_f#$REPO/}  $have of $size bytes"
                bad=1
            fi
        done <<< "$remote_sizes"
        [ "$bad" -eq 0 ] && echo "    all $(echo "$remote_sizes" | wc -l) files match the remote byte for byte."
    else
        echo "    could not stat the remote files; sizes unverified."
    fi
fi

echo
echo "=============================================================="
echo " the acceptance test"
echo "=============================================================="
PY="${PYTHON:-python3}"
"$PY" experiments/scoring/artifact_status.py || true
echo
echo "Every xco2 row must read  mc  yes  ~1.0. If one still says marglik/no, the job ran"
echo "with a stale oco2_code.tar.gz and the fix never travelled."
echo
echo "Then:  PYTHON=$PY ./experiments/scoring/rebuild.sh"

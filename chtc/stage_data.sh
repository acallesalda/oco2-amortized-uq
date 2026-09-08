#!/bin/bash
# Pull the OCO-2 dataset from Zenodo into CHTC /staging.
#
# RUN THIS ON THE TRANSFER SERVER, NOT ON AN ACCESS POINT:
#
#     ssh <netid>@transfer.chtc.wisc.edu
#     cd ~/OCO-2/chtc          # or wherever you cloned it
#     ./stage_data.sh
#
# CHTC asks that bulk data movement go through transfer.chtc.wisc.edu so the shared
# login nodes stay responsive. The whole record is ~3.3 GB, so this takes a few minutes
# on a good day; the script resumes partial downloads, so it is safe to re-run after an
# interruption and safe to run under tmux.
#
# Everything is verified against the SHA256SUMS published with Zenodo record 20617718.
# A file that already exists and matches is skipped, so re-running is cheap.
#
# Usage:
#     ./stage_data.sh                 # everything (3.3 GB)
#     ./stage_data.sh train test      # just what the training jobs need (780 MB)
#     ./stage_data.sh --list          # show what would be fetched, and current state
set -euo pipefail

ZENODO_RECORD=20617718
BASE_URL="https://zenodo.org/api/records/${ZENODO_RECORD}/files"

NETID=$(whoami)
STAGING="/staging/${NETID:0:1}/${NETID}/oco2"

# name|filename|sha256|bytes|what it is
# Checksums are from SHA256SUMS.txt in the record itself.
ITEMS=(
"train|oco2_processed_train.tar.gz|69f411cb8a80c14137f7e25d3a8d5e39050f85e458ec880ee0fe93da19699e32|624953386|processed training spectra + aux_data.csv  [REQUIRED for training]"
"test|oco2_processed_test.tar.gz|92abd8ee82e7c6258d0ba100563e6fd72fca6ea3abd4d4c2134830aeacc06846|155391914|processed test spectra + aux_data.csv + covariances  [REQUIRED for training]"
"stats|oco2_profile_stats.tar.gz|41ae7c9efa5717eb2e7932f69b3ba1ea535bfc9870cff093a7562e203f02442c|1564|profile_mu.npy / profile_sigma.npy  [REQUIRED for training]"
"weights|oco2_model_weights.tar.gz|c39962bd675ad4b62d5a8ee0b09c903391d8aef32aab633253b092069b0eb62b|319733926|published checkpoints (for scoring notebooks + Table A1 audit)"
"raw|oco2_unprocessed_og_data.tar.gz|a6a2df9e0062a65e9f854b959592dd8cd739e67ffd604527d03870b933a54008|2212659354|raw .h5 soundings (only needed to re-run extract_data_tree/clean_data)"
"manifest|MANIFEST.tsv|44126567ef59fea35df70388b77977d38374915b3606c900989402517e6164ae|7188031|file listing with sizes"
)

want=("$@")
list_only=0
if [ "${1:-}" = "--list" ]; then list_only=1; want=(); fi

selected() {
    if [ ${#want[@]} -eq 0 ]; then return 0; fi
    for w in "${want[@]}"; do
        if [ "$w" = "$1" ]; then return 0; fi
    done
    return 1
}

echo "=============================================================="
echo " OCO-2 data staging"
echo " host    : $(hostname)"
echo " netid   : $NETID"
echo " target  : $STAGING"
echo " source  : Zenodo record $ZENODO_RECORD"
echo "=============================================================="

case "$(hostname)" in
    transfer*) ;;
    *) echo
       echo "WARNING: this does not look like transfer.chtc.wisc.edu."
       echo "         CHTC asks that multi-GB transfers not run on the access points."
       echo "         Continuing in 10 seconds; Ctrl-C to stop."
       sleep 10 ;;
esac

if [ $list_only -eq 1 ]; then
    printf "\n%-10s %-34s %10s  %s\n" NAME FILE SIZE STATUS
    for item in "${ITEMS[@]}"; do
        IFS='|' read -r name file sha bytes desc <<< "$item"
        status="not fetched"
        if [ -f "$STAGING/$file" ]; then
            status="present ($(stat -c %s "$STAGING/$file") bytes)"
        fi
        printf "%-10s %-34s %10s  %s\n" "$name" "$file" "$bytes" "$status"
        printf "%-10s   %s\n" "" "$desc"
    done
    exit 0
fi

mkdir -p "$STAGING" "$STAGING/results" "$STAGING/logs"

fetched=0
skipped=0
for item in "${ITEMS[@]}"; do
    IFS='|' read -r name file sha bytes desc <<< "$item"
    selected "$name" || continue

    dest="$STAGING/$file"
    echo
    echo "--- $name: $file ($desc)"

    if [ -f "$dest" ]; then
        echo "    present; verifying checksum..."
        actual=$(sha256sum "$dest" | awk '{print $1}')
        if [ "$actual" = "$sha" ]; then
            echo "    OK, skipping."
            skipped=$((skipped + 1))
            continue
        fi
        echo "    checksum mismatch -- re-downloading."
        echo "      expected $sha"
        echo "      actual   $actual"
        rm -f "$dest"
    fi

    echo "    downloading $(numfmt --to=iec "$bytes" 2>/dev/null || echo "$bytes bytes")..."
    # -C - resumes a partial file; --retry rides out Zenodo's occasional 502s.
    curl -fL -C - --retry 5 --retry-delay 10 --retry-connrefused \
         -o "$dest" "$BASE_URL/$file/content"

    actual=$(sha256sum "$dest" | awk '{print $1}')
    if [ "$actual" != "$sha" ]; then
        echo "    ERROR: checksum mismatch after download."
        echo "      expected $sha"
        echo "      actual   $actual"
        echo "    Leaving the file in place so -C - can resume. Re-run this script."
        exit 1
    fi
    echo "    verified."
    fetched=$((fetched + 1))
done

echo
echo "=============================================================="
echo " $fetched downloaded, $skipped already present."
echo
ls -lh "$STAGING"
echo
echo " Quota check (default is 100 GB / 1000 items):"
du -sh "$STAGING" 2>/dev/null || true
echo
echo " Next: build the container (chtc/README.md step 3), then run the DAG."
echo "=============================================================="

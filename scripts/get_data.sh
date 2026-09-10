#!/bin/bash
# Download the dataset, the splits, the trained weights or the scoring artifacts from
# Zenodo, verify them, and unpack them where this repository's scripts already look.
#
#     ./scripts/get_data.sh --list                 what is in the record, and what you have
#     ./scripts/get_data.sh artifacts              enough to regenerate every table/figure
#     ./scripts/get_data.sh train test stats splits
#     ./scripts/get_data.sh weights                the published checkpoints
#     ./scripts/get_data.sh months                 March and July, the held-out months
#     ./scripts/get_data.sh all                    everything, including the raw HDF5
#
# Needs curl, tar and sha256sum (or shasum). No Python. Safe to interrupt and re-run:
# downloads resume, and a file that is already present and matches its checksum is skipped.
#
# Everything lands at the paths the code defaults to, so the commands in the documentation
# need no --files-root, --stats-dir or --weights-root of their own:
#
#     scratch/simulated/files/{train,test}/        spectra + aux_data.csv (+ ACOS covariances)
#     scratch/simulated/files/{202003,202007}/     the held-out months, the same layout
#     scratch/simulated/files/splits/              the split tables
#     scratch/simulated/og_data/                   raw .h5 soundings (only with `raw`)
#     experiments/pipelines/stats/<split>/         per-split standardization constants
#     experiments/weights/<split>/                 checkpoints
#     experiments/scoring/                         predictive artifacts (.npz + .meta.json)
#     group_split_v1_cov.csv                       the primary split table, at the root,
#                                                  where rebuild.sh expects it
#
# All of those are gitignored.
set -euo pipefail

# The record gains a new version -- and with it a NEW record id -- every time archives are
# added to it, and the id of an older version keeps serving that older version's file list
# forever. So a script that hardcodes a version id stops seeing new archives the moment a
# version is cut, and reports them as "not in the record yet" while they sit in the record.
#
# What this script wants is not a version but "whichever version is newest", so it asks for
# exactly that: the CONCEPT record id is the parent of every version and never changes, and
# `/versions/latest` resolves it to the newest one. No version id is hardcoded anywhere.
#
# `OCO2_ZENODO_RECORD=<id>` pins one version, which is what to do to reproduce against the
# record a paper cited rather than against whatever is newest.
ZENODO_CONCEPT=20617717
ZENODO_RECORD="${OCO2_ZENODO_RECORD:-}"
# Used only when the resolution call cannot be made at all. Being one version behind is a
# recoverable state -- every absent archive is named -- and being unable to run is not.
ZENODO_RECORD_FALLBACK=20617718
BASE_URL=""

cd "$(dirname "$0")/.."
ROOT=$PWD
DL="$ROOT/downloads"
FILES="$ROOT/scratch/simulated/files"
STATS="$ROOT/experiments/pipelines/stats"
WEIGHTS="$ROOT/experiments/weights"
SCORING="$ROOT/experiments/scoring"

# group|archive base name|unpack rule|what it is|needed for
#
# The BASE NAME carries no extension on purpose. Which compression the record actually
# uses is not this script's guess to make: `resolve_file` looks the base name up in the
# record's own SHA256SUMS.txt and takes whatever extension it finds there, and `untar`
# picks the decompressor from that extension. The record has held `.tar.gz` and its
# description has claimed `.tar.zst`; a script that hardcoded either would be wrong half
# the time, and wrong by 404 rather than by a message.
#
# The unpack rule is read by `unpack` below:
#   strip:<n>:<dir>      strip n leading path components, extract into dir
#   find:<needle>:<dir>  strip to just above <needle>, then extract into dir
#   plain:<dir>          extract into dir as-is
ITEMS=(
"train|oco2_processed_train|strip:4:$FILES|processed training spectra + aux_data.csv|training, and re-scoring from checkpoints"
"test|oco2_processed_test|strip:4:$FILES|processed test spectra, aux_data.csv, ACOS profile covariances|training and scoring"
"stats|oco2_profile_stats|find:experiments/pipelines/stats:$ROOT|profile_mu.npy / profile_sigma.npy for the original replicate-level split|the original-split models"
"splits|oco2_splits|plain:$ROOT|every split table (reference-disjoint, within-reference, all-February, the size subsets) with its own standardization constants and covariate encodings|any training or scoring run of the revised models"
"weights|oco2_model_weights_v2|find:experiments/weights:$ROOT|the published checkpoints: the six models of each split reported in the paper|scoring without training"
"weights_v1|oco2_model_weights|find:experiments/weights:$ROOT|the ORIGINAL submission's checkpoints, kept for provenance|nothing current; superseded by \`weights\`"
"artifacts|oco2_scoring_artifacts|plain:$SCORING|per-sounding predictive artifacts (.npz + .meta.json) for every experiment in the paper|regenerating every table and figure with no GPU"
"months|oco2_month_202003_full|plain:$ROOT|MARCH 2020: processed spectra, ACOS profile covariances, covariates and the month split table|re-scoring or re-fitting the temporal generalization experiment"
"months|oco2_month_202007_full|plain:$ROOT|JULY 2020: the same, for the second held-out month|re-scoring or re-fitting the temporal generalization experiment"
"raw|oco2_unprocessed_og_data|strip:3:$ROOT/scratch/simulated|raw .h5 soundings|re-running extract_data_tree.py / clean_data.py only"
)

SUMS="$DL/SHA256SUMS.txt"

sha_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
    else echo "no sha256sum or shasum on PATH; cannot verify downloads" >&2; return 1
    fi
}

# Resolved on first use rather than at startup, so that a re-run with everything already
# downloaded and verified needs no network at all.
resolve_record() {
    [ -n "$BASE_URL" ] && return 0
    if [ -z "$ZENODO_RECORD" ]; then
        local json id
        json=$(curl -fsSL --retry 3 --retry-delay 2 --max-time 30 \
               "https://zenodo.org/api/records/${ZENODO_CONCEPT}/versions/latest" 2>/dev/null \
               || true)
        # The record id is the one numeric `"id"` in the payload: every other id in it is a
        # quoted string -- the licence slug, the per-file UUIDs -- so the bare digits after
        # the colon are what identify it. Should Zenodo ever quote it too, the `self` link
        # carries the same number, hence the second pattern rather than a single one that
        # would have to be right about which form is served.
        id=$(printf '%s' "$json" | grep -o '"id"[[:space:]]*:[[:space:]]*[0-9]\{4,\}' \
             | head -1 | grep -o '[0-9]\{4,\}' || true)
        [ -z "$id" ] && id=$(printf '%s' "$json" | grep -o 'api/records/[0-9]\{4,\}' \
             | head -1 | grep -o '[0-9]\{4,\}' || true)
        if [ -n "$id" ]; then
            ZENODO_RECORD=$id
        else
            ZENODO_RECORD=$ZENODO_RECORD_FALLBACK
            echo "--- could not reach Zenodo to find the newest version of concept record" >&2
            echo "    $ZENODO_CONCEPT, so falling back to record $ZENODO_RECORD. If an archive" >&2
            echo "    below is reported as absent, it may exist in a newer version; check" >&2
            echo "    https://doi.org/10.5281/zenodo.$ZENODO_CONCEPT" >&2
        fi
    fi
    BASE_URL="https://zenodo.org/api/records/${ZENODO_RECORD}/files"
}

fetch_sums() {
    mkdir -p "$DL"
    if [ ! -s "$SUMS" ]; then
        resolve_record
        echo "--- fetching SHA256SUMS.txt from Zenodo record $ZENODO_RECORD ---"
        curl -fsSL --retry 5 --retry-delay 3 --max-time 120 -o "$SUMS" "$BASE_URL/SHA256SUMS.txt/content"
    fi
}

# The record's own checksum list is the authority, so nothing is hardcoded here and a new
# version of the record needs no change to this script. A base name with no entry in
# SHA256SUMS.txt has not been uploaded yet, and saying so is more useful than a 404.
#
# `resolve_file` turns a base name into the filename the record actually holds, so the
# compression is discovered rather than assumed. If the record ever holds two forms of the
# same archive, the first line wins and both are named in --list.
resolve_file() {
    awk -v b="$1" '{
        f = $2; sub(/^\*/, "", f)
        if (f == b || index(f, b ".tar") == 1 || index(f, b ".") == 1) { print f; exit }
    }' "$SUMS"
}

# For messages. The concept DOI is the honest answer before a version has been resolved:
# it always redirects to the newest version, which is the one being described.
record_url() {
    if [ -n "$ZENODO_RECORD" ]; then echo "https://zenodo.org/records/$ZENODO_RECORD"
    else echo "https://doi.org/10.5281/zenodo.$ZENODO_CONCEPT"
    fi
}

expected_sha() {
    awk -v f="$1" '{ g = $2; sub(/^\*/, "", g); if (g == f) { print $1; exit } }' "$SUMS"
}

# tar cannot infer zstd, and `-z` on a zstd stream fails with "not in gzip format" --
# which reads as a corrupt download rather than a wrong flag.
untar() {
    local src=$1; shift
    case "$src" in
        *.tar.gz|*.tgz)   tar -xzf "$src" "$@" ;;
        *.tar.zst|*.tzst) 
            if tar --zstd -tf "$src" >/dev/null 2>&1; then
                tar --zstd -xf "$src" "$@"
            elif command -v zstd >/dev/null 2>&1; then
                zstd -dc "$src" | tar -xf - "$@"
            else
                echo "ERROR: $src is zstd-compressed and neither tar --zstd nor the zstd"
                echo "       command is available. Install zstd, or ask for the .tar.gz form."
                return 1
            fi ;;
        *.tar.bz2)        tar -xjf "$src" "$@" ;;
        *.tar.xz)         tar -xJf "$src" "$@" ;;
        *.tar)            tar -xf "$src" "$@" ;;
        *) echo "ERROR: do not know how to unpack $src"; return 1 ;;
    esac
}

# The same question for listing, which the `find:` rule needs.
untar_list() {
    case "$1" in
        *.tar.gz|*.tgz)   tar -tzf "$1" ;;
        *.tar.zst|*.tzst) if tar --zstd -tf "$1" >/dev/null 2>&1; then tar --zstd -tf "$1"
                          else zstd -dc "$1" | tar -tf -; fi ;;
        *.tar.bz2)        tar -tjf "$1" ;;
        *.tar.xz)         tar -tJf "$1" ;;
        *.tar)            tar -tf "$1" ;;
    esac
}

download() {
    local file=$1 want=$2 dest="$DL/$file"
    if [ -s "$dest" ] && [ "$(sha_of "$dest")" = "$want" ]; then
        echo "    already downloaded and verified"
        return 0
    fi
    resolve_record
    echo "    downloading"
    curl -fL --progress-bar --retry 5 --retry-delay 3 -C - -o "$dest" "$BASE_URL/$file/content"
    local got
    got=$(sha_of "$dest")
    if [ "$got" != "$want" ]; then
        echo "ERROR: checksum mismatch for $file"
        echo "       expected $want"
        echo "       got      $got"
        echo "       Delete downloads/$file and re-run."
        return 1
    fi
    echo "    checksum ok"
}

unpack() {
    local file=$1 rule=$2 src="$DL/$file"
    local kind=${rule%%:*} rest=${rule#*:}
    case "$kind" in
        strip)
            local n=${rest%%:*} dir=${rest#*:}
            mkdir -p "$dir"
            untar "$src" --strip-components="$n" -C "$dir" ;;
        plain)
            mkdir -p "$rest"
            untar "$src" -C "$rest" ;;
        find)
            # Strip to just above <needle>, whatever depth the archive happens to be
            # rooted at. The archives were packed from three different working
            # directories over the life of the project; this is why nothing hardcodes a
            # fourth guess.
            local needle=${rest%%:*} dir=${rest#*:}
            local first strip
            first=$(untar_list "$src" | grep -m1 "$needle" || true)
            if [ -z "$first" ]; then
                echo "ERROR: $file contains no $needle path. First entries:"
                untar_list "$src" | head -5
                return 1
            fi
            strip=$(echo "${first%%$needle*}" | tr -cd '/' | wc -c)
            mkdir -p "$dir"
            untar "$src" --strip-components="$strip" -C "$dir" ;;
        *)  echo "ERROR: unknown unpack rule '$rule'"; return 1 ;;
    esac
}

# The split archive carries the tables under scratch/simulated/files/splits/. rebuild.sh
# and generalization_diagnostics.py read the primary one from the repository root, because
# that is where the pipeline writes it -- so link or copy it there rather than asking
# everyone to remember a --split-table.
place_primary_split() {
    local src="$FILES/splits/group_split_v1_cov.csv"
    if [ -f "$src" ] && [ ! -f "$ROOT/group_split_v1_cov.csv" ]; then
        cp "$src" "$ROOT/group_split_v1_cov.csv"
        echo "    copied group_split_v1_cov.csv to the repository root"
    fi
}

want=("$@")
if [ ${#want[@]} -eq 0 ]; then
    echo "usage: $0 [--list] <group> [<group> ...]   (or 'all')"
    echo
    printf "  %-11s %s\n" GROUP "WHAT IT IS"
    for item in "${ITEMS[@]}"; do
        IFS='|' read -r g f r desc need <<< "$item"
        printf "  %-11s %s\n" "$g" "$desc"
        printf "  %-11s   needed for: %s\n" "" "$need"
    done
    exit 1
fi

if [ "${want[0]}" = "--list" ]; then
    fetch_sums
    printf "\n%-11s %-36s %10s  %s\n" GROUP FILE "IN RECORD" "LOCAL"
    for item in "${ITEMS[@]}"; do
        IFS='|' read -r g base r desc need <<< "$item"
        f=$(resolve_file "$base")
        if [ -z "$f" ]; then
            printf "%-11s %-36s %10s  %s\n" "$g" "$base.*" "no" "-"
            continue
        fi
        sha=$(expected_sha "$f")
        local_state="-"
        if [ -s "$DL/$f" ]; then
            if [ -n "$sha" ] && [ "$(sha_of "$DL/$f")" = "$sha" ]; then local_state="verified"
            else local_state="present, UNVERIFIED"; fi
        fi
        printf "%-11s %-36s %10s  %s\n" "$g" "$f" "yes" "$local_state"
    done
    echo
    echo "record: $(record_url)"
    exit 0
fi

if [ "${want[0]}" = "all" ]; then
    want=()
    for item in "${ITEMS[@]}"; do want+=("${item%%|*}"); done
fi

fetch_sums

selected() {
    for w in "${want[@]}"; do [ "$w" = "$1" ] && return 0; done
    return 1
}

did_split=0
for item in "${ITEMS[@]}"; do
    IFS='|' read -r g base rule desc need <<< "$item"
    selected "$g" || continue
    f=$(resolve_file "$base")
    echo "=============================================================="
    echo " $g   (${f:-$base.*})"
    echo "=============================================================="
    if [ -z "$f" ]; then
        echo "    NOT IN THE RECORD YET."
        echo "    Zenodo record ${ZENODO_RECORD:-$ZENODO_CONCEPT} lists no archive named"
        echo "    $base.*, so there is nothing to fetch:"
        echo "    $(record_url)"
        continue
    fi
    sha=$(expected_sha "$f")
    download "$f" "$sha"
    echo "    unpacking"
    unpack "$f" "$rule"
    [ "$g" = "splits" ] && did_split=1
    echo "    done"
    echo
done

[ "$did_split" = 1 ] && place_primary_split

cat <<EOF

Downloads are kept in downloads/ so a re-run is free. Delete it to reclaim the space;
nothing reads from it after unpacking.

Next:
  tables and figures      ./experiments/scoring/rebuild.sh          (see docs/reproduce.md)
  score a checkpoint      python3 experiments/scoring/predict.py --help
  train a model           docs/training.md
EOF

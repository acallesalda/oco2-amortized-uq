#!/bin/bash
# Build the ALL-FEBRUARY training table, with statistics and covariates, so the same
# pipeline that trained the group split can train it.
#
#     condor_submit prep_full_feb.sub
#
# CPU only, a couple of minutes, nearly all of it file transfer. It opens no spectrum: the
# group split table already carries every aux_data column, the statistics come out of that
# table's own true_state_vector columns, and the covariate merge is two CSVs.
#
# Why this table exists
# ---------------------
# `group_split_v1` reserves 45 of February's 451 reference soundings as a test partition,
# because February was the only month there was. Once March and July are the test set that
# reservation buys nothing -- those 45 scenes are never scored again -- so this folds them
# back into training. The result is 406 training reference soundings instead of 361
# (+12.5 %), and 40,492 soundings instead of 36,003.
#
# The 45 VALIDATION scenes stay exactly where they were, and that is deliberate. They are
# what the searched configurations were ranked on, and -- the reason that settles it --
# predict.py calibrates the aleatoric diagonal of the Laplace predictive on them at scoring
# time. Calibrating that in-sample would make the Laplace's intervals narrower than they
# should be, in the one place where the paper's argument is that the flow's calibration
# beats the Laplace's. So "train on everything" stops at the test partition.
#
# See experiments/pipelines/make_full_feb_split.py for the argument in full.
#
# Produces: oco2_full_feb_v1.tar.gz
#     scratch/simulated/files/splits/full_feb_v1.csv
#     scratch/simulated/files/splits/full_feb_v1.summary.json
#     scratch/simulated/files/splits/full_feb_v1_cov.csv
#     experiments/pipelines/stats/full_feb_v1/{profile_mu,profile_sigma}.npy
#     experiments/pipelines/stats/full_feb_v1/{xco2_stats,provenance}.json
#     experiments/pipelines/stats/full_feb_v1/covariate_stats.json
set -euo pipefail

JOB="oco2_full_feb_v1"

# The split this job builds. Set here rather than in the submit file's `environment`
# because this is the one job whose split is not a choice -- it exists to make this table.
export OCO2_SPLIT=full_feb_v1

# shellcheck source=job_common.sh
source ./job_common.sh

export OCO2_PACK_PATHS="scratch/simulated/files/splits/full_feb_v1.csv scratch/simulated/files/splits/full_feb_v1.summary.json scratch/simulated/files/splits/full_feb_v1_cov.csv experiments/pipelines/stats/full_feb_v1 chtc/GIT_PROVENANCE.txt"

oco2_banner "$JOB (all-February split + statistics + covariates)"
oco2_trap_pack "$JOB"
oco2_setup

# NOT oco2_unpack_split: that would look for oco2_full_feb_v1.tar.gz, which is what this
# job produces. The group split's two archives are taken apart by hand instead, and only
# the two files that are actually inputs are extracted -- pulling the rest would put
# group-split statistics into this job's output directory, where the next job to unpack
# them would read them as full_feb_v1's.
for tarball in oco2_split_v1.tar.gz oco2_covariates_v1.tar.gz; do
    if [ ! -f "$tarball" ]; then
        echo "ERROR: $tarball was not transferred. It is an input to this job:"
        echo "         oco2_split_v1.tar.gz      the group split table (prep.sub)"
        echo "         oco2_covariates_v1.tar.gz covariates_v1.csv.gz (covariates.sub)"
        exit 1
    fi
done

echo "--- taking group_split_v1.csv out of the split archive ---"
tar -xzf oco2_split_v1.tar.gz -C "$PROJECT" \
    scratch/simulated/files/splits/group_split_v1.csv
rm -f oco2_split_v1.tar.gz

echo "--- taking covariates_v1.csv.gz out of the covariate archive ---"
tar -xzf oco2_covariates_v1.tar.gz -C "$PROJECT" \
    scratch/simulated/files/covariates_v1.csv.gz
rm -f oco2_covariates_v1.tar.gz

GROUP_SPLIT="$FILES_DIR/splits/group_split_v1.csv"
COV="$FILES_DIR/covariates_v1.csv.gz"
RAW_SPLIT="$FILES_DIR/splits/full_feb_v1.csv"
for f in "$GROUP_SPLIT" "$COV"; do
    if [ ! -s "$f" ]; then
        echo "ERROR: $f is missing or empty after extraction."
        exit 1
    fi
    echo "    $(basename "$f"): $(stat -c %s "$f") bytes"
done
echo

echo "=============================================================="
echo " 1/3  make_full_feb_split.py  (test -> train; val untouched)"
echo "=============================================================="
python3 experiments/pipelines/make_full_feb_split.py \
    --group-split "$GROUP_SPLIT" \
    --out         "$RAW_SPLIT"

echo
echo "=============================================================="
echo " 2/3  compute_split_stats.py  (training rows only)"
echo "=============================================================="
# Re-estimated, not copied. The training rows are 406 scenes rather than 361, so
# group_split_v1's constants were computed from a proper subset of what these models are
# fitted on -- reusing them would standardize with a mean nobody could reproduce from this
# table. The numbers move in the third decimal; the traceability is the point.
python3 experiments/pipelines/compute_split_stats.py \
    --split-csv "$RAW_SPLIT" \
    --out-dir   "$SPLIT_STATS" \
    --split     train

echo
echo "=============================================================="
echo " 3/3  augment_split.py  (merge covariates, encode from train)"
echo "=============================================================="
python3 experiments/pipelines/augment_split.py \
    --split-csv  "$RAW_SPLIT" \
    --covariates "$COV" \
    --out        "$SPLIT_CSV" \
    --stats-dir  "$SPLIT_STATS"

echo
echo "=============================================================="
echo " verification"
echo "=============================================================="
# Four things, and none of them is visible in the numbers that come out later. A stray
# `test` row would be a February sounding scored beside March's; validation scenes that are
# not group_split_v1's would mean the searched configurations were ranked on one set and
# their checkpoints selected on another, and would also move the scenes the Laplace's
# aleatoric diagonal is calibrated on. Every one of those trains fine and produces a
# plausible number. make_full_feb_split.py asserts them too; they are repeated here against
# the file that was actually written, because what the training jobs read is the file.
python3 - "$SPLIT_CSV" "$GROUP_SPLIT" <<'PY'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1], low_memory=False)
old = pd.read_csv(sys.argv[2], low_memory=False)
print(f"rows: {len(df)}")
print()
print(df["split"].value_counts().to_string())
print()

if (df["split"] == "test").any():
    raise SystemExit("FATAL: this table has a test partition. It is the ALL-February "
                     "training table; the test set is another month.")

refs = df.drop_duplicates(subset=["clst", "ref", "split"])
n = refs.groupby("split").size().to_dict()
old_train = old[old.split == "train"].drop_duplicates(["clst", "ref"]).shape[0]
print(f"reference soundings: train {n.get('train', 0)}, val {n.get('val', 0)}  "
      f"(group_split_v1 trained on {old_train})")


def scenes(frame, split):
    s = frame[frame["split"] == split]
    return set(map(tuple, s[["clst", "ref"]].drop_duplicates().to_numpy()))


# Train and val must stay scene-disjoint, or validation is a subsample of training and
# both the checkpoint selection and the Laplace's aleatoric calibration are in-sample.
if scenes(df, "train") & scenes(df, "val"):
    raise SystemExit("FATAL: reference soundings appear in both train and val.")

# And the validation scenes must be group_split_v1's, scene for scene.
if scenes(df, "val") != scenes(old, "val"):
    raise SystemExit("FATAL: the validation scenes are not group_split_v1's. The searched "
                     "configurations were ranked on those 45 scenes; selecting their "
                     "checkpoints on a different set is not the same experiment.")
print("OK: validation is group_split_v1's 45 scenes, unchanged and disjoint from training.")

# Every February sounding is somewhere. 44,982 in, 44,982 out.
if len(df) != len(old):
    raise SystemExit(f"FATAL: {len(old)} rows in, {len(df)} out. Nothing here drops rows.")
if df["ID"].duplicated().any():
    raise SystemExit("FATAL: duplicate sounding IDs in the table.")
print(f"OK: all {len(df)} February soundings kept; training gained "
      f"{n.get('train', 0) - old_train} reference soundings.")

cov = [c for c in df.columns if c.endswith("_fix")] + [c for c in ("XCO2Prior",
       "AODPrior1", "AODPrior2", "Latitude", "Longitude") if c in df.columns]
print(f"covariate columns present: {len(cov)}  {sorted(cov)}")
if "XCO2Prior" not in df.columns:
    raise SystemExit("FATAL: XCO2Prior is not in the augmented table, so the scene_prior "
                     "preset would train on something else.")
PY

echo
echo "--- stats written ---"
ls -l "$SPLIT_STATS"
cat "$SPLIT_STATS/xco2_stats.json"
echo
cat "$SPLIT_STATS/provenance.json"
echo

echo "=============================================================="
echo " $JOB finished"
echo "=============================================================="
ls -lh "$FILES_DIR/splits" "$SPLIT_STATS"

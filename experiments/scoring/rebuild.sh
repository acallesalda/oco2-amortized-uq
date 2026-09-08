#!/bin/bash
# Rebuild every table and figure in the paper from the artifacts on disk.
#
#     ./experiments/scoring/rebuild.sh
#     PYTHON=/c/Users/Aleja/anaconda3/python.exe ./experiments/scoring/rebuild.sh
#
# No GPU, no checkpoints, no queue, no data -- only the .npz files `predict.py` wrote.
# Ten to thirty minutes depending on the machine, three quarters of it the figures; the
# tables alone are the first two minutes. `./scripts/get_data.sh artifacts` downloads the
# artifacts it reads; on a cluster run,
# `chtc/fetch_results.sh` brings them back. Re-run this after either, and after any change
# to `score.py`, `figures.py` or `metrics.py`.
#
# It exists because the regeneration was five commands with long argument lists spread
# across two sets of notes, and two of them are easy to get subtly wrong:
#
#   * `score.py` with only the *_bo artifacts silently replaces the two-partition tables
#     with a one-partition version, and the within-reference rows -- where the one
#     resolved RMSE difference between the Laplace and the flow lives -- disappear;
#   * `figures.py` names its output by task plus split suffix, so pointing the
#     reference-disjoint and within-reference runs at one --out directory makes the
#     second overwrite the first with no warning.
#
# Both are handled here. Everything is written under `latex/figures/revision*` and
# `experiments/scoring/tables/`, and nothing existing is overwritten in place.
set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT=$PWD
PY="${PYTHON:-python3}"
S=experiments/scoring
TABLES=$S/tables
REP=$S/results/replicate_split_v1

# numpy on some Windows/conda installs needs this to import at all; harmless elsewhere.
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

have() { [ -f "$1" ]; }
missing=()
for f in "$S/xco2_bo.npz" "$S/profiles_bo.npz" "$S/last4_bo.npz"; do
    have "$f" || missing+=("$f")
done
if [ "${#missing[@]}" -gt 0 ]; then
    echo "ERROR: the reference-disjoint artifacts are absent:"
    printf '    %s\n' "${missing[@]}"
    echo "  ./scripts/get_data.sh artifacts        (or ./chtc/fetch_results.sh --best)"
    exit 1
fi

# Is anything here shadowed by a newer copy under results/?
#
# The group-split artifacts are read from $S directly while the replicate ones are read
# from $S/results/replicate_split_v1, and `fetch_results.sh` unpacks each archive wherever
# its own internal paths say -- so after a re-score it is entirely possible for
# results/group_split_v1/xco2.npz to be today's and $S/xco2.npz to be last week's, with
# this script reading the stale one and reporting last week's numbers.
#
# That failure is quiet in the direction that matters least (the not-reportable guard
# still fires on a stale contaminated artifact, so nothing wrong is emitted) and confusing in
# the direction that matters most: the guard staying put reads as "the CHTC job did not
# land" when the job landed and this script did not look at it. So say which file is being
# read and which one is newer, and let the reader decide.
# The mtime is only the cheap pre-filter, never the verdict. `diagnose_tail_scenes.py`
# copies artifacts into results/ as it runs, so on any tree where it has been used the
# results/ copy is newer than the original and byte-identical to it. A warning that fires
# on that is a warning nobody reads by the second week, so the difference is confirmed
# with `cmp` before anything is printed.
#
# `results/month_*/` is skipped, and not as a convenience: those directories hold a
# `xco2.npz` and a `last4.npz` that are a different EXPERIMENT, not a newer copy of
# February's. Without the skip this warns about all four month artifacts on every run --
# it did, the first time step 6 existed -- and a warning that is always wrong is a warning
# nobody reads when it is right. Step 6 scores them by their own paths.
shadowed=()
for f in xco2 profiles last4 xco2_bo profiles_bo last4_bo; do
    for d in "$S/results"/*/; do
        case "$d" in *"/month_"*) continue;; esac
        [ -f "$S/$f.npz" ] && [ -f "$d$f.npz" ] && [ "$d$f.npz" -nt "$S/$f.npz" ] \
            && ! cmp -s "$d$f.npz" "$S/$f.npz" \
            && shadowed+=("$d$f.npz is NEWER, and DIFFERS from, the $S/$f.npz this script reads")
    done
done
if [ "${#shadowed[@]}" -gt 0 ]; then
    echo "WARNING: a fetched artifact is not the one being scored:"
    printf '    %s\n' "${shadowed[@]}"
    echo "  Copy it over $S/ before trusting anything below."
    echo
fi

# The within-reference artifacts are optional: without them the tables lose their second
# block but everything else is unchanged, so this warns rather than stops.
REP_ARGS=()
for t in xco2 profiles last4 last4_marginal; do
    have "$REP/${t}_borep.npz" && REP_ARGS+=("$REP/${t}_borep.npz")
done
if [ "${#REP_ARGS[@]}" -eq 0 ]; then
    echo "NOTE: no within-reference artifacts under $REP."
    echo "      Tables will carry the reference-disjoint rows only."
    echo "      ./scripts/get_data.sh artifacts   (or ./chtc/fetch_results.sh --rerun)"
fi

echo "=============================================================="
echo " 1/8  tables and paired differences        (score.py)"
echo "=============================================================="
"$PY" $S/score.py \
    $S/xco2_bo.npz $S/profiles_bo.npz $S/last4_bo.npz \
    ${REP_ARGS[@]+"${REP_ARGS[@]}"} \
    --label group_split_v1_cov='reference-disjoint' \
    --label replicate_split_v1_cov='within-reference' \
    --paired --out "$TABLES"

echo
echo "=============================================================="
echo " 1b/8 joint vs marginal calibration        (joint_diagnostics.py)"
echo "=============================================================="
# Which coordinates every NLL and coverage number is computed in, and whether
# near-nominal MARGINAL coverage can be read as calibration of the joint predictive: mean
# ||z||^2/D and the chi2-PIT KS statistic, both invariant to the target's coordinate
# system, joint and marginal coverage side by side, and NLL in ppm per level, which is the
# only NLL column comparable across tasks of different D.
#
# Anything that reads the artifacts belongs in this script. This table did not use to be
# here, and it was the last float still carrying a not-reportable guard hours after the
# re-score that cleared it had landed.
"$PY" $S/joint_diagnostics.py \
    $S/xco2_bo.npz $S/profiles_bo.npz $S/last4_bo.npz \
    ${REP_ARGS[@]+"${REP_ARGS[@]}"} \
    --label group_split_v1_cov='reference-disjoint' \
    --label replicate_split_v1_cov='within-reference' \
    --out "$TABLES"

echo
echo "=============================================================="
echo " 2/8  where each method's error comes from  (delta_blocks.py)"
echo "=============================================================="
"$PY" $S/delta_blocks.py \
    $S/xco2_bo.npz $S/profiles_bo.npz $S/last4_bo.npz --out "$TABLES"

echo
echo "=============================================================="
echo " 2b/8 published inputs vs corrected inputs  (score.py)"
echo "=============================================================="
# What changing the inputs did. Same scorer, same test
# partition, same bootstrap unit -- the two families differ in the four scrambled
# covariates against `scene_prior`'s twelve, and in the hyperparameter search, which is
# why the caption has to say "and" rather than attribute the gain to the inputs alone.
# The covariate ablation (chtc/covariates.dag, collect_covariate_runs.py) is what
# separates them, and it is validation RMSE, so it cannot go in this table.
#
# Its own directory: score.py stems its filenames by task, so writing this beside the
# reference-disjoint/within-reference tables of step 1 would overwrite them with a
# same-named table of a different comparison -- the identical trap the two figure
# directories below exist for.
#
# The ACOS row is the control and must be identical between the two blocks to four
# significant figures: same retrieval, same soundings, two independently produced
# artifacts. If it is not, the artifacts are not the same soundings and nothing else in
# the table is readable.
PUB_ARGS=()
for t in xco2 profiles last4; do
    have "$S/${t}.npz" && PUB_ARGS+=("$S/${t}.npz" "$S/${t}_bo.npz")
done
if [ "${#PUB_ARGS[@]}" -eq 0 ]; then
    echo "  the published-input artifacts are absent (./scripts/get_data.sh artifacts);"
    echo "  skipping. Everything else is unaffected."
else
    "$PY" $S/score.py "${PUB_ARGS[@]}" \
        --label group_split_v1='published inputs (4, scrambled)' \
        --label group_split_v1_cov='corrected inputs (12)' \
        --out "$TABLES/inputs_compare"
fi

echo
echo "=============================================================="
echo " 3/8  the input table                      (make_input_table.py)"
echo "=============================================================="
if have group_split_v1_cov.csv; then
    "$PY" experiments/pipelines/make_input_table.py \
        --split-table group_split_v1_cov.csv --out "$TABLES"
else
    echo "  group_split_v1_cov.csv absent; writing the schema without ranges."
    "$PY" experiments/pipelines/make_input_table.py --out "$TABLES"
fi

echo
echo "=============================================================="
echo " 4/8  figures                              (figures.py)"
echo "=============================================================="
# --cases both writes the randomly selected panel AND the adversarially selected one the
# submitted Figure 7 actually used, so the caption and the figure can be made to agree.
# The seed is recorded in the filename's sibling log line, not in the PNG, so it is fixed
# here rather than defaulted.
#
# --dense and --flow-contours are what make the bivariate figure the one in the paper: the
# published panel is 20,000 draws per sounding with the flow's own 68/95% HDR contours over
# them, and neither is reachable from the 512 draws the scoring artifact carries. Without
# these two flags this step still writes a `pairs_profiles_random.png` -- a sparser,
# contourless one -- straight over the figure the manuscript includes, with no warning.
# The panel artifact is optional, so its absence degrades the figure rather than stopping
# the rebuild, and says so.
PANEL=$S/results/group_split_v1/profiles_panel.npz
DENSE=()
if have "$PANEL"; then
    DENSE=(--dense "$PANEL" --flow-contours)
else
    echo "NOTE: $PANEL absent, so the bivariate panel will be drawn from the 512 scoring"
    echo "      draws and without HDR contours -- not the published Figure 10."
    echo "      ./scripts/get_data.sh artifacts   (or ./chtc/pull_artifacts.sh --panel)"
fi
"$PY" $S/figures.py \
    $S/xco2_bo.npz $S/profiles_bo.npz $S/last4_bo.npz \
    ${DENSE[@]+"${DENSE[@]}"} \
    --out latex/figures/revision --cases both --seed 123

# Separate directory, not a separate name: figures.py stems its filenames by task plus the
# split's own suffix, and `*_borep` artifacts report `replicate_split_v1_cov` -- same stem
# as the reference-disjoint run, so a shared --out would overwrite silently.
if [ "${#REP_ARGS[@]}" -gt 0 ]; then
    echo
    echo "--- within-reference figures -> latex/figures/revision_within"
    "$PY" $S/figures.py ${REP_ARGS[@]+"${REP_ARGS[@]}"} \
        --out latex/figures/revision_within --cases random --seed 123
fi

echo
echo "=============================================================="
echo " 4b/8 published vs corrected inputs, per level  (figures.py)"
echo "=============================================================="
# The per-level companion to step 2b's table: the same two artifact families, drawn in one
# set of axes so the gap is visible level by level. Only the two profile tasks -- XCO2 has
# no level axis and `--overlay` refuses rather than writing a blank figure.
#
# Same guard as step 2b: without the published-input artifacts there is nothing to compare
# against, and that is a missing fetch rather than an error.
if [ "${#PUB_ARGS[@]}" -gt 0 ]; then
    for t in profiles last4; do
        if have "$S/${t}.npz" && have "$S/${t}_bo.npz"; then
            "$PY" $S/figures.py \
                --overlay "published inputs=$S/${t}.npz" \
                          "corrected inputs=$S/${t}_bo.npz" \
                --out latex/figures/revision --overlay-stem "${t}_inputs"
        fi
    done
else
    echo "  the published-input artifacts are absent; skipping (see step 2b)."
fi

echo
echo "=============================================================="
echo " 5/8  the error decomposition + its figure  (decompose_error.py)"
echo "=============================================================="
# The exact split of every method's error into the part that is constant within a reference
# sounding and the part that varies inside it, on all three held-out experiments at once.
# It is what the temporal and near-surface mechanism paragraphs are written from,
# and it reproduces every published RMSE cell to four decimals, which is the check that it
# is scoring the right soundings. The months are optional: the script prints which cases it
# skipped and the February block stands alone.
"$PY" $S/decompose_error.py --out "$TABLES" --fig latex/figures

echo
echo "=============================================================="
echo " 6/8  the temporal table                   (make_temporal_table.py)"
echo "=============================================================="
# The temporal table, in a directory of its own -- and that is not a stylistic choice.
# Each of the four month archives carries `experiments/scoring/tables/table_{xco2,last4}.csv`
# at exactly the path step 1 writes February's to, so unpacking one overwrites Tables 1-3
# with a single-month version, and re-running step 1 deletes the months again. The
# work-around was to unpack `results/*` only, which left the month numbers reachable only
# from inside a tarball. Scoring them here, into `$TABLES/temporal/`, is what makes the
# temporal table regenerable at all.
if [ -f "$S/results/month_202003_full/xco2.npz" ]; then
    "$PY" $S/make_temporal_table.py --out "$TABLES"
else
    echo "  the month artifacts are absent ($S/results/month_*_full/);"
    echo "  ./scripts/get_data.sh artifacts. Skipping: February is unaffected."
fi

echo
echo "=============================================================="
echo " 7/8  the appendix hyperparameter tables   (make_hparam_table.py)"
echo "=============================================================="
# Table A1 and the search table, from the six selection records rather than from the
# checkpoints -- the records are 2 kB each and tracked in hyperparameters/, the checkpoints
# are a 300 MB download, which is why the first version of Table A1 was typed by hand and
# went three model generations out of date. Standard library only.
HP=hyperparameters
[ -f "$HP/bo_det_xco2_best.json" ] || HP=chtc/bo/results
if [ -f "$HP/bo_det_xco2_best.json" ]; then
    "$PY" $S/make_hparam_table.py --results "$HP" --out "$TABLES"
else
    echo "  no selection records found; skipping. Table A1 stays whatever it was."
    echo "  They are tracked in hyperparameters/ -- pass --records hyperparameters"
fi

echo
echo "=============================================================="
echo " 8/8  generalization diagnostics"
echo "=============================================================="
# Three readers, in increasing order of what they need.
#
# The first needs the split table: the partition inventory, the standardization constants,
# and the no-spectrum baselines that say how much of the held-out skill is not available
# from scene metadata. Guarded rather than unconditional -- the table is a Zenodo item, and
# when it was absent this step's traceback took the three readers below it down too.
if have group_split_v1_cov.csv; then
    "$PY" $S/generalization_diagnostics.py --split-csv group_split_v1_cov.csv --out "$TABLES"
else
    echo "  group_split_v1_cov.csv absent; skipping the partition inventory and the"
    echo "  no-spectrum baselines (./scripts/get_data.sh splits)."
fi

# The second needs a trainer's stdout, which lives on the access point (chtc/logs/ is
# gitignored and never fetched). If the .out files have been copied down, this draws the
# learning curves of whatever it finds.
LC_LOGS=(chtc/logs/*_final*.out)
if [ -e "${LC_LOGS[0]}" ]; then
    "$PY" $S/learning_curves.py "${LC_LOGS[@]}" --out "$TABLES" --fig latex/figures
else
    echo "  no chtc/logs/*_final*.out here; skipping the learning curves."
    echo "  On the access point:  python3 $S/learning_curves.py chtc/logs/*_final*.out"
fi

# The third needs the size.dag artifacts. score.py has to see them and the published run in
# ONE call, because the curve is read off one table.
SIZE_ARGS=()
for k in 90 180 270; do
    f="$S/results/size_${k}_v1/xco2_s${k}.npz"
    have "$f" && SIZE_ARGS+=("$f")
done
if [ "${#SIZE_ARGS[@]}" -gt 0 ]; then
    # --out $TABLES/size, NOT $TABLES: this call sees the three subsets and the
    # published run and nothing else, so writing it into the paper's table directory
    # would replace table_xco2.{csv,tex} with a version that has lost the
    # within-reference rows -- the failure this script's header is about.
    "$PY" $S/score.py "${SIZE_ARGS[@]}" $S/xco2_bo.npz \
        --label group_split_v1_cov='reference-disjoint' --out "$TABLES/size"
    "$PY" $S/size_curve.py --task xco2 --tables "$TABLES/size" \
        --out "$TABLES/size" --fig latex/figures
else
    echo "  no size_*_v1 artifacts; skipping the learning curve in training scenes."
    echo "  ./scripts/get_data.sh artifacts, or run chtc/size.dag to produce them."
fi

# The offline-versus-online cost ledger. Standard library only, so it always runs, but what it is
# allowed to claim depends on what is on this machine: with chtc/logs/ or a directory of
# fetched job archives it *measures* the offline cost, and without them it falls back to the
# per-trial estimate and stamps every derived row `assumed` in amortization.json. Same for
# t_N: benchmark.csv if benchmark.sub has landed, the Table 8 flow row otherwise.
"$PY" $S/amortization.py \
    --condor-logs chtc/logs \
    --benchmark-csv "$TABLES/benchmark.csv" \
    --split-csv group_split_v1_cov.csv \
    --out "$TABLES" --fig latex/figures >"$TABLES/amortization.md" 2>&1 || true
if grep -q "ASSUMED" "$TABLES/amortization.md"; then
    echo "  amortization: written from ESTIMATED training cost -- see the tail of"
    echo "  $TABLES/amortization.md. On the access point, where the logs are, re-run:"
    echo "    python3 $S/amortization.py --condor-logs chtc/logs --out $TABLES"
else
    echo "  amortization: written from measured job wall clock"
fi

# The gap artifacts are scored here too when they are present: same scorer, both
# partitions in one table, so the in-sample and held-out rows are the same quantity.
GAP_ARGS=()
for t in xco2 last4 profiles; do
    f="$S/results/group_split_v1/${t}_gap_train.npz"
    have "$f" && GAP_ARGS+=("$f")
done
if [ "${#GAP_ARGS[@]}" -gt 0 ]; then
    "$PY" $S/score.py "${GAP_ARGS[@]}" \
        $S/xco2_bo.npz $S/profiles_bo.npz $S/last4_bo.npz \
        --label group_split_v1_cov='held-out scenes' \
        --label group_split_v1_cov+train='training rows (IN SAMPLE)' \
        --out "$TABLES/gap"
else
    echo "  no *_gap_train artifacts; skipping the generalization gap."
    echo "  ./scripts/get_data.sh artifacts, or run chtc/gap.dag to produce them."
fi

# The attribution figure is redrawn from its own two CSVs, with no GPU and no checkpoint --
# the same rule as every other figure here: the measurement costs a queue slot, the picture
# does not. `attribution.dag` writes the CSVs; this redraws whenever a colour or a caption
# changes.
ATTR_DREW=0
for t in xco2 last4 profiles; do
    if have "$TABLES/attribution_channels_${t}.csv"; then
        "$PY" $S/spectral_attribution.py --task "$t" --from-tables \
            --out "$TABLES" --fig latex/figures
        ATTR_DREW=1
    fi
done
if [ "$ATTR_DREW" -eq 0 ]; then
    echo "  no attribution_channels_*.csv; skipping the attribution figures."
    echo "  Run chtc/attribution.dag to measure them (it needs a GPU)."
fi

echo
echo "=============================================================="
echo " done"
echo "=============================================================="
echo "  tables : $TABLES/table_{xco2,profiles,last4}.{csv,tex}"
echo "           $TABLES/paired_*.csv   $TABLES/delta_blocks.md"
echo "           $TABLES/inputs_{spectral,aux}.tex"
echo "           $TABLES/model_{hyperparams,search}.tex   (Table A1)"
echo "           $TABLES/decomposition_*.csv               (mechanism)"
echo "           $TABLES/temporal/temporal_results.{csv,tex}   (the temporal table)"
echo "           $TABLES/temporal/decomposition{,_xco2}.tex"
if [ "${#PUB_ARGS[@]}" -gt 0 ]; then
    echo "           $TABLES/inputs_compare/    (published inputs vs corrected)"
fi
echo "           $TABLES/generalization_{partitions,baselines}.csv"
echo "           $TABLES/learning_curve{s,_gaps}.csv   $TABLES/size/size_curve_xco2.csv"
echo "           $TABLES/amortization.{csv,tex,json,md}      (the cost ledger)"
echo "           $TABLES/gap/table_*.csv                (in sample vs held out)"
echo "  figures: latex/figures/decomposition.png  (the temporal mechanism)"
echo "           latex/figures/revision/          (reference-disjoint)"
if [ "${#REP_ARGS[@]}" -gt 0 ]; then
    echo "           latex/figures/revision_within/   (within-reference)"
fi
echo
echo "Read the XCO2 Laplace row first. Scored from an artifact whose Laplace predictive"
echo "carries its observation-noise term, its coverage is near 95 % and its NLL near 1;"
echo "38 % and 10.9 mean the artifact predates that fix -- re-score with"
echo "predict.py --laplace-mode mc."

#!/bin/bash
# Train one model on the reference-sounding-disjoint split.
#
#     ./train.sh <job_name> [warm_start_tarball]
#
# job_name is one of the keys in the case statement below; it also names the output
# archive. The submit files pass it from the DAG's VARS lines, so the definitive list of
# what runs lives here, in one place, rather than being spread across the submit files.
#
# Every configuration is a single fixed-hyperparameter run with Bayesian optimization
# switched off, so that the only thing that differs from the published numbers is the
# split itself. Hyperparameters come from the checkpoints behind the published results --
# see the header comment of each *_grouped.yaml.
#
# ORDER OF OPERATIONS. The job selects and *validates* its configuration before unpacking
# any data. Unpacking 43k .npy files takes a minute or two; discovering afterwards that
# Hydra cannot find the config -- because oco2_code.tar.gz predates it -- wastes that, and
# buries the reason under a wall of setup output. Validation is cheap and comes first.
set -euo pipefail

JOB="${1:?usage: train.sh <job_name> [warm_start_tarball]}"
WARMSTART="${2:-}"

# shellcheck source=job_common.sh
source ./job_common.sh

# The trap goes on before anything that can fail, so that even a job dying during setup
# ships an archive -- HTCondor holds a job whose declared output file is missing, and a
# hold message is a much worse diagnostic than a .err file. The GPU check runs before the
# untarring, so a bad slot is discovered cheaply.
oco2_banner "$JOB"
oco2_trap_pack "$JOB"
oco2_gpu_check
oco2_setup

CONF="code/shared_conf/experiment"

# ---------------------------------------------------------------------------------
# Select. SCRIPT plus exactly one of EXP (a Hydra experiment name) or EXPCFG (a YAML the
# flow trainers read with --exp_cfg). REQUIRE_CKPT is the warm-start backbone, if any.
# ---------------------------------------------------------------------------------
EXP=""
EXPCFG=""
REQUIRE_CKPT=""
EXTRA=()
NEED_COVARIATES=0

case "$JOB" in
    # ---------------------------------------------------------------- deterministic --
    det_xco2)
        SCRIPT=experiments/train/train.py
        EXP=full_regressor_grouped ;;

    det_profiles)
        # Also the backbone nf_profiles warm-starts from.
        SCRIPT=experiments/train/train_profiles.py
        EXP=full_regressor_profiles_grouped ;;

    det_last4)
        # Near-surface (levels 17-20). Backbone for nf_last4.
        SCRIPT=experiments/train/train_profiles.py
        EXP=full_regressor_profiles_last4_grouped ;;

    det_last1)
        # Surface only (level 20), one output. The deterministic half of the scalar
        # surface task -- the near-surface counterpart of det_xco2. Nothing warm-starts
        # from it: the surface flow is train_nf.py's own backbone, trained from scratch.
        SCRIPT=experiments/train/train_profiles.py
        EXP=full_regressor_profiles_last1_grouped ;;

    # ------------------------------------------------------------------- controls --
    # Same models and hyperparameters as the three above, trained on the ORIGINAL
    # replicate-level split. Not in oco2.dag; nothing warm-starts from them. They exist to
    # separate "the split is hard" from "something is wrong", which the group-split runs
    # cannot do alone -- see full_regressor_profiles_last4_replicate.yaml.
    det_xco2_replicate)
        SCRIPT=experiments/train/train.py
        EXP=full_regressor_replicate ;;

    det_profiles_replicate)
        SCRIPT=experiments/train/train_profiles.py
        EXP=full_regressor_profiles_replicate ;;

    det_last4_replicate)
        SCRIPT=experiments/train/train_profiles.py
        EXP=full_regressor_profiles_last4_replicate ;;

    nf_xco2_replicate)
        # Optional: the XCO2 flow on the replicate split, so that row has the same
        # cross-split comparison the deterministic models do. A new number, not a
        # reproduction -- the published scalar flow modelled surface CO2.
        SCRIPT=experiments/train/train_nf.py
        EXPCFG="$CONF/nf_full_regressor_nflows_replicate.yaml" ;;

    # ----------------------------------------------------------- covariate runs --
    # One factor at a time, on the SAME reference-sounding-disjoint split, so each row of
    # the comparison isolates one change. All of them read the augmented split table
    # (group_split_v1_cov.csv), which is the base table plus columns -- so `_base2` must
    # reproduce det_profiles / det_last4 / det_xco2 exactly, and if it does not, the merge
    # rather than the covariates is what moved the number.
    #
    #   _base2     legacy covariates, standardized loss     the control
    #   _ppm       legacy covariates, ppm-weighted loss     metric alignment alone
    #   _covctl    the same four (published), new encoder   encoding change alone
    #   _covfix    the same four, CORRECTLY assigned         what the metadata bug cost
    #   _cov       + viewing geometry and prior psfc         the recovered geometry
    #   _covprior  + XCO2Prior and the aerosol AOD priors    the priors ACOS was given
    #   _covgeo    + latitude and longitude                  the climatology ablation
    #   _covacos   + the ACOS retrieval's own output        a post-processor, not a retrieval
    #
    # At D=1 the ppm and standardized metrics differ by a constant, so det_xco2 has no
    # _ppm variant and its loss/selection stay standardized.
    det_xco2_base2|det_xco2_covctl|det_xco2_covfix|det_xco2_covprior|det_xco2_covgeo|det_xco2_cov|det_xco2_covgeo|det_xco2_covacos)
        SCRIPT=experiments/train/train.py
        EXP=full_regressor_cov_grouped
        NEED_COVARIATES=1
        case "$JOB" in
            *_base2)   EXTRA=("covariates=legacy") ;;
            *_covctl)  EXTRA=("covariates=legacy_v2") ;;
            *_covfix)  EXTRA=("covariates=legacy_fix") ;;
            *_covprior) EXTRA=("covariates=scene_prior") ;;
            *_cov)     EXTRA=("covariates=scene") ;;
            *_covgeo)  EXTRA=("covariates=scene_geo") ;;
            *_covacos) EXTRA=("covariates=acos_informed") ;;
        esac ;;

    det_profiles_base2 | det_profiles_ppm | det_profiles_covctl | det_profiles_covfix | det_profiles_covprior | det_profiles_covgeo | \
    det_profiles_cov | det_profiles_covgeo | det_profiles_covacos | \
    det_last4_base2 | det_last4_ppm | det_last4_covctl | det_last4_covfix | det_last4_covprior | det_last4_covgeo | \
    det_last4_cov | det_last4_covgeo | det_last4_covacos)
        SCRIPT=experiments/train/train_profiles.py
        case "$JOB" in
            det_profiles_*) EXP=full_regressor_profiles_cov_grouped ;;
            det_last4_*)    EXP=full_regressor_profiles_last4_cov_grouped ;;
        esac
        NEED_COVARIATES=1
        case "$JOB" in
            *_base2)   EXTRA=("covariates=legacy"     "loss_weight=std" "select_metric=std") ;;
            *_ppm)     EXTRA=("covariates=legacy"     "loss_weight=ppm" "select_metric=ppm") ;;
            *_covctl)  EXTRA=("covariates=legacy_v2"  "loss_weight=ppm" "select_metric=ppm") ;;
            *_covfix)  EXTRA=("covariates=legacy_fix" "loss_weight=ppm" "select_metric=ppm") ;;
            *_covprior) EXTRA=("covariates=scene_prior" "loss_weight=ppm" "select_metric=ppm") ;;
            *_cov)     EXTRA=("covariates=scene"      "loss_weight=ppm" "select_metric=ppm") ;;
            *_covgeo)  EXTRA=("covariates=scene_geo"  "loss_weight=ppm" "select_metric=ppm") ;;
            *_covacos) EXTRA=("covariates=acos_informed" "loss_weight=ppm" "select_metric=ppm") ;;
        esac ;;

    # The Bayesian-optimization winners are NOT here. They used to be, as three
    # `det_*_best` cases reading a Hydra file that chtc/bo_collect.py generated by hand
    # off a finished study -- a path with three copies of the study table in it and a
    # human step in the middle. They are now nodes of chtc/bo/bo_all.dag: bo_final.sh
    # retrains a winner straight from the record the select node wrote, for all six
    # studies, using the one table in chtc/bo_studies.sh. Nothing dispatches through this
    # file any more, so a case here could only be a second, staler answer to the same
    # question.

    # ------------------------------------------------------------ normalizing flows --
    nf_xco2)
        # backbone_ckpt is null in this config, so unlike the other two flows it trains
        # from scratch and has no DAG parent.
        SCRIPT=experiments/train/train_nf.py
        EXPCFG="$CONF/nf_full_regressor_nflows_grouped.yaml" ;;

    nf_last1)
        # The same script as nf_xco2, with data.target=surface: a 1-d autoregressive
        # spline flow over level 20. Trains from scratch, so it has no DAG parent either.
        SCRIPT=experiments/train/train_nf.py
        EXPCFG="$CONF/nf_nflows_surface_grouped.yaml" ;;

    nf_profiles)
        SCRIPT=experiments/train/train_nf_profiles.py
        EXPCFG="$CONF/nf_profiles_grouped.yaml"
        REQUIRE_CKPT="$WEIGHTS_ROOT/full_regressor_profiles/net_profiles.pt" ;;

    nf_last4)
        SCRIPT=experiments/train/train_nf_profiles_4d.py
        EXPCFG="$CONF/nf_profiles4_grouped.yaml"
        REQUIRE_CKPT="$WEIGHTS_ROOT/full_regressor_profiles_last4/net_profiles.pt" ;;

    # ------------------------------------------------- flows with the covariates --
    # Same flow architecture and optimizer as the three above; the covariate set is the
    # only change, so the pair isolates it. Each warm-starts from the covariate-trained
    # deterministic backbone, and the aux encoder's shape must match -- the trainer
    # raises if the warm-start would have to drop it.
    nf_xco2_cov)
        SCRIPT=experiments/train/train_nf.py
        EXPCFG="$CONF/nf_full_regressor_nflows_cov_grouped.yaml"
        NEED_COVARIATES=1 ;;

    nf_profiles_cov)
        SCRIPT=experiments/train/train_nf_profiles.py
        EXPCFG="$CONF/nf_profiles_cov_grouped.yaml"
        NEED_COVARIATES=1
        REQUIRE_CKPT="$WEIGHTS_ROOT/full_regressor_profiles_scene_prior_ppm/net_profiles.pt" ;;

    nf_last4_cov)
        SCRIPT=experiments/train/train_nf_profiles_4d.py
        EXPCFG="$CONF/nf_profiles4_cov_grouped.yaml"
        NEED_COVARIATES=1
        REQUIRE_CKPT="$WEIGHTS_ROOT/full_regressor_profiles_last4_scene_prior_ppm/net_profiles.pt" ;;

    *)
        echo "ERROR: unknown job '$JOB'."
        echo "  training  : det_xco2 det_profiles det_last4 nf_xco2 nf_profiles nf_last4"
        echo "  surface   : det_last1 nf_last1        (scalar level-20 task; last1.dag)"
        echo "  controls  : det_xco2_replicate det_profiles_replicate det_last4_replicate"
        echo "              nf_xco2_replicate"
        echo "  covariates: det_{xco2,profiles,last4}_{base2,covctl,covfix,cov,covprior,covgeo,covacos}"
        echo "  flows+cov : nf_xco2_cov nf_profiles_cov nf_last4_cov"
        echo "              det_{profiles,last4}_ppm"
        echo "  bo winners: not here -- they are bo_final.sh, run as nodes of"
        echo "              chtc/bo/bo_all.dag. See docs/hpc.md."
        exit 2 ;;
esac

# ---------------------------------------------------------------------------------
# Validate, before the expensive part.
# ---------------------------------------------------------------------------------
echo "--- configuration ---"
echo "    job    : $JOB"
echo "    script : $SCRIPT"
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: $SCRIPT is not in oco2_code.tar.gz."
    exit 2
fi

if [ -n "$EXP" ]; then
    echo "    config : +experiment=$EXP"
    if [ ! -f "$CONF/$EXP.yaml" ]; then
        echo
        echo "ERROR: $CONF/$EXP.yaml is not in this code tarball."
        echo "       Hydra would fail with 'Could not find experiment/$EXP' about ninety"
        echo "       seconds from now, after untarring 43k files. Almost always this means"
        echo "       oco2_code.tar.gz predates the config:"
        echo
        echo "           cd ~/OCO-2 && git pull && ./chtc/make_code_tarball.sh"
        echo
        if [ -f chtc/GIT_PROVENANCE.txt ]; then
            echo "       This tarball was built from:"
            sed 's/^/         /' chtc/GIT_PROVENANCE.txt
        fi
        echo "       Configs it does contain:"
        find "$CONF" -name '*.yaml' -printf '         %f\n' | sort
        exit 2
    fi
elif [ -n "$EXPCFG" ]; then
    echo "    config : $EXPCFG"
    if [ ! -f "$EXPCFG" ]; then
        echo
        echo "ERROR: $EXPCFG is not in this code tarball."
        echo "       cd ~/OCO-2 && git pull && ./chtc/make_code_tarball.sh"
        exit 2
    fi
fi
echo

# ---------------------------------------------------------------------------------
# Data, then train.
# ---------------------------------------------------------------------------------
oco2_unpack_data
oco2_unpack_split
if [ "$NEED_COVARIATES" -eq 1 ]; then oco2_unpack_covariates; fi
oco2_unpack_warmstart "$WARMSTART"
if [ -n "$REQUIRE_CKPT" ]; then oco2_require_ckpt "$REQUIRE_CKPT"; fi

echo "=============================================================="
echo " training: $JOB"
echo "=============================================================="

if [ -n "$EXP" ]; then
    if [ "${#EXTRA[@]}" -gt 0 ]; then
        echo "overrides: ${EXTRA[*]}"
        python3 "$SCRIPT" "+experiment=$EXP" "${EXTRA[@]}"
    else
        python3 "$SCRIPT" "+experiment=$EXP"
    fi
else
    python3 "$SCRIPT" --exp_cfg "$EXPCFG"
fi

echo
echo "=============================================================="
echo " $JOB finished"
echo "=============================================================="
for root in "$WEIGHTS_ROOT" "$PROJECT/experiments/weights/replicate_control"; do
    if [ -d "$root" ]; then
        find "$root" -name '*.pt' -printf '%p  %s bytes\n' | sort
        find "$root" \( -name 'metrics.json' -o -name 'best_metric.txt' \) \
            -print -exec cat {} \; -exec echo \; || true
    fi
done

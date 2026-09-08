#!/bin/bash
# The six studies, in one place. Sourced by bo_trial.sh, bo_final.sh and bo_eval.sh.
#
# Every one of those three has to know the same six facts about a study -- which script
# trains it, which config that script reads, which key holds the epoch count, which key
# holds the weights directory, what switches the warm start off, and where the checkpoint
# lands. They used to know them separately: bo_trial.sh had a case statement, bo_collect.py
# had a FLOW dict and a WEIGHTS dict, and train.sh had a third copy under `bo winners`.
# Three copies of a table is three chances for a search trial and its own retrained winner
# to be different models, which is exactly the failure that leaves no trace in any log.
#
#     source ./bo_studies.sh
#     bo_study_spec nf_last4 /path/to/weights_dir
#
# sets, for the named study:
#
#   BO_ROLE          det | flow      which dialect the trainer speaks
#   BO_SCRIPT        the trainer
#   BO_EXP           Hydra experiment name           (det only, else empty)
#   BO_EXPCFG        YAML the trainer reads          (flow only, else empty)
#   BO_CONF          the config file to check exists, whichever kind it is
#   BO_EPOCH_KEY     the key that holds the epoch count
#   BO_SELECT        std | ppm -- the validation metric that picks the checkpoint (det)
#   BO_TASK          xco2 | profiles | last4 -- what predict.py calls it
#   BO_CKPT          the checkpoint, relative to the weights directory
#   BO_SCRATCH[]     the override that keeps the backbone untrained (flow only)
#   BO_DIR_OVERRIDE[] the overrides that point the run at $2, if $2 was given
#
# THE FLOWS TRAIN FROM SCRATCH. Every published flow was warm-started from the matching
# deterministic backbone, so the flow's own architecture was never searched against a fair
# alternative -- whatever the backbone had already learned, the flow inherited, and a
# search over the flow alone could only tune a head on somebody else's features. Here the
# warm start is off for all three, which is what makes the six studies six comparable
# searches rather than three models and three heads. It is switched off HERE, on the
# command line, rather than in the config, because the config is shared with the
# warm-started runs that these are the comparison for.

BO_STUDIES_ALL="det_xco2 det_profiles det_last4 nf_xco2 nf_profiles nf_last4"

# The covariate set, fixed for every study. Chosen by the covariate comparison
# (covariates.dag), not by the search: tuning over it would make the ablation incoherent.
BO_COVARIATES_DEFAULT="scene_prior"

# Which family of experiment configs to read. The six `*_cov_grouped` configs and both
# sets of twins -- `*_cov_replicate` and `*_cov_fullfeb` -- are identical except for the
# split table, the statistics directory and the weights directory;
# code/test/test_replicate_configs.py asserts that key by key, so switching families here
# is exactly and only a change of partition. `*_cov_ext` is the fourth member: February
# plus the extension delivery, 447 training scenes.
#
# Set from OCO2_SPLIT, which oco2_setup exports, so a study cannot be trained against one
# split and scored against another: both read the same variable, from the same submit
# file's `environment`.
bo_config_suffix() {
    case "${OCO2_SPLIT:-group_split_v1}" in
        replicate_split_v1) echo "cov_replicate" ;;
        full_feb_v1)        echo "cov_fullfeb" ;;
        ext_v1)             echo "cov_ext" ;;
        ext_cap_v1)         echo "cov_extcap" ;;
        *)                  echo "cov_grouped" ;;
    esac
}

bo_study_spec() {
    local study="${1:?usage: bo_study_spec <study> [weights_dir]}"
    local wd="${2:-}"
    local conf_dir="code/shared_conf/experiment"

    BO_ROLE=""; BO_SCRIPT=""; BO_EXP=""; BO_EXPCFG=""; BO_CONF=""
    BO_EPOCH_KEY=""; BO_SELECT=""; BO_TASK=""; BO_CKPT=""
    BO_SCRATCH=()
    BO_DIR_OVERRIDE=()

    case "$study" in
        # ---------------------------------------------------------------- deterministic
        det_xco2)
            BO_ROLE=det
            BO_SCRIPT=experiments/train/train.py
            BO_EXP=full_regressor_$(bo_config_suffix)
            BO_EPOCH_KEY=hyps.epochs
            # At D=1 the ppm and standardized RMSE differ by a constant, so it selects
            # the same checkpoint either way; it is std because that is what the config
            # declares and train.py does.
            BO_SELECT=std
            BO_TASK=xco2
            # train.py trains into the Hydra run directory and promotes the best trial
            # into <weight_path>/global_best.
            BO_CKPT=global_best/net.pt
            if [ -n "$wd" ]; then
                BO_DIR_OVERRIDE=("data.weight_path=$wd" "hydra.run.dir=$wd/run")
            fi ;;

        det_profiles)
            BO_ROLE=det
            BO_SCRIPT=experiments/train/train_profiles.py
            BO_EXP=full_regressor_profiles_$(bo_config_suffix)
            BO_EPOCH_KEY=hyps.epochs
            BO_SELECT=ppm
            BO_TASK=profiles
            BO_CKPT=global_best/net_profiles.pt
            if [ -n "$wd" ]; then
                BO_DIR_OVERRIDE=("weights_dir=$wd" "hydra.run.dir=$wd/run")
            fi ;;

        det_last4)
            BO_ROLE=det
            BO_SCRIPT=experiments/train/train_profiles.py
            BO_EXP=full_regressor_profiles_last4_$(bo_config_suffix)
            BO_EPOCH_KEY=hyps.epochs
            BO_SELECT=ppm
            BO_TASK=last4
            BO_CKPT=global_best/net_profiles.pt
            if [ -n "$wd" ]; then
                BO_DIR_OVERRIDE=("weights_dir=$wd" "hydra.run.dir=$wd/run")
            fi ;;

        # ----------------------------------------------------------------------- flows
        nf_xco2)
            BO_ROLE=flow
            BO_SCRIPT=experiments/train/train_nf.py
            BO_EXPCFG="$conf_dir/nf_full_regressor_nflows_$(bo_config_suffix).yaml"
            BO_EPOCH_KEY=optim.epochs
            BO_TASK=xco2
            BO_CKPT=best_flow_nflows.pt
            BO_SCRATCH=("setup.backbone_ckpt=null")
            if [ -n "$wd" ]; then
                BO_DIR_OVERRIDE=("weights_dir=$wd")
            fi ;;

        nf_profiles)
            BO_ROLE=flow
            BO_SCRIPT=experiments/train/train_nf_profiles.py
            BO_EXPCFG="$conf_dir/nf_profiles_$(bo_config_suffix).yaml"
            BO_EPOCH_KEY=optim.epochs
            BO_TASK=profiles
            BO_CKPT=best_flow_profiles.pt
            BO_SCRATCH=("setup.backbone_ckpt=null")
            if [ -n "$wd" ]; then
                BO_DIR_OVERRIDE=("weights_dir=$wd")
            fi ;;

        nf_last4)
            BO_ROLE=flow
            BO_SCRIPT=experiments/train/train_nf_profiles_4d.py
            BO_EXPCFG="$conf_dir/nf_profiles4_$(bo_config_suffix).yaml"
            BO_EPOCH_KEY=train.epochs
            BO_TASK=last4
            # cfg.train.ckpt_name; the config sets it and nothing here overrides it.
            BO_CKPT=best_nf4.pt
            BO_SCRATCH=("model.pretrained_ckpt=null")
            if [ -n "$wd" ]; then
                BO_DIR_OVERRIDE=("train.save_dir=$wd")
            fi ;;

        *)
            echo "ERROR: unknown study '$study'." >&2
            echo "       studies: $BO_STUDIES_ALL" >&2
            return 2 ;;
    esac

    if [ "$BO_ROLE" = det ]; then
        BO_CONF="$conf_dir/$BO_EXP.yaml"
    else
        BO_CONF="$BO_EXPCFG"
    fi
}

# Where a study's final model lives, relative to $WEIGHTS_ROOT -- which is
# experiments/weights/<OCO2_SPLIT>, so the same study on two splits lands in two trees.
#
# The `bo_final/` prefix is not decoration. The covariate comparison already wrote
# full_regressor_profiles_scene_prior_ppm/, and a searched winner trained with the same
# preset would land on exactly that directory -- two different models, one path, and
# whichever archive is unpacked second silently becomes "the" model. They are different
# objects and the difference between them is a result, so they get different directories.
bo_final_dir() {
    echo "bo_final/${1:?usage: bo_final_dir <study>}"
}

# The two studies whose models are scored together for one predict.py task.
#
# `last4_marginal` is the exception the published Table 3 also has: the 20-level models
# evaluated on levels 17-20, which is a slice of the profile models' predictive and not a
# separate pair of checkpoints.
bo_task_studies() {
    case "${1:?usage: bo_task_studies <task>}" in
        xco2)           echo "det_xco2 nf_xco2" ;;
        profiles)       echo "det_profiles nf_profiles" ;;
        last4)          echo "det_last4 nf_last4" ;;
        last4_marginal) echo "det_profiles nf_profiles" ;;
        *) echo "ERROR: unknown task '$1'." >&2; return 2 ;;
    esac
}

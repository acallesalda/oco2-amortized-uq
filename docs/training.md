# Training

Six models: a deterministic regressor and a conditional normalizing flow for each of three
targets. Every one of them is one command on one GPU.

| model | target | trainer | experiment config | epochs | hyperparameters |
|---|---|---|---|---|---|
| `det_xco2` | XCO₂ (D=1) | `train.py` | `full_regressor_cov_grouped` | 40 | `hyperparameters/bo_det_xco2_best.json` |
| `det_profiles` | 20-level profile | `train_profiles.py` | `full_regressor_profiles_cov_grouped` | 40 | `bo_det_profiles_best.json` |
| `det_last4` | levels 17–20 | `train_profiles.py` | `full_regressor_profiles_last4_cov_grouped` | 40 | `bo_det_last4_best.json` |
| `nf_xco2` | XCO₂ (D=1) | `train_nf.py` | `nf_full_regressor_nflows_cov_grouped` | 40 | `bo_nf_xco2_best.json` |
| `nf_profiles` | 20-level profile | `train_nf_profiles.py` | `nf_profiles_cov_grouped` | 40 | `bo_nf_profiles_best.json` |
| `nf_last4` | levels 17–20 | `train_nf_profiles_4d.py` | `nf_profiles4_cov_grouped` | 40 | `bo_nf_last4_best.json` |

All six train on the reference-sounding-disjoint split with the `scene_prior` covariates,
and **all six train from scratch** — no flow warm-starts from a deterministic backbone. See
[`../hyperparameters/README.md`](../hyperparameters/README.md) for what is fixed across the
six and why.

## Before you start

```bash
python3 -m pip install -r chtc/requirements.txt
./scripts/get_data.sh train test splits
python3 -m pytest code/test -q               # ~320 tests, no GPU needed
```

`splits` brings the split tables **and** the per-split standardization constants. A run
given the wrong statistics directory does not fail; it trains a different model. Both
values are printed at startup — read them.

## The two dialects

The trainers were written at different times and take configuration differently. This is
worth knowing before the first typo, because one of them fails loudly and the other does
not.

**The deterministic trainers are Hydra** (`@hydra.main`), composing in struct mode:

```bash
python3 experiments/train/train_profiles.py \
    +experiment=full_regressor_profiles_cov_grouped \
    covariates=scene_prior select_metric=ppm loss_weight=ppm \
    hyps.epochs=40 \
    $(python3 scripts/overrides.py hyperparameters/bo_det_profiles_best.json)
```

Overriding a key the config does not declare is a **hard error**; `++key=value` is the
add-or-override form.

**The flow trainers take a YAML plus `--set`**, into a plain dict:

```bash
python3 experiments/train/train_nf_profiles.py \
    --exp_cfg code/shared_conf/experiment/nf_profiles_cov_grouped.yaml \
    --set covariates=scene_prior optim.epochs=40 setup.backbone_ckpt=null \
    $(python3 scripts/overrides.py hyperparameters/bo_nf_profiles_best.json)
```

`--set` **creates** missing keys, so a typo there is silent: it lands in the checkpoint's
stored config as a setting that did nothing. And a leading `++` would be read as part of
the key name, so do not carry Hydra habits across.

The epoch key differs too: `hyps.epochs` for the deterministic trainers, `optim.epochs`
for `train_nf.py` and `train_nf_profiles.py`, `train.epochs` for the 4-D flow.

## The six commands

Assuming the defaults from `scripts/get_data.sh`. Each writes into
`experiments/weights/group_split_v1/<name>/`; add `data.weight_path=`, `weights_dir=` or
`train.save_dir=` (the trainers differ — the table below says which) to put it elsewhere.

```bash
SPLIT=scratch/simulated/files/splits/group_split_v1_cov.csv
STATS=experiments/pipelines/stats/group_split_v1
W=experiments/weights/group_split_v1
H=hyperparameters

# --- deterministic -------------------------------------------------------------
python3 experiments/train/train.py +experiment=full_regressor_cov_grouped \
    covariates=scene_prior hyps.epochs=40 \
    data.split_csv=$SPLIT data.stats_dir=$STATS \
    data.weight_path=$W/det_xco2 hydra.run.dir=$W/det_xco2/run \
    $(python3 scripts/overrides.py $H/bo_det_xco2_best.json)

python3 experiments/train/train_profiles.py +experiment=full_regressor_profiles_cov_grouped \
    covariates=scene_prior select_metric=ppm hyps.epochs=40 \
    data.split_csv=$SPLIT data.stats_dir=$STATS \
    weights_dir=$W/det_profiles hydra.run.dir=$W/det_profiles/run \
    $(python3 scripts/overrides.py $H/bo_det_profiles_best.json)

python3 experiments/train/train_profiles.py \
    +experiment=full_regressor_profiles_last4_cov_grouped \
    covariates=scene_prior select_metric=ppm hyps.epochs=40 \
    data.split_csv=$SPLIT data.stats_dir=$STATS \
    weights_dir=$W/det_last4 hydra.run.dir=$W/det_last4/run \
    $(python3 scripts/overrides.py $H/bo_det_last4_best.json)

# --- flows ---------------------------------------------------------------------
python3 experiments/train/train_nf.py \
    --exp_cfg code/shared_conf/experiment/nf_full_regressor_nflows_cov_grouped.yaml \
    --set covariates=scene_prior optim.epochs=40 setup.backbone_ckpt=null \
          data.split_csv=$SPLIT data.stats_dir=$STATS weights_dir=$W/nf_xco2 \
    $(python3 scripts/overrides.py $H/bo_nf_xco2_best.json)

python3 experiments/train/train_nf_profiles.py \
    --exp_cfg code/shared_conf/experiment/nf_profiles_cov_grouped.yaml \
    --set covariates=scene_prior optim.epochs=40 setup.backbone_ckpt=null \
          data.split_csv=$SPLIT data.stats_dir=$STATS weights_dir=$W/nf_profiles \
    $(python3 scripts/overrides.py $H/bo_nf_profiles_best.json)

python3 experiments/train/train_nf_profiles_4d.py \
    --exp_cfg code/shared_conf/experiment/nf_profiles4_cov_grouped.yaml \
    --set covariates=scene_prior train.epochs=40 model.pretrained_ckpt=null \
          data.split_csv=$SPLIT data.stats_dir=$STATS train.save_dir=$W/nf_last4 \
    $(python3 scripts/overrides.py $H/bo_nf_last4_best.json)
```

Where the checkpoint lands, per model:

| model | checkpoint, relative to its weights directory | directory override |
|---|---|---|
| `det_xco2` | `global_best/net.pt` | `data.weight_path=` |
| `det_profiles`, `det_last4` | `global_best/net_profiles.pt` | `weights_dir=` |
| `nf_xco2` | `best_flow_nflows.pt` | `weights_dir=` |
| `nf_profiles` | `best_flow_profiles.pt` | `weights_dir=` |
| `nf_last4` | `best_nf4.pt` | `train.save_dir=` |

The deterministic trainers train into a Hydra run directory and *promote* the best trial
into `global_best/`, alongside a `best_hparams.yaml` and a `metrics.json`. The flows write
their best checkpoint directly. Either way, `predict.py` finds them from
`--weights-root` — see [`evaluation.md`](evaluation.md).

## What the loss is measured in

Two knobs, and they matter more than they look:

* `loss_weight` — `std` weights every pressure level equally in standardized coordinates;
  `ppm` weights the loss in physical units, so the near-surface levels, whose spread is
  several times the mid-column's, dominate as they do in the reported RMSE.
* `select_metric` — which validation metric picks the checkpoint, on the same two scales.

At D=1 the two differ by a constant, so `det_xco2` has no `_ppm` variant and stays
standardized. For the two profile tasks the published models use `ppm` for both, which is
the setting that aligns the training objective with the metric the paper reports.

## Schedule

The published runs used the trainers' defaults: cosine to lr/10 for the deterministic
regressors, constant for `train_nf.py` and `train_nf_profiles.py`, cosine to 0 for the 4-D
flow, no warm-up and no extra gradient clipping.

The learning curves that come out of those runs are not flattering, and they are in the
paper for that reason: the three deterministic regressors are still descending at epoch 40,
two of them checkpointing at 38 and 39 of 40; the scalar flow's validation NLL ends above
its epoch-1 value; and the 20-level flow is best after 7 of 40 epochs and then diverges.
The first two are schedule problems and the third is a capacity problem, and none of them
is a reason to redo the search — the peak learning rate, the architecture and the weight
decay are what the search ranked.

Every trainer writes `history.json` beside its checkpoint: one record per epoch, training
loss and validation metric. It is the learning curve, it is impossible to recover from a
checkpoint afterwards, and `experiments/scoring/learning_curves.py` draws it.

If you want to do better than the published runs, `code/utils/schedule.py` is the one
schedule all five trainers share, and its knobs are config keys. Their defaults reproduce
the published behaviour exactly -- `warmup_epochs=0` and `eta_min_frac=0.1` are
`CosineAnnealingLR(T_max=T, eta_min=lr/10)` to within float error, and `grad_clip=None` is
no clipping -- so a run that sets none of them trains the model that was published:

| key (deterministic / flow) | |
|---|---|
| `hyps.warmup_epochs` / `optim.warmup_epochs` | linear warm-up, in epochs, before the cosine starts |
| `hyps.eta_min_frac` / `optim.eta_min_frac` | the cosine's floor, as a fraction of the peak learning rate |
| `hyps.grad_clip` / `optim.grad_clip` | gradient-norm clip |
| -- / `optim.sched=cosine` | turns annealing on for `train_nf.py` and `train_nf_profiles.py`, which have none |
| -- / `nflow.dropout`, `nflow.layers` | the 20-level flow's coupling stack |
| -- / `eval.rmse_samples` | posterior draws per validation sounding, for a per-epoch validation RMSE on a flow |

The peak learning rate is deliberately not among them: it is what the search ranked
configurations by, and changing it makes the selection record no longer describe the model.
`nflow.layers` changes the *architecture* the search selected, so a run using it is a
different model for that task and has to be reported as one.

On a cluster, `chtc/bo_final.sh` exposes the same knobs as `OCO2_WARMUP_EPOCHS`,
`OCO2_ETA_MIN_FRAC`, `OCO2_GRAD_CLIP`, `OCO2_NF_SCHED`, `OCO2_NF_DROPOUT`,
`OCO2_NF_LAYERS`, `OCO2_NF_WD` and `OCO2_RMSE_SAMPLES`, because a submit file can set an
environment variable but not a command line. They translate to exactly the keys above, in
whichever of the two dialects the study's trainer speaks.

## Other splits, and other experiments

The `*_cov_grouped` configs have twins that differ **only** in the split table, the
statistics directory and the weights directory — `code/test/test_replicate_configs.py`
asserts that key by key, so switching families is exactly and only a change of partition:

| family | split |
|---|---|
| `*_cov_grouped` | `group_split_v1` — reference-disjoint, the primary result |
| `*_cov_replicate` | `replicate_split_v1` — within-reference, the secondary result |
| `*_cov_fullfeb` | `full_feb_v1` — all of February that is not validation; the models scored on March and July |

For the size curve, train a `*_cov_grouped` config against a `size_<k>_v1` table by passing
`data.split_csv` and `data.stats_dir` explicitly, exactly as above. Nothing else changes,
which is what makes the curve a curve in training scenes.

## The baselines

`fc_aux` and `fc_no_aux` (`experiments/train/train.py +experiment=fc_aux`) are
fully-connected regressors on flattened spectra, with and without the auxiliary variables.
They are not in the paper; they are kept because they are the cheapest sanity check that
the data pipeline and the loss are doing what you think.

## Re-running the hyperparameter search

25 trials of 20 epochs per study, five rounds of five, then a retrain of the winner at 40
epochs — 160 GPU jobs in total. It is a cluster job and is described in
[`hpc.md`](hpc.md). It is not needed to reproduce the paper: the six records in
`hyperparameters/` are its output.

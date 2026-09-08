# The six selected configurations

One JSON per model. These are the hyperparameters behind every number in the paper, and
they are records rather than a table someone typed: each was written by the selection step
of its own Bayesian-optimization study, from the 25 trial records that study produced.

`experiments/scoring/make_hparam_table.py` renders them into the appendix table, and
`scripts/overrides.py` turns one back into command-line overrides, so the model that gets
trained and the configuration that was selected are the same object by construction.

```bash
python3 scripts/overrides.py hyperparameters/bo_nf_profiles_best.json --show
```

| study | task | trainer | experiment config | selected on | checkpoint |
|---|---|---|---|---|---|
| `det_xco2` | XCO₂ (D=1) | `train.py` | `full_regressor_cov_grouped` | validation RMSE (standardized) | `global_best/net.pt` |
| `det_profiles` | 20-level profile | `train_profiles.py` | `full_regressor_profiles_cov_grouped` | validation RMSE (ppm) | `global_best/net_profiles.pt` |
| `det_last4` | levels 17–20 | `train_profiles.py` | `full_regressor_profiles_last4_cov_grouped` | validation RMSE (ppm) | `global_best/net_profiles.pt` |
| `nf_xco2` | XCO₂ (D=1) | `train_nf.py` | `nf_full_regressor_nflows_cov_grouped` | validation NLL | `best_flow_nflows.pt` |
| `nf_profiles` | 20-level profile | `train_nf_profiles.py` | `nf_profiles_cov_grouped` | validation NLL | `best_flow_profiles.pt` |
| `nf_last4` | levels 17–20 | `train_nf_profiles_4d.py` | `nf_profiles4_cov_grouped` | validation NLL | `best_nf4.pt` |

Common to all six, and therefore **not** in the records:

* **split** — `group_split_v1`, the reference-sounding-disjoint one: 361 training scenes,
  45 validation, 45 test, and no reference sounding in more than one partition.
* **covariates** — the `scene_prior` preset (12 variables). Fixed by the covariate
  ablation rather than by the search: tuning over the input set would have made the
  ablation incoherent.
* **trial budget** — 25 trials of 20 epochs per study, five rounds of five.
* **final retrain** — the selected configuration at **40 epochs**, twice the trial count,
  because the trials ran short to rank configurations and a ranking is not a model. The
  trials keep no checkpoint.
* **warm start** — off. Every flow trains from scratch (`setup.backbone_ckpt=null`, or
  `model.pretrained_ckpt=null` for the 4-D flow). The published flows in the original
  submission warm-started from the matching deterministic backbone, which meant the flow's
  own architecture had never been searched against a fair alternative; switching it off is
  what makes the six studies six comparable searches rather than three models and three
  heads.
* **schedule** — the trainers' defaults: cosine to lr/10 for the deterministic
  regressors, constant for `train_nf.py` and `train_nf_profiles.py`, cosine to 0 for the
  4-D flow. No warm-up, no gradient clipping beyond what the configs declare.

## What each record contains

| key | |
|---|---|
| `study` | which of the six. `bo_final.sh` refuses to retrain a record whose study does not match the one it was asked for. |
| `overrides` | the configuration, as the exact strings the search produced. This is the part that reproduces the model. |
| `params` | the same values as a dict, for a reader. |
| `metric`, `value` | what it was selected on, and what it reached at 20 epochs on the validation scenes. Not a test number. |
| `round`, `trial` | where in the search it came from. |
| `baseline_value`, `baseline_overrides`, `search_beat_baseline` | the configuration behind the original submission's model, scored the same way, so "we never retuned" and "the search found nothing" are both answerable. |
| `n_trials`, `n_completed`, `n_failed` | how much of the study finished. All six completed 25 of 25. |
| `selected_utc` | when the selection ran. |

## Re-running the search

The search itself is a 160-job graph and is described in [`../docs/hpc.md`](../docs/hpc.md).
It is not needed to reproduce the paper — these records are its output.

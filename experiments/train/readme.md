# experiments/train/

Five trainers, six models. [`docs/training.md`](../../docs/training.md) has the exact
command for each, the hyperparameters, and where each checkpoint lands; this file is the
map.

| | |
|---|---|
| `train.py` | the scalar XCO₂ regressor. Hydra. |
| `train_profiles.py` | the 20-level and near-surface regressors, and the surface-only one. Hydra; `target_indices` selects the levels. |
| `train_nf.py` | the scalar conditional flow. `--exp_cfg` + `--set`. |
| `train_nf_profiles.py` | the 20-level conditional flow. |
| `train_nf_profiles_4d.py` | the four-level conditional flow. |
| `bo_nf4.py` | the standalone Bayesian-optimization driver for the 4-D flow, kept for reference; the searches reported in the paper ran as a job graph (see [`docs/hpc.md`](../../docs/hpc.md)). |
| `conf/` | the base configs the trainers compose. The experiment configs live in `code/shared_conf/experiment/`. |

## Configuration

[Hydra](https://hydra.cc/docs/intro/) for the deterministic trainers.
`code/shared_conf/` holds the shared groups (`data`, `model`, `covariates`) and
`code/shared_conf/experiment/` holds the
[experiments](https://hydra.cc/docs/patterns/configuring_experiments/) that compose them:

```bash
python3 experiments/train/train.py +experiment=full_regressor_cov_grouped
```

The flow trainers predate that and read one YAML directly, with dotted overrides:

```bash
python3 experiments/train/train_nf.py \
    --exp_cfg code/shared_conf/experiment/nf_full_regressor_nflows_cov_grouped.yaml \
    --set optim.epochs=40 optim.lr=4.0e-4
```

The difference matters: Hydra rejects an override of a key the config does not declare,
while `--set` creates it, so a typo in the second form is silent. `docs/training.md` says
more, including which key holds the epoch count in each (they differ).

## Baselines

`+experiment=fc_aux` and `+experiment=fc_no_aux` train fully-connected regressors on
flattened spectra, with and without the auxiliary variables. They are not in the paper;
they are the cheapest check that the data pipeline and the loss are doing what you think.

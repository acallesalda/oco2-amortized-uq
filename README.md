# Amortized probabilistic retrieval of atmospheric CO₂ from OCO-2 spectra

Code, configurations and reproduction instructions for:

> A. Calle-Saldarriaga, F. Jimenez, J. Grosskreuz, J. Wang, J. Hobbs and M. Katzfuss.
> *Amortized Probabilistic Retrieval of Atmospheric CO₂ from OCO-2 Spectra Using Deep
> Learning with Laplace Approximations and Normalizing Flows.*

A neural network maps an OCO-2 spectrum directly to a **probability distribution** over the
retrieved state — column-average CO₂ (XCO₂), the 20-level CO₂ profile, or the four
near-surface levels — in milliseconds, instead of running an iterative full-physics
inversion per sounding. Two ways of getting a distribution out of one network are compared
against each other and against the operational ACOS L2FP retrieval on the same simulated
soundings:

| | what it is | what it can represent |
|---|---|---|
| **Laplace approximation** | a Gaussian posterior over the last layer's weights, fitted after training, plus a calibrated aleatoric diagonal | a Gaussian predictive, cheap, no change to the trained network |
| **Conditional normalizing flow** | a rational-quadratic spline flow over the target, conditioned on the spectrum's embedding | skewed, curved and multi-modal predictives; between-level dependence |

The evaluation is deliberately hostile to the learned models: the primary test partition
holds out **whole reference soundings** (geophysical scenes the network never saw), and a
second experiment scores February-trained models on **March and July 2020**.

---

## Where everything lives

| | |
|---|---|
| **Code** (this repository) | <https://github.com/acallesalda/oco2-amortized-uq> |
| **Software archive** (citable, immutable) | Zenodo — archived at publication; the DOI is added here and to `CITATION.cff` when it is minted <!-- DOI-PENDING-SOFTWARE --> |
| **Data, splits and trained weights** | Zenodo record [20617718](https://zenodo.org/records/20617718), `10.5281/zenodo.20617718` |

The repository holds no data and no checkpoints — everything numerical comes from the
Zenodo record. See [`docs/data.md`](docs/data.md).

```
code/                the library: datasets, networks, Laplace, flows, metrics
  data/              loading and preprocessing of the simulated soundings
  modeling/          encoders, fusion, the deterministic and flow regressors
  evals/             Laplace fitting, metrics, artifact I/O, table rendering
  shared_conf/       Hydra configuration: data, model, experiment
  test/              the unit tests: no GPU, no data, ~1 min
experiments/
  pipelines/         raw HDF5 -> spectra + covariates + split tables
  train/             the six trainers
  scoring/           predict.py (GPU) -> .npz -> score.py / figures.py (laptop)
  figures/           the spectral-band figures, drawn from raw radiances
hyperparameters/     the six selected configurations behind the published models
chtc/                HTCondor submission layer, the Apptainer image, requirements.txt
docs/                how to get the data, train, evaluate, and reproduce each float
scripts/             data download, small helpers
release/             tooling that builds the public snapshot of this tree
```

---

## Three ways in, by what you have

| You have | You get | Time | Start at |
|---|---|---|---|
| a laptop | every table and figure in the paper, regenerated from the published predictive artifacts | 10-30 min | [Reproduce the paper](#reproduce-the-paper) |
| one GPU | a model trained and scored end to end on one task | 2–6 h | [`docs/training.md`](docs/training.md) |
| a cluster | all six models, the hyperparameter search, the temporal hold-out | ~1 day | [`docs/hpc.md`](docs/hpc.md) |

Nothing here needs the raw HDF5 soundings unless you want to change the preprocessing;
the processed spectra are a Zenodo download.

---

## Install

Python 3.10+, and a CUDA-capable GPU for anything that trains or samples a posterior.

```bash
git clone https://github.com/acallesalda/oco2-amortized-uq.git
cd oco2-amortized-uq
python3 -m pip install -r chtc/requirements.txt
```

`chtc/requirements.txt` is the canonical environment — the same file is baked into the
Apptainer image (`chtc/oco2-gpu.def`) that produced the published checkpoints. It pins
`torch==2.1.2+cu121`; read the reproducibility note at the top of it before changing a
version. The package is not installed: scripts put `code/` on `sys.path` themselves, so
run everything from the repository root.

The tests need neither the data nor a GPU, and they are the fastest check that the
environment is sound:

```bash
python3 -m pytest code/test -q      # ~320 tests, about a minute
```

---

## Reproduce the paper

Every table and figure is regenerated from the `.npz` **predictive artifacts** — one file
per task holding, for every held-out sounding, the truth, the ACOS retrieval, the Laplace
mean and covariance, and 512 draws from the flow. No GPU, no checkpoints, no data.

```bash
./scripts/get_data.sh artifacts            # from Zenodo, checksum-verified
./experiments/scoring/rebuild.sh           # standard scientific Python, no GPU
```

Tables land in `experiments/scoring/tables/` as both `.csv` and `.tex`; figures in
`latex/figures/`. [`docs/reproduce.md`](docs/reproduce.md) maps each float in the paper to
the command that writes it, and says which artifact it needs.

To go one step further back — from checkpoints to artifacts — you need a GPU and the
processed spectra:

```bash
./scripts/get_data.sh train test splits weights
python3 experiments/scoring/predict.py --task profiles \
    --split-csv    scratch/simulated/files/splits/group_split_v1_cov.csv \
    --stats-dir    experiments/pipelines/stats/group_split_v1 \
    --weights-root experiments/weights/group_split_v1 \
    --covariates   scene_prior \
    --laplace-mode mc \
    --out experiments/scoring/profiles_bo.npz
```

`--laplace-mode mc` is not optional. Without it the Laplace predictive carries only its
epistemic term, and its NLL and coverage columns are wrong in a way that still looks like a
result; `score.py` refuses to emit such a table without a `NOT REPORTABLE` banner. See
[`docs/evaluation.md`](docs/evaluation.md).

---

## Train

Six models, two per task, all on the reference-sounding-disjoint split. Their
hyperparameters are not typed into the configs — they are the records in
`hyperparameters/`, written by the Bayesian-optimization search that selected them, and
the trainers read them back:

```bash
# the 20-level deterministic regressor
python3 experiments/train/train_profiles.py +experiment=full_regressor_profiles_cov_grouped \
    covariates=scene_prior select_metric=ppm hyps.epochs=40 \
    $(python3 scripts/overrides.py hyperparameters/bo_det_profiles_best.json)
```

[`docs/training.md`](docs/training.md) has the same command for each of the six, the two
configuration dialects the trainers speak, the learning-rate schedule the published runs
used, and how to re-run the search itself.

---

## Data and software availability

* **Data, split tables, per-split standardization statistics and trained weights**:
  Zenodo record [20617718](https://zenodo.org/records/20617718)
  (`10.5281/zenodo.20617718`), CC-BY-4.0. The simulated OCO-2 ensemble was produced at the
  Jet Propulsion Laboratory, California Institute of Technology.
* **Software**: this repository, MIT licence, archived at Zenodo when the paper is
  published — that snapshot, not this branch, is the citable and immutable version, and
  its DOI appears here and in `CITATION.cff` as soon as it exists.
  <!-- DOI-PENDING-SOFTWARE -->

## Cite

`CITATION.cff` carries the machine-readable form. Please cite the paper, and the software
DOI if you use the code.

## Licence

MIT — see [`LICENSE`](LICENSE). The dataset is distributed separately under CC-BY-4.0.

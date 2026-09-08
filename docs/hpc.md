# Running it on a cluster

Nothing in this repository *needs* a cluster. One GPU trains any of the six models
([`training.md`](training.md)), and a laptop regenerates every table and figure
([`reproduce.md`](reproduce.md)). A cluster is what makes the full set — six models, the
160-job hyperparameter search, the temporal hold-out, the size curve — a day rather than a
fortnight.

## What is provided

`chtc/` is the **exact** submission layer used to produce the published results, on
UW-Madison's Center for High Throughput Computing (HTCondor + Apptainer). It is included
for provenance: it is what ran, with the resource requests, the dependency graphs and the
retry rules that got the models trained. [`chtc/README.md`](../chtc/README.md) is a
step-by-step guide to it, and if you have a CHTC account it should work as written.

| | |
|---|---|
| `chtc/oco2-gpu.def` | the Apptainer image: CUDA 12.3 runtime base, torch 2.1.2+cu121, `requirements.txt` |
| `chtc/requirements.txt` | the canonical Python environment, for containers and bare environments alike |
| `chtc/oco2.dag` | prep → four models in parallel → the two warm-started flows |
| `chtc/bo/bo_all.dag` | the search: five rounds of five trials per study, then the winner retrained at 40 epochs, then scoring |
| `chtc/eval.dag` | scoring: one GPU job per task, reading the finished models |
| `chtc/months_full.dag` | the temporal hold-out: retrain on all of February, score on March and July |
| `chtc/size.dag`, `chtc/size_tasks.dag` | the training-set-size curve |
| `chtc/gap.dag` | the same models scored on their own training rows |
| `chtc/attribution.dag` | where in the spectrum the retrieval reads |
| `chtc/covariates.dag` | the input ablation, one factor at a time |
| `chtc/status.py` | what failed and why, in one screen, from the HTCondor event logs |

The whole set of experiments in the paper is about **35 GPU-hours**, measured: 5.5 GPU-hours
per model at the median for the search plus the retrain, times six models, plus the scoring
jobs. `experiments/scoring/amortization.py` reports the ledger.

## Porting it to another scheduler

The submit files are HTCondor-specific; nothing else is. Four things a Slurm or PBS site
has to supply, and they are all in the same two places:

1. **A container, or an equivalent environment.** Build from `chtc/oco2-gpu.def`
   (`apptainer build oco2-gpu.sif chtc/oco2-gpu.def`), or `pip install -r
   chtc/requirements.txt` into a virtualenv. The GPU capability window matters: the cu121
   wheels carry SASS for `sm_50` through `sm_90` and will **not** run on `sm_100`/`sm_120`.
2. **The data, on node-local disk.** `scripts/get_data.sh` fetches and unpacks it. The
   43,000 `.npy` files are read one `np.load` per sample, so putting them on a shared
   network filesystem turns a training job into a filesystem benchmark.
3. **The commands.** `chtc/train.sh` holds all of them in one case statement, and
   `chtc/bo_studies.sh` holds the six-study table that `bo_trial.sh`, `bo_final.sh` and
   `bo_eval.sh` all read. Those two files are the definitive list of what runs; the submit
   files only pass a job name into them. A port that reads them rather than retyping them
   cannot drift.
4. **The dependency graph.** It is two layers deep and the `.dag` files state it:
   `prep` → `{det_xco2, det_profiles, det_last4, nf_xco2}` in parallel → `{nf_profiles,
   nf_last4}`. Scoring is an independent graph downstream of the checkpoints.

`chtc/job_common.sh` is worth reading before porting anything: it holds the unpacking, the
`/workspace/...` path rewriting that makes the hardcoded container paths work wherever a job
actually lands, and the result packing. Its `oco2_trap_pack` installs a trap before anything
that can fail, so a job that dies during setup still ships an archive — an HTCondor job whose
declared output is missing gets held, and a hold message is a much worse diagnostic than a
`.err` file.

## The one thing that is site-specific by design

`chtc/site.conf` is untracked. Copy `chtc/site.conf.example`, set your account name, and a
`git pull` can never reset it. `chtc/check_submit_macros.py` predicts HTCondor's
unused-macro warnings offline and refuses to pass on an unconfigured `site.conf`; run it
before submitting anything.

## The month deliveries

March and July 2020 are separate deliveries of raw HDF5 (~2.2 GB each) and are **not** in
the Zenodo record. `chtc/prep_month.sh` turns one into spectra, covariates and a split
table whose test partition is that whole month; `chtc/eval_month.sh` scores a
February-trained model on it. If you want to reproduce the temporal table without the raw
months, the month **artifacts** are in the `artifacts` download and
`make_temporal_table.py` reads those directly.

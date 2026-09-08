# Running the OCO-2 experiments on CHTC

Everything needed to run these experiments on UW-Madison's Center for High Throughput
Computing: one Apptainer image, the data staged from Zenodo, and a DAG that trains all
six group-split models — four of them at the same time.

The paper's models are the six trained on the reference-sounding-disjoint split. On a
two-GPU workstation that is six sequential runs. Here the dependency graph is only two
layers deep, so the wall time is roughly the slowest deterministic model plus the slowest
flow.

```
                              prep                       CPU, ~10 min
               (split table + training-only statistics)
                                |
           +---------+----------+----------+-----------+
           |         |          |          |
       det_xco2  det_profiles  det_last4  nf_xco2       4 GPUs, concurrent
                     |          |
                 nf_profiles  nf_last4                  2 GPUs, concurrent
```

Scoring is a second, independent graph (`eval.dag`), submitted once those archives exist.
Its three nodes read the finished models out of `/staging`, so nothing about how we
evaluate had to be settled before training started:

```
    eval_xco2      eval_profiles      eval_last4        3 GPUs, concurrent, ~30 min each
        |                |                 |
        +--------- one .npz each ----------+
                         |
              score.py / figures.py                     laptop, no GPU, seconds
```

---

## What is in this directory

| File | What it is |
|---|---|
| `site.conf.example` | Template for `site.conf`, which is **untracked**. Copy it, set your NetID, and a pull can never reset it. |
| `job_resources.conf` | `CPUS`, `MEM_GB`, `DISK_GB` — for the jobs that unpack the dataset. |
| `gpu_constraints.conf` | The `sm_50`–`sm_90` capability window and minimum GPU memory, for the jobs that ask for a card. Tied to the torch version in `requirements.txt`: raising one means raising the other, which is why they sit together in one file. |
| `check_submit_macros.py` | Predicts `condor_submit`'s unused-macro warnings offline, and refuses to pass on an unconfigured `site.conf`. Run before submitting. |
| `status.py` | **What failed and why, in one screen.** Reads the HTCondor event logs, failures first. `--archive` starts a clean `logs/`. |
| `sweep.sub`, `sweep.sh`, `make_sweep_grid.py`, `collect_sweep.py` | A coarse 16-point hyperparameter search on the group split, so "we never retuned" is not an open objection. |
| `audit.sub`, `audit.sh` | The Table A1 audit: two CPU-minutes that read every published checkpoint's architecture off its tensor shapes. Needs `./stage_data.sh weights` first. |
| `oco2-gpu.def` | Apptainer definition — CUDA 12.3 base, torch 2.1.2+cu121, the package set the published models were trained in. |
| `requirements.txt` | Python environment, `base_build` + `user_build` merged. Read the reproducibility note in it. |
| `build-container.sub`, `build_sif.sh` | One-time container build, as an interactive job. |
| `stage_data.sh` | Zenodo → `/staging`, with checksum verification. Run on the transfer node. |
| `make_code_tarball.sh` | Packages `code/` + `experiments/` for shipping to execute nodes. Re-run after every code change. |
| `interactive.sub` | A GPU shell in the real container. Run this once before the DAG. |
| `prep.sub`, `prep.sh` | Build the split and its statistics. Root node of the DAG. |
| `train.sub`, `train_dep.sub`, `train.sh` | The training jobs. `train.sh` holds the six commands. |
| `oco2.dag` | The graph above. |
| `job_common.sh` | Shared setup: unpacking, path rewriting, result packing. |
| `eval.sub`, `eval.sh`, `eval.dag` | Scoring. A second, independent graph — three short GPU jobs that read the finished models out of `/staging`. |
| `interactive-eval.sub` | A GPU shell with the trained checkpoints attached, for debugging the scoring code. |
| `fetch_results.sh` | Pull finished models and scoring artifacts back into the checkout. Works on the access point or the transfer node — CHTC shares `/home` between them, so it is the same checkout either way. |
| `score.sub`, `score.sh` | Tables and figures from the eval artifacts, **inside the container**. Two CPU-minutes. Use when the login node lacks numpy/matplotlib, or when you would rather not copy 250 MB of `.npz` anywhere. |
| `gauss.sub`, `gauss_submit.sh` | The moment-matched Gaussian control: the four scoring jobs re-run with `predict.py --gaussianize`, which streams thousands of flow samples per sounding, reduces them online and never stores them. Submit through the wrapper — it reads `bo_studies.sh` for the checkpoint pair, so the task and the model cannot come apart. |
| `prep_month.sub`, `prep_month_full.sub`, `prep_month.sh` | Turn one raw delivered month (March or July 2020) into spectra, covariates and a split table whose test partition is that whole month. Which February the month is appended to follows `OCO2_SPLIT`, which is the only difference between the two submit files. |
| `prep_full_feb.sub`, `prep_full_feb.sh` | Build `full_feb_v1`: `group_split_v1` with its test scenes folded back into training — 406 reference soundings instead of 361 — and its 45 validation scenes untouched, because the checkpoint is selected on them and the Laplace's aleatoric diagonal is calibrated on them. A few CPU-minutes; it opens no spectrum. Root node of `months_full.dag`. |
| `train_full.sub` | Retrain one searched winner on all of February that is not validation. Reads `bo_<study>_best.json` from the group-split search, so it is the same configuration on more data rather than a second search; the trainers themselves are untouched. |
| `eval_month.sub`, `eval_month_full.sub`, `eval_month.sh` | Score a February model on another month. No training. `n` must be ~45,000 in the log, not ~4,500. |
| `months.dag`, `months_full.dag` | The temporal hold-out. `months_full.dag` is the definitive one — it retrains on all of February first, because with March and July as the test set there is nothing left to hold February scenes back for. See `docs/hpc.md`. |
| `rescore_submit.sh` | Re-score a task on the test partition after a change to the SCORING code, on any of the three splits (`group`, `replicate`, `published`) or on a held-out month. Nothing retrains. It cleans the OSDF archive first, derives the study pair from `bo_studies.sh`, and reads `predict.py` out of the tarball that will actually be transferred — the three checks that otherwise fail silently. |
| `reprocess.sub`, `reprocess.sh` | **Optional**, not in the DAG. Rebuild the processed data to gain the lat/lon diagnostic columns. Read its header first. |

---

## Before you start

You need a CHTC account with HTC access and a `/staging` directory
(<https://chtc.cs.wisc.edu/uw-research-computing/form.html>). Personal `/staging` is
100 GB / 1000 items by default; this project uses about 8 GB, so no quota request is
needed.

Three machines are involved, and using the wrong one for bulk data is the most common
way to annoy the CHTC staff:

| Host | For |
|---|---|
| `ap2001.chtc.wisc.edu` or `ap2002.chtc.wisc.edu` — your welcome email says which | editing, `condor_submit`, `condor_q` |
| `transfer.chtc.wisc.edu` | anything that moves gigabytes — `stage_data.sh`, `fetch_results.sh` |
| execute nodes | your jobs. **No outbound network** — everything must arrive via `transfer_input_files`. |

---

## Step 1 — Clone and configure

On the access point:

```bash
git clone https://github.com/acallesalda/oco2-amortized-uq.git ~/OCO-2
cd ~/OCO-2/chtc
mkdir -p logs
```

Create `site.conf` from the template and set your NetID and its first letter:

```bash
cp site.conf.example site.conf
$EDITOR site.conf
```

```
NETID        = bucky
NETID_LETTER = b
```

That is the only edit. Everything else derives from it.

**`site.conf` is not tracked by git**, so a pull can never reset it. It used to be
committed with `NETID = CHANGEME`, which meant every pull that touched the file silently
reverted your NetID and the next job died a minute later with an OSDF `Permission denied`
naming `/staging/c/CHANGEME/`. Check it before submitting anything:

```bash
python3 chtc/check_submit_macros.py
```

That refuses to pass while the NetID is unset, and also catches a `NETID_LETTER` that does
not match the first letter of `NETID`.

## Step 2 — Stage the data from Zenodo

On the **transfer** node, because this moves 3.3 GB:

```bash
ssh <netid>@transfer.chtc.wisc.edu
cd ~/OCO-2/chtc
./stage_data.sh
```

Pulls all six files of Zenodo record
[20617718](https://zenodo.org/records/20617718) into `/staging/<l>/<netid>/oco2/` and
verifies every one against the published SHA256. Interrupted downloads resume; files that
are already correct are skipped, so re-running is cheap.

If you only want what the training jobs read (780 MB):

```bash
./stage_data.sh train test stats
```

`./stage_data.sh --list` shows what is there and what is not.

## Step 3 — Build the container

Back on the access point. This takes 15–30 minutes, nearly all of it pip pulling torch
and its vendored CUDA wheels:

```bash
cd ~/OCO-2/chtc
condor_submit -i build-container.sub     # waits for a build slot, then gives you a shell
bash build_sif.sh
exit
```

> `bash build_sif.sh`, not `./build_sif.sh`. The scripts are marked executable in the
> repository, but HTCondor does not always preserve the mode on `transfer_input_files`
> (it does for the file named by `executable`, which is why the training jobs are fine).
> If `./build_sif.sh` gives `Permission denied`, that is all it is — `chmod +x` first, or
> just use `bash`.

`build_sif.sh` builds the image, imports torch / xarray-DataTree / Laplace / nflows to
prove the environment resolved, copies the `.sif` to `/staging`, and tells you what it
did. Confirm it landed:

```bash
ls -lh /staging/$(whoami | cut -c1)/$(whoami)/oco2/oco2-gpu.sif
```

> **On exact versions.** The container is built from *constraints*, not a lock file, so
> unpinned packages may resolve to newer versions than the ones that trained the published
> checkpoints. If you have access to an environment that still runs them, `python3 -m pip
> freeze` inside it and commit the output as `chtc/requirements.lock.txt` — `oco2-gpu.def`
> prefers it automatically. Without it the drift is small and touches nothing numerical,
> but it is real and worth stating rather than assuming away.

## Step 4 — Package the code

Every time you change code or a config:

```bash
cd ~/OCO-2
./chtc/make_code_tarball.sh
```

Produces `chtc/oco2_code.tar.gz` (~15 MB) and records the git commit inside it, so a
result archive can always be traced to the source that produced it. The 235 MB of
published `.pt` checkpoints are excluded — no training job reads them.

## Step 5 — Smoke test (ten minutes, saves a day)

```bash
cd ~/OCO-2/chtc
condor_submit -i interactive.sub
```

Once you have a shell on a GPU node:

```bash
pwd                          # expect /workspace/project
nvidia-smi
source ./job_common.sh
oco2_gpu_check               # fails loudly if the card is one cu121 cannot drive
oco2_setup
oco2_unpack_data
ls scratch/simulated/files/train | head
exit
```

The one thing to actually look at is `pwd`. As of 2026-08-14 it prints
**`/workspace/project/scratch`** — one level below `container_target_dir`, because
HTCondor maps the sandbox root to the target and then works inside its `scratch`
subdirectory. That means the project's hardcoded `/workspace/project/...` paths do **not**
resolve as written, and `oco2_setup`'s rewrite is load-bearing rather than a safety net.
Nothing depends on which of the two you get — `PROJECT` is whatever `$PWD` turns out to
be — but if it ever prints a third thing, that is worth knowing before nine jobs rely on
it.

Two things this shell **cannot** do, both of which look like bugs when you try them:

- **`pytest` is not installed.** It was missing from `requirements.txt` when the image was
  built. It is there now, so the next rebuild will have it; until then run the evaluation
  tests wherever you have numpy, scipy, pandas and pytest — they import no torch, which is
  the whole reason they were written that way. The container-side check that genuinely
  needs the container is `predict.py --limit 2`, in `interactive-eval.sub`.
- **There are no `.pt` files.** `make_code_tarball.sh` excludes every checkpoint, so
  `experiments/weights/*/` holds only the `.yaml` and `.json` beside them. Anything that
  needs the published checkpoints — the Table A1 audit, scoring the replicate split —
  has to unpack `oco2_model_weights.tar.gz` from `/staging` first. That is what
  `audit.sub` does, and what `oco2_unpack_weights` in `job_common.sh` is for.

## Step 6 — Run everything

```bash
cd ~/OCO-2/chtc
condor_submit_dag oco2.dag
```

Watch it:

```bash
python3 chtc/status.py         # what failed and why, failures first -- start here
condor_q                       # your jobs
condor_q -dag -nobatch         # THE useful one: a line per node, named, with its state
condor_watch_q                 # the same thing, live-updating
cat oco2.dag.status            # node states, rewritten every 30 s
tail -f oco2.dag.dagman.out    # DAGMan's own reasoning about what to submit next
tail -f logs/det_xco2.err      # one node -- logs/<node>.{out,err,log}
```

`-nobatch` is what turns the default one-row summary back into a line per job. With
`-dag` it indents each node under its DAGMan job and labels it, which is the view you
want while six things are running:

```
 ID        OWNER          SUBMITTED      RUN_TIME ST PRI SIZE CMD
6036100.0  callesaldarr   8/14 13:02   0+00:00:31 R  0   0.0  condor_dagman
6036101.0   |-det_xco2    8/14 13:03   0+00:00:12 R  0   0.0  train.sh det_xco2
6036102.0   |-det_profiles 8/14 13:03  0+00:00:00 I  0   0.0  train.sh det_profiles
```

`ST` is the column to read: `I` idle, `R` running, `H` held, `X` removed. Node names are
`prep`, `det_xco2`, `det_profiles`, `det_last4`, `nf_xco2`, `nf_profiles`, `nf_last4` —
and `eval_xco2` / `eval_profiles` / `eval_last4` for `eval.dag`, whose status file is
`eval.dag.status`.

```bash
condor_q -nobatch -held        # held jobs with their hold reason
condor_rm <dagman_cluster>     # stop everything; removing DAGMan removes its children
```

`condor_q -better-analyze -job <id>` explains why something is idle — usually the GPU
capability bounds or waiting for a GPU Lab slot.

If a node fails, fix the cause and re-submit the same command. DAGMan wrote
`oco2.dag.rescue001` and will skip everything that already succeeded.

## Step 7 — Collect the results

On the transfer node:

```bash
ssh <netid>@transfer.chtc.wisc.edu
cd ~/OCO-2
./chtc/fetch_results.sh --status      # what has finished
./chtc/fetch_results.sh               # unpack all of it
```

Everything lands under `experiments/weights/group_split_v1/`. The published checkpoints
in `experiments/weights/` are never touched, so the old and new numbers stay side by
side, which is what the "within-reference interpolation" secondary result needs.

Each archive also carries `run_meta_<job>.json` with the real exit status, the host, the
GPU it ran on and the elapsed time. `fetch_results.sh` warns if a run exited nonzero and
you are about to unpack partial checkpoints.

## Step 8 — Score

Scoring is a **second DAG**, not a tail on the first, because its inputs are the result
archives in `/staging` rather than a DAGMan parent. That means training does not wait on
it, a re-score costs one short GPU job rather than a retrain, and a training node that has
to be repeated does not drag scoring along with it.

```bash
cd ~/OCO-2/chtc
./fetch_results.sh --status      # the six training archives must be there
condor_submit_dag eval.dag
```

Three nodes, one per table, running concurrently:

| Node | Reads | Produces |
|---|---|---|
| `eval_xco2` | `det_xco2`, `nf_xco2` | Table 1 |
| `eval_profiles` | `det_profiles`, `nf_profiles` | Table 2, Figures 4–5, 7 |
| `eval_last4` | `det_last4`, `nf_last4` | Table 3, Figure 6 |

Each is tens of minutes: a last-layer Laplace refit on a few thousand training soundings,
then 512 flow draws for every test sounding.

**What comes back is not a table.** Each job writes one `.npz` holding every predictive
quantity for every test sounding — the truth, the Laplace mean and covariance, the flow's
samples, its exact log-density at the truth, its latent `z = f_x(y)`, the ACOS mean and
covariance, and the per-sounding identifiers and diagnostics. The tables and figures are
made from that file afterwards, on your laptop:

```bash
ssh <netid>@transfer.chtc.wisc.edu
cd ~/OCO-2 && ./chtc/fetch_results.sh --eval

python3 experiments/scoring/score.py   experiments/scoring/results/group_split_v1/*.npz --out tables/
python3 experiments/scoring/figures.py experiments/scoring/results/group_split_v1/*.npz --out figures/
```

Neither needs a GPU, a checkpoint, or the data. That is the point: the half that costs
queue time runs once, and the half that gets re-run every time a caption or a font size
changes runs offline in seconds. It is also the archival record — the `.npz` is what a
reader would need to check any number in Tables 1–3 without retraining anything.

### Evaluating interactively

You can, and for *debugging* you should:

```bash
condor_submit -i interactive-eval.sub \
    -a 'det_tar = det_profiles.tar.gz' -a 'flow_tar = nf_profiles.tar.gz'
```

The header of that file has the sequence to run, ending in a two-batch `predict.py`
smoke test that takes about a minute and catches the things that actually go wrong: a
checkpoint path that does not exist, an architecture that disagrees with its state dict,
a missing ACOS covariance, a shape error.

What an interactive session is *not* good for is producing the numbers. It is capped at a
few hours, nothing is transferred back from it (`transfer_output_files = ""`), and a
result that exists only in the scrollback of an SSH session cannot be traced to a commit
or handed to a reader. Debug there, submit `eval.dag` for real.

---

## Covariates and the hyperparameter search

Two additions that run *after* the six jobs, and share their machinery. Both need
`oco2_split_v1.tar.gz` (from `prep.sub`) already on /staging.

### One-off: recover the covariates

```bash
./stage_data.sh raw              # once, on the transfer node: the 2.2 GB of .h5
condor_submit covariates.sub     # ~20 min, CPU -> oco2_covariates_v1.tar.gz
```

`covariates.sh` reads the raw soundings and writes a per-sounding covariate table plus an
augmented split table (`group_split_v1_cov.csv`) and the training-only statistics the
loader encodes with. It does not touch the `.npy` spectra and it does not change the
split: the same 383/48/48 reference soundings stay in the same partitions, so a covariate
run and the baseline differ in exactly one thing.

The two quantities the manuscript claims as inputs are recovered rather than read:

| covariate | how | check |
|---|---|---|
| viewing zenith angle | `airmass = sec(SZA) + sec(VZA)`, and `airmass` is constant across the 100 replicates of a scene | `sec(VZA) >= 1` must hold for every sounding; the extraction warns and writes NaN where it does not |
| prior surface pressure | `retrieved_psfc - 100 * delta_psfc` | constant across a scene to ~1e-8 relative, which is what identifies it as the met prior |

The second check is worth more than it looks: it is the only *independent* test that the
cluster metadata is paired with the right `.h5` files. `--verify-against` compares the new
table with the published `aux_data.csv`, but both come from the same pairing.

### Step A — the covariate comparison, no search

```bash
condor_submit_dag covariates.dag     # 10 GPU jobs, four at a time, 2-3 hours
./fetch_results.sh --cov
```

One factor at a time; the suffix table is in `train.sh` and in `covariates.dag`.
`det_*_base2` is the control and must reproduce the corresponding `det_*` number -- if it
does not, the merge moved it and nothing after that is interpretable.

### Step B — search, retrain, test set: one DAG

```bash
./make_code_tarball.sh
mkdir -p logs
NETID=$(whoami); mkdir -p "/staging/${NETID:0:1}/${NETID}/oco2/results/bo25"
cd bo && condor_submit_dag bo_all.dag
```

76 nodes, one submit. For each of the six models: 25 Bayesian-optimization trials as five
waves of five at 20 epochs, a one-minute CPU node that picks the winner out of the
records, and a retrain of that winner at 40 epochs. Then four scoring jobs, each waiting
on both models of its task, which are the only jobs in the project that read the test
partition.

Between waves a one-minute CPU job replays the finished trials into a fresh Optuna study
and asks TPE for the next five (`bo_propose.py`; `constant_liar=True`, or a batch of five
comes back as five copies of the same point). Round 0 trial 0 is always the published
configuration, so every study contains its own baseline and "the search helped" is a
comparison inside one study.

**All six models are searched from scratch.** The three flows have their warm start forced
off (`setup.backbone_ckpt=null`, `model.pretrained_ckpt=null`), which is what frees their
backbone width, heads and depth to be searched alongside the flow -- ten dimensions rather
than seven. A flow warm-started from a deterministic checkpoint cannot search its own
backbone at all: the state dict has to fit.

```bash
./fetch_results.sh --status    # trials finished per study, winners picked
./fetch_results.sh --bo        # records, winners, leaderboards -> chtc/bo/results/
./fetch_results.sh --best      # the six retrained models and the four .npz artifacts
```

`chtc/bo/results/bo_<study>_leaderboard.md` is written by the select node itself: 25 rows
ranked, the baseline marked, failures listed separately.

Trials keep no checkpoint -- 150 twenty-epoch checkpoints nobody will score is four GB of
transfer for nothing -- so `bo_final.sh` retraining the winner is the first time the
searched configuration exists as a model.

One table decides what each study runs, `chtc/bo_studies.sh`: the trainer, its config, the
epoch key, the weights directory, the switch that keeps the backbone untrained, and where
the checkpoint lands. `bo_trial.sh`, `bo_final.sh` and `bo_eval.sh` all read it, so a
trial and its own retrained winner cannot be different models.

To restart: `./fetch_results.sh --clean bo`, or regenerate with a fresh `--run` name.
OSDF refuses to overwrite an object, so a half-run study cannot simply be resubmitted.

---

## The six jobs

Defined in `train.sh`; hyperparameters in the `*_grouped.yaml` configs, all with Bayesian
optimization switched off so that the split is the only thing that differs from the
published runs.

| Node | Script | Config | Epochs | Warm start |
|---|---|---|---|---|
| `det_xco2` | `train.py` | `full_regressor_grouped` | 25 | — |
| `det_profiles` | `train_profiles.py` | `full_regressor_profiles_grouped` | 40 | — |
| `det_last4` | `train_profiles.py` | `full_regressor_profiles_last4_grouped` | 40 | — |
| `nf_xco2` | `train_nf.py` | `nf_full_regressor_nflows_grouped` | 50 | — (`backbone_ckpt: null`) |
| `nf_profiles` | `train_nf_profiles.py` | `nf_profiles_grouped` | 35 | `det_profiles` |
| `nf_last4` | `train_nf_profiles_4d.py` | `nf_profiles4_grouped` | 50 | `det_last4` |

> **Table A1 audit: done, and it found something.** `experiments/checkpoint_configs.json`
> already holds the answers, so the `!! CHECK BEFORE RUNNING !!` banners are resolved.
> Near-surface: Table A1 was right and the old YAMLs were stale — banner cleared, four
> settings corrected to the checkpoint, epochs back to 50 (which is why this job no longer
> asks for the `long` GPU class).
> **20-level: the config was wrong and had no banner.** `nf_profiles.yaml` runs Bayesian
> optimization over 100 trials, so its `nflow` block is the search *default*, not the
> selected model; the published checkpoint is trial 99 and uses 9 layers / 384 hidden /
> 10 bins / tail 8.391, against the 7 / 640 / 13 / 10.0 that had been copied across.
> Corrected. Only `nf_xco2` is still open —.

The two flows that warm-start will **refuse to run** if the backbone checkpoint is
missing, rather than silently training from scratch the way `train_nf_profiles_4d.py`
does on its own. A cold-started flow finishes and reports a plausible number that is not
the experiment we claimed to run; better to lose the job.

To run one model outside the DAG:

```bash
condor_submit train.sub -a 'job_name = det_xco2' -a 'gpu_length = short'
```

---

## How a job is put together

Worth knowing when something breaks.

**Paths.** The project hardcodes `/workspace/project`, `/workspace/data` and
`/workspace/outputs` — the development container's layout — in the configs, in the `--exp_cfg` YAMLs
the flow trainers read, and in `sys.path.append` lines at the top of the training
scripts. Two independent mechanisms handle this:

1. `container_target_dir = /workspace/project` in every submit file maps the HTCondor
   scratch directory to that path inside the container, so the hardcoded paths are simply
   correct. The image pre-creates `/workspace/project` and `/workspace/data`, because
   unprivileged Apptainer cannot always `mkdir` into a read-only image.
2. `oco2_setup` in `job_common.sh` rewrites every absolute `/workspace/...` in the
   extracted copy to wherever the job actually landed. This runs unconditionally — even
   in the happy path it is what redirects `conf/config_profiles.yaml`'s default
   `hydra.run.dir: /workspace/outputs/...` somewhere writable.

So an HTCondor too old to know `container_target_dir` costs nothing: it treats the line
as an unused macro, and mechanism 2 picks up the slack.

**Data.** The Zenodo archives are rooted at the development-container layout
(`data/oco2/simulated/files/train/...`), which is why `oco2_unpack_data` strips exactly
four path components. The 43k `.npy` files are untarred onto the execute node's local
disk, not read over a network filesystem — `SpectralDataset.__getitem__` does one
`np.load` per sample, so this is the difference between a training job and a filesystem
benchmark.

**Results.** `oco2_pack_results` runs from an `EXIT` trap, so a crashed run still ships
its partial checkpoints, its hydra run directory and its mlflow store. Those are usually
the only evidence available afterwards.

---

## Troubleshooting

**`bad interpreter: /bin/bash^M`** — the scripts were checked out with CRLF. `.gitattributes`
in this directory forces LF; if you cloned before it existed:
`git rm --cached -r . && git reset --hard`.

**`gpus_minimum_memory=16000 defaults to megabytes, but should contain a units suffix`.**
Fixed: it is `16000M` in `gpu_constraints.conf` now. Written as `16000M` and not `16G`
because HTCondor's `G` is binary — `16G` would ask for 16384 MB and match a slightly
narrower pool than the published runs did.

**Job sits idle forever.** `condor_q -better-analyze -job <id>`. Usually the GPU
capability window (`GPU_MIN_CAP` / `GPU_MAX_CAP` in `gpu_constraints.conf`) or a
`+GPUJobLength` class with no free slots. Dropping a job from `medium` to `short` often
finds a slot immediately.

**`no usable GPU` in the `.err`.** `oco2_gpu_check` refused to fall through to CPU. Either
the slot came without a GPU, or the card is newer than the cu121 wheels support
(sm_100+). Raising the torch version is a bigger change than it looks — it moves the
numerics — so prefer keeping the capability ceiling.

**The container build dies in `%post` with `exit status 100`**, after
`Couldn't create temporary file /tmp/apt.conf.XXXXXX for passing config to apt-key` and
`The repository '...' is not signed`. The signatures are fine — nothing could read them.
Apptainer bind-mounts the host `/tmp` into an unprivileged build, apt drops privileges to
the `_apt` user (uid 100), that uid is not mapped in the user namespace, and its first
write to `/tmp` fails. `oco2-gpu.def` now sets `APT::Sandbox::User "root"` before touching
apt, and `build_sif.sh` binds a mode-1777 scratch directory onto `/tmp`. **Do not** work
around a repeat of this with `AllowInsecureRepositories` or `--allow-unauthenticated` —
that turns "could not check the signatures" into "did not check the signatures", in the
image every number in the paper comes out of.

**`WARNING: the line 'NETID = ...' was unused by condor_submit. Is it a typo?`** Not a
typo. `condor_submit` warns about every macro a submit file defines and never references,
and an `include`d file counts as part of the submit file — so sharing one `site.conf`
across eight jobs bought a column of warnings on every job that did not use all of it.

The config is now split so that each job includes only what it uses:

| File | Holds | Included by |
|---|---|---|
| `site.conf` | NetID, `OSDF`, `SIF`, `STAGING_REQ` | every job that touches `/staging` |
| `job_resources.conf` | `CPUS`, `MEM_GB`, `DISK_GB` | the jobs that unpack the dataset: `prep`, `interactive`, `train`, `train_dep`, `eval`, `interactive-eval` |
| `gpu_constraints.conf` | `GPU_MIN_CAP`, `GPU_MAX_CAP`, `GPU_MIN_MEM` | the jobs that ask for a card — the five above, minus `prep` |
| — | — | `audit` and `reprocess` take `site.conf` only (8 GB and 100 GB, self-sized); `build-container` takes none |

`STAGING` was deleted outright: no submit file ever referenced it, and the shell scripts
that need the path derive it from `whoami`.

Check before submitting, from the access point or anywhere else:

```bash
python3 chtc/check_submit_macros.py
```

**All nine files should report `clean`, with no exceptions.** There were briefly two —
`audit.sub` and `reprocess.sub`, which size their own disk and so could not reference
`$(DISK_GB)` while it lived in `site.conf`. Tolerating them was the wrong call: every
submit then printed a warning, which is the exact state this is meant to prevent. Moving
`DISK_GB` into `job_resources.conf`, which those two do not include, removed the dilemma
rather than excusing it. If you ever want to add an exception, that is usually the same
signal — a macro is living somewhere more general than its users. Move it instead.

> `warn_on_unused_submit_file_macros` is a *configuration* knob, not a submit command.
> Putting it in a submit file earns a warning about itself.

**`oco2_split_v1.tar.gz was not transferred`.** The `prep` node has not finished, or its
output never reached `/staging/<l>/<netid>/oco2/`. Check `logs/prep.err`.

**Job goes to `held` with `Authorization Error ... Permission denied: token appears valid
but was rejected by the server`.** Read the `URL file =` at the end of the message. If the
path contains `CHANGEME`, or any NetID that is not yours, `site.conf` is not configured —
the token is fine, it just does not grant access to somebody else's `/staging`. Not a
credentials problem. `python3 chtc/check_submit_macros.py` catches this before you submit.
Fix `site.conf`, then `condor_rm <cluster>` and resubmit; releasing the held job would
re-run it with the same bad path.

**`Pelican Client Error: remote object already exists, upload aborted`** on *output*
transfer. OSDF will not overwrite. Since `when_to_transfer_output = ON_EXIT` uploads the
archive even when the job fails, any second run of a job that has already run once hits
this. Remove the old archive first:

```bash
./fetch_results.sh --clean det_profiles     # or: prep, audit, eval_profiles, ...
condor_submit train.sub -a 'job_name = det_profiles' -a 'gpu_length = medium'
```

This is why `max_retries = 0` in every submit file that writes to `/staging`. An automatic
retry could never succeed, and would replace the real failure with a confusing transfer
error. A DAGMan rescue is the same story — clean the failed node's archive, then resubmit
the DAG.

**Transfer errors mentioning `osdf://`.** Confirm `requirements = (HasCHTCStaging == true)`
survived in the submit file and that the file really is in `/staging` with the exact name.
`./stage_data.sh --list` will tell you.

**Out of disk.** `request_disk` in `site.conf` (`DISK_GB`). The container, two tarballs,
43k extracted files and the checkpoints all live in scratch together.

**`WARNING: /workspace paths the rewrite did not reach`.** A hardcoded `/workspace` path
in a file type `oco2_setup` does not rewrite — it handles `*.yaml`, `*.yml` and `*.py`,
while the check scans every text file. Such a path resolves against the read-only image
and fails at runtime. The warning prints the offenders.

**A job exits 141 having printed only the banner and the rewrite line.** Fixed on
2026-08-14; if you see it, your `oco2_code.tar.gz` predates the fix. Two bugs compounded.
The leftover check grepped for `/workspace/` without excluding `$PROJECT` — which *is*
`/workspace/project/scratch` when `container_target_dir` works, so every correctly
rewritten path matched and the warning fired on all of them. Then the `grep … | head -5`
that printed them returned 141 (SIGPIPE, once the output exceeds the pipe buffer) under
`set -o pipefail`, as the last command of `oco2_setup` — so `set -e` aborted the job
before it did any work. It would have taken down every node of both DAGs.

---

## Cost and housekeeping

`/staging` is not backed up and CHTC asks that data be removed once it is no longer
needed. After the revision is submitted:

```bash
rm -rf /staging/<l>/<netid>/oco2
```

Zenodo is the archive; nothing there is unique. Keep `oco2-gpu.sif` if you expect to
come back to this — rebuilding costs half an hour and, without a lock file, will not
resolve to exactly the same package versions.

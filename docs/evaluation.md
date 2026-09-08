# Evaluation

Scoring is two halves that share nothing operationally, and keeping them apart is what
makes the tables cheap to regenerate.

```
checkpoints + spectra                      a .npz artifact                tables, figures
        |                                        |                              |
   predict.py   ------------------------->   score.py                       figures.py
   GPU, ~30 min per task                    joint_diagnostics.py          decompose_error.py
                                            delta_blocks.py               size_curve.py
                                            make_temporal_table.py        ...
                                            seconds of numpy, on a laptop
```

`predict.py` needs a GPU, the training data, the checkpoints and tens of minutes. Everything
after it is numpy on a laptop, and gets re-run every time a caption changes. Merging the
two — as the original submission's notebooks did — means paying for the first every time
you want the second.

## The artifact

One `.npz` per task, holding for **every** held-out sounding:

| | |
|---|---|
| `y` | the simulated truth |
| ACOS | the operational retrieval's point estimate and its reported covariance |
| Laplace | predictive mean and covariance, epistemic and total |
| flow | `n_nf_samples` draws (512 in the published runs) |
| masks | which soundings are physical, which methods are valid on each |

Beside it, a `.meta.json` recording what produced it: the checkpoint paths and their
architectures read off the tensor shapes, the Laplace settings, the split table, the
standardization constants, the number of draws, the device, and the git commit. That file
is the only thing tying a number in the paper to a specific set of weights, and it is why
the `*.meta.json` files are tracked in this repository even though the `.npz` are not.

`code/evals/artifacts.py` defines the format; `Artifact.load` is what every scorer uses.

## Producing one

```bash
python3 experiments/scoring/predict.py --task profiles \
    --split-csv    scratch/simulated/files/splits/group_split_v1_cov.csv \
    --stats-dir    experiments/pipelines/stats/group_split_v1 \
    --weights-root experiments/weights/group_split_v1 \
    --covariates   scene_prior \
    --laplace-mode mc \
    --out experiments/scoring/profiles_bo.npz
```

Tasks: `xco2` (D=1), `profiles` (20 levels), `last4` (levels 17–20, from models trained
on those four natively), `last1` (level 20), and the two marginal ones — `last4_marginal`
and `last1_marginal`, the 20-level models evaluated on the near-surface levels. A marginal
task is a different quantity from its native counterpart and loads different checkpoints;
the artifact's metadata records which it was.

Flags that change what the numbers mean:

| flag | |
|---|---|
| `--laplace-mode mc` | **Use it.** The Laplace predictive then carries its observation-noise term as well as the epistemic one. Without it the XCO₂ predictive is ~27× too narrow in variance, and its NLL and coverage still look exactly like results. |
| `--laplace-hessian` | `diag`, `full` or `kron`. The published runs use `full` for XCO₂ (321 last-layer parameters) and `diag` for the two profile tasks. |
| `--nf-samples` | draws per sounding from the flow. 512 in the published artifacts; the bivariate figure needs 20,000 and gets its own artifact. |
| `--eval-partition` | `test` by default; `train` produces the in-sample side of the generalization gap, and the artifact records which it was. |
| `--gaussianize S` | replace the flow's predictive with the Gaussian matching its own mean and covariance, estimated from `S` draws. The moment-matched control: the two predictives then agree in their first two moments exactly, so any remaining gap is posterior *shape*. |
| `--ids`, `--limit` | a handful of soundings, for a smoke test. The artifact is marked partial. |

Omit `--split-csv` to score on the original replicate-level split.

## Reading one

```bash
python3 experiments/scoring/score.py \
    experiments/scoring/xco2_bo.npz \
    experiments/scoring/profiles_bo.npz \
    experiments/scoring/last4_bo.npz \
    --label group_split_v1_cov='reference-disjoint' \
    --paired --out experiments/scoring/tables
```

writes, per task, `table_<task>.csv` and `table_<task>.tex`, plus the paired method
differences. Columns:

| | |
|---|---|
| RMSE, MAE | in ppm, at **each method's own optimal point estimate** — the mean for the two Gaussians, the predictive median for the flow, since those minimize the two losses respectively |
| NLL | in standardized coordinates **and** in ppm per level. The change of variables is the constant Jacobian `sum(log(sigma_d * 1e6))`, so the two are both reported and the D=1, D=4 and D=20 columns can be compared with each other only in the ppm one |
| coverage | at nominal 95 %, **joint and marginal**. The joint figure is the χ² coverage of the whole D-dimensional predictive; the marginal one is the per-level rate averaged over levels. A predictive can be near-nominal marginally and badly wrong jointly |
| standard errors | bootstrap, resampling **whole reference soundings** by default (`--bootstrap sounding` for the other unit). Soundings within a scene are strongly correlated, so resampling soundings understates the uncertainty |

`joint_diagnostics.py` takes the same arguments and goes further into the joint/marginal
question: mean ‖z‖²/D and the χ²-PIT KS statistic, both invariant to the target's
coordinate system, and the off-diagonal of `corr(z)`, which is the between-level dependence
the predictive covariance failed to carry.

### The not-reportable guard

`score.py`, `joint_diagnostics.py` and `make_temporal_table.py` compare each of our own
predictives against the residuals it is placed against, and if the ratio is outside
[0.25, 4] they write a `NOT REPORTABLE` banner into the LaTeX naming the row.

It exists because a defect of exactly this kind survived a whole project undetected: the
scalar Laplace predictive carried no observation-noise term, so it was 27.6× too narrow,
and every column beside it — an NLL, a coverage percentage, a PIT histogram — looked like a
result. Nothing in the pipeline objected, because nothing compared a predictive to the
errors it was placed against.

ACOS is deliberately exempt. Its ratio is large too, and there it is the paper's central
measurement rather than a bug: an optimal-estimation posterior that carries no
forward-model-error term. `score.py` prints that one as a finding.

If the banner appears, re-score with `--laplace-mode mc`. `figures.py` runs the identical
test — imported from `code/evals/metrics.py` rather than reimplemented, so the figure and
the table cannot disagree about which rows are bad — and `--drop <method>` draws an interim
figure without one.

## Everything at once

```bash
./experiments/scoring/rebuild.sh
```

Regenerates every table and figure from whatever artifacts are on disk, skipping with a
message anything whose artifact is missing. It exists because the regeneration is a dozen
commands with long argument lists, two of which are easy to get subtly wrong in ways that
are silent: `score.py` given only the reference-disjoint artifacts quietly writes a
one-partition version of the tables over the two-partition ones, and `figures.py` names its
output by task plus split, so pointing two splits at one `--out` makes the second overwrite
the first. Both are handled there.

[`reproduce.md`](reproduce.md) maps each float in the paper to its command.

## Timing

`experiments/scoring/benchmark.py` measures test-time cost per sounding, per task and per
method, and records the machine it measured on: GPU model, driver, torch and CUDA versions,
batch size, precision, warm-up passes, repetitions, and whether the DataLoader was inside
the timed region (it reports both scopes, because "is preprocessing included" has no single
right answer).

It never writes an ACOS row. The operational figure quoted in the paper comes from the
group that produced the simulation, on their own hardware, and it is a different model of
computation — an iterative full-physics inversion against one amortized forward pass. It
belongs in a caption as an order of magnitude, not in a table as a row.

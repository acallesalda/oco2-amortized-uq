# The data

## What it is

A high-fidelity **simulation experiment** for OCO-2, produced at the Jet Propulsion
Laboratory. Its unit is a *reference sounding* — a geophysical scene: one true atmospheric
state, one surface, one aerosol configuration, one viewing geometry. Each reference
sounding is simulated **100 times** with independent instrument noise and a sampled
forward-model discrepancy, and every simulated spectrum is then put through the operational
ACOS L2FP retrieval.

So each row of the dataset carries, for one sounding:

| | |
|---|---|
| **inputs** | three spectral bands (O₂ A, weak CO₂, strong CO₂) plus scene metadata |
| **truth** | the simulated state: XCO₂ and the 20-level CO₂ profile |
| **the comparator** | ACOS L2FP's retrieved state, its reported XCO₂ variance, and its 20×20 retrieved profile covariance |

That last column is what makes this dataset unusual and what the paper is built on: for
every sounding there is a *truth*, a *learned predictive distribution* and an *operational
retrieval's own posterior*, all on the same quantity.

**February 2020** is the main delivery: 451 reference soundings, 44,982 soundings after
screening. **March 2020** and **July 2020** are two further deliveries from the same
simulation ensemble, used only as held-out months.

## Getting it

```bash
./scripts/get_data.sh --list                  # what the record holds, and what you have
./scripts/get_data.sh artifacts               # tables and figures, no GPU
./scripts/get_data.sh train test splits       # enough to train or to re-score
./scripts/get_data.sh weights                 # the published checkpoints
./scripts/get_data.sh months                  # March and July, the two held-out months
./scripts/get_data.sh raw                     # the raw .h5, only to redo preprocessing (2.2 GB)
```

Everything comes from Zenodo
[`10.5281/zenodo.20617717`](https://doi.org/10.5281/zenodo.20617717), verified against the
`SHA256SUMS.txt` published inside the record itself, and unpacked to the paths the code
already defaults to:

```
scratch/simulated/files/train/           aux_data.csv + one .npy per sounding
scratch/simulated/files/test/            the same, plus <id>_cov.npy ACOS covariances
scratch/simulated/files/202003/          March, the same layout as test/       [months]
scratch/simulated/files/202007/          July, likewise                        [months]
scratch/simulated/files/splits/          the split tables
scratch/simulated/og_data/               raw .h5 soundings                    [raw only]
experiments/pipelines/stats/<split>/     per-split standardization constants
experiments/weights/<split>/             checkpoints
experiments/scoring/                     predictive artifacts (.npz + .meta.json)
group_split_v1_cov.csv                   the primary split table, at the root
```

All of those are gitignored. `downloads/` keeps the tarballs so a re-run is free; delete it
when you are done.

That DOI is the record's **concept** DOI, which resolves to whichever version is newest.
`get_data.sh` hardcodes no version: it asks Zenodo which version is current and reads that
version's file list, so an archive added in a later version is fetchable without changing
the script. To reproduce against one specific version instead — the one a paper cited,
rather than the newest — pin it:

```bash
OCO2_ZENODO_RECORD=20617718 ./scripts/get_data.sh --list
```

### The held-out months

`months` brings down March and July 2020 as the two tarballs the scoring jobs themselves
consumed, rather than a repacking of them, so the bytes scored in the paper and the bytes
in the record are the same bytes. Each carries that month's spectra, its ACOS profile
covariances, its covariate table and its month split table — everything
`experiments/scoring/predict.py` needs to re-score a checkpoint on an unseen month. The
February archives (`train`, `test`) do not contain them.

Regenerating the paper's temporal-generalization table needs none of this: the per-sounding
predictions are already in `artifacts`, and `rebuild.sh` reads those. The months are for
re-running the network, not for re-making the table.

## The split tables

A **split table** is one CSV with one row per sounding, and it is the single source of
truth for an experiment: which partition a sounding is in, which reference sounding and
cluster it belongs to, its auxiliary variables, its truth, and ACOS's answer. Change the
experiment by changing the table, never by changing the loader.

| table | partitioning | what it is for |
|---|---|---|
| `group_split_v1_cov.csv` | **by reference sounding**: 361 train / 45 validation / 45 test scenes, 36,003 / 4,490 / 4,489 soundings. No scene appears in two partitions. | The paper's **primary** result. A test sounding's scene was never seen in any form. |
| `replicate_split_v1_cov.csv` | by replicate, as the original submission did: each of a scene's 100 replicates is assigned independently, so a test sounding's scene is in training under different noise. | The **secondary**, within-reference "interpolation" result. Identical code path, so the difference between the two numbers is the split and nothing else. |
| `full_feb_v1` | `group_split_v1` with its 45 test scenes folded back into training (406 scenes), the 45 validation scenes untouched. | The models scored on March and July. With another month as the test set there is nothing left to hold February scenes back for. |
| `size_{90,180,270}_v1` | `group_split_v1` with training subsampled to 90 / 180 / 270 whole scenes, stratified by cluster and **nested** across sizes. Validation and test untouched. | The learning curve in the number of training scenes. |

The `_cov` suffix means the table carries the recovered covariate columns. Everything in
the paper uses those; the un-suffixed tables are the same partitioning with the original
four auxiliary variables and are kept only for the input comparison.

### Covariate presets

Which of the table's columns the network actually sees is a named preset, defined in
`code/data/loading/covariates.py` (`PRESETS`) and selected with `covariates=<preset>`:

| preset | variables | |
|---|---|---|
| `legacy` | 4 | footprint, solar zenith angle, two aerosol types — the original submission's set, with its original encoding |
| `legacy_v2` | 4 | the same four through the current encoder, so an encoding change is not confounded with an input change |
| `legacy_fix` | 4 | the same four, correctly paired to their soundings |
| `scene` | 9 | + viewing geometry and airmass |
| `scene_prior` | 12 | + prior surface pressure, prior XCO₂ and the two prior aerosol optical depths — **what the paper uses** |
| `scene_geo` | 14 | + latitude and longitude, as a climatology ablation |
| `scene_l1b`, `scene_geo_l1b` | 18, 20 | + L1b-tier variables |
| `acos_informed` | 49 | + the ACOS retrieval's own output. A post-processor, not a retrieval; a control, not a result. |

`scene_prior`'s twelve are `Footprint`, `AerType1`, `AerType2` (categorical), and
`SlrZen`, `sec_sza`, `vza_deg`, `sec_vza`, `airmass`, `prior_psfc_pa`, `XCO2Prior`,
`AODPrior1`, `AODPrior2`. `experiments/pipelines/make_input_table.py` writes the
definitive table of all of them, with units and ranges, from the loader itself rather than
by hand.

The preset is fixed across the hyperparameter search on purpose: searching over the input
set as well would make the covariate ablation uninterpretable.

### Per-split standardization

Targets and covariates are standardized by constants computed from **that split's training
partition only** — `experiments/pipelines/stats/<split>/`. This is the reason each split
ships with its own statistics directory rather than sharing one: standardizing held-out
soundings by a mean that was estimated from data including them is leakage, and it is
invisible in every metric.

`predict.py` and the trainers take `--stats-dir` / `data.stats_dir`; passing the wrong one
does not fail, it reports a different experiment.

## Rebuilding from the raw HDF5

Only needed to change the preprocessing. Everything downstream reads the processed files.

```bash
./scripts/get_data.sh raw

cd experiments/pipelines
python3 extract_data_tree.py      # .h5 -> xarray DataTree (Zarr)
python3 clean_data.py             # -> one .npy per sounding + aux_data.csv
python3 get_mean_sd_profile.py    # -> profile_mu.npy / profile_sigma.npy
```

The HDF5 files must carry the groups `FilterVariables`, `RetrievalHeader`,
`RetrievalResults`, `SpectralParameters` and `StateVector`, and their filenames must carry
the `clst…` / `ref…` identifiers, which is how a sounding's scene is recovered.

Then build the split you want:

```bash
python3 experiments/pipelines/make_group_split.py      # reference-disjoint
python3 experiments/pipelines/make_replicate_split.py  # within-reference
python3 experiments/pipelines/extract_covariates.py    # the 12-variable table
python3 experiments/pipelines/make_full_feb_split.py   # all-February training
python3 experiments/pipelines/make_size_splits.py      # the nested subsets
```

`make_group_split.py` is deterministic given its seed, and
`experiments/scoring/generalization_diagnostics.py --split-csv <table>` prints the
partition inventory and the standardization constants, which is the cheapest way to check
that a rebuilt table is the published one.

## Licence and provenance

The dataset is CC-BY-4.0. It was produced at the Jet Propulsion Laboratory, California
Institute of Technology; the ACOS L2FP retrieval is the operational algorithm, run by the
same group, and no timing or accuracy figure attributed to it in the paper was measured
here.

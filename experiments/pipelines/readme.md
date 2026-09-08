# experiments/pipelines/

From raw HDF5 to the two things everything downstream reads: a directory of per-sounding
`.npy` spectra, and a **split table** — one CSV row per sounding carrying its partition,
its scene, its covariates, its truth and the operational retrieval's answer.

Only needed if you are changing the preprocessing or building a new experiment. The
processed spectra and every split table in the paper are a Zenodo download
(`scripts/get_data.sh`); see [`docs/data.md`](../../docs/data.md).

## Raw data → PyTorch-ready files

| | |
|---|---|
| `extract_data_tree.py` | the delivered `.h5` files → one [xarray DataTree](https://docs.xarray.dev/en/latest/generated/xarray.DataTree.html), written as a Zarr store at `cfg.data.tree` |
| `clean_data.py` | that store → one `.npy` per sounding plus `aux_data.csv`, at `cfg.data.files`, readable by `data.loading.spectral_dataset.SpectralDataset`. Drops missing radiance wavelengths and screens non-physical states |
| `get_mean_sd_profile.py` | `profile_mu.npy` / `profile_sigma.npy` for the 20-level targets |
| `prepare_month.py` | the same for one further delivered month, including the metadata pairing |

The heavy lifting is in `code/data/processing/`; these are the drivers.

## Split tables

| | |
|---|---|
| `make_group_split.py` | **reference-sounding-disjoint**: whole scenes to train / validation / test, stratified by cluster. The paper's primary partitioning |
| `make_replicate_split.py` | replicate-level, as the original submission did: a scene's 100 replicates assigned independently. The within-reference comparison |
| `make_full_feb_split.py` | the reference-disjoint split with its test scenes folded back into training, validation untouched. The models scored on another month |
| `make_size_splits.py` | nested subsets of the training scenes (90 / 180 / 270), for the learning curve in training-set size |
| `make_month_split.py` | a table whose test partition is a whole held-out month |
| `extract_covariates.py` | reads the 12 covariates out of the delivered HDF5, keyed by sounding ID. Split-independent, so it runs once |
| `augment_split.py` | merges that covariate table into a split table, producing the `_cov` variant |
| `compute_split_stats.py` | standardization constants from **that split's training partition only**. Every split needs its own; sharing one is leakage |
| `make_input_table.py` | the paper's definitive input table, generated from the loader rather than typed |

## Diagnostics

`characterize_discrepancy.py` describes the simulation's model discrepancy — whether its
mean and covariance differ across clusters, and whether it is correlated with the state.
`inspect_state_vector.py`, `survey_months.py`, `survey_extension.py` and
`diagnose_metadata_pairing.py` are one-off surveys of a raw delivery.

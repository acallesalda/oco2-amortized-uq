# Second-round revision experiments

These scripts produce the numbers added to the manuscript in the second round of review (JGR: Machine Learning and Computation, 2026JH001545R). Run them from the repository root.

| Script | Produces | Needs | GPU |
|---|---|---|---|
| `prior_baseline.py` | Prior XCO₂ RMSE/MAE in Tables 6–7; response Table 2 | split tables (`get_data.sh splits`) | no |
| `prior_gaussian.py` | Prior N(x_a, S_a) rows for the profile targets in Tables 6–7, and NLL/coverage for every prior row | scoring artifacts, a priori files (`get_data.sh artifacts apriori`) | no |
| `bias_qf.py` | Mean error, RMSE with the mean error removed, and quality-flag subsets for every XCO₂ method (Section 4.1, response R2.2) | scoring artifacts, split tables | no |
| `input_occlusion.py` | Auxiliary-input rows and "all three" row of Table 9; Table 10 | weights, splits, processed spectra (`weights`, `splits`, `train`, `test`) | yes |
| `mc_budget.py` | Table 8 (shape gap at S = 512, 2,048, 8,192) | scoring artifacts (`get_data.sh artifacts`) | no |

## Commands

Prior-only baseline (seconds):

```bash
python3 experiments/scoring/revision2/prior_baseline.py
```

The prior as a Gaussian retrieval (about a minute). It also rescores ACOS L2FP through the same path, which reproduces the published ACOS cells as a check:

```bash
./scripts/get_data.sh artifacts apriori
python3 experiments/scoring/revision2/prior_gaussian.py
```

The a priori files are one HDF5 per month and geophysical cluster, from the same simulation ensemble (data record, `oco2_apriori.tar.gz`). They unpack to `scratch/simulated/apriori/OCO2_RefPrior_{202002,202003,202007}/`, the script's default `--priors`. `OCO2_RefPrior_Augment_202002/` holds two further cluster 8–9 files that no table uses.

Mean error and quality-flag subsets (seconds):

```bash
python3 experiments/scoring/revision2/bias_qf.py
```

Input occlusion: one run per partition, each 10–15 minutes on a laptop GPU (RTX 4060).

```bash
# reference-disjoint: Table 9, Table 10 top block
python3 experiments/scoring/revision2/input_occlusion.py --setting ref_disjoint \
    --split-csv scratch/simulated/files/splits/group_split_v1_cov.csv \
    --stats-dir experiments/pipelines/stats/group_split_v1 \
    --weights-root experiments/weights/group_split_v1 \
    --out occlusion/ref_disjoint.csv

# within-reference: Table 10 bottom block
python3 experiments/scoring/revision2/input_occlusion.py --setting within_ref \
    --split-csv scratch/simulated/files/splits/replicate_split_v1_cov.csv \
    --stats-dir experiments/pipelines/stats/replicate_split_v1 \
    --weights-root experiments/weights/replicate_split_v1 \
    --out occlusion/within_ref.csv
```

In the output CSV:
- Table 9 uses the `[mean]` rows.
- Table 10 uses the `[mean]` and `[cluster]` rows.
- The `[zero]` rows were computed for completeness and are not reported.

Monte Carlo budget for Table 8 (a few minutes, CPU):

```bash
python3 experiments/scoring/revision2/mc_budget.py
```

## Notes

- **Single-band rows of Table 9.** These come from `experiments/scoring/spectral_attribution.py` and are unchanged from the first revision. That script substitutes the mean spectrum of its first 32 training batches (`--mean-batches 32`). The "all three" row and the auxiliary rows use the full training-partition mean computed by `input_occlusion.py`.
- **Standard errors.** All standard errors resample reference soundings, identified by cluster and ref number together. Ref numbers restart in every cluster.
- **Fix to `nongauss.py`.** `nongauss.py` previously grouped by ref number alone, which merged distinct reference soundings across clusters (37 groups instead of 45). That is fixed in the same release, and Table 8's standard errors are the corrected ones.
- **The July sounding.** One July sounding, `clst0002_ref0011_rep0010`, has a non-physical ACOS L2FP state. It is excluded from point errors, as in Table 7.

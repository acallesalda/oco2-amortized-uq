# experiments/

The entry points. The library they call lives in `code/`; the documentation for using them
is in `docs/`.

| | |
|---|---|
| `pipelines/` | raw HDF5 → spectra, covariates and split tables. [`docs/data.md`](../docs/data.md) |
| `train/` | the six trainers. [`docs/training.md`](../docs/training.md) |
| `scoring/` | `predict.py` (GPU) writes a predictive artifact; everything else reads one. [`docs/evaluation.md`](../docs/evaluation.md) |
| `figures/` | the three spectral-band figures, drawn from raw radiances rather than from an artifact |
| `weights/` | where checkpoints land. Not in git; `scripts/get_data.sh weights` fills it |
| `benchmark_test_time_oco2.py` | the standalone test-time benchmark |
| `dump_checkpoint_config.py` | read a checkpoint's architecture off its tensor shapes, without loading it into a model. What the appendix's configuration table is audited against |

To regenerate every table and figure in the paper from artifacts already on disk:

```bash
./experiments/scoring/rebuild.sh
```

[`docs/reproduce.md`](../docs/reproduce.md) maps each float to the command that writes it.

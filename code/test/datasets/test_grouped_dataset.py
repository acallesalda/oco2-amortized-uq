"""test_grouped_dataset.py
Tests for loading a group-disjoint split, where one annotations table draws rows from
both preprocessed partition directories via a per-row `source_dir` column.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))

from data.loading.spectral_dataset import SpectralDataset  # noqa: E402
from data.loading import spectral_loader as sl  # noqa: E402

# One wavelength in each of the three OCO-2 windows, as split by load_spectra
# (b1: <1.2 um, b2: 1.2-1.8 um, b3: >1.8 um).
BAND_WAVELENGTHS = [0.76, 1.61, 2.06]
N_PER_BAND = 8

AUX = {"Footprint": 3, "SlrZen": 45.0, "AerType1": "DU", "AerType2": "SO"}
PROFILE_COLS = [f"true_state_vector_{i:02d}" for i in range(1, 21)]
RETR_COLS = [f"retrieved_state_vector_{i:02d}" for i in range(1, 21)]


def _make_spectrum(seed: int) -> np.ndarray:
    """(L, 2) array of [wavelength, radiance], as written by save_data."""
    rng = np.random.default_rng(seed)
    wl = np.concatenate([np.full(N_PER_BAND, w) + 0.001 * np.arange(N_PER_BAND)
                         for w in BAND_WAVELENGTHS])
    rad = 1e-6 * (1.0 + rng.random(wl.size))
    return np.stack([wl, rad], axis=1)


@pytest.fixture
def files_root(tmp_path):
    """A miniature version of the processed data tree: files/{train,test}/*.npy."""
    root = tmp_path / "files"
    rows = []
    for i, (sub, clst, ref) in enumerate([
        ("train", 1, 1), ("train", 1, 2), ("test", 2, 1), ("test", 9, 7),
    ]):
        (root / sub).mkdir(parents=True, exist_ok=True)
        sounding_id = f"clst{clst:04d}_ref{ref:04d}_rep{i + 1:04d}"
        np.save(root / sub / f"{sounding_id}.npy", _make_spectrum(i))
        row = {
            "ID": sounding_id, "source_dir": sub, "clst": clst, "ref": ref,
            "true_xco2": 4.0e-4 + 1e-6 * i, "retrieved_xco2": 4.1e-4,
            **AUX,
        }
        row.update({c: 4.0e-4 + 1e-7 * j for j, c in enumerate(PROFILE_COLS)})
        row.update({c: 4.1e-4 for c in RETR_COLS})
        rows.append(row)
    df = pd.DataFrame(rows)
    # train/val/test assigned at the reference-sounding level
    df["split"] = ["train", "train", "val", "test"]
    split_csv = root / "splits" / "group_split_test.csv"
    split_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(split_csv, index=False)
    return root, split_csv, df


@pytest.fixture
def stats_dir(tmp_path):
    d = tmp_path / "stats"
    d.mkdir()
    np.save(d / "profile_mu.npy", np.full(20, 4.0e-4))
    np.save(d / "profile_sigma.npy", np.full(20, 1.0e-5))
    (d / "xco2_stats.json").write_text('{"mu": 0.0004, "sigma": 2.0e-05}')
    return d


def test_accepts_a_dataframe_instead_of_a_csv_path(files_root):
    root, _, df = files_root
    ds = SpectralDataset(df, root, spectra_dir_col="source_dir")
    assert len(ds) == len(df)


def test_source_dir_column_resolves_paths_across_partitions(files_root):
    root, _, df = files_root
    ds = SpectralDataset(df, root, spectra_dir_col="source_dir")
    for i in range(len(df)):
        path = ds.spectra_path(i)
        assert path.exists(), f"row {i} resolved to a missing file: {path}"
        assert path.parent.name == df.loc[i, "source_dir"]
    # rows really do come from both directories
    assert {ds.spectra_path(i).parent.name for i in range(len(df))} == {"train", "test"}


def test_without_source_dir_column_paths_stay_flat(files_root):
    root, _, df = files_root
    train = df[df.source_dir == "train"].reset_index(drop=True)
    ds = SpectralDataset(train, root / "train")
    assert ds.spectra_path(0) == root / "train" / f"{train.loc[0, 'ID']}.npy"


def test_unknown_source_dir_column_fails_loudly(files_root):
    root, _, df = files_root
    with pytest.raises(KeyError):
        SpectralDataset(df, root, spectra_dir_col="nope")


def test_grouped_profile_dataset_returns_standardized_20d_targets(files_root, stats_dir):
    root, split_csv, _ = files_root
    ds = sl.build_dataset_profiles_grouped(root, split_csv, "train", stats_dir=stats_dir)
    assert len(ds) == 2
    (bands, aux_cont, aux_cat), y, y_retr = ds[0]
    assert y.shape == (20,) and y_retr.shape == (20,)
    assert set(bands) == {"b1", "b2", "b3"}
    for b, const in bands.values():
        assert b.shape == (2, N_PER_BAND)
        assert const.shape == (2,)
    assert aux_cont.shape == (1,) and len(aux_cat) == 3
    # standardization is applied with the stats we supplied, not the legacy constants
    assert torch.isfinite(y).all() and y.abs().max() < 100


def test_grouped_scalar_dataset_uses_the_supplied_xco2_stats(files_root, stats_dir):
    root, split_csv, df = files_root
    ds = sl.build_dataset_grouped(root, split_csv, "train", stats_dir=stats_dir)
    (_, _, _), y, _ = ds[0]
    expected = (df.loc[0, "true_xco2"] - 0.0004) / 2.0e-05
    assert y.numel() == 1
    assert float(y.reshape(-1)[0]) == pytest.approx(expected, rel=1e-5)


def test_each_partition_selects_only_its_own_rows(files_root, stats_dir):
    root, split_csv, _ = files_root
    sizes = {
        s: len(sl.build_dataset_profiles_grouped(root, split_csv, s, stats_dir=stats_dir))
        for s in ("train", "val", "test")
    }
    assert sizes == {"train": 2, "val": 1, "test": 1}


def test_loaders_never_mix_partitions(files_root, stats_dir):
    root, split_csv, _ = files_root
    train_dl, val_dl = sl.train_val_loader_profiles(
        root, "aux_data.csv", train_batch_size=2, val_batch_size=2,
        stats_dir=stats_dir, split_csv=split_csv,
    )
    test_dl = sl.test_loader_profiles(
        root, "aux_data.csv", val_batch_size=2, stats_dir=stats_dir, split_csv=split_csv,
    )
    ids = {
        name: set(dl.dataset.files["ID"])
        for name, dl in (("train", train_dl), ("val", val_dl), ("test", test_dl))
    }
    assert ids["train"] & ids["val"] == set()
    assert ids["train"] & ids["test"] == set()
    assert ids["val"] & ids["test"] == set()


def test_missing_partition_in_split_table_raises(files_root, stats_dir):
    root, split_csv, _ = files_root
    with pytest.raises(ValueError):
        sl.build_dataset_profiles_grouped(root, split_csv, "nonexistent", stats_dir=stats_dir)


def test_scalar_stats_fall_back_to_legacy_constants_when_absent(tmp_path):
    mu, sigma = sl._load_scalar_stats(tmp_path)
    assert mu == sl._LEGACY_XCO2_MU
    assert sigma == sl._LEGACY_XCO2_SIGMA

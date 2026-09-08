"""test_covariates.py
Tests for the configurable auxiliary covariates.

The property that matters most is the first one: with the default spec, every model built
before covariates were configurable must see byte-identical inputs and have byte-identical
parameter names, or the published checkpoints stop loading and every table has to be
recomputed. The rest of the file checks the new path -- training-only statistics, unseen
categorical levels, missing values, and the shape contract between the spec and the
embedding tables.

Pure numpy/pandas except where marked; the torch parts skip if torch is absent.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from data.loading.covariates import (  # noqa: E402
    CovariateEncoder, CovariateSpec, PRESETS, cat_cardinalities, get_spec,
    load_covariate_stats,
)


def _load_module(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


aug = _load_module("experiments/pipelines/augment_split.py", "augment_split")


# ---------------------------------------------------------------- fixtures --
AER = ["DU", "SO", "SS", "OC", "BC"]


def make_frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ID": [f"clst0001_ref{1 + i // 10:04d}_rep{1 + i % 10:04d}" for i in range(n)],
        "split": ["train"] * (n // 2) + ["val"] * (n // 4) + ["test"] * (n - n // 2 - n // 4),
        "SlrZen": rng.uniform(10.0, 70.0, n),
        "Footprint": rng.integers(1, 9, n),
        "AerType1": rng.choice(AER, n),
        "AerType2": rng.choice(AER, n),
        "airmass": rng.uniform(2.0, 4.0, n),
        "sec_sza": rng.uniform(1.0, 3.0, n),
        "sec_vza": rng.uniform(1.0, 2.0, n),
        "vza_deg": rng.uniform(0.0, 40.0, n),
        "prior_psfc_pa": rng.uniform(7e4, 1.02e5, n),
        "Latitude": rng.uniform(-60, 60, n),
        "Longitude": rng.uniform(-180, 180, n),
        "outcome_flag": rng.integers(1, 4, n),
        # The corrected metadata family. The plain names above are the published
        # (scrambled) values; *_fix are the ones paired by reference-sounding number.
        "SlrZen_fix": rng.uniform(10.0, 70.0, n),
        "Footprint_fix": rng.integers(1, 9, n),
        "AerType1_fix": rng.choice(AER, n),
        "AerType2_fix": rng.choice(AER, n),
        "XCO2Prior": rng.uniform(405.0, 418.0, n),
        "AODPrior1": rng.uniform(0.0, 0.4, n),
        "AODPrior2": rng.uniform(0.0, 0.2, n),
    })


def make_stats(frame, spec):
    train = frame[frame["split"] == "train"]
    return {"columns": list(spec.columns),
            "stats": aug.build_stats(train, list(spec.columns))}


# ------------------------------------------------------- the legacy contract --
def test_legacy_is_the_default():
    assert get_spec(None).name == "legacy"
    assert get_spec(None).legacy is True
    assert get_spec("legacy").legacy is True


def test_legacy_encoding_is_unchanged():
    """SlrZen/90 - 0.5, Footprint - 1, and the published aerosol vocabulary."""
    df = make_frame()
    enc = CovariateEncoder(df, get_spec("legacy"))
    vocab = {"DU": 0, "SO": 1, "SS": 2, "OC": 3, "BC": 4}
    for i in range(len(df)):
        cont, cat = enc(i)
        assert cont.shape == (1,)
        assert np.isclose(float(cont[0]),
                          np.float32(df["SlrZen"].iloc[i]) / 90.0 - 0.5, atol=1e-6)
        assert len(cat) == 3
        assert int(cat[0]) == int(df["Footprint"].iloc[i]) - 1
        assert int(cat[1]) == vocab[df["AerType1"].iloc[i]]
        assert int(cat[2]) == vocab[df["AerType2"].iloc[i]]
    assert enc.cat_cardinalities == (8, 5, 5)


def test_legacy_needs_no_stats_file():
    """A legacy run must work against a split table with no covariate statistics."""
    CovariateEncoder(make_frame(), get_spec("legacy"), stats=None)


def test_non_legacy_requires_stats():
    with pytest.raises(ValueError, match="covariate_stats"):
        CovariateEncoder(make_frame(), get_spec("scene"), stats=None)


# ------------------------------------------------------------ the new path --
def test_continuous_standardized_with_training_statistics_only():
    df, spec = make_frame(), get_spec("scene")
    stats = make_stats(df, spec)
    enc = CovariateEncoder(df, spec, stats)

    train_mask = (df["split"] == "train").to_numpy()
    j = spec.continuous.index("prior_psfc_pa")
    col = df["prior_psfc_pa"].to_numpy()
    mu, sd = col[train_mask].mean(), col[train_mask].std()
    assert np.allclose(enc.cont[:, j], (col - mu) / sd, atol=1e-5)

    # The training rows -- and only they -- are centred.
    assert abs(enc.cont[train_mask, j].mean()) < 1e-5
    assert abs(enc.cont[train_mask, j].std() - 1.0) < 1e-5


def test_unseen_categorical_level_maps_to_the_reserved_index():
    """A level that appears only in validation must not collide with a trained one."""
    df, spec = make_frame(), get_spec("legacy_v2")
    df.loc[df["split"] == "train", "AerType1"] = "DU"     # train sees one level
    df.loc[df["split"] == "val", "AerType1"] = "BC"       # val sees another
    stats = make_stats(df, spec)
    enc = CovariateEncoder(df, spec, stats)

    j = spec.categorical.index("AerType1")
    unknown = stats["stats"]["AerType1"]["unknown_index"]
    assert set(enc.cat[(df["split"] == "train").to_numpy(), j]) == {0}
    assert set(enc.cat[(df["split"] == "val").to_numpy(), j]) == {unknown}
    # and the embedding table is big enough for it
    assert cat_cardinalities(spec, stats)[j] > unknown


def test_missing_values_are_imputed_at_the_training_median():
    df, spec = make_frame(), get_spec("scene")
    df.loc[df.index[-3:], "prior_psfc_pa"] = np.nan
    stats = make_stats(df, spec)
    enc = CovariateEncoder(df, spec, stats)
    j = spec.continuous.index("prior_psfc_pa")
    st = stats["stats"]["prior_psfc_pa"]
    expected = (st["impute"] - st["mean"]) / st["std"]
    assert np.allclose(enc.cont[-3:, j], expected, atol=1e-5)
    assert np.isfinite(enc.cont).all()


def test_missing_column_names_the_fix():
    df = make_frame().drop(columns=["prior_psfc_pa"])
    spec = get_spec("scene")
    with pytest.raises(KeyError, match="augment_split"):
        CovariateEncoder(df, spec, make_stats(make_frame(), spec))


def test_every_preset_declares_a_tier_and_is_self_consistent():
    for name, spec in PRESETS.items():
        assert spec.name == name
        assert spec.tier in {"prior", "l1b", "acos"}
        assert len(set(spec.columns)) == len(spec.columns), f"{name} repeats a covariate"
        assert spec.n_continuous + spec.n_categorical > 0
    # The presets nest the way the module documents. acos_informed deliberately does NOT
    # extend scene_geo: latitude and longitude are absent from the published aux_data.csv,
    # and making the ACOS post-processor depend on them would block it on a reprocess.
    assert set(PRESETS["scene"].columns) <= set(PRESETS["scene_geo"].columns)
    assert set(PRESETS["scene"].columns) <= set(PRESETS["scene_l1b"].columns)
    assert set(PRESETS["scene_l1b"].columns) <= set(PRESETS["acos_informed"].columns)
    assert set(PRESETS["acos_informed"].columns) <= set(PRESETS["acos_informed_geo"].columns)
    for geo in ("Latitude", "Longitude"):
        assert geo not in PRESETS["acos_informed"].columns
        assert geo in PRESETS["scene_geo"].columns
    # nothing outside acos_informed may carry a retrieval output
    for name, spec in PRESETS.items():
        if spec.tier != "acos":
            assert not any(c.startswith(("retr_", "retrieved_", "aod_", "albedo_"))
                           for c in spec.columns), f"{name} leaks an ACOS output"


def test_spec_from_mapping():
    spec = get_spec({"name": "custom", "continuous": ["SlrZen"], "categorical": ["Footprint"]})
    assert spec.n_continuous == 1 and spec.n_categorical == 1 and not spec.legacy
    assert get_spec({"name": "scene"}).name == "scene"


def test_load_covariate_stats_round_trip(tmp_path):
    df, spec = make_frame(), get_spec("scene")
    (tmp_path / "covariate_stats.json").write_text(json.dumps(make_stats(df, spec)))
    stats = load_covariate_stats(tmp_path)
    assert stats is not None and "prior_psfc_pa" in stats["stats"]
    assert load_covariate_stats(tmp_path / "nope") is None


def test_build_stats_splits_continuous_from_categorical():
    df, spec = make_frame(), get_spec("acos_informed")
    cols = ["SlrZen", "Footprint", "outcome_flag", "prior_psfc_pa"]
    stats = aug.build_stats(df[df["split"] == "train"], cols)
    assert stats["SlrZen"]["kind"] == "continuous"
    assert stats["prior_psfc_pa"]["kind"] == "continuous"
    assert stats["Footprint"]["kind"] == "categorical"
    assert stats["outcome_flag"]["kind"] == "categorical"
    assert spec.tier == "acos"


# ------------------------------------------------------------------- torch --
def test_aux_embedding_v2_shapes():
    torch = pytest.importorskip("torch")
    from modeling.components.encoders import AuxEmbedding, AuxEmbeddingV2, AuxEncoder

    emb = AuxEmbeddingV2(n_cont=6, cat_cardinalities=(8, 6, 6))
    assert emb.dim == 8 * 6 + 3 * 8
    cont = torch.zeros(4, 6)
    cat = [torch.zeros(4, 1, dtype=torch.long) for _ in range(3)]
    assert emb(cont, cat).shape == (4, emb.dim)

    # the legacy module is unchanged and still 32-wide
    assert AuxEmbedding().dim == 32
    legacy = AuxEncoder(32, 64)
    assert isinstance(legacy.embedding, AuxEmbedding)
    assert {n for n, _ in legacy.embedding.named_parameters()} == {
        "cat1.weight", "cat2.weight", "cat3.weight", "fc_cont.weight", "fc_cont.bias"}

    # a stale aux_dim in a config must not shape the projection
    v2 = AuxEncoder(32, 64, n_cont=6, cat_cardinalities=(8, 6, 6))
    assert v2.encoder.in_features == v2.embedding.dim == 72


def test_regressors_accept_a_covariate_description():
    torch = pytest.importorskip("torch")
    from modeling.full_regressor import FullRegressor
    from modeling.full_regressor_profiles import FullRegressorProfiles

    bands = {k: (torch.randn(2, 2, 128), torch.randn(2, 2)) for k in ("b1", "b2", "b3")}
    cont = torch.randn(2, 5)
    cat = [torch.zeros(2, 1, dtype=torch.long) for _ in range(2)]
    x = (bands, cont, cat)

    m = FullRegressorProfiles(embed_dim=32, nhead=4, nlayers=1, out_dim=20,
                              n_cont=5, cat_cardinalities=(8, 6))
    assert m(x).shape == (2, 20)
    s = FullRegressor(embed_dim=32, nhead=4, nlayers=1, n_cont=5, cat_cardinalities=(8, 6))
    assert s(x).shape == (2, 1)

    # default construction still takes the published 1 continuous / 3 categorical inputs
    legacy = FullRegressorProfiles(embed_dim=32, nhead=4, nlayers=1, out_dim=20)
    legacy_x = (bands, torch.randn(2, 1), [torch.zeros(2, 1, dtype=torch.long)] * 3)
    assert legacy(legacy_x).shape == (2, 20)


def test_level_weights_are_mean_one_and_reproduce_ppm_mse():
    pytest.importorskip("torch")
    from modeling import losses as tp

    sigma = np.linspace(1.4e-6, 2.0e-5, 20)
    idxs = list(range(20))
    assert tp.level_weights(sigma, "std", idxs) is None

    w = tp.level_weights(sigma, "ppm", idxs).numpy()
    assert np.isclose(w.mean(), 1.0)
    # weighted standardized MSE == ppm MSE up to the constant mean(sigma^2)
    err = np.random.default_rng(0).standard_normal((7, 20))
    lhs = (err ** 2 * w).mean()
    rhs = ((err * sigma * 1e6) ** 2).mean() / ((sigma * 1e6) ** 2).mean()
    assert np.isclose(lhs, rhs)

    # the last-4 slice weights only those levels
    w4 = tp.level_weights(sigma, "ppm", [16, 17, 18, 19]).numpy()
    assert w4.shape == (4,) and np.isclose(w4.mean(), 1.0)

    with pytest.raises(ValueError, match="unknown loss_weight"):
        tp.level_weights(sigma, "rmse", idxs)
    with pytest.raises(ValueError, match="non-positive"):
        tp.level_weights(np.zeros(20), "ppm", idxs)


def test_weighted_mse_matches_torch_mse_when_unweighted():
    torch = pytest.importorskip("torch")
    from modeling.losses import weighted_mse
    a, b = torch.randn(5, 20), torch.randn(5, 20)
    assert torch.allclose(weighted_mse(a, b), torch.nn.functional.mse_loss(a, b))


# ------------------------------------------------------------ end to end --
def test_loader_to_model_with_scene_covariates(tmp_path):
    """The whole chain: augmented split table -> dataset -> batch -> forward pass.

    Shape mismatches between the spec, the encoder and the embedding tables only show up
    when a real batch is collated, which is why this exists as well as the unit tests.
    """
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import torch
    from torch.utils.data import DataLoader
    from data.loading import spectral_loader as sl
    from modeling.full_regressor_profiles import FullRegressorProfiles

    profile_cols = [f"true_state_vector_{i:02d}" for i in range(1, 21)]
    retr_cols = [f"retrieved_state_vector_{i:02d}" for i in range(1, 21)]
    # 64 channels per band: the encoder halves the length four times and then max-pools,
    # so anything shorter than 32 leaves nothing to pool.
    wl = np.concatenate([np.full(64, w) + 1e-4 * np.arange(64) for w in (0.76, 1.61, 2.06)])

    root = tmp_path / "files"
    rng = np.random.default_rng(0)
    rows = []
    for i, (sub, clst, ref, split) in enumerate([
        ("train", 1, 1, "train"), ("train", 1, 2, "train"),
        ("train", 1, 3, "train"), ("test", 2, 1, "val"), ("test", 9, 7, "test"),
    ]):
        (root / sub).mkdir(parents=True, exist_ok=True)
        sid = f"clst{clst:04d}_ref{ref:04d}_rep{i + 1:04d}"
        # float32, as the h5 fields are and as save_data therefore writes
        np.save(root / sub / f"{sid}.npy",
                np.stack([wl, 1e-6 * (1.0 + rng.random(wl.size))], axis=1).astype(np.float32))
        row = {"ID": sid, "source_dir": sub, "clst": clst, "ref": ref, "split": split,
               "true_xco2": 4.0e-4, "retrieved_xco2": 4.1e-4,
               "SlrZen": 30.0 + i, "Footprint": 1 + i, "AerType1": "DU", "AerType2": "SO",
               "SlrZen_fix": 31.0 + i, "Footprint_fix": 1 + i,
               "AerType1_fix": "DU", "AerType2_fix": "SO",
               "XCO2Prior": 410.0 + i, "AODPrior1": 0.1 * i, "AODPrior2": 0.05 * i,
               "airmass": 2.4 + 0.1 * i, "sec_sza": 1.4 + 0.01 * i, "sec_vza": 1.0 + 0.01 * i,
               "vza_deg": 5.0 * i, "prior_psfc_pa": 9.9e4 + 100.0 * i,
               "Latitude": 10.0 * i, "Longitude": -20.0 * i}
        row.update({c: 4.0e-4 + 1e-7 * j for j, c in enumerate(profile_cols)})
        row.update({c: 4.1e-4 for c in retr_cols})
        rows.append(row)
    df = pd.DataFrame(rows)
    split_csv = root / "splits" / "group_split_cov.csv"
    split_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(split_csv, index=False)

    stats = tmp_path / "stats"
    stats.mkdir()
    np.save(stats / "profile_mu.npy", np.full(20, 4.0e-4))
    np.save(stats / "profile_sigma.npy", np.linspace(1.4e-6, 2.0e-5, 20))
    (stats / "xco2_stats.json").write_text('{"mu": 0.0004, "sigma": 3.1e-06}')
    spec = get_spec("scene_geo")
    (stats / "covariate_stats.json").write_text(json.dumps(
        {"columns": list(spec.columns),
         "stats": aug.build_stats(df[df["split"] == "train"], list(spec.columns))}))

    ds = sl.build_dataset_profiles_grouped(root, split_csv, "train",
                                           stats_dir=stats, cov_spec="scene_geo")
    (bands, cont, cat), y, y_retr = next(iter(DataLoader(ds, batch_size=3)))
    assert cont.shape == (3, spec.n_continuous)
    assert len(cat) == spec.n_categorical and all(c.shape == (3, 1) for c in cat)
    assert y.shape == (3, 20)

    cards = cat_cardinalities(spec, load_covariate_stats(stats))
    model = FullRegressorProfiles(embed_dim=32, nhead=4, nlayers=1, out_dim=20,
                                  n_cont=spec.n_continuous, cat_cardinalities=list(cards))
    with torch.no_grad():
        assert model((bands, cont, cat)).shape == (3, 20)

    # and the scalar loader takes the same route
    ds1 = sl.build_dataset_grouped(root, split_csv, "train",
                                   stats_dir=stats, cov_spec="scene_geo")
    (b1, c1, k1), y1, _ = next(iter(DataLoader(ds1, batch_size=2)))
    assert c1.shape == (2, spec.n_continuous) and y1.shape == (2, 1)

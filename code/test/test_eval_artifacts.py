"""Tests for evals.artifacts and evals.acos.

The ACOS tests are the important ones. `full_profiles.ipynb` reads the posterior
covariance from a fixed `test/` directory, which is correct for the replicate split and
badly wrong for the group split, where a test sounding usually lives under `train/`.
Nothing about that failure is visible at runtime -- the baseline row of the table is
simply computed on whichever soundings happened to be in the right folder -- so it is
pinned here.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals import acos
from evals.artifacts import Artifact, ArtifactMeta


# ---------------------------------------------------------------------------------
# Artifact round-trip
# ---------------------------------------------------------------------------------
def _artifact(n=5, d=3):
    meta = ArtifactMeta(
        task="profiles", split_name="group_split_v1", n=n, dim=d,
        y_mu=[4e-4] * d, y_sigma=[2e-5] * d, levels=list(range(1, d + 1)),
    )
    arrays = {
        "ids": np.array([f"s{i}" for i in range(n)], dtype="U8"),
        "y_std": np.arange(n * d, dtype=np.float32).reshape(n, d),
        "nf_samples_std": np.zeros((n, 4, d), dtype=np.float16),
    }
    return Artifact(meta, arrays)


def test_round_trip(tmp_path):
    art = _artifact()
    p = art.save(tmp_path / "profiles.npz")
    back = Artifact.load(p)

    assert back.meta.task == "profiles"
    assert back.meta.n == art.meta.n
    assert np.allclose(back["y_std"], art["y_std"])
    assert back["nf_samples_std"].dtype == np.float16
    assert np.allclose(back.y_sigma, 2e-5)


def test_sidecar_metadata_is_readable_without_loading_the_arrays(tmp_path):
    """A 200 MB file should still answer "what is this" with `cat`."""
    import json
    p = _artifact().save(tmp_path / "profiles.npz")
    side = p.with_suffix(".meta.json")
    assert side.exists()
    assert json.loads(side.read_text())["split_name"] == "group_split_v1"


def test_lazy_load_does_not_materialize_everything(tmp_path):
    p = _artifact().save(tmp_path / "profiles.npz")
    back = Artifact.load(p, lazy=True)
    assert back.has("nf_samples_std")
    assert not back.has("lap_mu_std")
    assert back.get("lap_mu_std") is None
    assert back["y_std"].shape == (5, 3)


def test_misaligned_array_is_rejected():
    """Everything in an artifact is per-sounding and positionally aligned; a length
    mismatch means the alignment is already broken and no metric computed from it means
    anything."""
    meta = ArtifactMeta(task="xco2", split_name="s", n=4, dim=1, y_mu=[0.0], y_sigma=[1.0])
    with pytest.raises(ValueError, match="leading dimension"):
        Artifact(meta, {"ids": np.arange(4).astype("U2"), "y_std": np.zeros((3, 1))})


def test_missing_required_array_is_rejected():
    meta = ArtifactMeta(task="xco2", split_name="s", n=2, dim=1, y_mu=[0.0], y_sigma=[1.0])
    with pytest.raises(KeyError, match="missing required"):
        Artifact(meta, {"y_std": np.zeros((2, 1))})


def test_unknown_array_name_says_what_is_there():
    art = _artifact()
    with pytest.raises(KeyError, match="nf_samples_std"):
        art["not_a_real_array"]


# ---------------------------------------------------------------------------------
# ACOS
# ---------------------------------------------------------------------------------
@pytest.fixture
def acos_tree(tmp_path):
    """A files root shaped like the real one: spectra and covariances in train/ and test/,
    and a group-split test partition that draws from both."""
    d = 20
    rng = np.random.default_rng(0)
    root = tmp_path / "files"
    for part in ("train", "test"):
        (root / part).mkdir(parents=True)

    rows = []
    covs = {}
    for i in range(10):
        part = "train" if i % 2 == 0 else "test"      # a group split mixes the two
        sid = f"snd{i:03d}"
        A = rng.normal(size=(d, d))
        cov = A @ A.T + d * np.eye(d)
        np.save(root / part / f"{sid}_cov.npy", cov)
        covs[sid] = cov
        row = {"ID": sid, "source_dir": part, "split": "test"}
        row.update({f"retrieved_state_vector_{j:02d}": 4e-4 + 1e-6 * j for j in range(1, d + 1)})
        row.update({"retrieved_xco2": 4.05e-4, "retrieved_xco2_variance": (2e-6) ** 2})
        rows.append(row)
    return root, pd.DataFrame(rows), covs


def test_covariances_are_found_in_both_partitions(acos_tree):
    root, frame, covs = acos_tree
    y_mu = np.full(20, 4e-4)
    y_sigma = np.full(20, 2e-5)

    mu, cov, has = acos.load_profiles(frame, root, y_mu, y_sigma)
    assert has.all(), "every covariance exists on disk and must be found"
    assert mu.shape == (10, 20) and cov.shape == (10, 20, 20)

    outer = np.outer(y_sigma, y_sigma)
    assert np.allclose(cov[0], covs[frame.loc[0, "ID"]] / outer)


def test_ignoring_source_dir_loses_most_of_the_baseline(acos_tree):
    """The regression this module exists for.

    With `source_dir_col=None` the loader looks directly under the root, finds nothing,
    and returns an all-NaN covariance -- which downstream becomes a baseline row computed
    on zero soundings rather than an error.
    """
    root, frame, _ = acos_tree
    y_mu, y_sigma = np.full(20, 4e-4), np.full(20, 2e-5)
    _, _, has = acos.load_profiles(frame, root, y_mu, y_sigma, source_dir_col=None)
    assert not has.any()


def test_missing_source_dir_column_is_an_error(acos_tree):
    root, frame, _ = acos_tree
    with pytest.raises(KeyError, match="source_dir"):
        acos.load_profiles(frame.drop(columns=["source_dir"]), root,
                           np.full(20, 4e-4), np.full(20, 2e-5))


def test_level_subset_takes_the_right_marginal_block(acos_tree):
    """The near-surface task is a marginal of the 20-level ACOS posterior, which for a
    Gaussian is the corresponding sub-block -- not a re-solve."""
    root, frame, covs = acos_tree
    y_mu, y_sigma = np.full(4, 4e-4), np.full(4, 2e-5)
    levels = [17, 18, 19, 20]

    mu, cov, has = acos.load_profiles(frame, root, y_mu, y_sigma, levels=levels)
    assert cov.shape == (10, 4, 4) and mu.shape == (10, 4)
    idx = np.array([l - 1 for l in levels])
    expected = covs[frame.loc[0, "ID"]][np.ix_(idx, idx)] / np.outer(y_sigma, y_sigma)
    assert np.allclose(cov[0], expected)


def test_malformed_covariance_is_dropped_not_reshaped(acos_tree):
    root, frame, _ = acos_tree
    bad = frame.loc[0, "ID"]
    np.save(root / frame.loc[0, "source_dir"] / f"{bad}_cov.npy", np.zeros((5, 5)))
    _, cov, has = acos.load_profiles(frame, root, np.full(20, 4e-4), np.full(20, 2e-5))
    assert not has[0]
    assert np.isnan(cov[0]).all()
    assert has[1:].all()


def test_xco2_variance_scales_by_sigma_squared(acos_tree):
    _, frame, _ = acos_tree
    mu, var, valid = acos.load_xco2(frame, y_mu=4e-4, y_sigma=2e-5)
    assert mu.shape == (10, 1) and var.shape == (10, 1, 1)
    assert valid.all()
    assert var[0, 0, 0] == pytest.approx((2e-6) ** 2 / (2e-5) ** 2)
    assert mu[0, 0] == pytest.approx((4.05e-4 - 4e-4) / 2e-5)

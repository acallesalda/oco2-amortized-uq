"""test_group_split.py
Tests for the reference-sounding-disjoint split builder and the joint calibration
diagnostics. Both are pure functions of arrays/frames, so they run without the
simulation data or any trained checkpoint.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))


def _load_module(relpath: str, name: str):
    """Import a script that lives outside the importable package tree."""
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mgs = _load_module("experiments/pipelines/make_group_split.py", "make_group_split")

from evals import joint_calibration as jc  # noqa: E402


# ---------------------------------------------------------------- fixtures --
def make_fake_annotations(refs_per_cluster, n_rep=100, seed=0):
    """Mimic the concatenated aux_data.csv frames: IDs, a target, a source_dir tag."""
    rng = np.random.default_rng(seed)
    rows = []
    for clst, n_refs in enumerate(refs_per_cluster, start=1):
        for ref in range(1, n_refs + 1):
            for rep in range(1, n_rep + 1):
                rows.append({
                    "ID": f"clst{clst:04d}_ref{ref:04d}_rep{rep:04d}",
                    "true_xco2": 4.0e-4 + 1e-5 * rng.standard_normal(),
                    # the original pipeline's replicate-level assignment
                    "source_dir": "train" if rng.random() < 0.8 else "test",
                })
    return pd.DataFrame(rows)


# 479 reference soundings across 9 clusters, deliberately unbalanced.
REFS_PER_CLUSTER = [80, 75, 70, 60, 55, 50, 45, 30, 14]


# ------------------------------------------------------------------- tests --
def test_refs_per_cluster_sums_to_479():
    assert sum(REFS_PER_CLUSTER) == 479


def test_add_group_keys_parses_ids():
    df = mgs.add_group_keys(make_fake_annotations([2, 1], n_rep=3))
    assert df["clst"].tolist() == [1] * 6 + [2] * 3
    assert df["ref"].tolist() == [1, 1, 1, 2, 2, 2, 1, 1, 1]
    assert df["rep"].tolist() == [1, 2, 3] * 3


def test_add_group_keys_rejects_malformed_ids():
    bad = pd.DataFrame({"ID": ["not_an_id"]})
    with pytest.raises(ValueError):
        mgs.add_group_keys(bad)


def test_apportion_sums_exactly_and_is_proportional():
    sizes = {c: n for c, n in enumerate(REFS_PER_CLUSTER, start=1)}
    alloc = mgs.apportion(sizes, 48)
    assert sum(alloc.values()) == 48
    # the largest cluster must not receive fewer than the smallest
    assert alloc[1] >= alloc[9]


def test_apportion_handles_zero_total():
    alloc = mgs.apportion({1: 10, 2: 5}, 0)
    assert alloc == {1: 0, 2: 0}


def test_cap_and_rebalance_respects_caps():
    alloc = mgs.cap_and_rebalance({1: 9, 2: 1}, caps={1: 4, 2: 8}, total=10)
    assert sum(alloc.values()) == 10
    assert alloc[1] <= 4 and alloc[2] <= 8


def test_split_is_group_disjoint_and_hits_target_sizes():
    df = mgs.add_group_keys(make_fake_annotations(REFS_PER_CLUSTER))
    groups = df.drop_duplicates(subset=["clst", "ref"])[["clst", "ref"]].reset_index(drop=True)
    assert len(groups) == 479

    groups = mgs.assign_splits(groups, val_frac=0.10, test_frac=0.10, seed=42)
    counts = groups.groupby("split").size().to_dict()
    assert counts == {"train": 383, "val": 48, "test": 48}

    merged = df.merge(groups, on=["clst", "ref"], how="left", validate="many_to_one")
    # must not raise: no reference sounding may span partitions
    mgs.validate(merged)

    # every replicate of a group lands in the same partition
    assert (merged.groupby(["clst", "ref"])["split"].nunique() == 1).all()
    # and the row counts follow the group counts exactly (100 replicates each)
    assert merged.groupby("split").size().to_dict() == {
        "train": 38300, "val": 4800, "test": 4800
    }


def test_every_cluster_represented_in_every_partition():
    df = mgs.add_group_keys(make_fake_annotations(REFS_PER_CLUSTER))
    groups = df.drop_duplicates(subset=["clst", "ref"])[["clst", "ref"]].reset_index(drop=True)
    groups = mgs.assign_splits(groups, 0.10, 0.10, seed=42)
    table = groups.groupby(["clst", "split"]).size().unstack(fill_value=0)
    assert (table["train"] > 0).all(), "a cluster lost all training soundings"
    for part in ("val", "test"):
        assert (table[part] > 0).all(), f"a cluster is missing from '{part}'"


def test_split_is_deterministic_given_seed():
    df = mgs.add_group_keys(make_fake_annotations(REFS_PER_CLUSTER))
    groups = df.drop_duplicates(subset=["clst", "ref"])[["clst", "ref"]].reset_index(drop=True)
    a = mgs.assign_splits(groups, 0.10, 0.10, seed=42)["split"].tolist()
    b = mgs.assign_splits(groups, 0.10, 0.10, seed=42)["split"].tolist()
    c = mgs.assign_splits(groups, 0.10, 0.10, seed=7)["split"].tolist()
    assert a == b
    assert a != c


def test_validate_catches_a_leaked_group():
    df = mgs.add_group_keys(make_fake_annotations([2], n_rep=4))
    df["split"] = ["train", "train", "train", "test"] + ["val"] * 4
    with pytest.raises(AssertionError):
        mgs.validate(df)


# ------------------------------------------------- joint calibration tests --
def _random_spd(d, rng):
    A = rng.standard_normal((d, d))
    return A @ A.T + d * np.eye(d)


def test_whiten_gaussian_recovers_standard_normal():
    rng = np.random.default_rng(0)
    n, d = 4000, 5
    cov = _random_spd(d, rng)
    mu = rng.standard_normal(d)
    x = rng.multivariate_normal(mu, cov, size=n)

    z = jc.whiten_gaussian(x, np.tile(mu, (n, 1)), np.tile(cov, (n, 1, 1)))
    assert np.isfinite(z).all()
    assert abs(z.mean()) < 0.05
    assert abs(z.std(ddof=1) - 1.0) < 0.05
    # ||z||^2 should average to d
    assert abs(jc.mahalanobis_sq(z).mean() - d) < 0.5


def test_joint_coverage_is_near_nominal_when_calibrated():
    rng = np.random.default_rng(1)
    n, d = 8000, 20
    z = rng.standard_normal((n, d))
    cov = jc.joint_coverage(z, levels=(0.68, 0.95))
    assert abs(cov[0.68] - 0.68) < 0.02
    assert abs(cov[0.95] - 0.95) < 0.01


def test_joint_coverage_detects_overconfidence():
    """Halving the predictive covariance must show up as severe undercoverage."""
    rng = np.random.default_rng(2)
    n, d = 8000, 20
    z = rng.standard_normal((n, d)) / 0.5   # residuals twice as large as claimed
    cov = jc.joint_coverage(z, levels=(0.95,))
    assert cov[0.95] < 0.2


def test_chi2_pit_is_uniform_when_calibrated():
    rng = np.random.default_rng(3)
    z = rng.standard_normal((5000, 20))
    pit = jc.chi2_pit(z)
    assert pit.min() >= 0.0 and pit.max() <= 1.0
    # mean of a Uniform(0,1) sample
    assert abs(pit.mean() - 0.5) < 0.02


def test_whiten_gaussian_marks_bad_rows_nan_without_dropping_good_ones():
    rng = np.random.default_rng(4)
    n, d = 50, 3
    cov = np.tile(_random_spd(d, rng), (n, 1, 1))
    cov[7] = np.nan                       # e.g. an ACOS sounding with no covariance
    cov[9] = np.zeros((d, d)) - 1.0       # not positive definite
    x = rng.standard_normal((n, d))
    z = jc.whiten_gaussian(x, np.zeros((n, d)), cov)

    assert np.isnan(z[7]).all() and np.isnan(z[9]).all()
    good = np.setdiff1d(np.arange(n), [7, 9])
    assert np.isfinite(z[good]).all()


def test_summarize_reports_dropped_rows_and_dimension():
    rng = np.random.default_rng(5)
    z = rng.standard_normal((200, 4))
    z[3] = np.nan
    s = jc.summarize(z, name="test")
    assert s["n"] == 199 and s["n_dropped"] == 1 and s["dim"] == 4
    assert abs(s["mahalanobis_mean"] - 4) < 1.0

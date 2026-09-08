"""test_balanced_sampler.py
Tests for the cluster-balanced training sampler.

The group-disjoint split is stratified by SOM cluster and the clusters are very unequal
(86 training reference soundings in cluster 1 against 12 and 14 in clusters 8 and 9).
`build_balanced_sampler` reweights training draws by n_group ** -alpha so the minority
regimes are not drowned out. These tests pin the weighting arithmetic, the tempering
exponent, and the failure modes -- not the downstream training behaviour.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))

from data.loading import spectral_loader as sl  # noqa: E402


class _FakeDataset:
    """Stands in for SpectralDataset: the sampler only reads `.files`."""

    def __init__(self, clusters):
        self.files = pd.DataFrame({
            "ID": [f"row{i:04d}" for i in range(len(clusters))],
            "clst": clusters,
        })

    def __len__(self):
        return len(self.files)


def _counts(clusters):
    return dict(zip(*np.unique(clusters, return_counts=True)))


# A miniature of the real imbalance: 86 / 12 / 14 reference soundings, one row each.
IMBALANCED = [1] * 86 + [8] * 12 + [9] * 14


def test_alpha_one_equalises_expected_draws_per_cluster():
    ds = _FakeDataset(IMBALANCED)
    sampler = sl.build_balanced_sampler(ds, "clst", alpha=1.0, seed=0)
    w = sampler.weights.numpy()
    clst = ds.files["clst"].to_numpy()
    mass = {c: w[clst == c].sum() for c in (1, 8, 9)}
    # full inverse frequency: every cluster carries the same total probability mass
    assert np.allclose(list(mass.values()), mass[1], rtol=1e-12)


def test_alpha_zero_is_uniform():
    ds = _FakeDataset(IMBALANCED)
    w = sl.build_balanced_sampler(ds, "clst", alpha=0.0, seed=0).weights.numpy()
    assert np.allclose(w, w[0])


def test_alpha_half_is_between_and_matches_sqrt_rule():
    ds = _FakeDataset(IMBALANCED)
    w = sl.build_balanced_sampler(ds, "clst", alpha=0.5, seed=0).weights.numpy()
    clst = ds.files["clst"].to_numpy()
    # per-row weight is n ** -0.5, so the ratio of a cluster-8 row to a cluster-1 row
    # is sqrt(86 / 12)
    ratio = w[clst == 8][0] / w[clst == 1][0]
    assert ratio == pytest.approx(np.sqrt(86 / 12))
    # and the total mass is tempered: cluster 8 gains, but not all the way to parity
    mass = {c: w[clst == c].sum() for c in (1, 8, 9)}
    assert mass[8] < mass[1]
    assert mass[8] / mass[1] > 12 / 86


def test_sampler_draws_minority_rows_more_often():
    ds = _FakeDataset(IMBALANCED)
    sampler = sl.build_balanced_sampler(ds, "clst", alpha=1.0, seed=7)
    drawn = np.array(list(sampler))
    clst = ds.files["clst"].to_numpy()[drawn]
    frac8 = (clst == 8).mean()
    # 12/112 = 10.7% unbalanced; ~1/3 balanced. Loose bound: this is a random draw of 112.
    assert 0.20 < frac8 < 0.47
    assert len(drawn) == len(ds)


def test_num_samples_preserves_epoch_length():
    ds = _FakeDataset(IMBALANCED)
    sampler = sl.build_balanced_sampler(ds, "clst", alpha=0.5, seed=1)
    assert sampler.num_samples == len(ds)
    assert sampler.replacement is True


def test_same_seed_same_draws():
    ds = _FakeDataset(IMBALANCED)
    a = list(sl.build_balanced_sampler(ds, "clst", alpha=0.5, seed=3))
    b = list(sl.build_balanced_sampler(ds, "clst", alpha=0.5, seed=3))
    c = list(sl.build_balanced_sampler(ds, "clst", alpha=0.5, seed=4))
    assert a == b
    assert a != c


def test_rejects_alpha_out_of_range():
    ds = _FakeDataset(IMBALANCED)
    with pytest.raises(ValueError, match=r"alpha"):
        sl.build_balanced_sampler(ds, "clst", alpha=1.5)
    with pytest.raises(ValueError, match=r"alpha"):
        sl.build_balanced_sampler(ds, "clst", alpha=-0.1)


def test_rejects_missing_column():
    ds = _FakeDataset(IMBALANCED)
    with pytest.raises(KeyError, match=r"no column 'template'"):
        sl.build_balanced_sampler(ds, "template", alpha=0.5)


def test_rejects_dataset_without_split_table():
    class NoFiles:
        def __len__(self):
            return 3
    with pytest.raises(TypeError, match=r"split table"):
        sl.build_balanced_sampler(NoFiles(), "clst", alpha=0.5)


def test_train_dataloader_is_unchanged_when_balancing_is_off():
    ds = _FakeDataset(IMBALANCED)
    dl = sl._train_dataloader(ds, batch_size=8, balance_by=None,
                              balance_alpha=0.5, seed=0)
    assert dl.sampler.__class__ is not torch.utils.data.WeightedRandomSampler
    assert dl.batch_size == 8


def test_train_dataloader_uses_the_sampler_when_balancing_is_on():
    ds = _FakeDataset(IMBALANCED)
    dl = sl._train_dataloader(ds, batch_size=8, balance_by="clst",
                              balance_alpha=0.5, seed=0)
    assert isinstance(dl.sampler, torch.utils.data.WeightedRandomSampler)
    assert dl.batch_size == 8

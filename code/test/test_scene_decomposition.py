"""Tests for the pieces the reference-disjoint analysis is built on.

`figures.error_decomposition` is the arithmetic behind the claim the new subsection rests
on -- that under a reference-disjoint split roughly half of the scalar XCO2 error is an
offset the model gets wrong *per scene*, and that the Laplace's reported variance does not
know about it. A decomposition whose parts do not sum to the total would make that
sentence unfalsifiable, so it is pinned here against constructed data where the answer is
known in advance.

The coverage masks are the other half: `score.py` aggregates coverage over every sounding
at once, and the figure needs the same quantity aggregated by held-out scene. Two
definitions of "inside the interval" would be a silent way for the table and the figure to
disagree, so the masks are tested to reproduce the pooled functions exactly.

The cluster bootstrap is tested here too, for the same reason: the reference-disjoint
tables carry its standard errors, and an error bar computed on the wrong resampling unit
is worse than none at all.

Pure numpy/scipy, like test_eval_metrics.py.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "code"))
sys.path.insert(0, str(_ROOT / "experiments" / "scoring"))

from evals import metrics as M                                  # noqa: E402
from evals.artifacts import scene_labels                        # noqa: E402
from figures import error_decomposition                         # noqa: E402


@pytest.fixture
def rng():
    return np.random.default_rng(7)


# ---------------------------------------------------------------------------------
# The decomposition
# ---------------------------------------------------------------------------------
def test_parts_sum_to_the_total(rng):
    """`offset + scene + within == total`, exactly, not to plotting precision."""
    scene = np.repeat([f"1/{i}" for i in range(9)], 40)
    y = rng.normal(400.0, 3.0, size=(len(scene), 5))
    mu = y + rng.normal(0.4, 1.5, size=y.shape)
    d = error_decomposition(y, mu, scene)
    assert d["offset"] + d["scene"] + d["within"] == pytest.approx(d["total"], rel=1e-12)


def test_a_pure_constant_offset_lands_entirely_in_offset():
    scene = np.repeat(["a", "b", "c"], 10)
    y = np.zeros((30, 2))
    d = error_decomposition(y, y + 2.0, scene)
    assert d["offset"] == pytest.approx(4.0)
    assert d["scene"] == pytest.approx(0.0)
    assert d["within"] == pytest.approx(0.0)


def test_a_pure_scene_offset_lands_entirely_in_scene():
    """The case the reference-disjoint split exists to expose.

    Each scene is predicted with its own constant error and no scatter at all. Nothing is
    wrong within any scene; everything is wrong about which scene it is. A replicate-level
    split cannot produce this pattern, because the offset of every test scene was seen in
    training.
    """
    scene = np.repeat(["a", "b", "c", "d"], 25)
    per_scene = {"a": -1.0, "b": 1.0, "c": -3.0, "d": 3.0}   # mean zero, so no offset term
    y = np.zeros((100, 1))
    mu = np.array([[per_scene[g]] for g in scene])
    d = error_decomposition(y, mu, scene)
    assert d["offset"] == pytest.approx(0.0)
    assert d["within"] == pytest.approx(0.0)
    assert d["scene"] == pytest.approx(np.mean([v ** 2 for v in per_scene.values()]))


def test_within_scene_scatter_is_not_attributed_to_the_scene(rng):
    """Unbalanced scenes included: the weighting is by sounding, not by scene."""
    scene = np.array(["a"] * 300 + ["b"] * 40 + ["c"] * 5)
    y = np.zeros((345, 3))
    mu = rng.normal(0.0, 2.0, size=y.shape)
    d = error_decomposition(y, mu, scene)
    # The group means of white noise shrink like 1/n, so almost everything is within.
    assert d["within"] / d["total"] > 0.9
    assert d["offset"] + d["scene"] + d["within"] == pytest.approx(d["total"], rel=1e-12)


def test_unbalanced_groups_keep_the_identity(rng):
    scene = np.repeat(["a", "b"], [90, 10])
    y = rng.normal(size=(100, 2))
    mu = y + np.where(scene[:, None] == "a", 0.5, -4.0) + rng.normal(size=y.shape)
    d = error_decomposition(y, mu, scene)
    assert d["offset"] + d["scene"] + d["within"] == pytest.approx(d["total"], rel=1e-12)


# ---------------------------------------------------------------------------------
# The masks
# ---------------------------------------------------------------------------------
def test_gaussian_mask_averages_to_the_pooled_coverage(rng):
    y = rng.normal(size=(500, 3))
    mu = rng.normal(size=(500, 3)) * 0.1
    var = rng.uniform(0.5, 2.0, size=(500, 3))
    for level in (0.68, 0.95):
        mask = M.gaussian_marginal_coverage_mask(y, mu, var, level)
        assert np.allclose(mask.mean(axis=0),
                           M.gaussian_marginal_coverage(y, mu, var, level))


def test_sample_mask_averages_to_the_pooled_coverage(rng):
    y = rng.normal(size=(200, 2))
    samples = rng.normal(size=(200, 400, 2))
    for level in (0.68, 0.95):
        mask = M.sample_marginal_coverage_mask(y, samples, level)
        assert np.allclose(mask.mean(axis=0),
                           M.sample_marginal_coverage(y, samples, level))


def test_masks_are_per_sounding(rng):
    """Shape `[N, D]`, so coverage can be grouped by anything -- scene, cluster, level."""
    y = rng.normal(size=(37, 4))
    mask = M.gaussian_marginal_coverage_mask(y, np.zeros_like(y), np.ones_like(y), 0.95)
    assert mask.shape == (37, 4)
    assert mask.dtype == bool


# ---------------------------------------------------------------------------------
# Scene labels
# ---------------------------------------------------------------------------------
def test_scene_labels_ignore_the_replicate_index():
    class _Art:
        def has(self, k):
            return k == "ids"

        def __getitem__(self, k):
            return np.array(["clst0001_ref0007_rep0001", "clst0001_ref0007_rep0100",
                             "clst0002_ref0007_rep0001"])

    labs = scene_labels(_Art())
    # Same cluster and reference -> one scene, whatever the replicate.
    assert labs[0] == labs[1]
    # Zero padding is dropped: these are axis labels and dictionary keys.
    assert labs[0] == "1/7"
    # The reference number is only unique inside its cluster, so the cluster is part of
    # the key. Dropping it would silently merge two different scenes.
    assert labs[0] != labs[2]


def test_scene_labels_return_none_when_the_ids_are_not_there():
    class _Art:
        def has(self, k):
            return False

    assert scene_labels(_Art()) is None


# ---------------------------------------------------------------------------------
# The cluster bootstrap
# ---------------------------------------------------------------------------------
def test_group_bootstrap_is_wider_than_the_sounding_bootstrap(rng):
    """The point of the whole exercise, as an inequality.

    100 replicates of one scene sharing a common offset are one observation with noise,
    not 100 observations. A bootstrap that resamples soundings treats them as 100 and
    reports an error bar that is too small by roughly the square root of the replicate
    count; resampling whole scenes does not.
    """
    n_scenes, n_rep = 40, 100
    scene = np.repeat([f"1/{i}" for i in range(n_scenes)], n_rep)
    offsets = rng.normal(0.0, 2.0, size=n_scenes)          # the scene-level signal
    v = np.repeat(offsets, n_rep) + rng.normal(0.0, 0.2, size=n_scenes * n_rep)
    se_sounding = M._boot_se(v, seed=0)
    se_group = M._boot_se(v, seed=0, groups=scene)
    assert se_group > 4 * se_sounding
    # and it is close to the textbook standard error over the 40 scene means
    assert se_group == pytest.approx(offsets.std(ddof=1) / np.sqrt(n_scenes), rel=0.3)


def test_group_bootstrap_reduces_to_the_sounding_one_for_singleton_groups(rng):
    """With one sounding per group the two resamplings are the same procedure."""
    v = rng.normal(size=400)
    groups = np.arange(400).astype(str)
    assert M._boot_se(v, seed=3, groups=groups) == pytest.approx(
        M._boot_se(v, seed=3), rel=0.15)


def test_group_bootstrap_of_the_rmse_takes_the_root_inside(rng):
    """As for the sounding version: the statistic is recomputed per replicate."""
    scene = np.repeat(["a", "b", "c", "d", "e", "f"], 50)
    mse = rng.gamma(2.0, 2.0, size=300)
    se = M._boot_se_rmse(mse, seed=0, groups=scene)
    # An SE of the RMSE is on the scale of the RMSE, not of the MSE. The MSE version is
    # roughly 2 * RMSE times larger, which is the bug this guards against.
    assert 0 < se < np.sqrt(mse.mean())


def test_nan_rows_are_dropped_from_both_the_values_and_the_labels():
    """A singular ACOS covariance drops its row, and its label has to go with it.

    If the labels are not subset the same way, the bootstrap resamples the wrong rows and
    the resulting number is not a standard error of anything.
    """
    scene = np.repeat(["a", "b", "c", "d"], 3)
    v = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, np.nan, 9.0, 10.0, 11.0, 12.0])
    se = M._boot_se(v, seed=0, groups=scene)
    assert np.isfinite(se) and se > 0


def test_group_bootstrap_is_deterministic_given_the_seed(rng):
    scene = np.repeat(["a", "b", "c", "d", "e"], 20)
    v = rng.normal(size=100)
    assert M._boot_se(v, seed=11, groups=scene) == M._boot_se(v, seed=11, groups=scene)

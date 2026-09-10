"""test_density_sampler.py
Tests for the covariate-density training sampler.

`build_balanced_sampler` equalises a categorical column; `scene_density_weights` weights
by the local density of the covariates the model is actually given. The two are different
interventions and the difference is the point of the module, so these tests pin the
properties the reply depends on rather than the downstream training behaviour:

  * an isolated scene gets more weight than a scene in a cluster of neighbours;
  * alpha=0 and beta=0 is exactly the published uniform sampler, to float error;
  * beta=1 makes a scene's total weight independent of its replicate count -- which is the
    property that lets this arm correct ext_v1's 42.5 %-of-rows imbalance without a prep
    job, and the one that would silently stop holding if the row broadcast were reversed;
  * a covariate that varies within a scene is refused rather than averaged, because a
    scene is one point in covariate space and a weight that tracked replicates would be a
    function of the target;
  * the clip is a max/min ratio on SCENE weights and binds where it says it does.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))

from data.loading import density_weights as dw  # noqa: E402
from data.loading import spectral_loader as sl  # noqa: E402

CONT = ("cov_a", "cov_b")
CAT = ("cov_c",)


def _table(scenes, reps=4):
    """One row per replicate. `scenes` is a list of (clst, ref, a, b, c) tuples."""
    rows = []
    for clst, ref, a, b, c in scenes:
        n = reps[(clst, ref)] if isinstance(reps, dict) else reps
        for r in range(n):
            rows.append({"clst": clst, "ref": ref, "rep": r,
                         "cov_a": a, "cov_b": b, "cov_c": c})
    return pd.DataFrame(rows)


# Nine scenes in a tight blob at the origin and one far away on cov_a. The far one is the
# analogue of cluster 9 -- isolated in a covariate the model sees -- and the blob is the
# majority regime.
BLOB = [(1, i, 0.1 * (i % 3), 0.1 * (i // 3), "x") for i in range(9)]
OUTLIER = [(9, 1, 40.0, 0.0, "x")]


class _FakeDataset:
    """Stands in for SpectralDataset: the sampler only reads `.files`."""

    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)


class _Spec:
    continuous = CONT
    categorical = CAT
    name = "test_spec"


def _weights(files, **kw):
    # k=3, not the production 15: these fixtures have ten scenes, and _bandwidth clamps k
    # to n-1, which sets the bandwidth to roughly the distance to the outlier and makes
    # every point look equally dense. The clamp is right -- a k that exceeds the sample is
    # not usable -- but it means a ten-scene fixture has to ask for a local neighbourhood
    # explicitly or it tests nothing.
    kw.setdefault("k", 3)
    return dw.scene_density_weights(files, CONT, CAT, **kw)


def test_an_isolated_scene_outweighs_the_blob():
    files = _table(BLOB + OUTLIER)
    w = _weights(files, alpha=0.5, beta=1.0)
    far = w[files["clst"].to_numpy() == 9]
    near = w[files["clst"].to_numpy() == 1]
    assert far.min() > near.max(), (
        f"the isolated scene got {far.mean():.3f} against the blob's {near.mean():.3f}; "
        f"an inverse-density weighting that does not favour the sparse region is not one"
    )


def test_alpha_zero_beta_zero_is_the_published_sampler():
    files = _table(BLOB + OUTLIER)
    w = _weights(files, alpha=0.0, beta=0.0)
    assert np.allclose(w, 1.0)


def test_alpha_orders_the_tilt():
    files = _table(BLOB + OUTLIER)
    far = files["clst"].to_numpy() == 9
    ratios = []
    for a in (0.25, 0.5, 1.0):
        w = _weights(files, alpha=a, beta=1.0, clip=1e9)
        ratios.append(w[far].mean() / w[~far].mean())
    assert ratios[0] < ratios[1] < ratios[2], ratios


def test_beta_one_makes_a_scene_weight_independent_of_its_replicate_count():
    # The same ten scenes, but the outlier is simulated four times as often -- which is
    # ext_v1's 200-against-100 imbalance in miniature.
    reps = {(c, r): 4 for c, r, *_ in BLOB}
    reps[(9, 1)] = 16
    files = _table(BLOB + OUTLIER, reps=reps)
    w = _weights(files, alpha=0.0, beta=1.0)
    totals = pd.DataFrame({"clst": files["clst"], "ref": files["ref"], "w": w}) \
        .groupby(["clst", "ref"])["w"].sum().to_numpy()
    assert np.allclose(totals, totals[0]), (
        "with beta=1 and no rarity term every scene must carry the same total weight, "
        f"got {totals.min():.4f}..{totals.max():.4f}"
    )


def test_beta_zero_leaves_the_replicate_imbalance_in_place():
    reps = {(c, r): 4 for c, r, *_ in BLOB}
    reps[(9, 1)] = 16
    files = _table(BLOB + OUTLIER, reps=reps)
    w = _weights(files, alpha=0.0, beta=0.0)
    totals = pd.DataFrame({"clst": files["clst"], "ref": files["ref"], "w": w}) \
        .groupby(["clst", "ref"])["w"].sum()
    assert totals.loc[(9, 1)] == pytest.approx(4 * totals.loc[(1, 0)])


def test_weights_are_mean_one():
    files = _table(BLOB + OUTLIER)
    for kw in ({"alpha": 0.5}, {"alpha": 1.0, "beta": 0.0}, {"alpha": 0.25, "beta": 1.0}):
        w = _weights(files, **kw)
        assert w.mean() == pytest.approx(1.0)


def test_clip_bounds_the_scene_weight_ratio_from_above():
    files = _table(BLOB + OUTLIER)
    # Equal replicate counts here, so the row ratio is the scene ratio.
    unclipped = _weights(files, alpha=1.0, beta=1.0, clip=1e9)
    clipped = _weights(files, alpha=1.0, beta=1.0, clip=2.0)
    r_unclipped = unclipped.max() / unclipped.min()
    r_clipped = clipped.max() / clipped.min()
    assert r_unclipped > 2.0, "the fixture must be skewed enough for the clip to bite"
    # An upper bound, not an equality: the weights are clamped a factor sqrt(clip) either
    # side of their geometric mean, and a skewed set need not saturate both ends.
    assert r_clipped <= 2.0 + 1e-9
    assert r_clipped < r_unclipped


def test_a_within_scene_varying_covariate_is_refused():
    files = _table(BLOB + OUTLIER)
    files.loc[0, "cov_a"] = 999.0  # one replicate of scene (1, 0) moved
    with pytest.raises(ValueError, match="vary within a scene"):
        _weights(files, alpha=0.5)


def test_the_listed_exception_is_allowed_and_reduced_by_median():
    files = _table(BLOB + OUTLIER)
    files["prior_psfc_pa"] = 1000.0
    files.loc[files.index[:2], "prior_psfc_pa"] = [900.0, 1100.0]
    # Listed in _WITHIN_SCENE_VARYING, so it passes the constancy check.
    w = dw.scene_density_weights(files, tuple(CONT) + ("prior_psfc_pa",), CAT, alpha=0.5)
    assert np.all(np.isfinite(w))


def test_ratio_mode_tilts_toward_the_target_covariates():
    train = _table(BLOB + OUTLIER)
    # A target made only of the outlier's neighbourhood: the outlier must gain.
    target = _table([(9, 2, 40.0, 0.0, "x"), (9, 3, 40.5, 0.1, "x")])
    w = _weights(train, alpha=1.0, beta=1.0, mode="ratio", target_files=target,
                 clip=1e9)
    far = train["clst"].to_numpy() == 9
    assert w[far].mean() > w[~far].mean()


def test_ratio_mode_needs_a_target():
    files = _table(BLOB + OUTLIER)
    with pytest.raises(ValueError, match="target_files"):
        _weights(files, mode="ratio")


@pytest.mark.parametrize("kw", [{"alpha": 1.5}, {"alpha": -0.1}, {"beta": 2.0},
                                {"clip": 0.5}, {"mode": "nope"}])
def test_out_of_range_settings_are_refused(kw):
    files = _table(BLOB + OUTLIER)
    with pytest.raises(ValueError):
        _weights(files, **kw)


def test_a_missing_covariate_column_names_itself():
    files = _table(BLOB + OUTLIER)
    with pytest.raises(KeyError, match="nosuch"):
        dw.scene_density_weights(files, ("nosuch",), CAT)


# ---------------------------------------------------------------- the loader-side adapter


def test_sampler_is_deterministic_in_the_seed():
    ds = _FakeDataset(_table(BLOB + OUTLIER))
    a = list(sl.build_density_sampler(ds, _Spec(), seed=3))
    b = list(sl.build_density_sampler(ds, _Spec(), seed=3))
    c = list(sl.build_density_sampler(ds, _Spec(), seed=4))
    assert a == b
    assert a != c


def test_sampler_draws_the_epoch_length_it_was_given():
    files = _table(BLOB + OUTLIER)
    ds = _FakeDataset(files)
    sampler = sl.build_density_sampler(ds, _Spec(), alpha=0.5)
    assert len(list(sampler)) == len(files)


def test_density_needs_a_covariate_spec():
    ds = _FakeDataset(_table(BLOB + OUTLIER))
    with pytest.raises(ValueError, match="covariate preset"):
        sl.build_density_sampler(ds, None)


def test_density_needs_a_split_table():
    class NoFiles:
        def __len__(self):
            return 0

    with pytest.raises(TypeError, match="split table"):
        sl.build_density_sampler(NoFiles(), _Spec())


def test_a_split_table_column_called_density_is_refused():
    files = _table(BLOB + OUTLIER)
    files[sl.DENSITY_KEY] = 1.0
    with pytest.raises(KeyError, match="reserved"):
        sl.build_density_sampler(_FakeDataset(files), _Spec())


def test_train_dataloader_dispatches_on_the_reserved_value():
    files = _table(BLOB + OUTLIER)
    ds = _FakeDataset(files)
    ds.__getitem__ = lambda self, i: i  # never called; the loader is not iterated here
    dl = sl._train_dataloader(ds, 4, sl.DENSITY_KEY, 0.5, 42, cov_spec=_Spec())
    assert isinstance(dl.sampler, torch.utils.data.WeightedRandomSampler)
    plain = sl._train_dataloader(ds, 4, None, 0.5, 42)
    assert not isinstance(plain.sampler, torch.utils.data.WeightedRandomSampler)


def test_density_settings_reads_a_plain_dict_and_fills_defaults():
    got = sl.density_settings({})
    assert got == {"mode": "inverse", "beta": 1.0, "k": 15, "clip": 10.0}
    got = sl.density_settings({"density_mode": "ratio", "density_k": 7})
    assert got["mode"] == "ratio" and got["k"] == 7


def test_density_settings_refuses_an_unknown_mode_before_the_gpu_is_spent():
    with pytest.raises(ValueError, match="density_mode"):
        sl.density_settings({"density_mode": "kde"})

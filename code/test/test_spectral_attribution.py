"""Tests for the pieces the spectral attribution rests on.

Three of them carry the argument and would fail silently if they were wrong:

* `occlude` has to replace the radiance of the named window and nothing else. If it also
  touched the wavelength channel or the scale constants, the "importance" of a window would
  include the network losing track of which band it is looking at, and the curve would be
  uninterpretable. If it mutated its input, every configuration after the first would be
  scored against an already-occluded spectrum and the whole sweep would be monotone by
  construction.

* `continuum` / absorption depth is the physics axis of the correlation statistic. It is
  estimated with no line list, so it has to be checked against a spectrum whose lines are
  known by construction: the depth must be large at the lines and near zero between them.

* `bootstrap_rmse` resamples reference soundings, not soundings. Replicates of one
  reference sounding share a state and a geometry, so the sounding-level bootstrap would
  report a standard error several times too small -- the same error the paper's own tables
  avoid, tested here so this script cannot reintroduce it.

`window_edges` and `n_real_channels` are cheap and pinned because an off-by-one in either
would shift every point of the curve against the wavelength axis.

torch is imported by the module under test (through predict.py). It is present in the
container the jobs run in and absent on the laptop, so the whole module skips rather than
erroring at collection.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "code"), str(_ROOT / "experiments" / "scoring")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import spectral_attribution as SA                                # noqa: E402


# --------------------------------------------------------------------------------------
# occlusion
# --------------------------------------------------------------------------------------
def _bands(n=4, length=64, real=50, seed=0):
    """A batch shaped like the loader's: {band: ([B, 2, L], [B, 2])}, zero-padded."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for j, b in enumerate(SA.BANDS):
        t = torch.zeros(n, 2, length)
        t[:, 0, :real] = torch.linspace(0.76 + j, 0.77 + j, real)          # wavelength
        t[:, 1, :real] = torch.rand(n, real, generator=g)                  # radiance
        out[b] = (t, torch.rand(n, 2, generator=g))
    return out


def test_occlude_replaces_only_the_named_window_of_the_radiance_channel():
    bands = _bands()
    mean = {b: torch.full((64,), -1.0) for b in SA.BANDS}
    before = {b: t.clone() for b, (t, _) in bands.items()}

    out = SA.occlude(bands, mean, [("b2", 10, 20)])

    # the window, and only the window, and only in the radiance channel
    assert torch.equal(out["b2"][0][:, SA.RAD, 10:20], mean["b2"][10:20].expand(4, 10))
    assert torch.equal(out["b2"][0][:, SA.RAD, :10], before["b2"][:, SA.RAD, :10])
    assert torch.equal(out["b2"][0][:, SA.RAD, 20:], before["b2"][:, SA.RAD, 20:])
    assert torch.equal(out["b2"][0][:, 0, :], before["b2"][:, 0, :])
    for b in ("b1", "b3"):
        assert torch.equal(out[b][0], before[b])
    # the scale constants pass through by reference, unchanged
    for b in SA.BANDS:
        assert out[b][1] is bands[b][1]


def test_occlude_does_not_mutate_its_input():
    bands = _bands(seed=1)
    keep = {b: t.clone() for b, (t, _) in bands.items()}
    SA.occlude(bands, {b: torch.zeros(64) for b in SA.BANDS},
               [(b, 0, 50) for b in SA.BANDS])
    for b in SA.BANDS:
        assert torch.equal(bands[b][0], keep[b]), f"{b} was written to in place"


def test_occlude_with_an_empty_spec_is_the_identity():
    bands = _bands(seed=2)
    out = SA.occlude(bands, {b: torch.zeros(64) for b in SA.BANDS}, [])
    assert out is bands


def test_n_real_channels_counts_the_unpadded_length():
    batches = [((_bands(length=64, real=37), None, None), torch.zeros(4, 1), None)]
    assert SA.n_real_channels(batches) == {b: 37 for b in SA.BANDS}


# --------------------------------------------------------------------------------------
# the physics axis
# --------------------------------------------------------------------------------------
def _synthetic(n=600, centres=(80, 200, 340, 500), slope=0.0, sigma=3.0, strength=0.5):
    """A flat-or-sloping continuum with Gaussian absorption lines at `centres`."""
    x = np.arange(n)
    base = 1.0 - slope * x / n
    rad = base.copy()
    for c in centres:
        rad -= strength * base[c] * np.exp(-0.5 * ((x - c) / sigma) ** 2)
    away = np.ones(n, dtype=bool)
    for c in centres:
        away[max(0, c - 5 * int(sigma)):c + 5 * int(sigma)] = False
    return x, base, rad, away


def test_continuum_finds_the_lines_and_not_the_spaces_between_them():
    centres = (80, 200, 340, 500)
    _x, base, rad, away = _synthetic(centres=centres)

    cont = SA.continuum(rad, width=160, q=0.9)
    depth = 1.0 - rad / cont

    # a flat continuum is recovered exactly enough that the depth away from the lines is
    # numerically zero rather than merely small
    assert np.allclose(cont[away], base[away], atol=0.01)
    assert np.abs(depth[away]).max() < 0.02
    # and each line is found at close to its constructed strength
    for c in centres:
        assert depth[c] > 0.4


def test_continuum_tracks_a_sloping_baseline():
    """The estimator is biased upward on a slope, by about a decile of the window's rise.

    Worth pinning rather than hiding: the bias is what sets the floor on the absorption
    depth of a line-free window, and it grows with --continuum-width. Over a 160-channel
    window of a baseline that falls 0.2 over 600 channels the rise across the window is
    0.053, so a bias of a few hundredths is expected and anything larger would mean the
    quantile is picking up the wrong part of the window.
    """
    _x, base, rad, away = _synthetic(slope=0.2)
    cont = SA.continuum(rad, width=160, q=0.9)
    err = cont[away] - base[away]
    assert err.max() < 0.05
    assert err.min() > -0.02


def test_continuum_of_a_flat_spectrum_is_that_spectrum():
    rad = np.full(300, 0.7)
    cont = SA.continuum(rad, width=64, q=0.9)
    assert np.allclose(cont, 0.7)
    assert np.allclose(1.0 - rad / cont, 0.0)


def test_continuum_has_no_nan_at_the_edges():
    rad = np.linspace(1.0, 0.5, 100)
    cont = SA.continuum(rad, width=256, q=0.9)      # window wider than the array
    assert np.isfinite(cont).all()


# --------------------------------------------------------------------------------------
# windows
# --------------------------------------------------------------------------------------
def test_non_overlapping_windows_tile_the_band_exactly_once():
    n, w = 835, 32
    edges = SA.window_edges(n, w, w)
    covered = np.zeros(n, dtype=int)
    for lo, hi in edges:
        covered[lo:hi] += 1
    assert (covered == 1).all()
    assert edges[0][0] == 0 and edges[-1][1] == n


def test_a_stride_below_the_width_overlaps_and_still_stops_at_the_end():
    edges = SA.window_edges(100, 20, 10)
    assert all(0 <= lo < hi <= 100 for lo, hi in edges)
    assert edges[0][0] == 0
    assert edges[-1][1] == 100
    assert edges[1][0] == 10                      # stride, not width
    assert edges[-1] == (90, 100)                 # the last window is short, not clipped off


# --------------------------------------------------------------------------------------
# the bootstrap
# --------------------------------------------------------------------------------------
def test_bootstrap_rmse_reproduces_the_pooled_number():
    rng = np.random.default_rng(3)
    se = rng.chisquare(2, size=500)
    groups = np.repeat(np.arange(50), 10)
    rmse, _, _ = SA.bootstrap_rmse(se, groups, ppm=3.142, n_boot=50, seed=0)
    assert rmse == pytest.approx(np.sqrt(se.mean()) * 3.142)


def test_the_paired_delta_error_is_far_tighter_than_the_marginal_one():
    """The reason `d_*_se` exists, pinned.

    An occlusion adds a small constant to every sounding's squared error. The RMSE's own
    bootstrap error is dominated by which scenes were drawn -- here the scene means differ
    a lot -- and that variation is identical in both terms of the difference, so it cancels
    on a shared resample. Reporting the marginal error beside a delta column would hide a
    real effect behind an error bar that has nothing to do with it.
    """
    rng = np.random.default_rng(11)
    n_scene, n_rep = 40, 100
    scene_level = rng.gamma(2.0, 2.0, size=n_scene)          # scenes differ strongly
    base = np.repeat(scene_level, n_rep) + rng.chisquare(1, size=n_scene * n_rep)
    occluded = base + 0.5                                    # the same shift everywhere
    groups = np.repeat(np.arange(n_scene), n_rep)

    _, se_marg, se_paired = SA.bootstrap_rmse(
        occluded, groups, ppm=1.0, n_boot=400, seed=0, base_se=base)

    assert np.isfinite(se_paired)
    assert se_paired < se_marg / 5


def test_the_delta_error_is_nan_when_no_baseline_is_given():
    _, se, dse = SA.bootstrap_rmse(np.ones(100), np.repeat(np.arange(10), 10),
                                   ppm=1.0, n_boot=20, seed=0)
    assert np.isfinite(se) or se == 0.0
    assert np.isnan(dse)


def test_bootstrap_mean_reproduces_the_pooled_mean_and_pairs_its_delta():
    rng = np.random.default_rng(5)
    n_scene, n_rep = 30, 100
    base = np.repeat(rng.normal(0, 3, size=n_scene), n_rep) + rng.normal(0, 1, n_scene * n_rep)
    shifted = base + 0.25
    groups = np.repeat(np.arange(n_scene), n_rep)

    m, se_marg, se_paired = SA.bootstrap_mean(shifted, groups, n_boot=400, seed=0, base=base)
    assert m == pytest.approx(shifted.mean())
    # the between-scene spread cancels in the difference, as for the RMSE
    assert se_paired < se_marg / 5


def test_bootstrap_rmse_resamples_scenes_not_soundings():
    """One extreme scene has to be able to leave the resample, and be able to repeat.

    Constructed so the two units disagree by construction: 20 scenes of 50 replicates, one
    of them carrying a hundred times the squared error of the others. Resampling scenes
    either draws that scene or does not, so the standard error is of the order of its
    contribution; resampling soundings would keep about 50 of its rows in every draw and
    report a far smaller number.
    """
    n_scene, n_rep = 20, 50
    se = np.ones(n_scene * n_rep)
    se[:n_rep] = 100.0
    groups = np.repeat(np.arange(n_scene), n_rep)

    _, se_scene, _ = SA.bootstrap_rmse(se, groups, ppm=1.0, n_boot=600, seed=0)
    _, se_sounding, _ = SA.bootstrap_rmse(se, np.arange(se.size), ppm=1.0,
                                          n_boot=600, seed=0)
    assert se_scene > 4 * se_sounding


def test_bootstrap_rmse_without_groups_reports_no_standard_error():
    rmse, se, dse = SA.bootstrap_rmse(np.ones(10), None, ppm=2.0, n_boot=100, seed=0)
    assert rmse == pytest.approx(2.0)
    assert np.isnan(se)
    assert np.isnan(dse)

"""test_characterize_discrepancy.py
Tests for the model-discrepancy recovery in experiments/pipelines/characterize_discrepancy.py.

The claims the script is used to make -- that delta is constant
within blocks of ten replicates, that its covariance is low rank, that it is drawn
independently of the state, and that it explains most of the ACOS XCO2 error variance --
all rest on the block detection and the variance decomposition being right. Here they are
exercised against synthetic data whose delta structure is known by construction.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))


def _load_module(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cd = _load_module("experiments/pipelines/characterize_discrepancy.py",
                  "characterize_discrepancy")

TEST_H5 = REPO / "code/test/data/lnd_nadir_202002_clst01_fp_aggregate_retrieval_ref46.h5"


# ---------------------------------------------------------------- fixtures --
def synth(n_rep=100, block=10, n_chan=400, mu_scale=0.7, delta_scale=1.0, seed=0):
    """Whitened (delta + eps): eps is standard normal, delta is constant within blocks."""
    rng = np.random.default_rng(seed)
    mu = mu_scale * np.sin(np.linspace(0, 6, n_chan))
    shape = np.cos(np.linspace(0, 4, n_chan))
    shape /= np.linalg.norm(shape) / np.sqrt(n_chan)
    draws = mu + delta_scale * rng.standard_normal((n_rep // block, 1)) * shape
    z = np.repeat(draws, block, axis=0) + rng.standard_normal((n_rep, n_chan))
    return z, mu, draws


# ------------------------------------------------------------ block detection --
@pytest.mark.parametrize("block", [2, 5, 10, 20])
def test_detects_the_block_size_used_to_generate_the_data(block):
    z, _, _ = synth(block=block, seed=block)
    detected, _ = cd.detect_block_size(z)
    assert detected == block


def test_reports_no_blocking_when_delta_is_redrawn_every_replicate():
    rng = np.random.default_rng(1)
    n_rep, n_chan = 100, 400
    shape = np.cos(np.linspace(0, 4, n_chan))
    z = rng.standard_normal((n_rep, 1)) * shape + rng.standard_normal((n_rep, n_chan))
    detected, _ = cd.detect_block_size(z)
    assert detected == 1


def test_reports_no_blocking_for_pure_noise():
    z = np.random.default_rng(2).standard_normal((100, 400))
    detected, scores = cd.detect_block_size(z)
    assert detected == 100          # a constant (here zero) delta for the whole sounding
    assert scores[10] == pytest.approx(1.0, abs=0.05)


def test_block_means_recover_the_delta_draws():
    z, _, draws = synth(seed=3)
    recovered = cd.delta_draws(z, 10)
    assert recovered.shape == draws.shape
    # each block mean carries noise of sd 1/sqrt(10) ~ 0.32 per channel
    assert np.abs(recovered - draws).mean() < 0.35


# ------------------------------------------------------------------ moments --
def test_recovers_the_mean_discrepancy():
    z, mu, _ = synth(mu_scale=0.7, seed=4)
    mu_hat = cd.delta_draws(z, 10).mean(axis=0)
    assert np.corrcoef(mu_hat, mu)[0, 1] > 0.9
    assert np.sqrt((mu_hat ** 2).mean()) == pytest.approx(
        np.sqrt((mu ** 2).mean()), rel=0.5)


def test_rank_one_covariance_is_reported_as_rank_one():
    z, _, _ = synth(delta_scale=3.0, seed=5)
    summary = cd.eigen_summary(cd.delta_draws(z, 10))
    assert summary["n_pc_for_90pct"] == 1
    assert summary["variance_share_top5"][0] > 0.8


def test_full_rank_covariance_is_not_reported_as_rank_one():
    rng = np.random.default_rng(6)
    draws = rng.standard_normal((40, 400)) * 3.0
    summary = cd.eigen_summary(draws)
    assert summary["n_pc_for_90pct"] > 5


# --------------------------------------------------------- state dependence --
def test_permutation_null_is_calibrated_for_state_independent_delta():
    """The flag fires at its nominal 5% rate, not more, when delta ignores the state."""
    fired = 0
    for seed in range(30):
        rng = np.random.default_rng(1000 + seed)
        z, _, _ = synth(seed=seed)
        x = rng.standard_normal((100, 49))
        info, cors, n = cd.state_dependence(cd.delta_draws(z, 10), x, 10)
        fired += info["exceeds_permutation_null"]
        assert cors.shape == (49,)
        assert n == 10
    assert fired <= 5          # 5% nominal over 30 trials


def test_state_dependent_delta_is_detected_when_pooled():
    """One reference sounding cannot resolve it; ninety of them can."""
    cors, ns, rng = [], [], np.random.default_rng(8)
    n_chan = 400
    shape = np.cos(np.linspace(0, 4, n_chan))
    for k in range(90):
        x = rng.standard_normal((100, 49))
        driver = x[:, 3].reshape(10, 10).mean(axis=1)          # delta follows state 4
        draws = driver[:, None] * shape * 3.0
        z = np.repeat(draws, 10, axis=0) + rng.standard_normal((100, n_chan))
        _, c, n = cd.state_dependence(cd.delta_draws(z, 10), x, 10)
        cors.append(c)
        ns.append(n)
    pooled = cd.pooled_state_dependence(cors, ns)
    assert pooled["n_pooled_delta_draws"] == 900
    assert not pooled["delta_appears_independent_of_state"]
    assert pooled["argmin_state_element"] == 4


def test_pooled_test_stays_null_when_delta_is_independent():
    scores, states, rng = [], [], np.random.default_rng(9)
    for k in range(90):
        z, _, _ = synth(seed=100 + k)
        x = rng.standard_normal((100, 49))
        _, c, n = cd.state_dependence(cd.delta_draws(z, 10), x, 10)
        scores.append(c)
        states.append(n)
    pooled = cd.pooled_state_dependence(scores, states)
    assert pooled["delta_appears_independent_of_state"]


# ------------------------------------------------------- error decomposition --
def test_error_decomposition_splits_between_and_within_block_variance():
    rng = np.random.default_rng(10)
    block, nb = 10, 10
    per_block = rng.standard_normal(nb) * 2.0        # delta-driven, sd 2 ppm
    err = (np.repeat(per_block, block)
           + rng.standard_normal(nb * block) * 0.5)  # state/noise, sd 0.5 ppm
    out = cd.error_decomposition(np.zeros(100), err * 1e-6, np.ones(100, dtype=int), block)
    assert out["sd_between_delta_blocks_corrected_ppm"] == pytest.approx(2.0, rel=0.5)
    assert out["sd_within_delta_block_ppm"] == pytest.approx(0.5, rel=0.4)
    assert out["share_of_error_variance_from_delta"] > 0.8


def test_error_decomposition_attributes_nothing_to_delta_when_blocks_agree():
    rng = np.random.default_rng(11)
    err = rng.standard_normal(100) * 0.5
    out = cd.error_decomposition(np.zeros(100), err * 1e-6, np.ones(100, dtype=int), 10)
    assert out["share_of_error_variance_from_delta"] < 0.4


# ------------------------------------------------------------- the real file --
@pytest.mark.skipif(not TEST_H5.exists(), reason="sample simulation file not present")
def test_committed_sample_file_has_ten_replicates_per_delta_draw():
    d = cd.read_file(TEST_H5)
    assert d is not None
    res = cd.analyse_file(d)
    assert res["replicates_per_delta_draw"] == 10
    assert res["n_distinct_delta_draws"] == 10
    # removing the per-block offset must leave exactly the reported instrument noise
    assert float(res["block_detection_within_sd"]["10"]) == pytest.approx(1.0, abs=0.02)
    # and delta must be a real effect, not a rounding artefact
    assert res["mu_delta_rms_over_sigma_eps_all_bands"] > 0.3
    assert res["sigma_delta"]["n_pc_for_90pct"] == 1

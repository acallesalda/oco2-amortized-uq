"""Tests for evals.gaussianize -- the moment-matched Gaussian control.

The streaming reduction is the part of this experiment with no visible output: it consumes
samples that are never stored and emits moments that look plausible whatever they are. A
transposed axis, a central-moment formula off by a term, or an accumulator that loses its
significant digits to cancellation would all produce a table rather than a crash. So the
central test does not check statistics at all -- it replays a *known* array of samples
through the accumulator and demands the same numbers numpy gets from that array directly.

Run:  python3 -m pytest code/test/test_gaussianize.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from evals import gaussianize as G          # noqa: E402
from evals import metrics as M              # noqa: E402


class ReplayFlow:
    """A `flow` whose samples are known in advance: successive chunks of a fixed pool.

    `stream_batch` only ever calls `sample(x_tuple, n=...)` and expects `[n, B, D]`, which
    is the whole interface the three real flow wrappers share.
    """

    def __init__(self, pool):
        self.pool = pool          # [S, B, D]
        self.pos = 0

    def sample(self, x_tuple, n=100):
        if self.pos + n > self.pool.shape[0]:
            raise AssertionError(
                f"asked for {n} samples at position {self.pos}, pool holds "
                f"{self.pool.shape[0]}"
            )
        out = self.pool[self.pos:self.pos + n]
        self.pos += n
        return out


def _pool(rng, s, b, d, loc=0.0, scale=1.0):
    return torch.as_tensor(rng.normal(loc, scale, size=(s, b, d)), dtype=torch.float64)


# ---------------------------------------------------------------------------------
# the schedule and the arithmetic around it
# ---------------------------------------------------------------------------------
def test_stage_schedule_doubles_and_ends_at_total():
    assert G.stage_schedule(8192) == [512, 1024, 2048, 4096, 8192]
    assert G.stage_schedule(1000) == [512, 1000]
    assert G.stage_schedule(512) == [512]
    assert G.stage_schedule(100) == [100]
    assert G.stage_schedule(6000, first=1000) == [1000, 2000, 4000, 6000]


def test_plugin_penalty_matches_parameter_count():
    # d means and d(d+1)/2 covariance entries, each costing 1/(2S).
    for d, s in ((1, 512), (4, 2048), (20, 512)):
        p = d + d * (d + 1) / 2
        assert G.plugin_penalty(d, s) == pytest.approx(p / (2 * s))
    assert G.plugin_penalty(20, 512) == pytest.approx(0.2246, abs=1e-4)


def test_extrapolate_recovers_a_known_curve():
    stages = [512, 1024, 2048, 4096, 8192]
    a, b = -3.25, 115.0
    nll = np.array([a + b / s for s in stages])
    fit = G.extrapolate(stages, nll)
    assert fit["a"] == pytest.approx(a, abs=1e-9)
    assert fit["b"] == pytest.approx(b, abs=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)


def test_extrapolate_survives_a_nan_stage():
    stages = [512, 1024, 2048]
    fit = G.extrapolate(stages, np.array([1.0, np.nan, 0.5]))
    assert np.isfinite(fit["a"]) and np.isfinite(fit["b"])


# ---------------------------------------------------------------------------------
# the reduction, against numpy on the same samples
# ---------------------------------------------------------------------------------
@pytest.mark.parametrize("d", [1, 4, 20])
def test_stream_batch_reproduces_numpy_on_the_same_samples(d):
    rng = np.random.default_rng(0)
    s, b = 2048, 6
    pool = _pool(rng, s, b, d)
    y = torch.as_tensor(rng.normal(size=(b, d)), dtype=torch.float64)

    stages = G.stage_schedule(s)
    got = G.stream_batch(ReplayFlow(pool), None, y, s, chunk=256, stages=stages)

    x = pool.numpy()                                   # [S,B,D]
    for k, stop in enumerate(stages):
        cut = x[:stop]
        assert got["mu"][:, k, :] == pytest.approx(cut.mean(axis=0), abs=1e-10)
        var = cut.var(axis=0, ddof=1)
        assert got["var"][:, k, :] == pytest.approx(var, rel=1e-10)
        pit = (cut <= y.numpy()[None]).mean(axis=0)
        assert got["pit"][:, k, :] == pytest.approx(pit, abs=1e-12)

    # Full covariance and the higher moments, at the final stage.
    cov = np.stack([np.cov(x[:, i, :], rowvar=False).reshape(d, d) for i in range(b)])
    assert got["cov"] == pytest.approx(cov, rel=1e-9, abs=1e-12)

    c = x - x.mean(axis=0, keepdims=True)
    m2 = (c ** 2).mean(axis=0)
    assert got["skew"] == pytest.approx((c ** 3).mean(axis=0) / m2 ** 1.5, rel=1e-8)
    assert got["exkurt"] == pytest.approx((c ** 4).mean(axis=0) / m2 ** 2 - 3.0, rel=1e-8)


def test_stream_batch_nll_matches_metrics_mvn_nll():
    """The in-job torch NLL and the CPU numpy one must be the same number.

    They are computed by independent code -- `gaussianize.mvn_nll_torch` inside the
    sampling pass, `metrics.mvn_nll` from the stored covariance in `nongauss.py` -- and
    the table prints both so that a disagreement is visible. This is that check.
    """
    rng = np.random.default_rng(1)
    s, b, d = 1024, 5, 4
    pool = _pool(rng, s, b, d, scale=0.7)
    y = torch.as_tensor(rng.normal(size=(b, d)), dtype=torch.float64)

    stages = G.stage_schedule(s)
    got = G.stream_batch(ReplayFlow(pool), None, y, s, chunk=512, stages=stages)

    want = M.mvn_nll(y.numpy(), got["mu"][:, -1, :], got["cov"])
    assert got["nll"][:, -1] == pytest.approx(want, abs=1e-9)


def test_higher_moments_survive_a_large_offset():
    """The reason the accumulator centres on the first chunk's mean before summing.

    Fourth moments about the origin of samples living at 1e4 with spread 1 have no
    significant digits left; about their own mean they have all of them.
    """
    rng = np.random.default_rng(2)
    s, b, d = 4096, 3, 2
    pool = _pool(rng, s, b, d, loc=1e4)
    y = torch.full((b, d), 1e4, dtype=torch.float64)

    got = G.stream_batch(ReplayFlow(pool), None, y, s, chunk=1024,
                         stages=G.stage_schedule(s))
    assert got["skew"] == pytest.approx(0.0, abs=0.12)
    assert got["exkurt"] == pytest.approx(0.0, abs=0.25)
    assert got["var"][:, -1, :] == pytest.approx(1.0, rel=0.1)


def test_m_idx_slices_the_scored_coordinates():
    """`last4_marginal`: the model emits 20 levels and 4 are scored."""
    rng = np.random.default_rng(3)
    s, b = 512, 4
    pool = _pool(rng, s, b, 20)
    keep = [16, 17, 18, 19]
    y = torch.as_tensor(rng.normal(size=(b, 4)), dtype=torch.float64)

    got = G.stream_batch(ReplayFlow(pool), None, y, s, chunk=256, stages=[512],
                         m_idx=torch.as_tensor(keep, dtype=torch.long))
    assert got["mu"].shape == (b, 1, 4)
    assert got["cov"].shape == (b, 4, 4)
    assert got["mu"][:, 0, :] == pytest.approx(
        pool.numpy()[:, :, keep].mean(axis=0), abs=1e-10)


def test_wrong_sample_axes_are_rejected():
    """A `[B, n, D]` flow would concatenate cleanly and transpose the whole table."""
    rng = np.random.default_rng(4)
    b, d, s = 4, 3, 512
    transposed = torch.as_tensor(rng.normal(size=(b, s, d)), dtype=torch.float64)

    class Bad:
        def sample(self, x_tuple, n=100):
            return transposed[:, :n, :]

    with pytest.raises(RuntimeError, match="transposed"):
        G.stream_batch(Bad(), None, torch.zeros(b, d, dtype=torch.float64), s,
                       chunk=512, stages=[512])


def test_stages_must_end_at_total():
    with pytest.raises(ValueError):
        G.stream_batch(ReplayFlow(torch.zeros(8, 2, 1)), None,
                       torch.zeros(2, 1), 8, chunk=4, stages=[4])


# ---------------------------------------------------------------------------------
# the statistic the experiment reports
# ---------------------------------------------------------------------------------
def test_control_recovers_the_truth_for_a_gaussian_flow():
    """A Gaussian predictive scored under its own moment-matched Gaussian.

    The shape gap must vanish: the control *is* the predictive here, so anything left is
    the plug-in penalty, and it must sit near d(d+3)/(4S) rather than anywhere else. This
    is the null case the reported gap is measured against.
    """
    rng = np.random.default_rng(5)
    b, d, s = 128, 4, 4096
    mu = rng.normal(size=(b, d))
    a = rng.normal(size=(b, d, d)) * 0.3
    cov = a @ np.swapaxes(a, -1, -2) + np.eye(d)
    chol = np.linalg.cholesky(cov)

    z = rng.normal(size=(s, b, d))
    pool = torch.as_tensor(mu[None] + np.einsum("bde,sbe->sbd", chol, z),
                           dtype=torch.float64)
    zt = rng.normal(size=(b, d))
    y = torch.as_tensor(mu + np.einsum("bde,be->bd", chol, zt), dtype=torch.float64)

    stages = G.stage_schedule(s)
    got = G.stream_batch(ReplayFlow(pool), None, y, s, chunk=1024, stages=stages)

    exact = M.mvn_nll(y.numpy(), mu, cov)                    # the flow's own density
    per_stage = got["nll"].mean(axis=0)
    gap = per_stage - exact.mean()

    # Falls with the budget, and lands within a factor of two of the predicted penalty at
    # the smallest stage -- 128 soundings is not enough to pin it tighter than that.
    assert 0 < gap[-1] < gap[0]
    assert gap[0] == pytest.approx(G.plugin_penalty(d, stages[0]), rel=1.0)

    fit = G.extrapolate(stages, per_stage)
    assert fit["a"] == pytest.approx(exact.mean(), abs=0.05)

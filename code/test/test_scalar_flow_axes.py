"""Pins the sample-axis convention of the scalar flow.

`nflows.Flow.sample(num_samples=n, context=ctx)` returns `[context_size, num_samples,
features]` -- batch first. `NFHead.sample` transposes it to `[n, B]`, which is what
`ScalarFlow.sample` and `predict.py` both assume.

That assumption failed silently for as long as every batch was full: the per-batch
arrays only disagree in shape on the short final batch, so the symptom was a
`np.concatenate` error at the end of `eval_xco2` rather than anything pointing at the
flow. Had the test set divided evenly by the batch size, the samples of sounding *i*
would have been reported as sounding *j* and the table would have looked plausible.

Hence a test on the axes rather than on any number.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.checkpoints import ScalarFlow  # noqa: E402


class _Encoder(torch.nn.Module):
    """Stands in for FullRegressorProfiles.encoder: x_tuple -> [B, C]."""

    def __init__(self, ctx_dim=7):
        super().__init__()
        self.ctx_dim = ctx_dim

    def forward(self, x_tuple):
        return torch.zeros(x_tuple.shape[0], self.ctx_dim)


class _Backbone(torch.nn.Module):
    def __init__(self, ctx_dim=7):
        super().__init__()
        self.encoder = _Encoder(ctx_dim)


class _NF(torch.nn.Module):
    """NFHead's contract, without nflows: `sample(n, context) -> [n, B]`."""

    def __init__(self, transposed=False):
        super().__init__()
        self.transposed = transposed

    def sample(self, n, context):
        b = context.shape[0]
        # The values encode their own (sample, batch) index so a transposition is
        # detectable from the contents, not just the shape.
        s = torch.arange(n, dtype=torch.float32).unsqueeze(1) * 1000.0 \
            + torch.arange(b, dtype=torch.float32).unsqueeze(0)      # [n, B]
        return s.t().contiguous() if self.transposed else s


@pytest.mark.parametrize("batch", [16, 9, 1])
def test_sample_is_n_by_batch(batch):
    """`ScalarFlow.sample` -> `[n, B, 1]` for any batch size, including a short one."""
    flow = ScalarFlow(_Backbone(), _NF())
    n = 32
    s = flow.sample(torch.zeros(batch, 3), n=n)
    assert tuple(s.shape) == (n, batch, 1)
    # Sample index varies down axis 0, sounding index across axis 1.
    assert s[5, 0, 0].item() == pytest.approx(5000.0)
    assert s[0, batch - 1, 0].item() == pytest.approx(float(batch - 1))


def test_predict_concatenation_agrees_across_ragged_batches():
    """The `predict.py` permutation must give a common trailing shape.

    Two full batches and one short one, exactly the shape of the test partition, then
    the concatenation that `predict.py` performs. This is the assertion `eval_xco2`
    failed on.
    """
    np = pytest.importorskip("numpy")
    flow = ScalarFlow(_Backbone(), _NF())
    n = 32
    parts = [
        flow.sample(torch.zeros(b, 3), n=n).permute(1, 0, 2).numpy()
        for b in (16, 16, 9)
    ]
    assert [p.shape[1:] for p in parts] == [(n, 1)] * 3
    assert np.concatenate(parts, axis=0).shape == (41, n, 1)


def test_transposed_flow_is_rejected_rather_than_reported():
    """A flow that regresses to the nflows convention must raise, not return."""
    flow = ScalarFlow(_Backbone(), _NF(transposed=True))
    with pytest.raises(RuntimeError, match="transposed"):
        flow.sample(torch.zeros(16, 3), n=32)

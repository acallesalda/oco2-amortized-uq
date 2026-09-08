"""gaussianize.py -- the flow against a Gaussian that agrees with it on two moments.

For each sounding, the flow's predictive is summarized by the mean and covariance of its
own samples, and the truth is scored under the Gaussian `N(m_i, C_i)` those two moments
define. The control and the flow agree *exactly* on first and second moments by
construction, so every difference between them -- in NLL, in coverage, in the shape of the
PIT -- is structure beyond second order and nothing else. It is the comparison that
separates "the flow is better calibrated in scale", which a rescaled Laplace approximation
could also achieve, from "the posterior is not Gaussian", which no Gaussian can.

Two things make this more than a call to `numpy.cov`:

**The plug-in penalty.** `C_i` is estimated from S samples, and a Gaussian fitted by
plug-in pays about `d(d+3)/(4S)` nats for that estimation. At `d=1, S=512` this is 0.002
nats and can be ignored; at `d=20, S=512` it is 0.22 nats, which is the same order as the
effect being measured -- so a control built from the 512 stored samples would attribute the
flow a shape advantage that is really the control's estimation error. Statistics are
therefore recorded at a schedule of sample budgets (512, 1024, 2048, ...) so that the
`a + b/S` trend is visible and can be extrapolated to `S -> infinity`. The fitted `b`
should land near `d(d+3)/4`; if it does not, the extrapolation is not to be trusted.

**Storage.** Reaching `S = 8192` for the 20-level task means 4,489 x 8192 x 20 samples,
0.7 GB in float16, which is not a thing to move across OSDF for a table with six rows. So
the samples are never stored: they are streamed in chunks and reduced online into sums that
are enough for everything reported here -- mean, covariance, per-level skewness and excess
kurtosis, the PIT (the fraction of samples below the truth, which determines marginal
coverage at *every* level exactly), and the control's NLL at each stage. The whole payload
is about 25 MB.

Nothing here draws the 512 samples `predict.py` already stores; that pass runs first and is
unaffected, so an artifact written with `--gaussianize` carries a bit-identical
`nf_samples_std` to one written without it.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

try:
    import torch
except ImportError:                 # pragma: no cover
    # `nongauss.py` imports this module on a laptop for `stage_schedule`,
    # `plugin_penalty` and `extrapolate`, which are numpy and arithmetic. Only the
    # sampling pass needs torch, and it says so when it is missing rather than failing at
    # the import line of a table script that was never going to sample anything.
    torch = None


def _require_torch():
    if torch is None:
        raise ImportError(
            "evals.gaussianize needs torch for the sampling pass. The numpy-only helpers "
            "(stage_schedule, plugin_penalty, extrapolate) do not."
        )

# The smallest budget in the schedule. Chosen to match `predict.py --nf-samples`, so the
# first stage reproduces the number the published tables were computed at and the trend
# starts from a point that can be checked against them.
STAGE_MIN = 512


def stage_schedule(total: int, first: int = STAGE_MIN) -> list[int]:
    """Sample budgets at which statistics are recorded: `first`, doubling, then `total`.

    Nested rather than independent: stage k uses the first `S_k` samples of the same
    stream. That makes the stages correlated, which is what is wanted -- the object being
    estimated is a deterministic bias curve in `1/S`, not a set of independent replicates.
    """
    if total < 1:
        raise ValueError("total must be >= 1")
    stages, s = [], min(first, total)
    while s < total:
        stages.append(s)
        s *= 2
    stages.append(total)
    return stages


def plugin_penalty(d: int, s: int) -> float:
    """Expected NLL excess of a plug-in Gaussian fitted from `s` samples, in nats.

    `d(d+3)/(4S)`: the usual `p/(2S)` for a model with `p = d + d(d+1)/2` free parameters.
    """
    return d * (d + 3) / (4.0 * s)


# ---------------------------------------------------------------------------------
# The streaming reduction
# ---------------------------------------------------------------------------------
def _mean_cov(shift: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor, n: int):
    """Mean and unbiased covariance from sums of shift-centered samples."""
    mbar = s1 / n
    mu = shift + mbar
    cov = (s2 - n * torch.einsum("bd,be->bde", mbar, mbar)) / max(n - 1, 1)
    return mu, cov


def _skew_exkurt(s2: torch.Tensor, s3: torch.Tensor, s4: torch.Tensor,
                 mbar: torch.Tensor, n: int):
    """Per-coordinate skewness and excess kurtosis from the same shifted sums."""
    d2 = torch.diagonal(s2, dim1=-2, dim2=-1) / n
    m2 = (d2 - mbar ** 2).clamp_min(1e-300)
    m3 = s3 / n - 3 * mbar * d2 + 2 * mbar ** 3
    m4 = s4 / n - 4 * mbar * (s3 / n) + 6 * mbar ** 2 * d2 - 3 * mbar ** 4
    return m3 / m2 ** 1.5, m4 / m2 ** 2 - 3.0


def mvn_nll_torch(y: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor,
                  jitters: Sequence[float] = (0.0, 1e-8, 1e-6, 1e-4, 1e-3)) -> torch.Tensor:
    """Per-sounding joint Gaussian NLL, `[B]`, NaN where the Cholesky fails.

    The torch counterpart of `metrics.mvn_nll`, with the same jitter ladder and the same
    refusal to regularize a broken covariance into a number. It lives here rather than
    there because it runs inside the sampling pass, where the covariance exists only as a
    device tensor; `nongauss.py` recomputes the final stage from the stored covariance
    through `metrics.mvn_nll` and the two must agree.
    """
    b, d = y.shape
    r = (y.to(torch.float64) - mu).unsqueeze(-1)
    out = torch.full((b,), float("nan"), dtype=torch.float64, device=y.device)
    todo = torch.ones(b, dtype=torch.bool, device=y.device)
    scale = torch.diagonal(cov, dim1=-2, dim2=-1).mean(-1).clamp_min(1e-30)
    eye = torch.eye(d, dtype=torch.float64, device=y.device)
    const = d * math.log(2.0 * math.pi)

    for j in jitters:
        idx = torch.nonzero(todo, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            break
        c = cov[idx] + j * scale[idx].view(-1, 1, 1) * eye
        chol, info = torch.linalg.cholesky_ex(c)
        ok = info == 0
        if not bool(ok.any()):
            continue
        good = idx[ok]
        low = chol[ok]
        z = torch.linalg.solve_triangular(low, r[good], upper=False)
        logdet = 2.0 * torch.log(torch.diagonal(low, dim1=-2, dim2=-1)).sum(-1)
        out[good] = 0.5 * (const + logdet + (z ** 2).sum((-2, -1)))
        todo[good] = False
    return out


def stream_batch(flow, x_tuple, y, total: int, chunk: int,
                 stages: Sequence[int], m_idx=None) -> dict:
    """Draw `total` flow samples for one batch and reduce them online.

    `y` is the truth for this batch in the scored coordinates, `[B, D]`. `m_idx` slices the
    model's outputs down to those coordinates when the two differ (`last4_marginal`), and
    is applied to the samples exactly as `predict.py` applies it to the stored ones.

    Returns numpy arrays whose leading dimension is the batch, so that a caller can
    concatenate them into per-sounding artifact arrays:

        mu     [B,K,D]   posterior mean at each stage
        var    [B,K,D]   marginal variance at each stage
        nll    [B,K]     NLL of the truth under the moment-matched Gaussian at each stage
        pit    [B,K,D]   fraction of samples at or below the truth
        cov    [B,D,D]   full covariance at the final stage
        skew   [B,D]     per-level skewness at the final stage
        exkurt [B,D]     per-level excess kurtosis at the final stage
    """
    _require_torch()
    stages = list(stages)
    if not stages or stages[-1] != total:
        raise ValueError(f"stages {stages} must be ascending and end at total={total}")

    with torch.no_grad():
        return _stream_batch(flow, x_tuple, y, chunk, stages, m_idx)


def _stream_batch(flow, x_tuple, y, chunk: int, stages: Sequence[int], m_idx) -> dict:
    y = y.to(torch.float64)
    b, d = y.shape
    dev = y.device
    shift = None
    s1 = torch.zeros(b, d, dtype=torch.float64, device=dev)
    s2 = torch.zeros(b, d, d, dtype=torch.float64, device=dev)
    s3 = torch.zeros(b, d, dtype=torch.float64, device=dev)
    s4 = torch.zeros(b, d, dtype=torch.float64, device=dev)
    below = torch.zeros(b, d, dtype=torch.float64, device=dev)
    seen = 0
    mus, variances, nlls, pits = [], [], [], []
    mu = cov = mbar = None

    for target in stages:
        while seen < target:
            n = min(chunk, target - seen)
            s = flow.sample(x_tuple, n=n)                       # [n,B,D'] in model space
            if m_idx is not None:
                s = s.index_select(2, m_idx)
            s = s.to(torch.float64)
            if s.shape[1:] != (b, d):
                raise RuntimeError(
                    f"flow.sample returned {tuple(s.shape)}; expected (n, {b}, {d}). "
                    "The sample and batch axes may be transposed -- see the check in "
                    "evals.checkpoints.ScalarFlow.sample."
                )
            # The shift keeps the higher moments conditioned. Sample values live at a few
            # standardized units from zero with a spread of a fraction of one, so raw
            # fourth moments about the origin lose most of their significant digits to
            # cancellation; about the first chunk's mean they lose none.
            if shift is None:
                shift = s.mean(dim=0)
            c = s - shift
            c2 = c * c
            s1 += c.sum(0)
            s2 += torch.einsum("nbd,nbe->bde", c, c)
            s3 += (c2 * c).sum(0)
            s4 += (c2 * c2).sum(0)
            below += (s <= y.unsqueeze(0)).to(torch.float64).sum(0)
            seen += n

        mbar = s1 / seen
        mu, cov = _mean_cov(shift, s1, s2, seen)
        mus.append(mu.clone())
        variances.append(torch.diagonal(cov, dim1=-2, dim2=-1).clone())
        nlls.append(mvn_nll_torch(y, mu, cov))
        pits.append(below / seen)

    skew, exkurt = _skew_exkurt(s2, s3, s4, mbar, seen)
    stack = lambda xs: torch.stack(xs, dim=1).cpu().numpy()     # noqa: E731  [B,K,...]
    return {
        "mu": stack(mus),
        "var": stack(variances),
        "nll": torch.stack(nlls, dim=1).cpu().numpy(),
        "pit": stack(pits),
        "cov": cov.cpu().numpy(),
        "skew": skew.cpu().numpy(),
        "exkurt": exkurt.cpu().numpy(),
    }


# ---------------------------------------------------------------------------------
# The extrapolation
# ---------------------------------------------------------------------------------
def extrapolate(stages: Sequence[int], nll: np.ndarray) -> dict:
    """Fit `NLL(S) = a + b/S` over the stages and return the intercept.

    `nll` is `[K]` -- the control's mean NLL at each stage. `a` is what the control would
    score with the moments known exactly, which is the number the shape gap should be
    measured against; `b` should come out near `d(d+3)/4` and is reported so that a fit
    that did not behave is visible rather than absorbed.
    """
    s = np.asarray(stages, dtype=np.float64)
    v = np.asarray(nll, dtype=np.float64)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return {"a": float("nan"), "b": float("nan"), "r2": float("nan")}
    x = 1.0 / s[ok]
    y = v[ok]
    a_mat = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(a_mat, y, rcond=None)
    resid = y - a_mat @ coef
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "a": float(coef[0]),
        "b": float(coef[1]),
        "r2": float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
    }

# flow_regressor.py
"""
FlowRegressor (scalar) built on top of your FullRegressor.encoder.
Implements conditional 1D normalizing flow with rational–quadratic splines.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# 1) RQS building blocks
# ----------------------------

def _normalize_bins(w, h, eps: float = 1e-5):
    w = F.softmax(w, dim=-1)
    h = F.softmax(h, dim=-1)
    K = w.size(-1)
    w = (w + eps) / (1 + K * eps)
    h = (h + eps) / (1 + K * eps)
    return w, h

def _constrain_slopes(s, min_s: float = 1e-3):
    return F.softplus(s) + min_s

def rational_quadratic_spline(
    y, widths, heights, slopes_left, slopes_right,
    x_min: float = -10.0, x_max: float = 10.0,
    inverse: bool = False, eps: float = 1e-6
):
    """
    1D monotonic rational–quadratic spline with K bins on [x_min, x_max].
    widths, heights: (..., K); slopes_left/right: (..., K+1)
    Returns (transformed, log|det J|).
    Shapes expected in our use:
      y:        [B, 1]
      widths:   [B, K]
      heights:  [B, K]
      slopes_*: [B, K+1]
    """
    y = torch.clamp(y, x_min + eps, x_max - eps)

    K = widths.size(-1)
    widths, heights = _normalize_bins(widths, heights)      # each [B, K]
    cumw = torch.cumsum(widths, dim=-1)                     # [B, K]
    cumh = torch.cumsum(heights, dim=-1)                    # [B, K]
    cumw = F.pad(cumw, (1, 0), value=0.0)                   # [B, K+1]
    cumh = F.pad(cumh, (1, 0), value=0.0)                   # [B, K+1]

    if not inverse:
        # forward: y -> x
        y01 = (y - x_min) / (x_max - x_min)                 # [B, 1]
        # Find which height-bin y01 falls into; keep shapes [B, 1]
        bin_idx = torch.searchsorted(cumh, y01, right=False) - 1  # [B, 1]
        bin_idx = bin_idx.clamp(0, K - 1).long()

        def g(x):  # gather along last dim by bin; returns [B, 1]
            return torch.gather(x, -1, bin_idx)

        w_k   = g(widths)                            # [B, 1]
        h_k   = g(heights)                           # [B, 1]
        s_l   = g(slopes_left)                       # [B, 1]
        s_r   = g(slopes_right)                      # [B, 1]
        h_st  = g(cumh[..., :-1])                    # [B, 1]
        w_st  = g(cumw[..., :-1])                    # [B, 1]

        t = (y01 - h_st) / (h_k + 1e-12)             # [B, 1] in (0,1)
        numer = h_k * (w_st + w_k * t) + (y01 - h_st) * (s_l * (1 - t) + s_r * t - 2 * h_k)
        denom = s_l * (1 - t) + s_r * t + 2 * (w_k - h_k)
        x01 = numer / (denom + 1e-12)                # [B, 1]
        x = x01 * (x_max - x_min) + x_min            # [B, 1]

        # log|dx/dy|
        dt_dy   = 1.0 / (h_k * (x_max - x_min) + 1e-12)
        dnum_dt = h_k * w_k + (s_r - s_l - 2 * h_k) * (y01 - h_st) + (s_l - s_r + 2 * h_k) * h_k
        dden_dt = (s_r - s_l) + 2 * (w_k - h_k)
        dx_dy   = (dnum_dt * denom - numer * dden_dt) * (dt_dy / (denom**2 + 1e-12)) * (x_max - x_min)
        logabsdet = torch.log(torch.abs(dx_dy) + 1e-12)     # [B, 1]
        return x, logabsdet

    else:
        # inverse: x -> y (bisection on t)
        x01 = (y - x_min) / (x_max - x_min)                 # [B, 1]
        bin_idx = torch.searchsorted(cumw, x01, right=False) - 1  # [B, 1]
        bin_idx = bin_idx.clamp(0, K - 1).long()

        def g(x):
            return torch.gather(x, -1, bin_idx)

        w_k   = g(widths)                            # [B, 1]
        h_k   = g(heights)                           # [B, 1]
        s_l   = g(slopes_left)                       # [B, 1]
        s_r   = g(slopes_right)                      # [B, 1]
        h_st  = g(cumh[..., :-1])                    # [B, 1]
        w_st  = g(cumw[..., :-1])                    # [B, 1]

        # Bisection on t in [0,1]
        t_lo, t_hi = torch.zeros_like(x01), torch.ones_like(x01)
        for _ in range(30):
            t_mid = 0.5 * (t_lo + t_hi)
            numer = h_k * (w_st + w_k * t_mid) + (h_st + h_k * t_mid - h_st) * (
                s_l * (1 - t_mid) + s_r * t_mid - 2 * h_k
            )
            denom = s_l * (1 - t_mid) + s_r * t_mid + 2 * (w_k - h_k)
            x01_mid = numer / (denom + 1e-12)
            t_lo = torch.where(x01_mid < x01, t_mid, t_lo)
            t_hi = torch.where(x01_mid >= x01, t_mid, t_hi)

        t = 0.5 * (t_lo + t_hi)
        y01 = h_st + h_k * t
        y = y01 * (x_max - x_min) + x_min

        numer = h_k * (w_st + w_k * t) + (y01 - h_st) * (s_l * (1 - t) + s_r * t - 2 * h_k)
        denom = s_l * (1 - t) + s_r * t + 2 * (w_k - h_k)
        dt_dy   = 1.0 / (h_k * (x_max - x_min) + 1e-12)
        dnum_dt = h_k * w_k + (s_r - s_l - 2 * h_k) * (y01 - h_st) + (s_l - s_r + 2 * h_k) * h_k
        dden_dt = (s_r - s_l) + 2 * (w_k - h_k)
        dx_dy   = (dnum_dt * denom - numer * dden_dt) * (dt_dy / (denom**2 + 1e-12)) * (x_max - x_min)
        logabsdet = -torch.log(torch.abs(dx_dy) + 1e-12)
        return y, logabsdet


class ConditionalRQS1D(nn.Module):
    """
    Single conditional RQS transform y<->z in 1D.
    Conditioner outputs (widths, heights, slopes_left, slopes_right) given context.
    """
    def __init__(self, context_dim: int, K: int = 8, hidden: int = 128,
                 x_min: float = -10.0, x_max: float = 10.0):
        super().__init__()
        self.K, self.x_min, self.x_max = K, x_min, x_max
        out_dim = 2 * K + 2 * (K + 1)
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def _split(self, params):
        K = self.K
        widths   = params[..., :K]
        heights  = params[..., K:2*K]
        slopes_l = params[..., 2*K:3*K+1]
        slopes_r = params[..., 3*K+1:4*K+2]
        return widths, heights, slopes_l, slopes_r

    def forward(self, y, context, inverse: bool = False):
        params = self.net(context)
        widths, heights, sl, sr = self._split(params)
        sl = _constrain_slopes(sl)
        sr = _constrain_slopes(sr)
        return rational_quadratic_spline(
            y, widths, heights, sl, sr, x_min=self.x_min, x_max=self.x_max, inverse=inverse
        )


# ----------------------------
# 2) Conditional Flow (stack)
# ----------------------------

class ConditionalFlow1D(nn.Module):
    """
    Stack of T conditional RQS transforms; base z ~ N(0,1).
    """
    def __init__(self, context_dim: int, T: int = 3, K: int = 8, hidden: int = 128,
                 x_min: float = -10.0, x_max: float = 10.0):
        super().__init__()
        self.transforms = nn.ModuleList([
            ConditionalRQS1D(context_dim, K=K, hidden=hidden, x_min=x_min, x_max=x_max)
            for _ in range(T)
        ])
        self.base = torch.distributions.Normal(0.0, 1.0)

    def _forward(self, y, context):
        # forward: y -> z, accumulate log|dz/dy|
        logabsdet = 0.0
        z = y
        for tr in self.transforms:
            z, lad = tr(z, context, inverse=False)
            logabsdet = logabsdet + lad
        return z, logabsdet

    def _inverse(self, z, context):
        # inverse: z -> y, accumulate log|dy/dz|
        logabsdet = 0.0
        y = z
        for tr in reversed(self.transforms):
            y, lad = tr(y, context, inverse=True)
            logabsdet = logabsdet + lad
        return y, logabsdet

    def log_prob(self, y, context):
        z, lad = self._forward(y, context)
        log_pz = self.base.log_prob(z)
        return (log_pz + lad).squeeze(-1)

    def sample(self, context, n: int = 1):
        B = context.size(0)
        z = self.base.sample((n, B)).to(context.device)
        ctx = context.unsqueeze(0).expand(n, B, -1).reshape(-1, context.size(-1))
        zf = z.reshape(-1, 1)
        y, _ = self._inverse(zf, ctx)
        return y.reshape(n, B)


# --------------------------------------------------------
# 3) Feature wrapper for your FullRegressor.encoder
# --------------------------------------------------------

class FeatureNetFromFullRegressor(nn.Module):
    """
    Wraps your FullRegressor.encoder so that forward((bands, cont, cat)) -> [B, C].
    If the encoder returns something with extra dims, we flatten to [B, C].
    """
    def __init__(self, full_regressor_encoder: nn.Module, context_dim: Optional[int] = None):
        super().__init__()
        self.encoder = full_regressor_encoder
        self.context_dim = context_dim  # optional hint

    def _flatten_if_needed(self, z):
        # Expect [B, C] ideally; otherwise flatten trailing dims
        if z.dim() > 2:
            return z.flatten(1)
        return z

    def infer_context_dim(self, sample_x: Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, List[torch.Tensor]]):
        self.eval()
        with torch.no_grad():
            z = self.encoder(sample_x)       # whatever shape
            z = self._flatten_if_needed(z)   # -> [B, C]
        return z.shape[-1]

    def forward(self, x_tuple):
        z = self.encoder(x_tuple)
        z = self._flatten_if_needed(z)
        return z


# --------------------------------------------------------
# 4) Full Flow Regressor (scalar y)
# --------------------------------------------------------

class FlowRegressorScalar(nn.Module):
    """
    (x) --encoder--> context  --flow--> p(y|x)
    """
    def __init__(self, feature_net: FeatureNetFromFullRegressor, context_dim: int,
                 T: int = 3, K: int = 8, hidden: int = 128,
                 x_min: float = -10.0, x_max: float = 10.0,
                 standardize_y: bool = True, y_mu: float = 0.0, y_sigma: float = 1.0):
        super().__init__()
        self.feature_net = feature_net
        self.flow = ConditionalFlow1D(context_dim, T=T, K=K, hidden=hidden, x_min=x_min, x_max=x_max)
        self.standardize_y = standardize_y
        self.register_buffer("y_mu", torch.tensor(float(y_mu)))
        self.register_buffer("y_sigma", torch.tensor(float(y_sigma)))

    # y transforms
    def _y_to_std(self, y):
        if self.standardize_y:
            return (y - self.y_mu) / (self.y_sigma + 1e-12)
        return y

    def _y_from_std(self, y):
        if self.standardize_y:
            return y * (self.y_sigma + 1e-12) + self.y_mu
        return y

    # API
    def log_prob(self, x_tuple, y):
        ctx = self.feature_net(x_tuple)          # [B, C]
        y = y.view(-1, 1).to(ctx.device)
        ys = self._y_to_std(y)
        return self.flow.log_prob(ys, ctx)

    @torch.no_grad()
    def sample(self, x_tuple, n: int = 100):
        ctx = self.feature_net(x_tuple)
        ys = self.flow.sample(ctx, n=n)          # [n, B]
        return self._y_from_std(ys)

    @torch.no_grad()
    def predict_mean(self, x_tuple, n: int = 200):
        s = self.sample(x_tuple, n=n)            # [n, B]
        return s.mean(0)                          # [B]


# --------------------------------------------------------
# 5) Factory helper
# --------------------------------------------------------

def build_flow_from_fullregressor(
    full_regressor_cls,                    # e.g., modeling.full_regressor.FullRegressor
    full_regressor_kwargs: Dict[str, Any], # embed_dim, nhead, nlayers, aux_dim, ...
    T: int = 4, K: int = 8, hidden: int = 256,
    x_min: float = -10.0, x_max: float = 10.0,
    standardize_y: bool = True,
    y_mu: float = 0.0004108437,           # <-- replace with your scalar stats
    y_sigma: float = 2.1281e-05
) -> Tuple[FlowRegressorScalar, int]:
    """
    Construct a FlowRegressorScalar using FullRegressor.encoder as context network.
    Returns (model, context_dim).
    """
    base = full_regressor_cls(**full_regressor_kwargs)
    feat = FeatureNetFromFullRegressor(base.encoder)

    # Infer context dim with a tiny synthetic pass (robust if out_dim not exposed)
    # Caller should do a real pass to set it precisely; we fallback to base.encoder.out_dim if present.
    context_dim = getattr(base.encoder, "out_dim", None)
    if context_dim is None:
        # We expect caller to call feat.infer_context_dim on a real batch during train init.
        # Temporarily set a dummy, will overwrite later.
        context_dim = full_regressor_kwargs.get("embed_dim", 256)

    model = FlowRegressorScalar(
        feature_net=feat, context_dim=context_dim,
        T=T, K=K, hidden=hidden, x_min=x_min, x_max=x_max,
        standardize_y=standardize_y, y_mu=y_mu, y_sigma=y_sigma
    )
    return model, context_dim

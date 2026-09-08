# modeling/flow_regressor_profiles_nf.py
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn

from nflows import flows, distributions, transforms
from nflows.nn import nets


class FeatureNetProfiles(nn.Module):
    """
    Wrap your FullRegressorProfiles backbone. Produces a flat context vector [B, C].
    Expects backbone.embed((bands, aux_cont, aux_cat)) -> [B, 4*E] or similar.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x_tuple) -> torch.Tensor:
        z = self.backbone.embed(x_tuple)          # e.g. [B, 4, E] or [B, 4*E]
        if z.dim() > 2:
            z = z.flatten(1)
        return z

    @torch.no_grad()
    def infer_context_dim(self, x_tuple) -> int:
        z = self.forward(x_tuple)
        return int(z.shape[-1])


def _alt_mask(D: int, even: bool) -> torch.Tensor:
    """Alternate 0/1 mask along dims so each coupling transforms ~half the coords."""
    idx = torch.arange(D)
    m = (idx % 2 == (0 if even else 1)).to(torch.float32)
    return m


def _make_tnet_factory(hidden_features: int, context_features: int, num_blocks: int = 2,
                       dropout_probability: float = 0.0):
    """
    Factory returning a transform-net builder that *respects* the out_features
    computed by the coupling transform (i.e., out_features = tf * (3*K + 2)).

    `dropout_probability` regularizes the coupling nets and defaults to 0.0, which is what
    every published checkpoint was trained with. It is exposed because this is the model
    This is where over-fitting bites hardest -- validation NLL best after 7 of 40 epochs, then
    18 nats of divergence while the training likelihood improved by another 35 -- and
    dropout inside the coupling networks is the one regularizer that acts on the flow's own
    capacity rather than on the backbone's. nflows applies it in the residual blocks only,
    so it is inactive under .eval() and the density at scoring time is unchanged.
    """
    def make_tnet(in_features: int, out_features: int) -> nn.Module:
        return nets.ResidualNet(
            in_features=in_features,
            out_features=out_features,            # <- CRITICAL: do not recompute!
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            activation=nn.ReLU(),
            dropout_probability=float(dropout_probability),
            use_batch_norm=False,
        )
    return make_tnet


class NFProfiles(nn.Module):
    """
    Conditional 20-D PRQ-coupling flow: p(y | x), with context from FeatureNetProfiles.
    Uses alternating masks + random permutations. Base is StandardNormal(D).
    """
    def __init__(
        self,
        feature_net: FeatureNetProfiles,
        context_dim: int,
        out_dim: int = 20,
        num_layers: int = 7,
        hidden_features: int = 640,
        num_bins: int = 13,
        tail_bound: float = 10.0,
        standardize_y: bool = True,
        y_mu: Optional[torch.Tensor] = None,      # expect [D] if provided
        y_sigma: Optional[torch.Tensor] = None,   # expect [D] if provided
        min_bin_width: float = 5e-3,
        min_bin_height: float = 5e-3,
        min_derivative: float = 1e-3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.feature_net = feature_net
        self.out_dim = int(out_dim)
        self.standardize_y = bool(standardize_y)
        self._min_bin_width = float(min_bin_width)
        self._min_bin_height = float(min_bin_height)
        self._min_derivative = float(min_derivative)

        # y standardization buffers (no-op if mu=0, sigma=1)
        if self.standardize_y:
            self.register_buffer("y_mu",    (torch.zeros(self.out_dim) if y_mu is None else y_mu.detach().clone()))
            self.register_buffer("y_sigma", (torch.ones(self.out_dim)  if y_sigma is None else y_sigma.detach().clone()))

        # Build coupling stack
        tfs: List[nn.Module] = []
        make_tnet = _make_tnet_factory(hidden_features=hidden_features,
                                       context_features=context_dim,
                                       dropout_probability=self.dropout)
        for l in range(num_layers):
            mask = _alt_mask(self.out_dim, even=(l % 2 == 0))
            tfs.append(
                transforms.PiecewiseRationalQuadraticCouplingTransform(
                    mask=mask,
                    transform_net_create_fn=make_tnet,      # <- out_features honored
                    num_bins=int(num_bins),                  # <- wired through
                    tails="linear",
                    tail_bound=float(tail_bound),
                    min_bin_width=self._min_bin_width,
                    min_bin_height=self._min_bin_height,
                    min_derivative=self._min_derivative,
                )
            )
            tfs.append(transforms.RandomPermutation(features=self.out_dim))

        transform = transforms.CompositeTransform(tfs)
        base = distributions.StandardNormal(shape=[self.out_dim])
        self.flow = flows.Flow(transform, base)

    # -------- utilities --------
    def _y_to_std(self, y: torch.Tensor) -> torch.Tensor:
        if not self.standardize_y:
            return y
        return (y - self.y_mu) / (self.y_sigma + 1e-12)

    def _y_from_std(self, ys: torch.Tensor) -> torch.Tensor:
        if not self.standardize_y:
            return ys
        return ys * (self.y_sigma + 1e-12) + self.y_mu

    # -------- public API --------
    def log_prob(self, x_tuple, y: torch.Tensor) -> torch.Tensor:
        """
        x_tuple: (bands, aux_cont, aux_cat)
        y: [B, D]
        returns: log p(y|x) [B]
        """
        ctx = self.feature_net(x_tuple)                               # [B, C]
        y   = y.to(ctx.device, dtype=ctx.dtype).view(-1, self.out_dim)
        ys  = self._y_to_std(y)
        return self.flow.log_prob(inputs=ys, context=ctx)             # [B]

    @torch.no_grad()
    def latent(self, x_tuple, y: torch.Tensor) -> torch.Tensor:
        """Map observations to the flow's base space: z = f_x(y), shape [B, D].

        Under a correctly calibrated conditional density the pooled z are i.i.d.
        N(0, I_D), which makes this the flow's counterpart of the Cholesky whitening
        used for the Gaussian predictors, and the basis for a *joint* calibration check
        rather than a per-level marginal one.
        """
        ctx = self.feature_net(x_tuple)                               # [B, C]
        y   = y.to(ctx.device, dtype=ctx.dtype).view(-1, self.out_dim)
        ys  = self._y_to_std(y)
        z, _ = self.flow._transform(ys, context=ctx)                  # [B, D]
        return z

    @torch.no_grad()
    def sample(self, x_tuple, n: int = 100) -> torch.Tensor:
        """
        returns samples in ORIGINAL y-space: [n, B, D]
        """
        ctx = self.feature_net(x_tuple)                               # [B, C]
        s_std = self.flow.sample(num_samples=n, context=ctx)          # [n,B,D] or [B,n,D] or [B,D]
        # --- normalize to [n, B, D] ---
        if s_std.dim() == 2:
            # some nflows builds return [B, D] if n==1
            s_std = s_std.unsqueeze(0)                                # -> [1, B, D]
        elif s_std.shape[0] != n and s_std.shape[1] == n:
            # got [B, n, D] -> permute
            s_std = s_std.permute(1, 0, 2).contiguous()               # -> [n, B, D]
        # else: already [n, B, D]

        return self._y_from_std(s_std)

    @torch.no_grad()
    def mean(self, x_tuple, n: int = 200) -> torch.Tensor:
        s = self.sample(x_tuple, n=n)                                 # [n, B, D]
        return s.mean(0)                                              # [B, D]

# modeling/flow_regressor_profiles_nf.py
"""
20-D conditional normalizing flow for CO2 profiles, reusing your
FullRegressorProfiles backbone as the context network.

Key pieces
----------
- FeatureNetProfiles: wraps FullRegressorProfiles to output a [B, C] context.
- CouplingFlowND: nflows-based PRQ coupling stack with context.
- NFProfiles: convenience module exposing log_prob / sample / predict_mean.

Dependencies: nflows (already used in your scalar NF code).
"""
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from nflows import flows, distributions, transforms
from nflows.nn import nets

# --- context wrapper -----------------------------------------------------------

class FeatureNetProfiles(nn.Module):
    """Use your FullRegressorProfiles *backbone* (encoder+fusion) as feature net.

    We assume the module provides an `.embed(x)` method that returns [B, C].
    (Your `FullRegressorProfiles.embed` does exactly that.)
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x_tuple):
        z = self.backbone.embed(x_tuple)  # [B, C]
        if z.dim() > 2:
            z = z.flatten(1)
        return z

    @torch.no_grad()
    def infer_context_dim(self, sample_x_tuple) -> int:
        z = self.forward(sample_x_tuple)
        return int(z.shape[-1])


# --- nflows coupling stack -----------------------------------------------------

def _build_coupling_stack(
    features: int,
    context_dim: int,
    num_layers: int = 8,
    hidden_features: int = 512,
    num_bins: int = 8,
    tail_bound: float = 10.0,
    dropout: float = 0.0,
    use_batch_norm: bool = False,
):
    """
    Piecewise-Rational-Quadratic (PRQ) *coupling* stack with context.

    We alternate masks and add random permutations between layers to mix dims.
    ResidualNet is used for the conditioner (stable + expressive).
    """
    assert features > 1, "Use the scalar NF for features==1."

    def make_tnet(in_features: int, out_features: int):
        return nets.ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            context_features=context_dim,
            num_blocks=2,
            activation=F.relu,
            dropout_probability=dropout,
            use_batch_norm=use_batch_norm,
        )

    layers = []
    # two complementary binary masks we will alternate
    mask_even = torch.zeros(features)
    mask_even[::2] = 1.0
    mask_odd = 1.0 - mask_even

    for i in range(num_layers):
        mask = mask_even if (i % 2 == 0) else mask_odd
        layers.append(
            transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=make_tnet,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
            )
        )
        # permutation increases mixing; for features=20 this is cheap
        layers.append(transforms.RandomPermutation(features=features))

    return transforms.CompositeTransform(layers)


class CouplingFlowND(nn.Module):
    """Context-conditional ND flow using PRQ coupling layers."""
    def __init__(
        self,
        features: int,
        context_dim: int,
        num_layers: int = 8,
        hidden_features: int = 512,
        num_bins: int = 8,
        tail_bound: float = 10.0,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        transform = _build_coupling_stack(
            features=features,
            context_dim=context_dim,
            num_layers=num_layers,
            hidden_features=hidden_features,
            num_bins=num_bins,
            tail_bound=tail_bound,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )
        base = distributions.StandardNormal(shape=[features])
        self.flow = flows.Flow(transform, base)

    def log_prob(self, y: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Return log p(y | context) for y of shape [B, D]."""
        return self.flow.log_prob(inputs=y, context=context)

    def sample(self, n: int, context: torch.Tensor) -> torch.Tensor:
        """Sample [n, B, D] from p(y | context)."""
        s = self.flow.sample(num_samples=n, context=context)  # [n, B, D]
        return s


# --- full model wrapper --------------------------------------------------------

class NFProfiles(nn.Module):
    """
    (x) --FeatureNetProfiles--> context  --CouplingFlowND--> p(y|x) on R^D.

    Optionally standardize outputs inside the module using per-dim y_mu/y_sigma.
    """
    def __init__(
        self,
        feature_net: FeatureNetProfiles,
        context_dim: int,
        out_dim: int = 20,
        num_layers: int = 8,
        hidden_features: int = 512,
        num_bins: int = 8,
        tail_bound: float = 10.0,
        standardize_y: bool = True,
        y_mu: Optional[torch.Tensor] = None,
        y_sigma: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.feature_net = feature_net
        self.out_dim = int(out_dim)
        self.flow = CouplingFlowND(
            features=self.out_dim,
            context_dim=context_dim,
            num_layers=num_layers,
            hidden_features=hidden_features,
            num_bins=num_bins,
            tail_bound=tail_bound,
        )
        self.standardize_y = bool(standardize_y)
        if y_mu is None:
            y_mu = torch.zeros(self.out_dim)
        if y_sigma is None:
            y_sigma = torch.ones(self.out_dim)
        assert y_mu.shape[-1] == self.out_dim and y_sigma.shape[-1] == self.out_dim
        self.register_buffer("y_mu", y_mu.float())
        self.register_buffer("y_sigma", y_sigma.float())

    # y transforms --------------------------------------------------------------
    def _to_std(self, y: torch.Tensor) -> torch.Tensor:
        if self.standardize_y:
            return (y - self.y_mu) / (self.y_sigma + 1e-12)
        return y

    def _from_std(self, y: torch.Tensor) -> torch.Tensor:
        if self.standardize_y:
            return y * (self.y_sigma + 1e-12) + self.y_mu
        return y

    # API ----------------------------------------------------------------------
    def log_prob(self, x_tuple, y: torch.Tensor) -> torch.Tensor:
        ctx = self.feature_net(x_tuple)               # [B, C]
        y   = y.to(ctx.device)
        ys  = self._to_std(y)
        return self.flow.log_prob(ys, ctx)            # [B]

    @torch.no_grad()
    def sample(self, x_tuple, n: int = 64) -> torch.Tensor:
        ctx = self.feature_net(x_tuple)               # [B, C]
        ys  = self.flow.sample(n, ctx)                # [n, B, D]
        return self._from_std(ys)

    @torch.no_grad()
    def predict_mean(self, x_tuple, n: int = 256) -> torch.Tensor:
        s = self.sample(x_tuple, n=n)                 # [n, B, D]
        return s.mean(0)                              # [B, D]
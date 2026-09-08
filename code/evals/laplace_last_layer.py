"""laplace_last_layer.py

Last-layer Laplace over the deterministic regressors, extracted from the scoring
notebooks so that all three tasks (scalar XCO2, 20-level profile, near-surface 4-level)
go through one code path.

That uniformity is the point. In the notebooks the scalar table used
`laplace.Laplace(..., hessian_structure="full")` and read the library's own predictive
variance, while the two profile tables fitted a diagonal Hessian and then did their own
Monte-Carlo over last-layer weights plus a residual-calibrated aleatoric diagonal. Two
different estimators, three tables, one manuscript -- and no way to tell whether a
difference between Table 1 and Table 2 was the task or the estimator. Both recipes live
here now, selected by `mode`, and whichever was used is recorded in the artifact.

The awkward machinery in this file exists because `laplace-torch` assumes `x` is a
tensor it can slice, move and batch. Our input is a nested
`(dict[str, (tensor, tensor)], tensor, list[tensor])`. `BatchAdapter` gives that nest the
three methods the library actually calls -- `__len__`, `__getitem__`, `.to()` -- and
`ModelWrapper` unpacks it again on the way into the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler
from torch.utils.data._utils.collate import default_collate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------
# Nested-batch plumbing
# ---------------------------------------------------------------------------------
def nested_to(obj, device, non_blocking: bool = True):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: nested_to(v, device, non_blocking) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(nested_to(v, device, non_blocking) for v in obj)
    return obj


def nested_slice(obj, slc):
    if torch.is_tensor(obj):
        return obj[slc]
    if isinstance(obj, dict):
        return {k: nested_slice(v, slc) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(nested_slice(v, slc) for v in obj)
    return obj


def move_batch(batch, device):
    """`((bands, aux_cont, aux_cat), y, y_retrieved)` -> the same, on `device`."""
    x, y, retr = batch
    if isinstance(x, dict):
        x = (x["bands"], x["aux_continuous"], x["aux_categorical"])
    bands, aux_cont, aux_cat = x
    bands = {k: (v[0].to(device), v[1].to(device)) for k, v in bands.items()}
    return (
        (bands, aux_cont.to(device), [t.to(device) for t in aux_cat]),
        y.to(device),
        None if retr is None else retr.to(device),
    )


class BatchAdapter:
    """Makes a nested batch look enough like a tensor for `laplace-torch`."""

    def __init__(self, batch):
        self.batch = batch

    @staticmethod
    def _first_tensor(obj):
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                t = BatchAdapter._first_tensor(v)
                if t is not None:
                    return t
        if isinstance(obj, (list, tuple)):
            for v in obj:
                t = BatchAdapter._first_tensor(v)
                if t is not None:
                    return t
        return None

    def __len__(self) -> int:
        t = self._first_tensor(self.batch)
        if t is None:
            raise RuntimeError("BatchAdapter: no tensor found inside the batch.")
        return int(t.shape[0])

    def __getitem__(self, slc):
        return BatchAdapter(nested_slice(self.batch, slc))

    def to(self, device, non_blocking: bool = True):
        return BatchAdapter(nested_to(self.batch, device, non_blocking))

    @property
    def device(self):
        return self._first_tensor(self.batch).device

    @property
    def shape(self):
        return self._first_tensor(self.batch).shape


class _LaplaceDatasetWrapper(Dataset):
    """Drops the retrieved-baseline third element and names the input fields, so that
    `default_collate` can batch the nest without a bespoke collate for every level."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        (bands, aux_continuous, aux_categorical), label, _ = self.dataset[idx]
        return {
            "bands": bands,
            "aux_continuous": aux_continuous.float(),
            "aux_categorical": aux_categorical,
        }, label.float()


class _CollateForLaplace:
    def __init__(self, target_idx: Optional[Sequence[int]] = None):
        self.target_idx = None if target_idx is None else list(target_idx)

    def __call__(self, batch):
        xs, ys = zip(*batch)
        xb = default_collate(xs)
        yb = default_collate(ys).float()
        if self.target_idx is not None:
            yb = yb[:, self.target_idx]
        return BatchAdapter(xb), yb


class ModelWrapper(nn.Module):
    """Unpacks a `BatchAdapter`/dict back into the tuple the backbone expects."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        if isinstance(x, BatchAdapter):
            x = x.batch
        if isinstance(x, dict):
            x = (x["bands"], x["aux_continuous"], x["aux_categorical"])
        return self.model(x)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def _sod_loader_keeping_collate(self, train_loader):
    """Replacement for `Laplace._get_SoD_data_loader`.

    The library's own version rebuilds the loader with the default collate, which throws
    away `_CollateForLaplace` and then fails on the nested input. Subsampling is done
    here instead, preserving the collate.
    """
    n = len(train_loader.dataset)
    if self.n_subset >= n:
        return train_loader
    idx = torch.randperm(n)[: self.n_subset].tolist()
    return DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        sampler=SubsetRandomSampler(idx),
        num_workers=0,
        pin_memory=False,
        collate_fn=train_loader.collate_fn,
    )


# ---------------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------------
@dataclass
class LaplaceFit:
    """A fitted last-layer Laplace, plus what is needed to sample from it.

    `mode` is either "mc" (sample last-layer weights, propagate, take the sample mean and
    covariance; aleatoric noise added afterwards from held-out residuals) or "marglik"
    (let `laplace-torch` produce the predictive itself, with the observation noise fitted
    by marginal-likelihood optimization of the prior precision).
    """

    la: object
    backbone: nn.Module
    last_layer: nn.Linear
    last_layer_name: str
    mean_w: torch.Tensor
    std_w: Optional[torch.Tensor]
    mode: str
    hessian_structure: str
    n_fit: int
    prior_precision: Optional[float] = None
    sigma_noise: Optional[float] = None
    aleatoric_var: Optional[torch.Tensor] = None   # [D], standardized units

    def describe(self) -> dict:
        d = {
            "mode": self.mode,
            "hessian_structure": self.hessian_structure,
            "last_layer": self.last_layer_name,
            "n_posterior_params": int(self.mean_w.numel()),
            "n_fit_samples": int(self.n_fit),
        }
        if self.prior_precision is not None:
            d["prior_precision"] = float(self.prior_precision)
        if self.sigma_noise is not None:
            d["sigma_noise"] = float(self.sigma_noise)
        if self.aleatoric_var is not None:
            d["aleatoric_var_std"] = [float(v) for v in self.aleatoric_var.cpu()]
        return d


def _identify_last_layer(backbone: nn.Module, n_params: int):
    """Find the `nn.Linear` whose parameter count matches the Laplace posterior.

    `laplace-torch` reports the posterior over a flat vector; to sample and propagate by
    hand we need the module it belongs to. Matching on parameter count and taking the
    last match is what the notebooks did, and it is unambiguous for these architectures
    (the head is `Linear(embed_dim, D)` and nothing else in the network has that size).
    """
    candidates = [
        (name, m)
        for name, m in backbone.named_modules()
        if isinstance(m, nn.Linear)
        and int(m.weight.numel() + (0 if m.bias is None else m.bias.numel())) == n_params
    ]
    if not candidates:
        raise RuntimeError(
            f"No nn.Linear in the backbone has {n_params} parameters, so the Laplace "
            "posterior cannot be mapped back onto a module. The architecture and the "
            "checkpoint probably disagree."
        )
    return candidates[-1]


def build_laplace_loader(
    dataset,
    batch_size: int = 128,
    n_subset: Optional[int] = None,
    seed: int = 0,
    shuffle: bool = True,
    target_idx: Optional[Sequence[int]] = None,
) -> DataLoader:
    """Loader over `dataset` in the form `laplace-torch` needs.

    `n_subset` subsamples deterministically: the full 20-level Hessian over ~34k training
    soundings is minutes of GPU for a quantity that is stable at a few thousand, and the
    notebooks used 6000.
    """
    if n_subset is not None and n_subset < len(dataset):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(dataset), size=int(n_subset), replace=False)
        dataset = Subset(dataset, sorted(int(i) for i in idx))
    return DataLoader(
        _LaplaceDatasetWrapper(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,          # the nested collate is not worth a worker pool
        pin_memory=False,
        collate_fn=_CollateForLaplace(target_idx),
    )


def fit_laplace(
    backbone: nn.Module,
    fit_loader: DataLoader,
    device: torch.device,
    mode: str = "mc",
    hessian_structure: str = "diag",
    prior_precision: float = 1.0,
    optimize_prior: bool = False,
    prior_opt_steps: int = 100,
    prior_opt_lr: float = 5e-2,
) -> LaplaceFit:
    """Fit a last-layer Laplace approximation to `backbone`.

    `mode="marglik"` additionally optimizes the prior precision by marginal likelihood
    and leaves the predictive to the library; `mode="mc"` keeps the fitted precision as
    an epistemic-only posterior and expects `calibrate_aleatoric` to supply the rest.
    """
    from laplace import Laplace

    if mode not in ("mc", "marglik"):
        raise ValueError(f"mode must be 'mc' or 'marglik', got {mode!r}")

    wrapped = ModelWrapper(backbone).to(device)
    wrapped.eval()

    kwargs = dict(
        likelihood="regression",
        subset_of_weights="last_layer",
        hessian_structure=hessian_structure,
        prior_precision=prior_precision,
    )
    try:
        from laplace.curvature import CurvlinopsGGN
        kwargs["backend"] = CurvlinopsGGN
    except Exception:      # older laplace-torch; the default backend is fine
        logger.info("CurvlinopsGGN unavailable; using the default curvature backend.")

    la = Laplace(wrapped, **kwargs)
    # Subset-of-data selection must not discard our collate; see the note on the patch.
    la._get_SoD_data_loader = _sod_loader_keeping_collate.__get__(la, type(la))

    logger.info("fitting Laplace (%s, %s) ...", mode, hessian_structure)
    la.fit(fit_loader)

    if mode == "marglik" or optimize_prior:
        logger.info("optimizing prior precision by marginal likelihood ...")
        la.optimize_prior_precision(
            method="marglik", n_steps=prior_opt_steps, lr=prior_opt_lr,
        )

    n_params = int(la.mean.numel())
    name, layer = _identify_last_layer(backbone, n_params)
    logger.info("Laplace posterior over '%s' (%d parameters)", name, n_params)

    mean_w = la.mean.detach().to(device)
    std_w = None
    if mode == "mc":
        prec = getattr(la, "posterior_precision", None)
        if prec is None:
            raise RuntimeError("the fitted Laplace object exposes no posterior_precision.")
        prec = prec.detach()
        if prec.ndim == 2:                     # full/kron structures -> take the diagonal
            prec = torch.diagonal(prec)
        std_w = torch.rsqrt(torch.clamp(prec.reshape(-1).to(device), min=1e-12))

    def _scalar(x):
        if x is None:
            return None
        t = x.detach().cpu().reshape(-1) if torch.is_tensor(x) else torch.as_tensor(x).reshape(-1)
        return float(t[0]) if t.numel() else None

    return LaplaceFit(
        la=la,
        backbone=backbone,
        last_layer=layer,
        last_layer_name=name,
        mean_w=mean_w,
        std_w=std_w,
        mode=mode,
        hessian_structure=hessian_structure,
        n_fit=len(fit_loader.dataset),
        prior_precision=_scalar(getattr(la, "prior_precision", None)),
        sigma_noise=_scalar(getattr(la, "sigma_noise", None)),
    )


# ---------------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------------
@torch.no_grad()
def posterior_mean_cov(
    fit: LaplaceFit,
    x_tuple,
    n_samples: int = 256,
    target_idx: Optional[Sequence[int]] = None,
    generator: Optional[torch.Generator] = None,
):
    """Predictive mean and covariance in standardized units, `[B,D]` and `[B,D,D]`.

    In `mc` mode this is the epistemic part only: last-layer weights are drawn from the
    fitted Gaussian, the head is re-run for each draw, and the sample moments are taken.
    In `marglik` mode the library's predictive is used directly and already includes the
    fitted observation noise, so `calibrate_aleatoric` must not be applied on top.
    """
    if fit.mode == "marglik":
        pred = fit.la(BatchAdapter({
            "bands": x_tuple[0],
            "aux_continuous": x_tuple[1],
            "aux_categorical": list(x_tuple[2]),
        }))
        mean, var = pred[0], pred[1]
        mean = mean.reshape(mean.shape[0], -1)
        if var.ndim == 3:
            cov = var
        else:                                   # [B,D] variances -> diagonal covariance
            var = var.reshape(mean.shape[0], -1)
            cov = torch.diag_embed(var)
        if target_idx is not None:
            idx = torch.as_tensor(list(target_idx), device=mean.device, dtype=torch.long)
            mean = mean.index_select(1, idx)
            cov = cov.index_select(1, idx).index_select(2, idx)
        return mean, cov

    layer = fit.last_layer
    W0 = layer.weight.detach().clone()
    b0 = None if layer.bias is None else layer.bias.detach().clone()
    n_w = W0.numel()
    n_b = 0 if b0 is None else b0.numel()

    eps = torch.randn(
        n_samples, fit.mean_w.numel(),
        device=fit.mean_w.device, dtype=fit.mean_w.dtype, generator=generator,
    )
    vecs = fit.mean_w[None, :] + eps * fit.std_w[None, :]

    outs = []
    try:
        for s in range(n_samples):
            v = vecs[s]
            layer.weight.data.copy_(v[:n_w].view_as(W0))
            if b0 is not None:
                layer.bias.data.copy_(v[n_w:n_w + n_b].view_as(b0))
            outs.append(fit.backbone(x_tuple))
    finally:
        # Restore the MAP weights whatever happens; leaving a sampled head in place would
        # silently corrupt every later batch.
        layer.weight.data.copy_(W0)
        if b0 is not None:
            layer.bias.data.copy_(b0)

    Y = torch.stack(outs, dim=0)                       # [S,B,D]
    if target_idx is not None:
        idx = torch.as_tensor(list(target_idx), device=Y.device, dtype=torch.long)
        Y = Y.index_select(2, idx)
    mean = Y.mean(dim=0)
    Yc = Y - mean.unsqueeze(0)
    cov = torch.einsum("sbd,sbe->bde", Yc, Yc) / max(n_samples - 1, 1)
    return mean, cov


@torch.no_grad()
def calibrate_aleatoric(
    fit: LaplaceFit,
    loader: Iterable,
    device: torch.device,
    n_samples: int = 256,
    y_idx: Optional[Sequence[int]] = None,
    target_idx: Optional[Sequence[int]] = None,
    max_batches: Optional[int] = None,
    clamp_min: float = 1e-10,
    move_fn: Callable = move_batch,
) -> torch.Tensor:
    """Per-level observation-noise variance, from held-out residuals.

    A last-layer Laplace gives epistemic uncertainty only; on this problem it is far too
    small to cover the residuals on its own. The remaining variance is estimated as
    `E[(y - m)^2] - E[Var_epi]` per level on the *validation* partition, which is the
    method behind Tables 2-3. It is a diagonal, so it widens the marginals and leaves the
    correlation structure to the epistemic term -- a limitation worth stating rather than
    hiding, and one the flow does not share.

    The two index arguments are not the same thing and conflating them is a shape error
    waiting to happen. `y_idx` cuts the loader's 20-level *target* down to the levels this
    task predicts; `target_idx` cuts the *model's output*. The near-surface model needs
    the first and not the second -- it was trained with `target_indices=last4` and already
    emits four numbers -- so they must be settable independently.
    """
    resid_sum = None
    epi_sum = None
    n_total = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x_tuple, y, _ = move_fn(batch, device)
        y = y.reshape(y.shape[0], -1)
        if y_idx is not None:
            idx = torch.as_tensor(list(y_idx), device=y.device, dtype=torch.long)
            y = y.index_select(1, idx)
        mean, cov = posterior_mean_cov(fit, x_tuple, n_samples=n_samples, target_idx=target_idx)
        if mean.shape[1] != y.shape[1]:
            raise RuntimeError(
                f"the model emits {mean.shape[1]} outputs but the target has "
                f"{y.shape[1]} columns after slicing. Check y_idx (cuts the target) "
                "against target_idx (cuts the model output)."
            )
        v_epi = torch.diagonal(cov, dim1=-2, dim2=-1)
        r2 = (y - mean).pow(2)
        if resid_sum is None:
            resid_sum = r2.sum(dim=0)
            epi_sum = v_epi.sum(dim=0)
        else:
            resid_sum += r2.sum(dim=0)
            epi_sum += v_epi.sum(dim=0)
        n_total += int(y.shape[0])

    if n_total == 0:
        raise RuntimeError("calibrate_aleatoric saw no data.")

    sigma2 = torch.clamp(resid_sum / n_total - epi_sum / n_total, min=clamp_min)
    fit.aleatoric_var = sigma2.detach()
    return fit.aleatoric_var

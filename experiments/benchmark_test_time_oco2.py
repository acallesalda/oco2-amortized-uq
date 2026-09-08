#!/usr/bin/env python3
"""
Benchmark test-time wall-clock cost for the OCO-2 methods used in the manuscript.

What is timed
-------------
Only the *test-time predictive computation* is timed.
Model loading, figure generation, and CSV writing are excluded.
For Laplace methods, fitting the Laplace approximation is performed once up front
(because there is no saved Laplace artifact in the current workflow) and is NOT
included in the timed region.

What is compared
----------------
- xco2:
    * Laplace (scalar xCO2)
    * NF (scalar xCO2)
- profile20:
    * Laplace (20-output profile)
    * NF (20-output profile)
- last4:
    * Laplace (4-output)
    * NF (4-output)
    * Laplace (20->4)
    * NF (20->4)

Notes
-----
1. The timings are intended to match the evaluation notebooks as closely as is
   practical in a single script.
2. For profile / last4 Laplace, the benchmark mirrors the notebook's posterior
   summary computation based on Monte Carlo draws from the fitted last-layer
   Laplace approximation.
3. NASA / ACOS runtime is not measured here, because the operational solver is
   external to the notebook workflow. If you later obtain a per-sounding NASA
   runtime, pass --nasa-seconds-per-sounding and it will be added as a reference
   row / line in the outputs.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Defaults matched to the uploaded notebooks as closely as possible
# -----------------------------------------------------------------------------
DEFAULT_CODE_ROOT = Path("/workspace/project/code")
DEFAULT_DATA_ROOT = Path("/workspace/project/scratch/simulated/files")
DEFAULT_STATS_DIR = Path("/workspace/project/experiments/pipelines/stats")

DEFAULT_XCO2_WEIGHTS_DIR = Path("/workspace/project/experiments/weights/full_regressor")
DEFAULT_XCO2_NF_CKPT = Path("/workspace/project/experiments/weights/flow_regressor_nflow/best_flow_nflows.pt")

DEFAULT_PROFILE20_NF_CKPT = Path("/workspace/project/experiments/weights/flow_regressor_profiles/best_flow_profiles.pt")
DEFAULT_PROFILE20_LAPLACE_BACKBONE_CKPT = Path("/workspace/project/experiments/weights/full_regressor_profiles/global_best/net_profiles_old.pt")
DEFAULT_LAST4_BACKBONE_CANDIDATES = [
    "/workspace/project/experiments/weights/full_regressor_profiles_last4_bo/net_profiles.pt",
    "/workspace/project/experiments/weights/full_regressor_profiles_last4/net_profiles.pt",
]
DEFAULT_LAST4_NF_CANDIDATES = [
    "/workspace/project/experiments/weights/flow_regressor_profiles_nf4_bo_warm_v1/best_nf4.pt",
    "/workspace/project/experiments/weights/flow_regressor_profiles_nf4_bo_warm/best_nf4.pt",
    "/workspace/project/experiments/weights/flow_regressor_profiles_last4/best_flow_profiles.pt",
]
DEFAULT_FULL20_BACKBONE_CANDIDATES = [
    "/workspace/project/experiments/weights/full_regressor_profiles/global_best/net_profiles_old.pt",
    "/workspace/project/experiments/weights/full_regressor_profiles/global_best/net_profiles.pt",
]
DEFAULT_FULL20_NF_CANDIDATES = [
    "/workspace/project/experiments/weights/flow_regressor_profiles/best_flow_profiles.pt",
]

# scalar standardization from xco2.ipynb
Y_MU = 0.0004108437
Y_SIGMA = 2.1281e-05

# last-4 levels from the refit notebook
LAST4_IDX0 = [16, 17, 18, 19]
LAST4_LEVELS = [i + 1 for i in LAST4_IDX0]


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_first_existing(candidates: Sequence[str | Path], label: str) -> Path:
    cands = [Path(p) for p in candidates]
    for p in cands:
        if p.exists():
            print(f"[resolved] {label}: {p}")
            return p
    print(f"[warning] no candidate exists for {label}; falling back to {cands[0]}")
    return cands[0]


def maybe_subset_indices(n_total: int, n_subset: Optional[int], seed: int) -> np.ndarray:
    if n_subset is None or n_subset >= n_total:
        return np.arange(n_total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n_subset, replace=False))


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_nf_sample_shape(samples: torch.Tensor, expected_s: Optional[int] = None) -> torch.Tensor:
    if samples.dim() == 2:
        samples = samples.unsqueeze(0)
    if expected_s is not None and samples.shape[0] != expected_s and samples.shape[1] == expected_s:
        samples = samples.permute(1, 0, 2).contiguous()
    return samples


def sample_moments(samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    samples = normalize_nf_sample_shape(samples)
    mu = samples.mean(dim=0)
    xc = samples - mu.unsqueeze(0)
    cov = torch.einsum("sbd,sbe->bde", xc, xc) / max(samples.shape[0] - 1, 1)
    return mu, cov


def _nested_to(obj: Any, device: torch.device, non_blocking: bool = True) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: _nested_to(v, device, non_blocking) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_nested_to(v, device, non_blocking) for v in obj)
    return obj


def _nested_slice(obj: Any, slc: slice) -> Any:
    if torch.is_tensor(obj):
        return obj[slc]
    if isinstance(obj, dict):
        return {k: _nested_slice(v, slc) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_nested_slice(v, slc) for v in obj)
    return obj


# -----------------------------------------------------------------------------
# Scalar xCO2 helpers (mirrors xco2.ipynb)
# -----------------------------------------------------------------------------
class ScalarModelInput:
    """Wrapper so Laplace can slice and move nested spectral inputs."""

    def __init__(self, bands, aux_continuous, aux_categorical):
        self.bands = bands
        self.aux_continuous = aux_continuous
        self.aux_categorical = aux_categorical
        self.shape = aux_continuous.shape

    @property
    def device(self):
        return self.aux_continuous.device

    def to(self, device):
        bands_device = {k: (v[0].to(device), v[1].to(device)) for k, v in self.bands.items()}
        aux_continuous_device = self.aux_continuous.to(device)
        aux_categorical_device = [tensor.to(device) for tensor in self.aux_categorical]
        return ScalarModelInput(bands_device, aux_continuous_device, aux_categorical_device)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.aux_continuous.shape[0])
            bands_sliced = {
                k: (v[0][start:stop:step], v[1][start:stop:step]) for k, v in self.bands.items()
            }
            aux_cont_sliced = self.aux_continuous[start:stop:step]
            aux_cat_sliced = [tensor[start:stop:step] for tensor in self.aux_categorical]
            return ScalarModelInput(bands_sliced, aux_cont_sliced, aux_cat_sliced)
        bands_idx = {k: (v[0][idx], v[1][idx]) for k, v in self.bands.items()}
        aux_cont_idx = self.aux_continuous[idx]
        aux_cat_idx = [tensor[idx] for tensor in self.aux_categorical]
        return ScalarModelInput(bands_idx, aux_cont_idx, aux_cat_idx)

    def unpack(self):
        return (self.bands, self.aux_continuous, self.aux_categorical)


class ScalarLaplaceDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        (bands, aux_continuous, aux_categorical), label, _ = self.dataset[idx]
        model_input = ScalarModelInput(bands, aux_continuous, aux_categorical)
        return model_input, label


class ScalarModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if isinstance(x, ScalarModelInput):
            return self.model(x.unpack())
        return self.model(x)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def scalar_custom_collate_fn(batch):
    inputs, labels = zip(*batch)
    all_bands = [inp.bands for inp in inputs]
    all_aux_continuous = [inp.aux_continuous for inp in inputs]
    all_aux_categorical = [inp.aux_categorical for inp in inputs]

    batched_bands = {}
    band_names = all_bands[0].keys()
    for band_name in band_names:
        band_data = [bands[band_name][0] for bands in all_bands]
        band_constants = [bands[band_name][1] for bands in all_bands]
        batched_bands[band_name] = (torch.stack(band_data), torch.stack(band_constants))

    batched_aux_continuous = torch.stack(all_aux_continuous)
    num_cat_features = len(all_aux_categorical[0])
    batched_aux_categorical = []
    for i in range(num_cat_features):
        cat_feature_batch = torch.stack([aux_cat[i] for aux_cat in all_aux_categorical])
        batched_aux_categorical.append(cat_feature_batch)

    batched_input = ScalarModelInput(batched_bands, batched_aux_continuous, batched_aux_categorical)
    batched_labels = torch.stack(labels)
    return batched_input, batched_labels


def move_to_device_scalar(batch, device: torch.device):
    (bands, aux_continuous, aux_categorical), label, retr_label = batch
    bands_gpu = {k: (v[0].to(device), v[1].to(device)) for k, v in bands.items()}
    aux_continuous_gpu = aux_continuous.to(device)
    aux_categorical_gpu = [b.to(device) for b in aux_categorical]
    return (bands_gpu, aux_continuous_gpu, aux_categorical_gpu), label.to(device), retr_label.to(device)


# -----------------------------------------------------------------------------
# Profile helpers (shared by profile20 and last4)
# -----------------------------------------------------------------------------

def move_to_device_profiles(batch, device: torch.device):
    x_tuple, y, y_retr = batch
    bands_dict, aux_cont, aux_cat = x_tuple
    bands = {k: (b.to(device), c.to(device)) for k, (b, c) in bands_dict.items()}
    aux_cont = aux_cont.to(device)
    aux_cat = [t.to(device) for t in aux_cat]
    y = y.to(device)
    y_retr = None if y_retr is None else y_retr.to(device)
    return (bands, aux_cont, aux_cat), y, y_retr


class BatchAdapter:
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

    def __len__(self):
        t = self._first_tensor(self.batch)
        if t is None:
            raise RuntimeError("BatchAdapter: could not find a Tensor inside the batch.")
        return int(t.shape[0])

    def __getitem__(self, slc):
        return BatchAdapter(_nested_slice(self.batch, slc))

    def to(self, device, non_blocking=True):
        return BatchAdapter(_nested_to(self.batch, device, non_blocking))


class TargetSliceDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, idx0: Optional[Sequence[int]] = None):
        self.dataset = dataset
        self.idx0 = None if idx0 is None else list(map(int, idx0))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        (bands, aux_continuous, aux_categorical), y, _ = self.dataset[idx]
        y = y.float()
        if self.idx0 is not None:
            y = y[self.idx0]
        return {
            "bands": bands,
            "aux_continuous": aux_continuous.float(),
            "aux_categorical": aux_categorical,
        }, y


class CollateForLaplace:
    def __call__(self, batch):
        from torch.utils.data._utils.collate import default_collate

        xs, ys = zip(*batch)
        xb = default_collate(xs)
        yb = default_collate(ys).float()
        return BatchAdapter(xb), yb


class ModelWrapperProfiles(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, xb):
        if isinstance(xb, BatchAdapter):
            xb = xb.batch
        x_tuple = (xb["bands"], xb["aux_continuous"], xb["aux_categorical"])
        return self.backbone(x_tuple)


def _get_sod_data_loader_keep_collate(self, data_loader, batch_size=None, subset_of_data=None, shuffle=True):
    """Patch for laplace package so custom collate_fn is preserved."""
    from torch.utils.data import DataLoader

    if batch_size is None:
        batch_size = data_loader.batch_size
    collate_fn = getattr(data_loader, "collate_fn", None)

    if subset_of_data is not None:
        if isinstance(subset_of_data, int):
            n_data = len(data_loader.dataset)
            subset_of_data = min(n_data, subset_of_data)
            subset_indices = np.random.choice(n_data, size=subset_of_data, replace=False)
            dataset = torch.utils.data.Subset(data_loader.dataset, subset_indices)
        else:
            dataset = subset_of_data
    else:
        dataset = data_loader.dataset

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )


def make_laplace_train_loader_profiles(base_dataset, idx0=None, n_subset=None, batch_size=128, seed=0):
    from torch.utils.data import DataLoader, Subset

    ds = TargetSliceDatasetWrapper(base_dataset, idx0=idx0)
    idx = maybe_subset_indices(len(ds), n_subset=n_subset, seed=seed)
    if len(idx) < len(ds):
        ds = Subset(ds, idx.tolist())
    collate_for_laplace = CollateForLaplace()
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_for_laplace,
    )


def linear_param_count(m: nn.Linear) -> int:
    return int(m.weight.numel() + (m.bias.numel() if m.bias is not None else 0))


def identify_laplace_last_layer(backbone, la, device: torch.device):
    n_params = int(la.mean.numel())
    linears = [
        (name, module)
        for (name, module) in backbone.named_modules()
        if isinstance(module, nn.Linear) and linear_param_count(module) == n_params
    ]
    if not linears:
        raise RuntimeError(f"Could not find nn.Linear with param_count={n_params}.")

    last_name, last_layer = linears[-1]
    mean_w = la.mean.detach().to(device)

    prec = getattr(la, "posterior_precision", None)
    if prec is None:
        raise RuntimeError("Laplace object missing posterior_precision.")
    prec = prec.detach()
    if prec.ndim == 2:
        prec = torch.diag(prec)
    prec = prec.reshape(-1).to(device)
    std_w = torch.rsqrt(torch.clamp(prec, min=1e-12))

    print(f"[Laplace] using last_layer='{last_name}' with n_params={n_params}")
    return {
        "name": last_name,
        "layer": last_layer,
        "mean_w": mean_w,
        "std_w": std_w,
        "n_params": n_params,
    }


@torch.no_grad()
def laplace_batch_mean_cov_std(backbone, lapmeta, x_tuple, device: torch.device, s: int = 256):
    last_layer = lapmeta["layer"]
    mean_w = lapmeta["mean_w"]
    std_w = lapmeta["std_w"]
    n_params = lapmeta["n_params"]

    w0 = last_layer.weight.detach().clone()
    b0 = last_layer.bias.detach().clone() if last_layer.bias is not None else None
    w_n = w0.numel()
    b_n = 0 if b0 is None else b0.numel()

    eps = torch.randn(s, n_params, device=device, dtype=mean_w.dtype)
    vecs = mean_w[None, :] + eps * std_w[None, :]

    outs = []
    for i in range(s):
        v = vecs[i]
        last_layer.weight.data.copy_(v[:w_n].view_as(w0))
        if b0 is not None:
            last_layer.bias.data.copy_(v[w_n:w_n + b_n].view_as(b0))
        outs.append(backbone(x_tuple))

    y = torch.stack(outs, dim=0)
    m = y.mean(dim=0)
    xc = y - m.unsqueeze(0)
    cov = torch.einsum("sbd,sbe->bde", xc, xc) / max(s - 1, 1)

    last_layer.weight.data.copy_(w0)
    if b0 is not None:
        last_layer.bias.data.copy_(b0)

    return m, cov


def infer_backbone_dims_from_sd(sd: Dict[str, torch.Tensor]) -> Dict[str, Optional[int]]:
    embed_dim = None
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and re.search(r"encoder\.band_encoders\.b\d+\.mlp\.2\.weight$", k):
            embed_dim = v.shape[0]
            break
    if embed_dim is None and "predictor.0.weight" in sd:
        embed_dim = sd["predictor.0.weight"].shape[0]
    if embed_dim is None and "predictor.0.bias" in sd:
        embed_dim = sd["predictor.0.bias"].shape[0]

    nlayers = 1
    layer_idxs = []
    for k in sd.keys():
        m = re.search(r"fusion\.layers\.(\d+)\.", k)
        if m:
            layer_idxs.append(int(m.group(1)))
    if layer_idxs:
        nlayers = max(layer_idxs) + 1

    out_dim = None
    if "predictor.4.weight" in sd:
        out_dim = sd["predictor.4.weight"].shape[0]

    return {
        "embed_dim": int(embed_dim) if embed_dim is not None else 256,
        "nlayers": int(nlayers),
        "out_dim": int(out_dim) if out_dim is not None else None,
    }


def _strip_module_prefix(sd):
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _load_state_dict_blob(ckpt_path):
    blob = torch.load(ckpt_path, map_location="cpu")
    if isinstance(blob, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in blob and isinstance(blob[key], dict):
                return _strip_module_prefix(blob[key])
    return _strip_module_prefix(blob)


def _load_matching(module, state_dict: dict, name: str):
    own = module.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in own and v.shape == own[k].shape}
    skipped = [k for k in own.keys() if k not in filtered]
    unexpected = [k for k in state_dict.keys() if k not in own]
    module.load_state_dict(filtered, strict=False)
    print(f"[load:{name}] copied={len(filtered)}/{len(own)} skipped={len(skipped)} unexpected={len(unexpected)}")


def _nf_ckpt_get_cfg(payload):
    cfg = payload.get("cfg", {}) if isinstance(payload, dict) else {}
    model_cfg = dict(cfg.get("model", {}))
    nflow_cfg = dict(cfg.get("nflow", {}))
    task_cfg = dict(cfg.get("task", {}))

    if "layers" in cfg and "num_layers" not in nflow_cfg:
        nflow_cfg["num_layers"] = cfg["layers"]
    if "hidden" in cfg and "hidden_features" not in nflow_cfg:
        nflow_cfg["hidden_features"] = cfg["hidden"]
    if "bins" in cfg and "num_bins" not in nflow_cfg:
        nflow_cfg["num_bins"] = cfg["bins"]
    if "tail_bound" in cfg and "tail_bound" not in nflow_cfg:
        nflow_cfg["tail_bound"] = cfg["tail_bound"]
    if "standardize_y" in cfg and "standardize_y" not in task_cfg:
        task_cfg["standardize_y"] = cfg["standardize_y"]

    return cfg, model_cfg, nflow_cfg, task_cfg


def slice_last4_mean(mu: torch.Tensor) -> torch.Tensor:
    if mu.shape[-1] == 4:
        return mu
    idx = torch.tensor(LAST4_IDX0, dtype=torch.long, device=mu.device)
    return mu.index_select(-1, idx)


def slice_last4_cov(cov: torch.Tensor) -> torch.Tensor:
    if cov.shape[-1] == 4 and cov.shape[-2] == 4:
        return cov
    idx = torch.tensor(LAST4_IDX0, dtype=torch.long, device=cov.device)
    return cov.index_select(-2, idx).index_select(-1, idx)


# -----------------------------------------------------------------------------
# Benchmark container
# -----------------------------------------------------------------------------
@dataclass
class BenchmarkCase:
    name: str
    scenario: str
    predictor: Callable[[Any], Any]
    make_loader: Callable[[int], Iterable]
    notes: str = ""
    posterior_samples: Optional[int] = None


# -----------------------------------------------------------------------------
# Scenario setup: xCO2
# -----------------------------------------------------------------------------

def setup_xco2_cases(args, device: torch.device) -> List[BenchmarkCase]:
    import yaml
    from laplace import Laplace
    from data.loading.spectral_loader import build_dataset
    from modeling.full_regressor import FullRegressor
    from nflows import flows, distributions, transforms

    class NFHead(nn.Module):
        def __init__(self, context_dim: int, num_layers: int, hidden_features: int, num_bins: int, tail_bound: float):
            super().__init__()
            layers = []
            for _ in range(num_layers):
                layers.append(
                    transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                        features=1,
                        hidden_features=hidden_features,
                        context_features=context_dim,
                        num_bins=num_bins,
                        tails="linear",
                        tail_bound=tail_bound,
                    )
                )
                layers.append(transforms.RandomPermutation(features=1))
            transform = transforms.CompositeTransform(layers)
            base = distributions.StandardNormal(shape=[1])
            self.flow = flows.Flow(transform, distribution=base)

        def log_prob(self, y_std: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
            return self.flow.log_prob(inputs=y_std.unsqueeze(-1), context=context).squeeze(-1)

        def sample(self, n: int, context: torch.Tensor) -> torch.Tensor:
            s = self.flow.sample(num_samples=n, context=context)
            if s.dim() == 3 and s.shape[-1] == 1:
                s = s.squeeze(-1)
            if s.shape[0] == context.shape[0]:
                s = s.T
            return s

    class LoadedNFScalar(nn.Module):
        def __init__(self, backbone: FullRegressor, nf: NFHead):
            super().__init__()
            self.backbone = backbone
            self.nf = nf

        def context(self, x_tuple):
            z = self.backbone.encoder(x_tuple)
            if z.dim() > 2:
                z = z.flatten(1)
            return z

        @torch.no_grad()
        def sample_std(self, x_tuple, n: int = 100):
            ctx = self.context(x_tuple)
            return self.nf.sample(n=n, context=ctx)

    def load_full_regressor(weights_dir: Path):
        ckpt_path = weights_dir / "net.pt"
        hparams_path = weights_dir / "best_hparams.yaml"
        model_params = dict(embed_dim=256, nhead=8, nlayers=1, aux_dim=32)
        if hparams_path.exists():
            with open(hparams_path, "r") as f:
                hp = yaml.safe_load(f) or {}
            bp = hp.get("best_params", hp)
            model_params["embed_dim"] = int(bp.get("hyps.embed_dim", bp.get("embed_dim", model_params["embed_dim"])))
            model_params["nhead"] = int(bp.get("hyps.nhead", bp.get("nhead", model_params["nhead"])))
            model_params["nlayers"] = int(bp.get("hyps.nlayers", bp.get("nlayers", model_params["nlayers"])))
            model_params["aux_dim"] = int(bp.get("hyps.aux_dim", bp.get("aux_dim", model_params["aux_dim"])))

        model = FullRegressor(**model_params)
        obj = torch.load(ckpt_path, map_location="cpu")
        state = obj.get("model_state_dict", obj)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        return model, model_params

    def _infer_scalar_nf_backbone_cfg(backbone_sd: dict, fallback_cfg: dict) -> dict:
        """Infer scalar NF backbone dimensions from the checkpoint when possible.

        This avoids assuming that the scalar NF checkpoint uses the same encoder
        width as the deterministic scalar regressor used for Laplace.
        """
        cfg = dict(fallback_cfg)

        aux_w = backbone_sd.get("encoder.aux_encoder.encoder.weight")
        if torch.is_tensor(aux_w) and aux_w.ndim == 2:
            cfg["embed_dim"] = int(aux_w.shape[0])
            cfg["aux_dim"] = int(aux_w.shape[1])

        layer_pat = re.compile(r"encoder\.fusion\.transformer\.layers\.(\d+)\.")
        layer_ids = []
        for k in backbone_sd.keys():
            m = layer_pat.match(k)
            if m:
                layer_ids.append(int(m.group(1)))
        if layer_ids:
            cfg["nlayers"] = max(layer_ids) + 1

        cfg["nhead"] = int(cfg.get("nhead", 8))
        return cfg

    def load_nf_from_train_nf_checkpoint(ckpt_path, backbone_cfg, nflow_cfg, sample_x_tuple):
        ckpt = torch.load(ckpt_path, map_location=device)
        if "state_dict" not in ckpt:
            raise ValueError(f"Expected checkpoint with key 'state_dict'. Got keys: {list(ckpt.keys())}")

        state = ckpt["state_dict"]
        if "backbone" not in state or "nf" not in state:
            raise ValueError(f"Expected state_dict with 'backbone' and 'nf'. Got keys: {list(state.keys())}")

        backbone_sd = state["backbone"]
        nf_sd = state["nf"]

        ckpt_cfg, model_cfg_ckpt, nflow_cfg_ckpt, _ = _nf_ckpt_get_cfg(ckpt)

        resolved_backbone_cfg = dict(backbone_cfg)
        for k in ("aux_dim", "embed_dim", "nhead", "nlayers"):
            if k in model_cfg_ckpt:
                resolved_backbone_cfg[k] = model_cfg_ckpt[k]
            elif isinstance(ckpt_cfg, dict) and k in ckpt_cfg:
                resolved_backbone_cfg[k] = ckpt_cfg[k]
        resolved_backbone_cfg = _infer_scalar_nf_backbone_cfg(backbone_sd, resolved_backbone_cfg)

        resolved_nflow_cfg = dict(nflow_cfg)
        if "hidden_features" in nflow_cfg_ckpt:
            resolved_nflow_cfg["hidden"] = nflow_cfg_ckpt["hidden_features"]
        elif isinstance(ckpt_cfg, dict) and "hidden" in ckpt_cfg:
            resolved_nflow_cfg["hidden"] = ckpt_cfg["hidden"]
        if "num_layers" in nflow_cfg_ckpt:
            resolved_nflow_cfg["layers"] = nflow_cfg_ckpt["num_layers"]
        elif isinstance(ckpt_cfg, dict) and "layers" in ckpt_cfg:
            resolved_nflow_cfg["layers"] = ckpt_cfg["layers"]
        if "num_bins" in nflow_cfg_ckpt:
            resolved_nflow_cfg["bins"] = nflow_cfg_ckpt["num_bins"]
        elif isinstance(ckpt_cfg, dict) and "bins" in ckpt_cfg:
            resolved_nflow_cfg["bins"] = ckpt_cfg["bins"]
        if "tail_bound" in nflow_cfg_ckpt:
            resolved_nflow_cfg["tail_bound"] = nflow_cfg_ckpt["tail_bound"]
        elif isinstance(ckpt_cfg, dict) and "tail_bound" in ckpt_cfg:
            resolved_nflow_cfg["tail_bound"] = ckpt_cfg["tail_bound"]

        print("Scalar NF backbone cfg (resolved):", resolved_backbone_cfg)
        print("Scalar NF flow cfg (resolved):", resolved_nflow_cfg)

        backbone = FullRegressor(
            aux_dim=int(resolved_backbone_cfg["aux_dim"]),
            embed_dim=int(resolved_backbone_cfg["embed_dim"]),
            nhead=int(resolved_backbone_cfg["nhead"]),
            nlayers=int(resolved_backbone_cfg["nlayers"]),
        ).to(device)
        backbone.eval()

        encoder_sd = {k[len("encoder."):]: v for k, v in backbone_sd.items() if k.startswith("encoder.")}
        if len(encoder_sd) == 0:
            raise ValueError("No encoder.* keys found in scalar NF backbone checkpoint.")
        backbone.encoder.load_state_dict(encoder_sd, strict=True)

        with torch.no_grad():
            context_dim = backbone.encoder(sample_x_tuple)
            if context_dim.dim() > 2:
                context_dim = context_dim.flatten(1)
            context_dim = context_dim.shape[-1]

        nf = NFHead(
            context_dim=context_dim,
            num_layers=int(resolved_nflow_cfg["layers"]),
            hidden_features=int(resolved_nflow_cfg["hidden"]),
            num_bins=int(resolved_nflow_cfg["bins"]),
            tail_bound=float(resolved_nflow_cfg["tail_bound"]),
        ).to(device)
        nf.eval()
        nf.load_state_dict(nf_sd, strict=True)
        return LoadedNFScalar(backbone, nf).to(device).eval()

    weights_dir = Path(args.xco2_weights_dir)
    nf_ckpt = Path(args.xco2_nf_ckpt)
    data_root = Path(args.data_root)

    model, model_params = load_full_regressor(weights_dir)
    annotations_file = "aux_data.csv"
    train_dataset = build_dataset(data_root / "train", annotations_file)
    test_dataset = build_dataset(data_root / "test", annotations_file)

    generator = torch.Generator().manual_seed(args.seed)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, _ = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator=generator)

    laplace_train_dataset = ScalarLaplaceDatasetWrapper(train_subset)
    if args.laplace_fit_subset is not None and args.laplace_fit_subset < len(laplace_train_dataset):
        idx = maybe_subset_indices(len(laplace_train_dataset), args.laplace_fit_subset, args.seed)
        laplace_train_dataset = torch.utils.data.Subset(laplace_train_dataset, idx.tolist())

    laplace_train_loader = torch.utils.data.DataLoader(
        laplace_train_dataset,
        batch_size=args.scalar_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=scalar_custom_collate_fn,
    )

    wrapped_model = ScalarModelWrapper(model).to(device).eval()
    print("Fitting scalar Laplace (not timed)...")
    la = Laplace(
        wrapped_model,
        "regression",
        hessian_structure=args.scalar_laplace_hessian,
        subset_of_weights=args.scalar_laplace_subset,
    )
    la.fit(laplace_train_loader)
    if args.scalar_optimize_prior:
        print("Optimizing scalar Laplace prior precision (not timed)...")
        la.optimize_prior_precision(
            method="marglik",
            n_steps=args.scalar_prior_opt_steps,
            lr=args.scalar_prior_opt_lr,
            init_prior_prec=args.scalar_prior_opt_init,
        )

    tmp_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=max(args.batch_sizes),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    warm_batch = next(iter(tmp_loader))
    (x_bands, x_aux_cont, x_aux_cat), _, _ = move_to_device_scalar(warm_batch, device=device)
    sample_x_tuple = (x_bands, x_aux_cont, x_aux_cat)

    backbone_cfg = {
        "aux_dim": model_params["aux_dim"],
        "embed_dim": model_params["embed_dim"],
        "nhead": model_params["nhead"],
        "nlayers": model_params["nlayers"],
    }
    nflow_cfg = {
        "hidden": args.scalar_nf_hidden,
        "layers": args.scalar_nf_layers,
        "bins": args.scalar_nf_bins,
        "tail_bound": args.scalar_nf_tail_bound,
    }
    scalar_nf = load_nf_from_train_nf_checkpoint(nf_ckpt, backbone_cfg, nflow_cfg, sample_x_tuple)

    def make_loader(batch_size: int):
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    @torch.no_grad()
    def predict_laplace(batch):
        (x_bands, x_aux_cont, x_aux_cat), _, _ = move_to_device_scalar(batch, device=device)
        pred = la(ScalarModelInput(x_bands, x_aux_cont, x_aux_cat))
        return pred[0], pred[1]

    @torch.no_grad()
    def predict_nf(batch):
        x_tuple, _, _ = move_to_device_scalar(batch, device=device)
        nf_samples = scalar_nf.sample_std(x_tuple, n=args.scalar_nf_samples)
        nf_samples = nf_samples.to(device=device)
        nf_mean = nf_samples.mean(dim=0)
        nf_var = nf_samples.var(dim=0, unbiased=False)
        return nf_mean, nf_var

    return [
        BenchmarkCase(
            name="xCO2 Laplace",
            scenario="xco2",
            predictor=predict_laplace,
            make_loader=make_loader,
            notes="scalar posterior predictive via la(...)",
            posterior_samples=None,
        ),
        BenchmarkCase(
            name="xCO2 NF",
            scenario="xco2",
            predictor=predict_nf,
            make_loader=make_loader,
            notes="scalar NF samples -> mean/variance",
            posterior_samples=args.scalar_nf_samples,
        ),
    ]


# -----------------------------------------------------------------------------
# Scenario setup: 20D profile
# -----------------------------------------------------------------------------

def setup_profile20_cases(args, device: torch.device) -> List[BenchmarkCase]:
    from laplace import Laplace
    from laplace.curvature import CurvlinopsGGN
    from data.loading.spectral_loader import test_loader_profiles, train_val_loader_profiles
    from modeling.full_regressor_profiles import FullRegressorProfiles
    from modeling.flow_regressor_profiles_nf import FeatureNetProfiles, NFProfiles

    data_root = Path(args.data_root)
    stats_dir = Path(args.stats_dir)
    nf_ckpt = Path(args.profile20_nf_ckpt)
    laplace_ckpt = None if args.profile20_laplace_backbone_ckpt in ("", "none", "None") else Path(args.profile20_laplace_backbone_ckpt)

    def make_test_loader(batch_size: int):
        return test_loader_profiles(
            root=str(data_root),
            annotations_filename="aux_data.csv",
            val_batch_size=batch_size,
            seed=args.seed,
            stats_dir=stats_dir,
        )

    train_dl, _ = train_val_loader_profiles(
        root=str(data_root),
        annotations_filename="aux_data.csv",
        train_batch_size=args.profile_train_batch_size,
        val_batch_size=args.profile_val_batch_size,
        seed=args.seed,
        train_size=0.9,
        stats_dir=stats_dir,
    )

    warm_loader = make_test_loader(max(args.batch_sizes))
    batch0 = next(iter(warm_loader))
    x0, _, _ = move_to_device_profiles(batch0, device)

    print("Loading 20D NF checkpoint...")
    payload = torch.load(nf_ckpt, map_location="cpu")
    nf_cfg = payload.get("cfg", {})
    nf_backbone = FullRegressorProfiles(
        aux_dim=int(nf_cfg.get("aux_dim", 32)),
        embed_dim=int(nf_cfg.get("embed_dim", 256)),
        nhead=int(nf_cfg.get("nhead", 8)),
        nlayers=int(nf_cfg.get("nlayers", 1)),
        out_dim=int(nf_cfg.get("out_dim", 20)),
    ).to(device)
    feat = FeatureNetProfiles(nf_backbone).to(device)
    ctx_dim = feat.infer_context_dim(x0)
    nf = NFProfiles(
        feature_net=feat,
        context_dim=ctx_dim,
        out_dim=int(nf_cfg.get("out_dim", 20)),
        num_layers=int(nf_cfg.get("layers", 7)),
        hidden_features=int(nf_cfg.get("hidden", 640)),
        num_bins=int(nf_cfg.get("bins", 13)),
        tail_bound=float(nf_cfg.get("tail_bound", 10.0)),
        standardize_y=bool(nf_cfg.get("standardize_y", True)),
    ).to(device)
    nf.load_state_dict(payload["nf"], strict=True)
    feat.load_state_dict(payload["feat"], strict=True)
    nf_backbone.load_state_dict(payload["backbone"], strict=False)
    nf.eval()

    print("Loading / building 20D Laplace backbone...")
    if laplace_ckpt is None:
        laplace_backbone = copy.deepcopy(nf_backbone).to(device)
    else:
        from modeling.legacy_full_regressor_profiles import FullRegressorProfilesLegacy

        laplace_backbone = FullRegressorProfilesLegacy(
            aux_dim=32,
            embed_dim=256,
            nhead=8,
            nlayers=1,
            out_dim=20,
        ).to(device)

        _ = laplace_backbone(x0)  # warm forward for lazy modules

        ckpt = torch.load(laplace_ckpt, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        state = _strip_module_prefix(state)

        incompatible = laplace_backbone.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print("[profile20 laplace backbone]")
            print("  missing:", incompatible.missing_keys[:10])
            print("  unexpected:", incompatible.unexpected_keys[:10])

    laplace_backbone.eval()

    lap_train_loader = make_laplace_train_loader_profiles(
        train_dl.dataset,
        idx0=None,
        n_subset=args.laplace_fit_subset,
        batch_size=args.profile_train_batch_size,
        seed=args.seed,
    )
    wrapped = ModelWrapperProfiles(laplace_backbone).to(device)
    la = Laplace(
        wrapped,
        likelihood="regression",
        subset_of_weights="last_layer",
        hessian_structure="diag",
        backend=CurvlinopsGGN,
        prior_precision=1.0,
    )
    la._get_SoD_data_loader = _get_sod_data_loader_keep_collate.__get__(la, type(la))
    print("Fitting 20D Laplace (not timed)...")
    la.fit(lap_train_loader)
    lapmeta = identify_laplace_last_layer(laplace_backbone, la, device)

    def make_loader(batch_size: int):
        return make_test_loader(batch_size)

    @torch.no_grad()
    def predict_laplace(batch):
        x_tuple, _, _ = move_to_device_profiles(batch, device)
        mu, cov = laplace_batch_mean_cov_std(
            laplace_backbone,
            lapmeta,
            x_tuple,
            device=device,
            s=args.profile_laplace_samples,
        )
        return mu, cov

    @torch.no_grad()
    def predict_nf(batch):
        x_tuple, _, _ = move_to_device_profiles(batch, device)
        s = normalize_nf_sample_shape(nf.sample(x_tuple, n=args.profile_nf_samples), expected_s=args.profile_nf_samples)
        mu, cov = sample_moments(s)
        return mu, cov

    return [
        BenchmarkCase(
            name="Profile20 Laplace",
            scenario="profile20",
            predictor=predict_laplace,
            make_loader=make_loader,
            notes="20D profile Laplace via MC weight samples",
            posterior_samples=args.profile_laplace_samples,
        ),
        BenchmarkCase(
            name="Profile20 NF",
            scenario="profile20",
            predictor=predict_nf,
            make_loader=make_loader,
            notes="20D NF samples -> moments",
            posterior_samples=args.profile_nf_samples,
        ),
    ]


# -----------------------------------------------------------------------------
# Scenario setup: last4 vs marginalized20
# -----------------------------------------------------------------------------

def setup_last4_cases(args, device: torch.device) -> List[BenchmarkCase]:
    from laplace import Laplace
    from laplace.curvature import CurvlinopsGGN
    from data.loading.spectral_loader import test_loader_profiles, train_val_loader_profiles
    from modeling.full_regressor_profiles import FullRegressorProfiles
    from modeling.flow_regressor_profiles_nf import FeatureNetProfiles, NFProfiles

    data_root = Path(args.data_root)
    stats_dir = Path(args.stats_dir)

    last4_backbone_ckpt = resolve_first_existing(DEFAULT_LAST4_BACKBONE_CANDIDATES, "LAST4_BACKBONE_CKPT")
    last4_nf_ckpt = resolve_first_existing(DEFAULT_LAST4_NF_CANDIDATES, "LAST4_NF_CKPT")
    full20_backbone_ckpt = resolve_first_existing(DEFAULT_FULL20_BACKBONE_CANDIDATES, "FULL20_BACKBONE_CKPT")
    full20_nf_ckpt = resolve_first_existing(DEFAULT_FULL20_NF_CANDIDATES, "FULL20_NF_CKPT")

    def make_test_loader(batch_size: int):
        return test_loader_profiles(
            root=str(data_root),
            annotations_filename="aux_data.csv",
            val_batch_size=batch_size,
            seed=args.seed,
            stats_dir=stats_dir,
        )

    train_dl, _ = train_val_loader_profiles(
        root=str(data_root),
        annotations_filename="aux_data.csv",
        train_batch_size=args.profile_train_batch_size,
        val_batch_size=args.profile_val_batch_size,
        seed=args.seed,
        train_size=0.9,
        stats_dir=stats_dir,
    )

    warm_loader = make_test_loader(max(args.batch_sizes))
    warm_batch = next(iter(warm_loader))
    warm_x_tuple, _, _ = move_to_device_profiles(warm_batch, device)

    def load_backbone_last4(ckpt_path, device):
        payload = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload:
            sd = _strip_module_prefix(payload["state_dict"])
            cfg_meta = payload.get("config", {})
        else:
            sd = _load_state_dict_blob(str(ckpt_path))
            cfg_meta = {}

        dims = infer_backbone_dims_from_sd(sd)
        model = FullRegressorProfiles(
            aux_dim=int(cfg_meta.get("aux_dim", 32)),
            embed_dim=int(cfg_meta.get("embed_dim", dims["embed_dim"])),
            nhead=int(cfg_meta.get("nhead", 8)),
            nlayers=int(cfg_meta.get("nlayers", dims["nlayers"])),
            out_dim=int(cfg_meta.get("out_dim", 4) or 4),
        ).to(device)
        model.load_state_dict(sd, strict=False)
        model.eval()
        return model

    def load_backbone_full20(ckpt_path, device):
        if args.last4_full20_variant == "legacy":
            from modeling.legacy_full_regressor_profiles import FullRegressorProfilesLegacy
            backbone = FullRegressorProfilesLegacy(
                aux_dim=32,
                embed_dim=256,
                nhead=8,
                nlayers=1,
                out_dim=20,
            ).to(device)
        else:
            backbone = FullRegressorProfiles(
                aux_dim=32,
                embed_dim=256,
                nhead=8,
                nlayers=1,
                out_dim=20,
            ).to(device)
        _ = backbone(warm_x_tuple)
        sd = _load_state_dict_blob(str(ckpt_path))
        backbone.load_state_dict(sd, strict=False)
        backbone.eval()
        return backbone

    def build_nf_last4_from_ckpt(ckpt_path, warm_x_tuple):
        payload = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise RuntimeError("Expected a dict checkpoint for last4 NF.")
        cfg_ck, model_cfg, nflow_cfg, task_cfg = _nf_ckpt_get_cfg(payload)
        sd_for_dims = payload.get("backbone") or payload.get("feat")
        dims = infer_backbone_dims_from_sd(sd_for_dims)
        backbone = FullRegressorProfiles(
            aux_dim=int(model_cfg.get("aux_dim", 32)),
            embed_dim=int(model_cfg.get("embed_dim", dims["embed_dim"])),
            nhead=int(model_cfg.get("nhead", 8)),
            nlayers=int(model_cfg.get("nlayers", dims["nlayers"])),
            out_dim=int(model_cfg.get("out_dim", 4)),
        ).to(device)
        feat = FeatureNetProfiles(backbone).to(device)
        ctx_dim = feat.infer_context_dim(warm_x_tuple)
        nf = NFProfiles(
            feature_net=feat,
            context_dim=ctx_dim,
            out_dim=int(model_cfg.get("out_dim", 4)),
            num_layers=int(nflow_cfg.get("num_layers", 7)),
            hidden_features=int(nflow_cfg.get("hidden_features", 640)),
            num_bins=int(nflow_cfg.get("num_bins", 13)),
            tail_bound=float(nflow_cfg.get("tail_bound", 10.0)),
            standardize_y=bool(task_cfg.get("standardize_y", nflow_cfg.get("standardize_y", False))),
            min_bin_width=float(nflow_cfg.get("min_bin_width", 5e-3)),
            min_bin_height=float(nflow_cfg.get("min_bin_height", 5e-3)),
            min_derivative=float(nflow_cfg.get("min_derivative", 1e-3)),
        ).to(device)
        if "backbone" in payload:
            _load_matching(feat.backbone, payload["backbone"], "last4_backbone")
        if "feat" in payload:
            _load_matching(feat, payload["feat"], "last4_feat")
        if "nf" in payload:
            _load_matching(nf, payload["nf"], "last4_nf")
        feat.eval()
        nf.eval()
        return nf

    def build_nf_full20_from_ckpt(ckpt_path, warm_x_tuple):
        payload = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise RuntimeError("Expected a dict checkpoint for full20 NF.")
        cfg_ck, model_cfg, nflow_cfg, task_cfg = _nf_ckpt_get_cfg(payload)
        out_dim = int(model_cfg.get("out_dim", cfg_ck.get("out_dim", 20)))
        backbone_nf = FullRegressorProfiles(
            aux_dim=int(model_cfg.get("aux_dim", 32)),
            embed_dim=int(model_cfg.get("embed_dim", 256)),
            nhead=int(model_cfg.get("nhead", 8)),
            nlayers=int(model_cfg.get("nlayers", 1)),
            out_dim=out_dim,
        ).to(device)
        feat_nf = FeatureNetProfiles(backbone_nf).to(device)
        ctx_dim = feat_nf.infer_context_dim(warm_x_tuple)
        nf = NFProfiles(
            feature_net=feat_nf,
            context_dim=ctx_dim,
            out_dim=out_dim,
            num_layers=int(nflow_cfg.get("num_layers", cfg_ck.get("layers", 7))),
            hidden_features=int(nflow_cfg.get("hidden_features", cfg_ck.get("hidden", 640))),
            num_bins=int(nflow_cfg.get("num_bins", cfg_ck.get("bins", 13))),
            tail_bound=float(nflow_cfg.get("tail_bound", cfg_ck.get("tail_bound", 10.0))),
            standardize_y=bool(task_cfg.get("standardize_y", cfg_ck.get("standardize_y", True))),
            min_bin_width=float(nflow_cfg.get("min_bin_width", 5e-3)),
            min_bin_height=float(nflow_cfg.get("min_bin_height", 5e-3)),
            min_derivative=float(nflow_cfg.get("min_derivative", 1e-3)),
        ).to(device)
        if "nf" in payload:
            _load_matching(nf, payload["nf"], "full20_nf")
        if "feat" in payload:
            _load_matching(feat_nf, payload["feat"], "full20_feat")
        if "backbone" in payload:
            _load_matching(feat_nf.backbone, payload["backbone"], "full20_backbone")
        nf.eval()
        return nf

    print("Loading last4 / full20 backbones and NFs...")
    backbone_last4 = load_backbone_last4(last4_backbone_ckpt, device)
    backbone_full20 = load_backbone_full20(full20_backbone_ckpt, device)
    nf_last4 = build_nf_last4_from_ckpt(last4_nf_ckpt, warm_x_tuple)
    nf_full20 = build_nf_full20_from_ckpt(full20_nf_ckpt, warm_x_tuple)

    lap_train_loader_last4 = make_laplace_train_loader_profiles(
        train_dl.dataset,
        idx0=LAST4_IDX0,
        n_subset=args.laplace_fit_subset,
        batch_size=args.profile_train_batch_size,
        seed=args.seed,
    )
    lap_train_loader_full20 = make_laplace_train_loader_profiles(
        train_dl.dataset,
        idx0=None,
        n_subset=args.laplace_fit_subset,
        batch_size=args.profile_train_batch_size,
        seed=args.seed,
    )

    wrapped_last4 = ModelWrapperProfiles(backbone_last4).to(device)
    wrapped_full20 = ModelWrapperProfiles(backbone_full20).to(device)

    la_last4 = Laplace(
        wrapped_last4,
        likelihood="regression",
        subset_of_weights="last_layer",
        hessian_structure="diag",
        backend=CurvlinopsGGN,
        prior_precision=1.0,
    )
    la_full20 = Laplace(
        wrapped_full20,
        likelihood="regression",
        subset_of_weights="last_layer",
        hessian_structure="diag",
        backend=CurvlinopsGGN,
        prior_precision=1.0,
    )
    la_last4._get_SoD_data_loader = _get_sod_data_loader_keep_collate.__get__(la_last4, type(la_last4))
    la_full20._get_SoD_data_loader = _get_sod_data_loader_keep_collate.__get__(la_full20, type(la_full20))

    print("Fitting Laplace for dedicated 4-output model (not timed)...")
    la_last4.fit(lap_train_loader_last4)
    print("Fitting Laplace for full 20-output model (not timed)...")
    la_full20.fit(lap_train_loader_full20)

    lapmeta_last4 = identify_laplace_last_layer(backbone_last4, la_last4, device)
    lapmeta_full20 = identify_laplace_last_layer(backbone_full20, la_full20, device)

    last4_t = torch.tensor(LAST4_IDX0, dtype=torch.long, device=device)

    def make_loader(batch_size: int):
        return make_test_loader(batch_size)

    @torch.no_grad()
    def predict_laplace4(batch):
        x_tuple, _, _ = move_to_device_profiles(batch, device)
        return laplace_batch_mean_cov_std(backbone_last4, lapmeta_last4, x_tuple, device=device, s=args.profile_laplace_samples)

    @torch.no_grad()
    def predict_nf4(batch):
        x_tuple, _, _ = move_to_device_profiles(batch, device)
        s4 = normalize_nf_sample_shape(nf_last4.sample(x_tuple, n=args.profile_nf_samples), expected_s=args.profile_nf_samples)
        s4 = s4 if s4.shape[-1] == 4 else s4.index_select(-1, last4_t)
        return sample_moments(s4)

    @torch.no_grad()
    def predict_laplace20to4(batch):
        x_tuple, _, _ = move_to_device_profiles(batch, device)
        mu20, cov20 = laplace_batch_mean_cov_std(backbone_full20, lapmeta_full20, x_tuple, device=device, s=args.profile_laplace_samples)
        return slice_last4_mean(mu20), slice_last4_cov(cov20)

    @torch.no_grad()
    def predict_nf20to4(batch):
        x_tuple, _, _ = move_to_device_profiles(batch, device)
        s20 = normalize_nf_sample_shape(nf_full20.sample(x_tuple, n=args.profile_nf_samples), expected_s=args.profile_nf_samples)
        s20 = s20.index_select(-1, last4_t)
        return sample_moments(s20)

    return [
        BenchmarkCase(
            name="Last4 Laplace (4-output)",
            scenario="last4",
            predictor=predict_laplace4,
            make_loader=make_loader,
            notes="dedicated 4-output Laplace via MC weight samples",
            posterior_samples=args.profile_laplace_samples,
        ),
        BenchmarkCase(
            name="Last4 NF (4-output)",
            scenario="last4",
            predictor=predict_nf4,
            make_loader=make_loader,
            notes="dedicated 4-output NF samples -> moments",
            posterior_samples=args.profile_nf_samples,
        ),
        BenchmarkCase(
            name="Last4 Laplace (20->4)",
            scenario="last4",
            predictor=predict_laplace20to4,
            make_loader=make_loader,
            notes="full20 Laplace then slice last 4 levels",
            posterior_samples=args.profile_laplace_samples,
        ),
        BenchmarkCase(
            name="Last4 NF (20->4)",
            scenario="last4",
            predictor=predict_nf20to4,
            make_loader=make_loader,
            notes="full20 NF samples then slice last 4 levels",
            posterior_samples=args.profile_nf_samples,
        ),
    ]


# -----------------------------------------------------------------------------
# Timing / reporting
# -----------------------------------------------------------------------------

def run_case(case: BenchmarkCase, batch_size: int, args, device: torch.device) -> List[Dict[str, Any]]:
    loader = case.make_loader(batch_size)

    # warmup
    if args.warmup_batches > 0:
        warm_n = 0
        for batch in loader:
            _ = case.predictor(batch)
            warm_n += 1
            if warm_n >= args.warmup_batches:
                break
        sync_device(device)

    rows: List[Dict[str, Any]] = []
    for rep in range(args.num_repeats):
        n_soundings = 0
        n_batches = 0
        sync_device(device)
        t0 = time.perf_counter()
        for batch in loader:
            out = case.predictor(batch)
            if out is None:
                raise RuntimeError(f"Predictor for {case.name} returned None.")
            if isinstance(batch, (list, tuple)):
                # the first target tensor has batch size in dim 0 for all loaders here
                try:
                    if case.scenario == "xco2":
                        n_soundings += int(batch[1].shape[0])
                    else:
                        n_soundings += int(batch[1].shape[0])
                except Exception:
                    pass
            n_batches += 1
            if args.max_batches is not None and n_batches >= args.max_batches:
                break
        sync_device(device)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        rows.append(
            {
                "scenario": case.scenario,
                "method": case.name,
                "batch_size": int(batch_size),
                "repeat": int(rep),
                "elapsed_sec": float(elapsed),
                "n_soundings": int(n_soundings),
                "n_batches": int(n_batches),
                "sec_per_sounding": float(elapsed / max(n_soundings, 1)),
                "ms_per_sounding": float(1000.0 * elapsed / max(n_soundings, 1)),
                "soundings_per_sec": float(n_soundings / max(elapsed, 1e-12)),
                "posterior_samples": case.posterior_samples,
                "notes": case.notes,
                "device": str(device),
            }
        )
    return rows


def summarize_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    g = raw_df.groupby(["scenario", "method", "batch_size", "posterior_samples", "notes", "device"], dropna=False)
    out = g.agg(
        elapsed_sec_mean=("elapsed_sec", "mean"),
        elapsed_sec_sd=("elapsed_sec", "std"),
        elapsed_sec_median=("elapsed_sec", "median"),
        ms_per_sounding_mean=("ms_per_sounding", "mean"),
        ms_per_sounding_sd=("ms_per_sounding", "std"),
        ms_per_sounding_median=("ms_per_sounding", "median"),
        soundings_per_sec_mean=("soundings_per_sec", "mean"),
        soundings_per_sec_sd=("soundings_per_sec", "std"),
        soundings_per_sec_median=("soundings_per_sec", "median"),
        n_soundings=("n_soundings", "first"),
        n_batches=("n_batches", "first"),
        n_repeats=("repeat", "count"),
    ).reset_index()
    return out.sort_values(["scenario", "batch_size", "method"]).reset_index(drop=True)


def add_optional_nasa_reference(summary_df: pd.DataFrame, args) -> pd.DataFrame:
    if args.nasa_seconds_per_sounding is None:
        return summary_df
    rows = []
    for bs in sorted(summary_df["batch_size"].unique()):
        rows.append(
            {
                "scenario": "reference",
                "method": "NASA full physics (external)",
                "batch_size": int(bs),
                "posterior_samples": np.nan,
                "notes": "external reference added by --nasa-seconds-per-sounding",
                "device": str(summary_df["device"].iloc[0]),
                "elapsed_sec_mean": np.nan,
                "elapsed_sec_sd": np.nan,
                "elapsed_sec_median": np.nan,
                "ms_per_sounding_mean": 1000.0 * args.nasa_seconds_per_sounding,
                "ms_per_sounding_sd": np.nan,
                "ms_per_sounding_median": 1000.0 * args.nasa_seconds_per_sounding,
                "soundings_per_sec_mean": 1.0 / args.nasa_seconds_per_sounding,
                "soundings_per_sec_sd": np.nan,
                "soundings_per_sec_median": 1.0 / args.nasa_seconds_per_sounding,
                "n_soundings": np.nan,
                "n_batches": np.nan,
                "n_repeats": np.nan,
            }
        )
    return pd.concat([summary_df, pd.DataFrame(rows)], ignore_index=True)


def write_latex_table(summary_df: pd.DataFrame, out_path: Path, batch_size: int) -> None:
    df = summary_df.loc[summary_df["batch_size"] == batch_size].copy()
    df = df[["method", "scenario", "ms_per_sounding_median", "soundings_per_sec_median", "posterior_samples", "notes"]]
    df = df.sort_values(["scenario", "ms_per_sounding_median", "method"]).reset_index(drop=True)

    lines = []
    lines.append("\\begin{tabular}{llrrl}")
    lines.append("\\toprule")
    lines.append("Method & Scenario & ms/sounding & soundings/s & Posterior samples " + r"\\")
    lines.append("\\midrule")
    for _, r in df.iterrows():
        ps = "--" if pd.isna(r["posterior_samples"]) else f"{int(r['posterior_samples'])}"
        ms = "nan" if pd.isna(r["ms_per_sounding_median"]) else f"{r['ms_per_sounding_median']:.3f}"
        thr = "nan" if pd.isna(r["soundings_per_sec_median"]) else f"{r['soundings_per_sec_median']:.1f}"
        lines.append(f"{r['method']} & {r['scenario']} & {ms} & {thr} & {ps} " + r"\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    ensure_parent_dir(out_path)
    out_path.write_text("\n".join(lines))


def make_bar_plot(summary_df: pd.DataFrame, out_path: Path, batch_size: int, nasa_ms_per_sounding: Optional[float]) -> None:
    df = summary_df.loc[summary_df["batch_size"] == batch_size].copy()
    df = df.sort_values(["scenario", "ms_per_sounding_median", "method"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(df))
    y = df["ms_per_sounding_median"].to_numpy(dtype=float)
    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], rotation=35, ha="right")
    ax.set_ylabel("Median ms per sounding")
    ax.set_title(f"Test-time cost at batch size = {batch_size}")
    ax.set_yscale("log")
    if nasa_ms_per_sounding is not None:
        ax.axhline(nasa_ms_per_sounding, linestyle="--", linewidth=1)
        ax.text(len(df) - 0.5, nasa_ms_per_sounding, " NASA reference", va="bottom", ha="right")
    fig.tight_layout()
    ensure_parent_dir(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_throughput_plot(summary_df: pd.DataFrame, out_path: Path, nasa_soundings_per_sec: Optional[float]) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    method_order = list(dict.fromkeys(summary_df["method"].tolist()))
    for method in method_order:
        sub = summary_df.loc[summary_df["method"] == method].sort_values("batch_size")
        ax.plot(
            sub["batch_size"].to_numpy(dtype=float),
            sub["soundings_per_sec_median"].to_numpy(dtype=float),
            marker="o",
            label=method,
        )
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Median soundings / sec")
    ax.set_title("Throughput vs batch size")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    if nasa_soundings_per_sec is not None:
        ax.axhline(nasa_soundings_per_sec, linestyle="--", linewidth=1, label="NASA reference")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    ensure_parent_dir(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark test-time runtime for OCO-2 manuscript methods.")

    p.add_argument("--scenario", choices=["all", "xco2", "profile20", "last4"], default="all")
    p.add_argument("--outdir", type=str, default="./benchmark_runtime_outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128, 256])
    p.add_argument("--num-repeats", type=int, default=5)
    p.add_argument("--warmup-batches", type=int, default=2)
    p.add_argument("--max-batches", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--plot-batch-size", type=int, default=None)
    p.add_argument("--nasa-seconds-per-sounding", type=float, default=None)

    p.add_argument("--code-root", type=str, default=str(DEFAULT_CODE_ROOT))
    p.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--stats-dir", type=str, default=str(DEFAULT_STATS_DIR))

    p.add_argument("--laplace-fit-subset", type=int, default=6000)

    p.add_argument("--xco2-weights-dir", type=str, default=str(DEFAULT_XCO2_WEIGHTS_DIR))
    p.add_argument("--xco2-nf-ckpt", type=str, default=str(DEFAULT_XCO2_NF_CKPT))
    p.add_argument("--scalar-train-batch-size", type=int, default=512)
    p.add_argument("--scalar-laplace-hessian", type=str, default="full")
    p.add_argument("--scalar-laplace-subset", type=str, default="last_layer")
    p.add_argument("--scalar-optimize-prior", action="store_true")
    p.add_argument("--scalar-prior-opt-steps", type=int, default=100)
    p.add_argument("--scalar-prior-opt-lr", type=float, default=5e-2)
    p.add_argument("--scalar-prior-opt-init", type=float, default=0.1)
    p.add_argument("--scalar-nf-samples", type=int, default=512)
    p.add_argument("--scalar-nf-hidden", type=int, default=384)
    p.add_argument("--scalar-nf-layers", type=int, default=5)
    p.add_argument("--scalar-nf-bins", type=int, default=9)
    p.add_argument("--scalar-nf-tail-bound", type=float, default=6.3108919641)

    p.add_argument("--profile20-nf-ckpt", type=str, default=str(DEFAULT_PROFILE20_NF_CKPT))
    p.add_argument("--profile20-laplace-backbone-ckpt", type=str, default=str(DEFAULT_PROFILE20_LAPLACE_BACKBONE_CKPT))
    p.add_argument("--profile-train-batch-size", type=int, default=128)
    p.add_argument("--profile-val-batch-size", type=int, default=256)
    p.add_argument("--profile-laplace-samples", type=int, default=256)
    p.add_argument("--profile-nf-samples", type=int, default=512)

    p.add_argument("--last4-full20-variant", choices=["legacy", "modern"], default="legacy")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    set_global_seed(args.seed)
    torch.set_default_dtype(torch.float32)

    code_root = Path(args.code_root)
    if str(code_root) not in sys.path:
        sys.path.append(str(code_root))

    print("device:", device)
    print("code root exists:", code_root.exists(), "->", code_root)
    print("data root exists:", Path(args.data_root).exists(), "->", args.data_root)
    print("outdir:", outdir.resolve())

    cases: List[BenchmarkCase] = []
    if args.scenario in ("all", "xco2"):
        print("\n=== Setting up xCO2 cases ===")
        cases.extend(setup_xco2_cases(args, device))
    if args.scenario in ("all", "profile20"):
        print("\n=== Setting up profile20 cases ===")
        cases.extend(setup_profile20_cases(args, device))
    if args.scenario in ("all", "last4"):
        print("\n=== Setting up last4 cases ===")
        cases.extend(setup_last4_cases(args, device))

    print("\n=== Running timed benchmarks ===")
    rows: List[Dict[str, Any]] = []
    for case in cases:
        for bs in args.batch_sizes:
            print(f"[benchmark] {case.name} | batch_size={bs}")
            rows.extend(run_case(case, batch_size=bs, args=args, device=device))

    raw_df = pd.DataFrame(rows)
    summary_df = summarize_results(raw_df)
    summary_df = add_optional_nasa_reference(summary_df, args)

    raw_csv = outdir / "benchmark_raw.csv"
    summary_csv = outdir / "benchmark_summary.csv"
    raw_df.to_csv(raw_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    print("saved:", raw_csv)
    print("saved:", summary_csv)

    plot_batch_size = args.plot_batch_size if args.plot_batch_size is not None else max(args.batch_sizes)
    table_tex = outdir / f"benchmark_table_batch{plot_batch_size}.tex"
    bar_png = outdir / f"benchmark_bar_batch{plot_batch_size}.png"
    line_png = outdir / "benchmark_throughput_vs_batch.png"

    write_latex_table(summary_df, table_tex, batch_size=plot_batch_size)
    make_bar_plot(
        summary_df,
        bar_png,
        batch_size=plot_batch_size,
        nasa_ms_per_sounding=None if args.nasa_seconds_per_sounding is None else 1000.0 * args.nasa_seconds_per_sounding,
    )
    make_throughput_plot(
        summary_df,
        line_png,
        nasa_soundings_per_sec=None if args.nasa_seconds_per_sounding is None else 1.0 / args.nasa_seconds_per_sounding,
    )
    print("saved:", table_tex)
    print("saved:", bar_png)
    print("saved:", line_png)

    print("\nDone.")
    print("Suggested manuscript-facing artifact:")
    print(f"  - LaTeX table: {table_tex}")
    print(f"  - Bar plot:    {bar_png}")
    print(f"  - CSV summary: {summary_csv}")


if __name__ == "__main__":
    main()

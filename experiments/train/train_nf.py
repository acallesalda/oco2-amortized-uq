# experiments/train/train_nf.py
import sys
from pathlib import Path
sys.path.append("/workspace/project/code")

import yaml
import json
import argparse
import numpy as np

from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import optuna
except ImportError:
    optuna = None

from nflows import flows, distributions, transforms

from data.loading.spectral_loader import (DENSITY_KEY, density_settings,
                                          train_val_loader_profiles, train_val_loader)
from data.loading.covariates import get_spec, load_covariate_stats, model_aux_kwargs
from utils.schedule import cosine_with_warmup, describe as describe_sched
from modeling.full_regressor_profiles import FullRegressorProfiles
from utils.cfg import dotted_overrides

# What this "scalar" flow actually models.
#
# It has always been "surface": the file loaded the PROFILE dataset and then took
# `label[:, 19]`, so the model behind Table 1's "Normalizing flow" row is the
# level-20 (surface) CO2 concentration -- not the column-average XCO2 that the same
# table's Laplace and ACOS rows report. Three things differ, not one: the target
# variable, the standardization constant (profile_sigma[19] = 19.2 ppm rather than the
# XCO2 sigma), and the input preprocessing (the profile loader does not pad the bands to
# 1024, the scalar loader does). The padding difference is what finally surfaced this, as
# a shape mismatch when scoring.
#
# `data.target: xco2` selects the scalar dataset and the true_xco2 target, which is what
# Table 1 claims to be about. The default stays "surface" so that re-running an existing
# config reproduces what it did before rather than silently changing meaning.
TARGET_IDX = 19
TARGET_MODE = "surface"

import copy


# --------------------------------------------------------------------------------------
# Config helpers
# --------------------------------------------------------------------------------------
def _get_model_cfg(cfg: dict) -> dict:
    """
    Return a dict with keys:
      - 'full_backbone': {...}
      - either 'nflow' or 'flow': {...}
    Accepts inline 'model_cfg_obj' or loads from 'model_cfg_path'.
    """
    if cfg.get("model_cfg_obj") is not None:
        mcfg = cfg["model_cfg_obj"]
    elif cfg.get("model_cfg_path"):
        with open(cfg["model_cfg_path"], "r") as f:
            mcfg = yaml.safe_load(f)
    else:
        raise KeyError("Provide either 'model_cfg_obj' or 'model_cfg_path' in the exp cfg.")

    # Normalize: support either block name
    if "nflow" in mcfg:
        mcfg["flow_like"] = mcfg["nflow"]
    elif "flow" in mcfg:
        mcfg["flow_like"] = mcfg["flow"]
    else:
        raise KeyError("Model cfg must contain a 'nflow' or 'flow' section.")
    return mcfg


def _normalize_model_cfg(cfg: dict) -> dict:
    """
    For logging/debug. Produces a resolved model cfg object (inline).
    """
    obj = None
    mpath = cfg.get("model_cfg_path", None)
    if mpath:
        with open(mpath, "r") as f:
            obj = yaml.safe_load(f)
    else:
        obj = cfg.get("model_cfg_obj", None)
    assert obj is not None, "Provide model_cfg_path or model_cfg_obj in the exp cfg."

    # Provide defaults if missing
    obj.setdefault("nflow", {})
    nfd = obj["nflow"]
    nfd.setdefault("hidden", 256)
    nfd.setdefault("layers", 4)
    nfd.setdefault("bins", 8)
    nfd.setdefault("tail_bound", 10.0)
    obj.setdefault("full_backbone", {})
    return obj


# --------------------------------------------------------------------------------------
# Dataloader adapters
# --------------------------------------------------------------------------------------
def move_to_device(batch, device):
    (bands, aux_cont, aux_cat), label, _retr = batch
    bands_gpu = {k: (v[0].to(device), v[1].to(device)) for k, v in bands.items()}
    aux_cont  = aux_cont.to(device)
    aux_cat   = [b.to(device) for b in aux_cat]

    label = label.to(device)
    if TARGET_MODE == 'xco2':
        # scalar loader: [B, 1] already standardized by the XCO2 mu/sigma
        label = label.reshape(-1)
    elif label.ndim == 2:
        label = label[:, TARGET_IDX]   # pick surface layer only
    else:
        label = label.squeeze(-1)

    return (bands_gpu, aux_cont, aux_cat), label


# --------------------------------------------------------------------------------------
# Model building
# --------------------------------------------------------------------------------------
def get_context(backbone: FullRegressorProfiles, x_tuple):
    """
    backbone.encoder returns something like [B, N, D]; flatten to [B, C].
    """
    z = backbone.encoder(x_tuple)
    if z.dim() > 2:
        z = z.flatten(1)
    return z


def build_nflows_transform(
    features: int,
    context_dim: int,
    num_layers: int = 4,
    hidden_features: int = 256,
    num_bins: int = 8,
    tail_bound: float = 10.0
):
    """
    1D conditional autoregressive spline stack. Works for features==1 and uses context.
    """
    layers = []
    for _ in range(num_layers):
        layers.append(
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=features,
                hidden_features=hidden_features,
                context_features=context_dim,
                num_bins=num_bins,
                tails='linear',
                tail_bound=tail_bound
            )
        )
        # For 1D this permutation is a no-op, but harmless and keeps the pattern uniform.
        layers.append(transforms.RandomPermutation(features=features))
    return transforms.CompositeTransform(layers)


class NFHead(nn.Module):
    """
    Wraps an nflows.Flow. We work directly in standardized y space (dataset does it).
    """
    def __init__(self, context_dim: int, num_layers: int, hidden_features: int, num_bins: int, tail_bound: float):
        super().__init__()
        transform = build_nflows_transform(
            features=1,
            context_dim=context_dim,
            num_layers=num_layers,
            hidden_features=hidden_features,
            num_bins=num_bins,
            tail_bound=tail_bound
        )
        base = distributions.StandardNormal(shape=[1])
        self.flow = flows.Flow(transform, distribution=base)

    def log_prob(self, y_std: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # y_std: [B], context: [B, C] -> [B]
        # nflows.Flow.log_prob already returns [B]; the trailing squeeze that used to be
        # here was a no-op except when B == 1, where it collapsed the result to a scalar.
        return self.flow.log_prob(inputs=y_std.unsqueeze(-1), context=context)

    def sample(self, n: int, context: torch.Tensor) -> torch.Tensor:
        # returns [n, B] (still standardized space)
        #
        # nflows returns [context_size, num_samples, features] -- that is [B, n, 1], NOT
        # [n, B, 1] as this comment used to claim. Callers (ScalarFlow.sample, and through
        # it predict.py, which permutes 1 <-> 0) all assumed the sample axis came first, so
        # the batch axis ended up where the sample axis belonged and the per-batch arrays
        # only agreed in shape while every batch was full. It surfaced as a concatenate
        # error on the short final batch. NFProfiles.sample normalizes the same convention;
        # this is the scalar counterpart.
        s = self.flow.sample(num_samples=n, context=context)   # [B, n, 1]
        return s.squeeze(-1).transpose(0, 1).contiguous()      # [n, B]


def epoch_loop(backbone, nf, loader, device, optimizer=None, grad_clip=None):
    train_mode = optimizer is not None
    if train_mode:
        backbone.train(); nf.train()
    else:
        backbone.eval();  nf.eval()

    total = 0.0
    nobs = 0
    for batch in loader:
        x, y_std = move_to_device(batch, device)   # y already standardized
        ctx = get_context(backbone, x)             # [B, C]

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        nll = -nf.log_prob(y_std, ctx).mean()
        if train_mode:
            nll.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.encoder.parameters()) + list(nf.parameters()),
                    grad_clip
                )
            optimizer.step()

        bs = y_std.shape[0]
        total += float(nll.detach().cpu()) * bs
        nobs  += bs
    return total / max(nobs, 1)


def build_models(backbone_cfg, nflow_cfg, device, sample_batch, freeze_backbone: bool,
                 aux_kwargs: Optional[dict] = None):
    # aux_kwargs carries the covariate set's shape (n_cont, cat_cardinalities). Empty for
    # the published four inputs, so an omitted argument reproduces the old behaviour.
    backbone = FullRegressorProfiles(
        aux_dim=int(backbone_cfg["aux_dim"]),
        embed_dim=int(backbone_cfg["embed_dim"]),
        nhead=int(backbone_cfg["nhead"]),
        nlayers=int(backbone_cfg["nlayers"]),
        out_dim=1,
        **(aux_kwargs or {}),
    ).to(device)

    x_tuple, _ = sample_batch
    with torch.no_grad():
        ctx_dim = get_context(backbone, x_tuple).shape[-1]

    nf = NFHead(
        context_dim=ctx_dim,
        num_layers=int(nflow_cfg["layers"]),
        hidden_features=int(nflow_cfg["hidden"]),
        num_bins=int(nflow_cfg["bins"]),
        tail_bound=float(nflow_cfg["tail_bound"]),
    ).to(device)

    if freeze_backbone:
        for p in backbone.encoder.parameters():
            p.requires_grad = False

    return backbone, nf, ctx_dim


def save_checkpoint(module_dict: dict, out_path: Path, extra: dict | None = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.state_dict() for k, v in module_dict.items()}
    payload = {
        "state_dict": state,
        "extra": (extra or {}),
    }
    torch.save(payload, out_path)


def load_backbone_ckpt_if_any(backbone, path: str | None):
    if not path:
        return
    p = Path(path)
    if not p.exists():
        print(f"[warn] backbone_ckpt {p} not found; skipping.")
        return
    try:
        ckpt = torch.load(p, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        backbone.load_state_dict(state, strict=False)
        print(f"[info] loaded backbone weights (strict=False) from {p}")
    except Exception as e:
        print(f"[warn] could not load backbone weights from {p}: {e}")


# --------------------------------------------------------------------------------------
# Train one (possibly BO) trial
# --------------------------------------------------------------------------------------
def train_one_trial(
    cfg,
    device,
    trial_idx: int,
    lr=None,
    wd=None,
    epochs=None,
    grad_clip=None,
    freeze_backbone=None,
    backbone_ckpt=None,
    save_intermediate: bool = False
):
    # Data
# inside train_one_trial(...)
    global TARGET_MODE
    TARGET_MODE = str(cfg["data"].get("target", "surface")).lower()
    if TARGET_MODE not in ("xco2", "surface"):
        raise ValueError(f"data.target must be 'xco2' or 'surface', got {TARGET_MODE!r}")
    print(f"[data] target = {TARGET_MODE}"
          + ("  (column-average XCO2, scalar loader, bands padded to 1024)"
             if TARGET_MODE == "xco2" else
             "  (profile level 20, profile loader, bands NOT padded)"))

    cov_spec = get_spec(cfg.get("covariates", None))
    cov_stats = load_covariate_stats(cfg["data"]["stats_dir"])
    print(f"[covariates] {cov_spec.name} (tier={cov_spec.tier}, "
          f"{cov_spec.n_continuous} continuous, {cov_spec.n_categorical} categorical)")

    # Cluster-balanced training. data.balance_by names a column of the split table to
    # equalise (use "clst"); data.balance_alpha tempers it, 0 = off, 1 = full inverse
    # frequency, 0.5 = square root. Unset -> uniform shuffling, i.e. published behaviour.
    # `balance_by="density"` selects the covariate-density sampler instead of a column;
    # data.density_{mode,beta,k,clip} are its own settings and alpha is shared. See
    # code/data/loading/density_weights.py for what that intervention is and is not.
    balance_by = cfg["data"].get("balance_by", None)
    balance_alpha = float(cfg["data"].get("balance_alpha", 0.5))
    density = density_settings(cfg["data"]) if balance_by == DENSITY_KEY else None
    if density is not None:
        print(f"[balance] covariate-density sampling in {cov_spec.name} space "
              f"(alpha={balance_alpha:.2f}, {density})")
    loader_fn = train_val_loader if TARGET_MODE == "xco2" else train_val_loader_profiles
    train_dl, val_dl = loader_fn(
        root=cfg["data"]["root"],
        annotations_filename=cfg["data"]["annotations"],
        train_batch_size=cfg["data"]["train_batch_size"],
        val_batch_size=cfg["data"]["val_batch_size"],
        seed=cfg["data"]["seed"],
        train_size=cfg["data"]["train_size"],
        stats_dir=cfg["data"]["stats_dir"],
        # set data.split_csv to use a reference-sounding-disjoint split; None keeps the
        # original replicate-level split with validation drawn at random from training
        split_csv=cfg["data"].get("split_csv", None),
        cov_spec=cov_spec,
        balance_by=balance_by,
        balance_alpha=balance_alpha,
        density=density,
    )

    # Warm batch
    first = next(iter(train_dl))
    sample_x, sample_y = move_to_device(first, device)

    # Configs
    mcfg = _get_model_cfg(cfg)
    flow_cfg = copy.deepcopy(mcfg["flow_like"])
    backbone_cfg = copy.deepcopy(mcfg["full_backbone"])

    # Defaults from cfg.setup if not provided
    if freeze_backbone is None:
        freeze_backbone = cfg.get("setup", {}).get("freeze_backbone", False)
    if backbone_ckpt is None:
        backbone_ckpt = cfg.get("setup", {}).get("backbone_ckpt", None)

    # Build
    backbone, nf, ctx_dim = build_models(backbone_cfg, flow_cfg, device, (sample_x, sample_y),
                                         freeze_backbone=bool(freeze_backbone),
                                         aux_kwargs=model_aux_kwargs(cov_spec, cov_stats))
    load_backbone_ckpt_if_any(backbone, backbone_ckpt)

    # Optim
    lr = lr if lr is not None else cfg["optim"]["lr"]
    wd = wd if wd is not None else cfg["optim"]["wd"]
    epochs = epochs if epochs is not None else cfg["optim"]["epochs"]
    grad_clip = grad_clip if grad_clip is not None else cfg["optim"]["grad_clip"]

    params = [{"params": nf.parameters(), "lr": lr, "weight_decay": wd}]
    if not freeze_backbone:
        # smaller LR for encoder when training jointly
        params.append({"params": backbone.encoder.parameters(), "lr": lr * 0.25, "weight_decay": wd})
    opt = optim.AdamW(params)

    # Learning-rate schedule. `optim.sched` is "none" by default, which is what every
    # published flow was trained with: AdamW at a constant learning rate for the whole
    # budget. The learning curves show what that costs -- the scalar flow's validation NLL wanders
    # by 0.24 nats an epoch over the last third and ends at 1.162, ABOVE its epoch-1 value
    # of 1.033, with its best (0.698) at epoch 32. Setting sched=cosine anneals to
    # eta_min_frac * lr with an optional linear warm-up; see code/utils/schedule.py. The
    # peak learning rate is untouched, because that is the number the search ranked.
    sched_name = str(cfg["optim"].get("sched", "none")).lower()
    warmup_epochs = int(cfg["optim"].get("warmup_epochs", 0))
    eta_min_frac = float(cfg["optim"].get("eta_min_frac", 0.1))
    sched = None
    if sched_name in ("cosine", "cosine_warmup"):
        sched = cosine_with_warmup(opt, epochs=int(epochs), warmup_epochs=warmup_epochs,
                                   eta_min_frac=eta_min_frac)
        print("[optim] " + describe_sched(int(epochs), float(lr), warmup_epochs,
                                          eta_min_frac, grad_clip))
    elif sched_name not in ("none", "", "constant"):
        raise SystemExit(f"unknown optim.sched={sched_name!r}; use 'cosine' or 'none'")
    else:
        print(f"[optim] constant lr {lr:.3e} for {epochs} epochs"
              + (f" | grad clip {grad_clip}" if grad_clip is not None else ""))

    best_val = float("inf")
    best_state = None

    # One record per epoch, written beside the checkpoint as best_flow_nflows.history.json.
    # The two numbers are already printed; keeping them in a file is what makes the
    # learning curve recoverable after the job's stdout is gone.
    history = []

    for ep in range(1, epochs + 1):
        tr_nll = epoch_loop(backbone, nf, train_dl, device, optimizer=opt, grad_clip=grad_clip)
        va_nll = epoch_loop(backbone, nf, val_dl,   device, optimizer=None)
        print(f"[ep {ep:03d}] train NLL: {tr_nll:.6f} | val NLL: {va_nll:.6f}")
        history.append({"epoch": int(ep), "train_nll_std": float(tr_nll),
                        "val_nll_std": float(va_nll),
                        "lr": float(opt.param_groups[0]["lr"])})
        if sched is not None:
            sched.step()
        if va_nll < best_val - 1e-4:
            best_val = va_nll
            best_state = {
                "backbone": {k: v.cpu() for k, v in backbone.state_dict().items()},
                "nf":       {k: v.cpu() for k, v in nf.state_dict().items()},
            }

    # Persist
    weights_dir = Path(cfg["weights_dir"])
    best_path   = weights_dir / "best_flow_nflows.pt"
    resolved_extra = dict(
            val_nll=float(best_val),
            lr=float(lr),
            wd=float(wd),
            epochs=int(epochs),
            grad_clip=None if grad_clip is None else float(grad_clip),
            trial_idx=int(trial_idx),
            freeze_backbone=bool(freeze_backbone),
            backbone_ckpt=backbone_ckpt,
            # The covariate set sets the auxiliary encoder's parameter shapes, so scoring
            # cannot rebuild this network without it. checkpoints.py already looks for it
            # here first (`extra.get("covariates")`) and only then falls back to whatever
            # --covariates the caller passed -- a fallback that is right until the day
            # somebody omits the flag, at which point the encoder is built for `legacy`
            # and the load fails on shapes. The other two flow trainers record it; this
            # one did not, which made its checkpoint the only one of the three that is
            # not self-describing.
            covariates=cov_spec.name,
            context_dim=int(ctx_dim),
            backbone_cfg=copy.deepcopy(backbone_cfg),
            nflow_cfg=copy.deepcopy(flow_cfg),
            model_cfg_resolved=copy.deepcopy(mcfg),
            data_cfg=copy.deepcopy(cfg["data"]),
            optim_cfg=dict(
                lr=float(lr),
                wd=float(wd),
                epochs=int(epochs),
                grad_clip=None if grad_clip is None else float(grad_clip),
            ),
        )
    if best_state is not None:
        backbone.load_state_dict(best_state["backbone"])
        nf.load_state_dict(best_state["nf"])
        save_checkpoint(
            {"backbone": backbone, "nf": nf},
            best_path,
            extra=resolved_extra
        )
        best_cfg_path = weights_dir / "best_flow_nflows_config.yaml"
        with open(best_cfg_path, "w") as f:
            yaml.safe_dump(resolved_extra, f, sort_keys=False)

        # The validation NLL, on its own, in a file that can be read without torch.
        # It is already inside the checkpoint's `extra`, but a Bayesian-optimization
        # trial's whole output is this one number and loading a 200 MB checkpoint to
        # recover a float -- on a job that then deletes the checkpoint -- is not a
        # sensible way to get it. Same name and shape as train_nf_profiles_4d.py's.
        with open(best_path.with_suffix(".history.json"), "w") as f:
            json.dump({"study": "nf", "target": TARGET_MODE,
                       "metric": "NLL, standardized units",
                       "covariates": cov_spec.name,
                       "best_val_nll": float(best_val),
                       "history": history}, f, indent=2)

        metrics_path = best_path.with_suffix(".metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({
                "best_val_nll": float(best_val),
                "epochs": int(epochs),
                "lr": float(lr),
                "wd": float(wd),
                "trial_idx": int(trial_idx),
                "backbone_ckpt": backbone_ckpt,
                "ckpt": str(best_path),
            }, f, indent=2)

    # Save *one* intermediate (trial 0) if asked
    if save_intermediate:
        save_checkpoint(
            {"backbone": backbone, "nf": nf},
            weights_dir / "intermediate.pt",
            extra=resolved_extra
        )

    return best_val, best_path


# --------------------------------------------------------------------------------------
# Optuna objective
# --------------------------------------------------------------------------------------
def objective(trial, base_cfg, device):
    # Search space
    lr     = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
    wd     = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
    hidden = trial.suggest_categorical("hidden", [128, 192, 256, 384])
    layers = trial.suggest_int("layers", 2, 8)
    bins   = trial.suggest_int("bins", 6, 24)
    tail_b = trial.suggest_float("tail_bound", 6.0, 12.0)

    # Pull + patch model cfg (supports inline or path)
    base_mcfg  = _get_model_cfg(base_cfg)
    trial_mcfg = copy.deepcopy(base_mcfg)
    trial_mcfg["flow_like"]["hidden"]     = int(hidden)
    trial_mcfg["flow_like"]["layers"]     = int(layers)
    trial_mcfg["flow_like"]["bins"]       = int(bins)
    trial_mcfg["flow_like"]["tail_bound"] = float(tail_b)

    # Build a cfg copy that carries the patched inline model cfg
    cfg_local = dict(base_cfg)
    cfg_local["model_cfg_obj"]  = trial_mcfg  # inline for this trial
    cfg_local["model_cfg_path"] = None        # force inline use

    # Inherit setup defaults
    freeze_backbone = base_cfg.get("setup", {}).get("freeze_backbone", False)
    backbone_ckpt   = base_cfg.get("setup", {}).get("backbone_ckpt", None)

    # Train; save an intermediate only for the first trial so you have a checkpoint to test
    val_nll, _ = train_one_trial(
        cfg_local, device, trial_idx=trial.number,
        lr=lr, wd=wd,
        freeze_backbone=freeze_backbone,
        backbone_ckpt=backbone_ckpt,
        save_intermediate=(trial.number == 0)
    )
    trial.set_user_attr("val_nll", val_nll)
    return val_nll


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_cfg", type=str,
                        default="code/shared_conf/experiment/nf_full_regressor_nflows.yaml")
    parser.add_argument("--set", nargs="*", default=[],
                        help="Dotted overrides like model_cfg_obj.nflow.layers=7 "
                             "optim.lr=1e-3 setup.backbone_ckpt=null")
    args = parser.parse_args()

    with open(args.exp_cfg, "r") as f:
        cfg = yaml.safe_load(f)
    overrides = args.__dict__.get("set")
    if overrides:
        print("[cfg] overrides:", " ".join(overrides))
    cfg = dotted_overrides(cfg, overrides)

    # Log a normalized view of the model cfg (for sanity)
    mobj = _normalize_model_cfg(cfg)
    print("[cfg] normalized model_cfg_obj:", mobj)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    bo = cfg.get("bo", {})
    if bo.get("enabled", False):
        assert optuna is not None, "pip install optuna to run BO."
        n_trials = int(bo.get("n_trials", 100))
        timeout  = bo.get("timeout", None)
        study = optuna.create_study(direction="minimize", study_name="nf_flow_bo")
        study.optimize(lambda tr: objective(tr, cfg, device), n_trials=n_trials, timeout=timeout)
        best_meta = {
            "best_value": float(study.best_value),
            "best_params": study.best_params,
            "n_trials": int(n_trials),
            "timeout": timeout,
        }
        out = Path(cfg["weights_dir"]) / "bo_best.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            yaml.safe_dump(best_meta, f, sort_keys=False)

        print("Best:", study.best_value, study.best_params)
    else:
        # Single run (no BO)
        train_one_trial(
            cfg, device, trial_idx=0,
            freeze_backbone=cfg.get("setup", {}).get("freeze_backbone", False),
            backbone_ckpt=cfg.get("setup", {}).get("backbone_ckpt", None)
        )


if __name__ == "__main__":
    main()

# experiments/train/train_nf_profiles.py
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List

import yaml
import numpy as np
from tqdm import tqdm
from nflows.transforms.coupling import CouplingTransform

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import optuna
except Exception:
    optuna = None

# --- project imports ---
import sys
sys.path.append("/workspace/project/code")

from modeling.full_regressor_profiles import FullRegressorProfiles
from modeling.flow_regressor_profiles_nf import FeatureNetProfiles, NFProfiles
from data.loading.spectral_loader import (DENSITY_KEY, density_settings,
                                          train_val_loader_profiles)
from data.loading.covariates import get_spec, load_covariate_stats, model_aux_kwargs
from utils.cfg import dotted_overrides
from utils.targets import resolve_target_indices as _resolve_target_indices
from utils.schedule import cosine_with_warmup, describe as describe_sched


# ---------------------------
# helpers
# ---------------------------
def _move_to_device_profiles(batch, device):
    """
    Loader yields LISTS:
      batch = [(bands_dict, aux_cont, aux_cat), y, y_retr]
    """
    x_tuple, y, y_retr = batch
    bands_dict, aux_cont, aux_cat = x_tuple
    bands = {k: (b.to(device), c.to(device)) for k, (b, c) in bands_dict.items()}
    aux_cont = aux_cont.to(device)
    aux_cat  = [t.to(device) for t in aux_cat]
    y = y.to(device)
    y_retr = None if y_retr is None else y_retr.to(device)
    return (bands, aux_cont, aux_cat), y, y_retr

from collections.abc import Sequence


def resolve_target_indices(cfg, D_full=20):
    """Which profile levels this run predicts. Lookup here, resolution in utils.targets.

    The resolution used to be inlined and was wrong: it tested `isinstance(t, Sequence)`
    without handling strings first, and `str` is a Sequence, so the default `"all"` was
    iterated character by character and every run died on `int('a')`. See
    code/utils/targets.py for both halves of that history.
    """
    return _resolve_target_indices(
        cfg.get("model", {}).get("target_indices", "all"), D_full)

@torch.no_grad()
def _rmse_from_samples(nf: NFProfiles, x_tuple, y_true, n_samples: int) -> torch.Tensor:
    if n_samples <= 0:
        return torch.tensor(float("nan"))
    try:
        s = nf.sample(x_tuple, n=n_samples)     # [n,B,D] or [B,n,D] or [B,D]
    except (AssertionError, RuntimeError, ValueError):
        return torch.tensor(float("nan"))
    if s.dim() == 2:
        s = s.unsqueeze(0)
    sample_axis = 0 if s.shape[0] == n_samples else (1 if s.shape[1] == n_samples else 0)
    m = s.mean(dim=sample_axis)                 # [B,D]
    se = (m - y_true).pow(2).mean(dim=-1)      # [B]
    return torch.sqrt(se.mean())

def build_models(cfg, device, warm_x_tuple, idxs: List[int]):
    # The auxiliary encoder's shape follows the covariate set, and the backbone this
    # warm-starts from must have been trained with the same one -- see
    # _require_aux_keys_loaded.
    cov_spec = get_spec(cfg.get("covariates", None))
    cov_stats = load_covariate_stats(Path(cfg["data"]["stats_dir"]))
    backbone = FullRegressorProfiles(
        aux_dim=int(cfg["model"]["aux_dim"]),
        embed_dim=int(cfg["model"]["embed_dim"]),
        nhead=int(cfg["model"]["nhead"]),
        nlayers=int(cfg["model"]["nlayers"]),
        out_dim=int(cfg["model"]["out_dim"]),
        **model_aux_kwargs(cov_spec, cov_stats),
    ).to(device)
    feat = FeatureNetProfiles(backbone).to(device)

    # TRY LOADING BEFORE THE FIRST FORWARD
    _maybe_load_backbone_into_feature_net(
        feature_net=feat,
        ckpt_path=cfg.get("setup", {}).get("backbone_ckpt", None),
        drop_band_mlp0=True,
        verbose=True,
    )

    # Now it’s safe to run a warm forward to get context dim
    ctx_dim = feat.infer_context_dim(warm_x_tuple)

    nf = NFProfiles(
        feature_net=feat,
        context_dim=ctx_dim,
        out_dim=len(idxs),
        num_layers=int(cfg["nflow"]["layers"]),
        hidden_features=int(cfg["nflow"]["hidden"]),
        num_bins=int(cfg["nflow"]["bins"]),
        tail_bound=float(cfg["nflow"]["tail_bound"]),
        standardize_y=bool(cfg["nflow"].get("standardize_y", True)),
        min_bin_width=float(cfg["nflow"].get("min_bin_width", 5e-3)),
        min_bin_height=float(cfg["nflow"].get("min_bin_height", 5e-3)),
        min_derivative=float(cfg["nflow"].get("min_derivative", 1e-3)),
        # 0.0 is the published setting. Raise it to regularize the coupling nets: this is
        # the model whose validation NLL is best after 7 of 40 epochs.
        dropout=float(cfg["nflow"].get("dropout", 0.0)),
    ).to(device)

    return feat, nf

def save_checkpoint(nf: NFProfiles, out_path: Path, cfg: dict, extra: Optional[dict] = None):
    """
    Save a self-describing checkpoint. Counts coupling layers by walking the transform tree,
    so it works across nflows versions (no private attrs).
    """
    def _flatten_transforms(t):
        # Recursively yield leaves in a CompositeTransform tree
        if hasattr(t, "_transforms"):           # CompositeTransform
            for s in t._transforms:
                yield from _flatten_transforms(s)
        else:
            yield t

    # how many couplings do we actually have?
    flat = list(_flatten_transforms(nf.flow._transform))
    n_couplings = sum(isinstance(t, CouplingTransform) for t in flat)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "nf": nf.state_dict(),
        "feat": nf.feature_net.state_dict(),
        "backbone": nf.feature_net.backbone.state_dict(),
        "cfg": {
            # store what we KNOW from the model/config
            "layers": int(n_couplings),                 # ← robust coupling count
            "hidden": int(cfg["nflow"]["hidden"]),
            "bins":   int(cfg["nflow"]["bins"]),
            "tail_bound": float(cfg["nflow"]["tail_bound"]),
            "out_dim": int(getattr(nf, "out_dim", 0)),
            "standardize_y": bool(cfg["nflow"].get("standardize_y", True)),
            # The spline clamps do not change any tensor shape, so a checkpoint loaded
            # with the wrong ones still loads cleanly and still returns a log-density --
            # a different one. nf_profiles_grouped.yaml sets min_derivative to 0.01, ten
            # times the class default, so this is not hypothetical.
            "min_bin_width":  float(cfg["nflow"].get("min_bin_width", 5e-3)),
            "min_bin_height": float(cfg["nflow"].get("min_bin_height", 5e-3)),
            "min_derivative": float(cfg["nflow"].get("min_derivative", 1e-3)),
            # Backbone dimensions, so scoring does not have to assume the defaults.
            "covariates": get_spec(cfg.get("covariates", None)).name,
            "model": {k: cfg["model"][k] for k in
                      ("aux_dim", "embed_dim", "nhead", "nlayers")
                      if k in cfg.get("model", {})},
        },
        "extra": extra or {},
    }
    torch.save(payload, out_path)

def _require_aux_keys_loaded(kept: dict, dropped: list, ckpt_path) -> None:
    """Fail if the warm-start dropped the auxiliary encoder.

    `_maybe_load_backbone_into_feature_net` skips shape-mismatched tensors deliberately:
    the band MLP's first layer depends on the flattened spectrum length and legitimately
    differs. But the auxiliary encoder's shape is set by the COVARIATE SET, so a mismatch
    there means the flow and its backbone were built from different covariates. Skipping
    it quietly warm-starts the flow from a backbone whose auxiliary pathway is random,
    prints a reassuring count of loaded keys, and trains to a plausible number.
    """
    aux_dropped = [k for k, _ in dropped if k.startswith("encoder.aux_encoder.")]
    aux_kept = [k for k in kept if k.startswith("encoder.aux_encoder.")]
    if aux_dropped or not aux_kept:
        raise SystemExit(
            "backbone warm-start dropped the auxiliary encoder: {} (from {}).\n"
            "The checkpoint was trained with a different covariate set than this "
            "config asks for. Set `covariates:` in the flow config to whatever the "
            "backbone used -- it is recorded in the backbone's best_hparams.yaml."
            .format(aux_dropped or "no aux keys present at all", ckpt_path)
        )



def _maybe_load_backbone_into_feature_net(
    feature_net: FeatureNetProfiles,
    ckpt_path: Optional[str],
    drop_band_mlp0: bool = True,
    verbose: bool = True,
) -> bool:
    """
    Load as many backbone weights as possible, skipping shape-mismatched tensors
    and (optionally) the band mlp.0.* heads whose input dim depends on flattened size.
    """
    if not ckpt_path:
        return False
    p = Path(ckpt_path)
    if not p.exists():
        if verbose: print(f"[warn] backbone_ckpt not found: {p}")
        return False

    try:
        state = torch.load(p, map_location="cpu")
        sd_src = state.get("model_state_dict", state.get("state_dict", state))
        sd_tgt = feature_net.backbone.state_dict()

        drop_patterns = []
        if drop_band_mlp0:
            # these are the fragile shape keys you’re hitting
            drop_patterns += [
                "encoder.band_encoders.b1.mlp.0.",
                "encoder.band_encoders.b2.mlp.0.",
                "encoder.band_encoders.b3.mlp.0.",
            ]

        keep, dropped = {}, []
        for k, v in sd_src.items():
            if any(pat in k for pat in drop_patterns):
                dropped.append((k, "pattern"))
                continue
            if k in sd_tgt and tuple(sd_tgt[k].shape) == tuple(v.shape):
                keep[k] = v
            else:
                # skip any other shape mismatch quietly
                if k in sd_tgt:
                    dropped.append((k, f"shape src={tuple(v.shape)} tgt={tuple(sd_tgt[k].shape)}"))

        _require_aux_keys_loaded(keep, dropped, p)
        feature_net.backbone.load_state_dict(keep, strict=False)
        if verbose:
            print(f"[backbone] loaded {len(keep)} keys, skipped {len(dropped)} "
                  f"({sum(1 for _,r in dropped if r=='pattern')} pattern, "
                  f"{sum(1 for _,r in dropped if r!='pattern')} shape).")
        return True
    except Exception as e:
        if verbose: print(f"[warn] could not load backbone weights from {p}: {e}")
        return False


def epoch_loop(
    nf: NFProfiles,
    loader,
    device: torch.device,
    optimizer: Optional[optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
    n_samples_rmse: int = 0,
    target_idxs: Optional[List[int]] = None,
):
    train_mode = optimizer is not None
    nf.train(train_mode)

    total_nll = 0.0
    n_obs = 0
    total_rmse = 0.0
    n_rmse_batches = 0

    for batch in loader:
        x_tuple, y_true, _ = _move_to_device_profiles(batch, device)
        if target_idxs is not None:
            y_true = y_true[:, target_idxs]
        assert y_true.shape[1] == nf.out_dim, f"y dim {y_true.shape[1]} != nf.out_dim {nf.out_dim}"
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        nll = -nf.log_prob(x_tuple, y_true).mean()  # scalar
        if train_mode:
            nll.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(nf.parameters(), grad_clip)
            optimizer.step()

        B = y_true.shape[0]
        total_nll += float(nll.detach().cpu()) * B
        n_obs += B

        if n_samples_rmse and not train_mode:
            rmse_b = _rmse_from_samples(nf, x_tuple, y_true, n_samples=n_samples_rmse)
            if torch.isfinite(rmse_b):
                total_rmse += float(rmse_b.cpu())
                n_rmse_batches += 1

    avg_nll = total_nll / max(n_obs, 1)
    avg_rmse = (total_rmse / max(n_rmse_batches, 1)) if n_rmse_batches > 0 else float("nan")
    return avg_nll, avg_rmse


def train_one(cfg: dict, device: torch.device, trial_idx: int = 0, idxs: Optional[List[int]] = None) -> Tuple[float, Path]:
    # data
    # Cluster-balanced training. data.balance_by names a column of the split table to
    # equalise (use "clst"); data.balance_alpha tempers it, 0 = off, 1 = full inverse
    # frequency, 0.5 = square root. Unset -> uniform shuffling, i.e. published behaviour.
    # `balance_by="density"` selects the covariate-density sampler instead of a column;
    # data.density_{mode,beta,k,clip} are its own settings and alpha is shared. See
    # code/data/loading/density_weights.py for what that intervention is and is not.
    balance_by = cfg["data"].get("balance_by", None)
    balance_alpha = float(cfg["data"].get("balance_alpha", 0.5))
    density = density_settings(cfg["data"]) if balance_by == DENSITY_KEY else None
    cov_spec = get_spec(cfg.get("covariates", None))
    if density is not None:
        print(f"[balance] covariate-density sampling in {cov_spec.name} space "
              f"(alpha={balance_alpha:.2f}, {density})")
    train_dl, val_dl = train_val_loader_profiles(
        root=cfg["data"]["root"],
        annotations_filename=cfg["data"]["annotations"],
        train_batch_size=cfg["data"]["train_batch_size"],
        val_batch_size=cfg["data"]["val_batch_size"],
        seed=cfg["data"]["seed"],
        stats_dir = Path(cfg["data"]["stats_dir"]),
        # set data.split_csv to use a reference-sounding-disjoint split; None keeps the
        # original replicate-level split with validation drawn at random from training
        split_csv=cfg["data"].get("split_csv", None),
        cov_spec=cov_spec,
        balance_by=balance_by,
        balance_alpha=balance_alpha,
        density=density,
    )
    # warm batch for context dim
    warm_batch = next(iter(train_dl))
    warm_x_tuple, _, _ = _move_to_device_profiles(warm_batch, device)

    # models
    feat, nf = build_models(cfg, device, warm_x_tuple, idxs)

    # freeze backbone if requested (IMPORTANT: NF contains FeatureNet; only one param group!)
    if bool(cfg["setup"].get("freeze_backbone", False)):
        for p in nf.feature_net.parameters():
            p.requires_grad = False

    # optimizer (single param group to avoid duplicates)
    opt = optim.AdamW(
        [{"params": [p for p in nf.parameters() if p.requires_grad],
          "lr": float(cfg["optim"]["lr"]), "weight_decay": float(cfg["optim"]["wd"])}]
    )

    best_val = float("inf")
    best_path = Path(cfg["weights_dir"]) / "best_flow_profiles.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = int(cfg["optim"]["epochs"])
    grad_clip = cfg["optim"].get("grad_clip", None)
    n_samples_rmse = int(cfg["eval"].get("rmse_samples", 0))

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
        sched = cosine_with_warmup(opt, epochs=epochs, warmup_epochs=warmup_epochs,
                                   eta_min_frac=eta_min_frac)
        print("[optim] " + describe_sched(epochs, float(cfg["optim"]["lr"]), warmup_epochs,
                                          eta_min_frac, grad_clip))
    elif sched_name not in ("none", "", "constant"):
        raise SystemExit(f"unknown optim.sched={sched_name!r}; use 'cosine' or 'none'")
    print(f"[optim] dropout={float(cfg['nflow'].get('dropout', 0.0))} "
          f"wd={cfg['optim']['wd']} | rmse_samples={n_samples_rmse}")

    print(f"[fit] trial={trial_idx} | epochs={epochs} | layers={cfg['nflow']['layers']} "
          f"| hidden={cfg['nflow']['hidden']} | bins={cfg['nflow']['bins']} | lr={cfg['optim']['lr']}")

    history = []   # per-epoch learning curve; see train_nf.py

    for ep in range(1, epochs + 1):
        tr_nll, _ = epoch_loop(nf, train_dl, device, optimizer=opt, 
                               grad_clip=grad_clip, n_samples_rmse=0, target_idxs=idxs)
        va_nll, va_rmse = epoch_loop(nf, val_dl, device, optimizer=None, grad_clip=None,
                                     n_samples_rmse=n_samples_rmse, target_idxs=idxs)
        print(f"[ep {ep:03d}] train NLL={tr_nll:.6f} | val NLL={va_nll:.6f}"
              f"{'' if np.isnan(va_rmse) else f' | val RMSE≈{va_rmse:.5f}'}")

        history.append({"epoch": int(ep), "train_nll_std": float(tr_nll),
                        "val_nll_std": float(va_nll),
                        "val_rmse_std": (None if np.isnan(va_rmse) else float(va_rmse)),
                        "lr": float(opt.param_groups[0]["lr"])})
        if sched is not None:
            sched.step()

        if va_nll + 1e-5 < best_val:
            best_val = va_nll
            save_checkpoint(nf, best_path, cfg, extra=dict(val_nll=best_val, epoch=ep, trial=trial_idx))
    print(f"[best] val NLL={best_val:.6f} | saved -> {best_path}")

    # The validation NLL in a file that can be read without torch -- see the note in
    # train_nf.py. A search trial's entire output is this number.
    with open(best_path.with_suffix(".history.json"), "w") as fh:
        json.dump({"study": "nf", "target": "profiles",
                   "metric": "NLL, standardized units",
                   "best_val_nll": float(best_val),
                   "history": history}, fh, indent=2)

    with open(best_path.with_suffix(".metrics.json"), "w") as fh:
        json.dump({
            "best_val_nll": float(best_val),
            "epochs": int(cfg["optim"]["epochs"]),
            "lr": float(cfg["optim"]["lr"]),
            "wd": float(cfg["optim"]["wd"]),
            "trial_idx": int(trial_idx),
            "backbone_ckpt": cfg.get("setup", {}).get("backbone_ckpt", None),
            "ckpt": str(best_path),
        }, fh, indent=2)
    return best_val, best_path


def objective(trial, base_cfg: dict, device: torch.device, idxs: List[int]):
    # sample hyperparams
    layers = trial.suggest_int("layers", 5, 9)
    hidden = trial.suggest_categorical("hidden", [384, 512, 640, 768])
    bins   = trial.suggest_int("bins", 8, 16)
    lr     = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
    wd     = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
    tail_b = trial.suggest_float("tail_bound", 6.0, 10.0)

    # patch cfg
    cfg = dict(base_cfg)
    cfg["nflow"] = dict(cfg["nflow"])
    cfg["optim"] = dict(cfg["optim"])
    cfg["nflow"]["layers"] = int(layers)
    cfg["nflow"]["hidden"] = int(hidden)
    cfg["nflow"]["bins"]   = int(bins)
    cfg["optim"]["lr"]     = float(lr)
    cfg["optim"]["wd"]     = float(wd)
    cfg["nflow"]["tail_bound"] = float(tail_b)

    val_nll, _ = train_one(cfg, device, trial_idx=trial.number, idxs=idxs)
    trial.set_user_attr("val_nll", val_nll)
    return val_nll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_cfg", type=str, required=True,
                        help="Path to YAML config (nf_profiles.yaml)")
    parser.add_argument("--set", nargs="*", default=[],
                        help="Dotted overrides like nflow.layers=7 optim.lr=1e-3 "
                             "setup.backbone_ckpt=null")
    args = parser.parse_args()

    with open(args.exp_cfg, "r") as f:
        cfg = yaml.safe_load(f)
    overrides = args.__dict__.get("set")
    if overrides:
        print("[cfg] overrides:", " ".join(overrides))
    cfg = dotted_overrides(cfg, overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg["data"]["seed"]))
    idxs = resolve_target_indices(cfg, D_full=20)
    print(f"[cfg] target_indices = {idxs} (out_dim={len(idxs)})")

    bo = cfg.get("bo", {})
    if bo.get("enabled", False):
        assert optuna is not None, "pip install optuna to run BO."
        study = optuna.create_study(direction="minimize", study_name=bo.get("study_name", "nf_profiles_bo"))
        study.optimize(lambda tr: objective(tr, cfg, device, idxs),
                       n_trials=int(bo.get("n_trials", 50)))
        print("[BO] Best value:", study.best_value, "| params:", study.best_params)
    else:
        train_one(cfg, device, trial_idx=0, idxs=idxs)

if __name__ == "__main__":
    main()

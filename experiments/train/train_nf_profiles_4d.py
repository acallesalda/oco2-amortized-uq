from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append("/workspace/project/code")
from data.loading.spectral_loader import train_val_loader_profiles, test_loader_profiles
from data.loading.covariates import get_spec, load_covariate_stats, model_aux_kwargs
from modeling.full_regressor_profiles import FullRegressorProfiles
from modeling.flow_regressor_profiles_nf import FeatureNetProfiles, NFProfiles
from utils.cfg import dotted_overrides
from utils.schedule import cosine_with_warmup, describe as describe_sched

try:
    # if available, reuse your existing mover
    from experiments.train.train_nf_profiles import _move_to_device_profiles as move_to_dev_x
except Exception:
    move_to_dev_x = None

import yaml

# ------------------------- utils -------------------------
def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)

# dotted_overrides used to be defined here. It now lives in code/utils/cfg.py, imported
# above, because train_nf.py and train_nf_profiles.py need the same thing and three copies
# of a coercion rule is three chances for a trial to be run with a string where the config
# wanted a float.

def fallback_move_to_device_profiles(batch, device):
    x, y, *rest = batch
    if isinstance(x, tuple) and len(x) == 3:
        bands, aux_cont, aux_cat = x
    elif isinstance(x, dict):
        bands = x["bands"]; aux_cont = x["aux_continuous"]; aux_cat = x["aux_categorical"]
    else:
        raise TypeError(f"Unexpected X type: {type(x)}")

    bands = {k: (b.to(device), m.to(device)) for k,(b,m) in bands.items()}
    aux_cont = aux_cont.to(device)
    aux_cat  = [t.to(device) for t in aux_cat]
    return (bands, aux_cont, aux_cat), y.to(device), (rest[0] if rest else None)

def _move(batch, device):
    if move_to_dev_x is not None:
        return move_to_dev_x(batch, device)
    return fallback_move_to_device_profiles(batch, device)

# ------------------------- build -------------------------
def _maybe_load_pretrained_backbone(backbone, cfg: dict, device: torch.device) -> dict:
    mcfg = cfg.get("model", {})
    ckpt = mcfg.get("pretrained_ckpt", None)
    if not ckpt:
        return {"used_pretrained_backbone": False, "pretrained_ckpt": None}

    ckpt_path = Path(ckpt)
    if not ckpt_path.exists():
        print(f"[warmstart] checkpoint not found: {ckpt_path} -> training from scratch")
        return {
            "used_pretrained_backbone": False,
            "pretrained_ckpt": str(ckpt_path),
            "reason": "missing",
        }

    obj = torch.load(ckpt_path, map_location=device)

    # support both a plain state_dict and a richer checkpoint dict
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj

    missing, unexpected = backbone.load_state_dict(state, strict=False)
    print(f"[warmstart] loaded backbone from {ckpt_path}")
    print(f"[warmstart] missing={missing}")
    print(f"[warmstart] unexpected={unexpected}")

    return {
        "used_pretrained_backbone": True,
        "pretrained_ckpt": str(ckpt_path),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def build_nf4_from_cfg(cfg: dict, device: torch.device):
    mcfg = cfg.get("model", {})

    nhead = int(mcfg.get("nhead", 4))
    embed_dim = int(mcfg.get("embed_dim", 256))

    # Coerce embed_dim so it's divisible by nhead
    if embed_dim % nhead != 0:
        new_embed_dim = ((embed_dim + nhead - 1) // nhead) * nhead
        print(f"[coerce] embed_dim {embed_dim} -> {new_embed_dim} to be divisible by nhead={nhead}")
        mcfg["embed_dim"] = new_embed_dim
        embed_dim = new_embed_dim

    # The auxiliary encoder's shape follows the covariate set, and it has to match the
    # backbone this warm-starts from -- see _require_aux_keys_loaded in
    # train_nf_profiles.py for what a mismatch looks like when it is not checked.
    cov_spec = get_spec(cfg.get("covariates", None))
    cov_stats = load_covariate_stats(Path(cfg["data"]["stats_dir"]))
    backbone = FullRegressorProfiles(
        aux_dim=int(mcfg.get("aux_dim", 32)),
        embed_dim=embed_dim,
        nhead=nhead,
        nlayers=int(mcfg.get("nlayers", 1)),
        out_dim=4,
        **model_aux_kwargs(cov_spec, cov_stats),
    ).to(device)

    last4 = cfg["task"]["last4"]
    dims0 = torch.tensor([i - 1 for i in last4], dtype=torch.long, device=device)

    # Cluster-balanced training. data.balance_by names a column of the split table to
    # equalise (use "clst"); data.balance_alpha tempers it, 0 = off, 1 = full inverse
    # frequency, 0.5 = square root. Unset -> uniform shuffling, i.e. published behaviour.
    balance_by = cfg["data"].get("balance_by", None)
    balance_alpha = float(cfg["data"].get("balance_alpha", 0.5))
    train_dl, val_dl = train_val_loader_profiles(
        root=cfg["data"]["root"],
        annotations_filename=cfg["data"]["annotations_filename"],
        train_batch_size=int(cfg["data"]["train_batch_size"]),
        val_batch_size=int(cfg["data"]["val_batch_size"]),
        seed=int(cfg["data"]["seed"]),
        stats_dir=Path(cfg["data"]["stats_dir"]),
        # set data.split_csv to use a reference-sounding-disjoint split; None keeps the
        # original replicate-level split with validation drawn at random from training
        split_csv=cfg["data"].get("split_csv", None),
        cov_spec=cov_spec,
        balance_by=balance_by,
        balance_alpha=balance_alpha,
    )

    feat = FeatureNetProfiles(backbone).to(device)

    # warm once so lazy modules are initialized and ctx_dim can be inferred
    warm_batch = next(iter(train_dl))
    x_warm, _, _ = _move(warm_batch, device)
    ctx_dim = feat.infer_context_dim(x_warm)

    build_info = _maybe_load_pretrained_backbone(backbone, cfg, device)
    # The resolved covariate preset travels out with the rest of the build report, because
    # the checkpoint records it and train_one is where the checkpoint is written. `cov_spec`
    # itself is local to this function and is not JSON-serializable; its name is both.
    build_info["covariates"] = cov_spec.name

    nfcfg = cfg["nflow"]
    nf = NFProfiles(
        feature_net=feat,
        context_dim=int(ctx_dim),
        out_dim=int(mcfg["out_dim"]),
        num_layers=int(nfcfg["num_layers"]),
        hidden_features=int(nfcfg["hidden_features"]),
        num_bins=int(nfcfg["num_bins"]),
        tail_bound=float(nfcfg["tail_bound"]),
        standardize_y=bool(cfg["task"].get("standardize_y", False)),
        min_bin_width=float(nfcfg["min_bin_width"]),
        min_bin_height=float(nfcfg["min_bin_height"]),
        min_derivative=float(nfcfg["min_derivative"]),
    ).to(device)

    return nf, feat, train_dl, val_dl, dims0, build_info

# ------------------------- train -------------------------
def train_one(cfg: dict) -> dict:
    device = torch.device(cfg["train"]["device"] if (cfg["train"]["device"] == "cuda" and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(int(cfg["data"]["seed"]))

    nf, feat, train_dl, val_dl, dims0, build_info = build_nf4_from_cfg(cfg, device)

    opt = torch.optim.AdamW(
        nf.parameters(),
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"]["weight_decay"])
    )
    grad_clip = float(cfg["optim"]["grad_clip"])
    # This trainer is the only flow that already annealed, but to eta_min=0 and with no
    # warm-up. Routing it through the shared helper puts the three flows on one schedule
    # and makes eta_min_frac / warmup_epochs settable here too; eta_min_frac=0.0 is the
    # published behaviour and stays the default for THIS trainer alone.
    epochs_4d = max(1, int(cfg["train"]["epochs"]))
    warmup_epochs = int(cfg["optim"].get("warmup_epochs", 0))
    eta_min_frac = float(cfg["optim"].get("eta_min_frac", 0.0))
    sched = cosine_with_warmup(opt, epochs=epochs_4d, warmup_epochs=warmup_epochs,
                               eta_min_frac=eta_min_frac)
    print("[optim] " + describe_sched(epochs_4d, float(cfg["optim"]["lr"]),
                                      warmup_epochs, eta_min_frac, grad_clip))

    best_val = float("inf")
    history = []   # per-epoch learning curve; see train_nf.py
    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / cfg["train"]["ckpt_name"]

    t0 = time.time()
    for ep in range(1, int(cfg["train"]["epochs"]) + 1):
        nf.train()
        train_loss_sum, n_train = 0.0, 0
        for it, batch in enumerate(train_dl, 1):
            x, y, _ = _move(batch, device)
            y4 = y.index_select(1, dims0)
            opt.zero_grad(set_to_none=True)
            loss = - nf.log_prob(x, y4).mean()
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(nf.parameters(), grad_clip)
            opt.step()
            train_loss_sum += float(loss.item()) * y4.size(0)
            n_train += y4.size(0)
            if cfg["train"]["log_every"] and (it % int(cfg["train"]["log_every"]) == 0):
                print(f"[ep {ep:02d} it {it:04d}] train_nll={train_loss_sum/max(1,n_train):.4f}")

        if sched is not None:
            sched.step()

        # validation (exact NLL)
        nf.eval()
        with torch.no_grad():
            val_sum, n_val = 0.0, 0
            for batch in val_dl:
                x, y, _ = _move(batch, device)
                y4 = y.index_select(1, dims0)
                nll_b = - nf.log_prob(x, y4)   # [B]
                val_sum += float(nll_b.sum().item())
                n_val += y4.size(0)
            val_nll = val_sum / max(1, n_val)

        print(f"[ep {ep:02d}] train_nll={train_loss_sum/max(1,n_train):.4f} | "
              f"val_nll={val_nll:.4f} | best={min(best_val, val_nll):.4f}")
        history.append({"epoch": int(ep),
                        "train_nll_std": float(train_loss_sum / max(1, n_train)),
                        "val_nll_std": float(val_nll)})

        if val_nll < best_val:
            best_val = val_nll
            torch.save(
                {
                    "nf": nf.state_dict(),
                    "feat": feat.state_dict(),
                    "cfg": {
                        "data": cfg["data"],
                        "task": cfg["task"],
                        # NOT cov_spec.name: cov_spec is local to build_nf4_from_cfg, and
                        # naming it here raised NameError on the first epoch that improved
                        # -- i.e. always, at epoch 1, before any nf_last4 run could
                        # produce a checkpoint. checkpoints.py reads this key to rebuild
                        # the auxiliary encoder, so it cannot simply be dropped.
                        "covariates": build_info["covariates"],
                        "model": cfg["model"],
                        "nflow": cfg["nflow"],
                        "optim": cfg["optim"],
                        "train": {k:v for k,v in cfg["train"].items() if k != "device"},
                        "out_dim": 4,
                    },
                },
                ckpt_path
            )

    elapsed = time.time() - t0
    print(f"[done] best_val_nll={best_val:.6f} | ckpt={str(ckpt_path)} | elapsed={elapsed:.1f}s")

    metrics = {
        "best_val_nll": best_val,
        "ckpt": str(ckpt_path),
        "elapsed_s": elapsed,
        "cfg": cfg,
        **build_info,
    }

    metrics_path = ckpt_path.with_suffix(".metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    with open(ckpt_path.with_suffix(".history.json"), "w") as fh:
        json.dump({"study": "nf", "target": "last4",
                   "metric": "NLL, standardized units",
                   "best_val_nll": float(best_val),
                   "history": history}, fh, indent=2)

    return metrics

# ------------------------- CLI -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_cfg", required=True, type=str, help="YAML config path")
    ap.add_argument("--set", nargs="*", help="Dotted overrides like nflow.num_layers=11 optim.lr=1e-3", default=[])
    args = ap.parse_args()

    cfg = load_yaml(args.exp_cfg)
    cfg = dotted_overrides(cfg, args.__dict__.get("set"))
    _ = train_one(cfg)

if __name__ == "__main__":
    main()

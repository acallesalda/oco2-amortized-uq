from __future__ import annotations

import math
import os
import random
import sys
import time
import json
import shutil
from copy import deepcopy
from pathlib import Path

import yaml
import torch

sys.path.append("/workspace/project")
from experiments.train.train_nf_profiles_4d import train_one as train_fn


def load_yaml(path):
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


SPACE = {
    "nflow.num_layers":      ("int", 5, 13),
    "nflow.hidden_features": ("int", 128, 640),
    "nflow.num_bins":        ("int", 6, 16),
    "nflow.tail_bound":      ("float", 4.0, 8.0),
    "optim.lr":              ("logfloat", 1e-4, 2e-3),
    "optim.weight_decay":    ("logfloat", 1e-8, 1e-4),
    "model.embed_dim":       ("choice", [128, 256, 384]),
    "model.nhead":           ("choice", [4, 8]),
    "model.nlayers":         ("choice", [1, 2, 3]),
    "task.standardize_y":    ("bool", 0, 1),
}


def _set_dotted(cfg: dict, key: str, val):
    node = cfg
    *prefix, last = key.split(".")
    for p in prefix:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[last] = val


def _get_dotted(cfg: dict, key: str):
    node = cfg
    for p in key.split("."):
        node = node[p]
    return node


def sample_cfg(base_cfg: dict) -> dict:
    cfg = deepcopy(base_cfg)
    for k, spec in SPACE.items():
        typ = spec[0]

        if typ == "int":
            a, b = spec[1], spec[2]
            val = int(random.randint(a, b))
        elif typ == "float":
            a, b = spec[1], spec[2]
            val = float(a + (b - a) * random.random())
        elif typ == "logfloat":
            a, b = spec[1], spec[2]
            la, lb = math.log(a), math.log(b)
            val = float(math.exp(la + (lb - la) * random.random()))
        elif typ == "bool":
            a, b = spec[1], spec[2]
            val = bool(random.randint(a, b))
        elif typ == "choice":
            choices = spec[1]
            val = random.choice(choices)
        else:
            raise ValueError(f"Unknown SPACE type: {typ}")

        _set_dotted(cfg, k, val)
    return cfg


def perturb_cfg(cfg: dict, frac=0.25) -> dict:
    new = deepcopy(cfg)
    for k, spec in SPACE.items():
        typ = spec[0]
        cur = _get_dotted(new, k)

        if typ == "int":
            a, b = spec[1], spec[2]
            span = b - a
            val = int(round(cur + random.uniform(-frac, frac) * span))
            val = max(a, min(b, val))

        elif typ == "float":
            a, b = spec[1], spec[2]
            span = b - a
            val = float(cur + random.uniform(-frac, frac) * span)
            val = max(a, min(b, val))

        elif typ == "logfloat":
            a, b = spec[1], spec[2]
            la, lb = math.log(a), math.log(b)
            v = math.log(max(a, min(b, cur)))
            v = v + random.uniform(-frac, frac) * (lb - la)
            val = float(math.exp(max(la, min(lb, v))))

        elif typ == "bool":
            val = bool(random.choice([cur, not cur]))

        elif typ == "choice":
            choices = spec[1]
            val = cur if random.random() < 0.5 else random.choice(choices)

        else:
            raise ValueError(f"Unknown SPACE type: {typ}")

        _set_dotted(new, k, val)
    return new


def load_regressor_signature(base_cfg: dict):
    ws = base_cfg.get("warm_start", {})
    if not ws.get("enabled", False):
        return None

    hp_path = Path(ws["regressor_hparams"])
    if not hp_path.exists():
        print(f"[warmstart] hparams file not found: {hp_path}")
        return None

    hp = load_yaml(hp_path)
    sig = {
        "embed_dim": int(hp["embed_dim"]),
        "nhead": int(hp["nhead"]),
        "nlayers": int(hp["nlayers"]),
        "aux_dim": int(hp["aux_dim"]),
    }
    return {
        "signature": sig,
        "ckpt": str(ws["regressor_ckpt"]),
    }


def maybe_attach_warmstart(cfg: dict, reg_info) -> dict:
    cfg = deepcopy(cfg)
    cfg.setdefault("model", {})
    cfg["model"].pop("pretrained_ckpt", None)

    if reg_info is None:
        return cfg

    sig = {
        "embed_dim": int(cfg["model"]["embed_dim"]),
        "nhead": int(cfg["model"]["nhead"]),
        "nlayers": int(cfg["model"]["nlayers"]),
        "aux_dim": int(cfg["model"].get("aux_dim", 32)),
    }

    if sig == reg_info["signature"]:
        cfg["model"]["pretrained_ckpt"] = reg_info["ckpt"]
        print(f"[warmstart] compatible backbone -> using pretrained regressor ckpt")
    else:
        print(f"[warmstart] incompatible backbone -> training from scratch")

    return cfg


def run_one_trial(cfg: dict, save_dir: Path, label: str):
    cfg = deepcopy(cfg)
    cfg["train"]["save_dir"] = str(save_dir)
    cfg["train"]["ckpt_name"] = "_trial_latest.pt"
    print(label)
    return train_fn(cfg)


def run_bo(exp_cfg_path: str, total_trials=20, seed=42, epochs_override=None, save_dir_override=None):
    random.seed(seed)
    torch.manual_seed(seed)

    base = load_yaml(exp_cfg_path)
    if epochs_override is not None:
        base["train"]["epochs"] = int(epochs_override)
    if save_dir_override is not None:
        base["train"]["save_dir"] = save_dir_override

    save_dir = Path(base["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    reg_info = load_regressor_signature(base)

    best_ckpt = save_dir / "best_nf4.pt"
    best_metrics = save_dir / "best_nf4.metrics.json"
    best_cfg_yaml = save_dir / "best_nf4_cfg.yaml"
    summary_path = save_dir / "bo_summary.json"

    trial_ckpt = save_dir / "_trial_latest.pt"
    trial_metrics = save_dir / "_trial_latest.metrics.json"

    results = []
    best = None

    def maybe_update_incumbent(rec):
        nonlocal best
        if (best is None) or (rec["best_val_nll"] < best["best_val_nll"]):
            best = rec
            if trial_ckpt.exists():
                shutil.copy2(trial_ckpt, best_ckpt)
            if trial_metrics.exists():
                shutil.copy2(trial_metrics, best_metrics)
            with open(best_cfg_yaml, "w") as fh:
                yaml.safe_dump(rec["cfg"], fh, sort_keys=False)

    # Trial 0: exactly the base config, so it can warm-start if compatible
    cfg0 = maybe_attach_warmstart(deepcopy(base), reg_info)
    rec0 = run_one_trial(cfg0, save_dir, "[BO base 1/1]")
    results.append({"cfg": cfg0, **rec0})
    maybe_update_incumbent(results[-1])
    with open(summary_path, "w") as fh:
        json.dump({"best": best, "results": results}, fh, indent=2)

    remaining = max(0, total_trials - 1)
    n_init_random = max(4, remaining // 3) if remaining > 0 else 0

    # Random init
    for i in range(n_init_random):
        cfg = sample_cfg(base)
        cfg = maybe_attach_warmstart(cfg, reg_info)
        rec = run_one_trial(cfg, save_dir, f"[BO init {i+1}/{n_init_random}]")
        results.append({"cfg": cfg, **rec})
        maybe_update_incumbent(results[-1])

        with open(summary_path, "w") as fh:
            json.dump({"best": best, "results": results}, fh, indent=2)

    # Local exploit around incumbent
    for j in range(remaining - n_init_random):
        cfg = perturb_cfg(best["cfg"], frac=0.25)
        cfg = maybe_attach_warmstart(cfg, reg_info)
        rec = run_one_trial(cfg, save_dir, f"[BO exploit {j+1}/{remaining - n_init_random}]")
        results.append({"cfg": cfg, **rec})
        maybe_update_incumbent(results[-1])

        with open(summary_path, "w") as fh:
            json.dump({"best": best, "results": results}, fh, indent=2)

    print("\n[BO] Best val_nll:", best["best_val_nll"])
    print("[BO] Best ckpt:", str(best_ckpt))
    print("[BO] Saved summary:", str(summary_path))
    return best, results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_cfg", required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--save_dir", type=str, default=None)
    args = ap.parse_args()

    run_bo(
        args.exp_cfg,
        total_trials=args.trials,
        seed=args.seed,
        epochs_override=args.epochs,
        save_dir_override=args.save_dir,
    )
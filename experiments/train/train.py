"""train.py
Modular training script.
Author(s): Felix Jimenez (edited)
"""

import logging
logger = logging.getLogger(__name__)

import sys
sys.path.append("/workspace/project/code")

import os
import json
import shutil
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import hydra
import mlflow
import yaml

from data.loading.spectral_loader import DENSITY_KEY, density_settings, train_val_loader
from data.loading.covariates import get_spec, cat_cardinalities, load_covariate_stats
from utils.schedule import cosine_with_warmup, clip_gradients, describe as describe_sched

# ----------------------------
# Helpers
# ----------------------------
def rmse(x: float) -> float:
    return float(np.sqrt(x))

def _atomic_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)

def _read_global_best(global_best_dir: Path):
    m = global_best_dir / "metrics.json"
    if not m.exists():
        return None
    try:
        with open(m, "r") as f:
            data = json.load(f)
        return float(data.get("val_rmse"))
    except Exception:
        return None

def _maybe_promote_to_global_best(trial_dir: Path, best_val_rmse: float, weights_root: Path):
    global_best_dir = weights_root / "global_best"
    current_best = _read_global_best(global_best_dir)

    improved = (current_best is None) or (best_val_rmse < current_best)
    if not improved:
        logger.info(f"[global_best] No improvement (current best={current_best:.6f} <= {best_val_rmse:.6f}).")
        return

    logger.info(f"[global_best] Improved! Promoting trial ({best_val_rmse:.6f}) -> global_best/")
    # copy files
    _atomic_copy(trial_dir / "net.pt", global_best_dir / "net.pt")
    _atomic_copy(trial_dir / "best_hparams.yaml", global_best_dir / "best_hparams.yaml")
    # write metrics
    with open(global_best_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "val_rmse": float(best_val_rmse),
                "timestamp": int(datetime.now().timestamp()),
                "source_run_dir": str(trial_dir),
            },
            f,
            indent=2,
        )


def move_to_device(batch, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # band data is tuple(torch.Tensor, torch.Tensor)
    # with first tensor being the spectra and the second being the constants.
    bands, aux_cont, aux_cat = batch
    bands_gpu = {k: (v[0].to(device), v[1].to(device)) for k, v in bands.items()}
    aux_cont_gpu = aux_cont.to(device)
    aux_cat_gpu  = [b.to(device) for b in aux_cat]
    return (bands_gpu, aux_cont_gpu, aux_cat_gpu)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    # Hydra gives each trial a unique working directory:
    trial_dir = Path(os.getcwd())
    trial_net_pt = trial_dir / "net.pt"                 # per-trial checkpoint (safe)
    trial_net_prof_pt = trial_dir / "net_profiles.pt"   # optional duplicate for older loaders
    trial_hparams = trial_dir / "best_hparams.yaml"
    trial_metrics = trial_dir / "metrics.json"

    # Shared weights root (used only for global_best/ promotion)
    weights_root = Path(cfg.data.weight_path).expanduser()
    weights_root.mkdir(parents=True, exist_ok=True)

    # Seed & device
    torch.manual_seed(cfg.hyps.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data. data.split_csv selects a reference-sounding-disjoint split (built by
    # make_group_split.py); leaving it unset keeps the original replicate-level split.
    stats_dir = cfg.data.get("stats_dir", None)
    split_csv = cfg.data.get("split_csv", None)
    if split_csv is not None:
        logger.info(f"Using group-disjoint split from {split_csv}")
    # cfg.covariates is a preset name (see code/data/loading/covariates.py) or an inline
    # {continuous, categorical} mapping. Absent -> the published four inputs.
    cov_spec = get_spec(cfg.get("covariates", None))
    logger.info("covariates: %s (tier=%s, %d continuous, %d categorical)",
                cov_spec.name, cov_spec.tier, cov_spec.n_continuous, cov_spec.n_categorical)
    # Cluster-balanced training. data.balance_by names a column of the split table to
    # equalise (use "clst"); data.balance_alpha tempers it, 0 = off, 1 = full inverse
    # frequency, 0.5 = square root. Unset -> uniform shuffling, i.e. published behaviour.
    # `balance_by="density"` selects the covariate-density sampler instead of a column:
    # weights come from the local density of `covariates` in scene space, tempered by the
    # same alpha, with data.density_{mode,beta,k,clip} for the rest. See
    # code/data/loading/density_weights.py for what that intervention is and is not.
    balance_by = cfg.data.get("balance_by", None)
    balance_alpha = float(cfg.data.get("balance_alpha", 0.5))
    density = density_settings(cfg.data) if balance_by == DENSITY_KEY else None
    if balance_by == DENSITY_KEY:
        logger.info("covariate-density sampling in %s space with alpha=%.2f, %s",
                    cov_spec.name, balance_alpha, density)
    elif balance_by:
        logger.info("cluster-balanced sampling on '%s' with alpha=%.2f",
                    balance_by, balance_alpha)
    train_dl, val_dl = train_val_loader(
        cfg.data.files,
        cfg.data.annotations,
        seed=cfg.data.seed,
        stats_dir=Path(stats_dir) if stats_dir else None,
        split_csv=split_csv,
        cov_spec=cov_spec,
        balance_by=balance_by,
        balance_alpha=balance_alpha,
        density=density,
    )

    # Model via Hydra instantiate so you can swap architectures in YAML
    model_kwargs = {}
    if not cov_spec.legacy:
        cov_stats = load_covariate_stats(Path(stats_dir) if stats_dir else None)
        model_kwargs["n_cont"] = cov_spec.n_continuous
        model_kwargs["cat_cardinalities"] = list(cat_cardinalities(cov_spec, cov_stats))
    model = hydra.utils.instantiate(cfg.model, **model_kwargs)
    logger.info(f"model being trained: {type(model).__name__}")
    if torch.cuda.is_available():
        model = model.cuda()

    # The band encoders use nn.LazyLinear, whose weights do not exist until the
    # first forward pass infers in_features. Materialize them on a single sounding
    # (cheap, and avoids OOM) before counting parameters or handing them to AdamW.
    with torch.no_grad():
        model.eval()
        warm_bands, warm_aux_cont, warm_aux_cat = next(iter(train_dl))[0]
        warm_x = (
            {k: (v[0][:1], v[1][:1]) for k, v in warm_bands.items()},
            warm_aux_cont[:1],
            [t[:1] for t in warm_aux_cat],
        )
        _ = model(move_to_device(warm_x, device))
    model.train()

    n_param = sum(p.numel() for p in model.parameters())
    logger.info(f"Total trainable params: {n_param:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.hyps.lr, weight_decay=cfg.hyps.wd)
    # Schedule knobs, all optional and all defaulting to the published behaviour:
    # warmup_epochs=0 and eta_min_frac=0.1 reproduce CosineAnnealingLR(T_max=epochs,
    # eta_min=lr/10) to float error, and grad_clip=None is no clipping. They exist because
    # The published learning curves show the three deterministic regressors still descending at epoch 40 and
    # visibly noisy epoch to epoch -- both properties of the schedule rather than of the
    # searched configuration, which is why the peak learning rate is left alone. See
    # code/utils/schedule.py.
    # `k in cfg.hyps`, not `cfg.hyps.get(k, d)`: Hydra composes in STRUCT mode, where
    # .get() on an undeclared key raises rather than returning the default. Membership is
    # the idiom the rest of this file already uses, and it is why these arrive on the
    # command line as `++hyps.warmup_epochs=3` -- the add-or-override form.
    warmup_epochs = int(cfg.hyps.warmup_epochs) if "warmup_epochs" in cfg.hyps else 0
    eta_min_frac = float(cfg.hyps.eta_min_frac) if "eta_min_frac" in cfg.hyps else 0.1
    grad_clip = cfg.hyps.grad_clip if "grad_clip" in cfg.hyps else None
    grad_clip = None if grad_clip in (None, 0, 0.0) else float(grad_clip)
    sched = cosine_with_warmup(opt, epochs=int(cfg.hyps.epochs),
                               warmup_epochs=warmup_epochs, eta_min_frac=eta_min_frac)
    logger.info(describe_sched(int(cfg.hyps.epochs), float(cfg.hyps.lr),
                               warmup_epochs, eta_min_frac, grad_clip))
    mse_loss = torch.nn.MSELoss()

    best_val_rmse = float("inf")
    last_retr_rmse = None  # for logging to mlflow at the end
    # One record per epoch, written to history.json beside the trial checkpoint. It is the
    # learning curve: cheap to keep, impossible to recover afterwards from a checkpoint,
    # and the thing to plot to see whether the model over-fits.
    history = []

    with mlflow.start_run():
        # Log static params once
        mlflow.log_params({
            "lr": float(cfg.hyps.lr),
            "wd": float(cfg.hyps.wd),
            "embed_dim": int(cfg.hyps.embed_dim),
            "nhead": int(cfg.hyps.nhead) if "nhead" in cfg.hyps else None,
            "nlayers": int(cfg.hyps.nlayers) if "nlayers" in cfg.hyps else None,
            "epochs": int(cfg.hyps.epochs),
            "warmup_epochs": warmup_epochs,
            "eta_min_frac": eta_min_frac,
            "grad_clip": grad_clip if grad_clip is not None else 0.0,
            "seed": int(cfg.hyps.seed),
            "covariates": cov_spec.name,
            "covariate_tier": cov_spec.tier,
            "balance_by": balance_by or "none",
            "balance_alpha": balance_alpha if balance_by else 0.0,
            # The density sampler's own settings, so the run record says which
            # intervention was applied and not merely that one was.
            "density_mode": density["mode"] if density else "none",
            "density_beta": density["beta"] if density else 0.0,
            "density_k": density["k"] if density else 0,
            "density_clip": density["clip"] if density else 0.0,
        })

        for epoch in range(cfg.hyps.epochs):
            # ----------------- train -----------------
            model.train()
            # The epoch's mean training loss, accumulated for `history` below. Without it
            # the only per-epoch number this trainer records is a validation one, and a
            # validation curve on its own cannot say whether the gap to training is
            # widening, which is the over-fitting diagnostic.
            tr_mse_accum, n_tr_batches = 0.0, 0
            for x, y, y_retr in train_dl:
                if torch.cuda.is_available():
                    x = move_to_device(x, device)
                    y = y.cuda()
                opt.zero_grad(set_to_none=True)
                y_hat = model(x)
                loss = mse_loss(y_hat, y)
                loss.backward()
                clip_gradients(model.parameters(), grad_clip)
                opt.step()
                tr_mse_accum += float(loss.detach())
                n_tr_batches += 1
            sched.step()
            train_rmse = rmse(tr_mse_accum / max(n_tr_batches, 1))

            # ----------------- eval ------------------
            model.eval()
            with torch.no_grad():
                mse_accum = 0.0
                retr_mse_accum = 0.0
                n_batches = 0
                n_retr = 0
                for x, y, y_retr in val_dl:
                    if torch.cuda.is_available():
                        x = move_to_device(x, device)
                        y = y.cuda()
                        y_retr = y_retr.cuda()
                    y_hat = model(x)
                    mse_accum += mse_loss(y_hat, y).item()
                    n_batches += 1
                    # retrieval baseline RMSE (for reference)
                    retr_mse_accum += mse_loss(y_retr, y).item()
                    n_retr += 1

                val_mse = mse_accum / max(n_batches, 1)
                retr_mse = (retr_mse_accum / max(n_retr, 1)) if n_retr > 0 else float("nan")
                val_rmse = rmse(val_mse)
                last_retr_rmse = rmse(retr_mse) if np.isfinite(retr_mse) else None

            logger.info(f"epoch = {epoch}, train loss = {train_rmse:.2E}, "
                        f"val loss = {val_rmse:.2E}, retr_loss = "
                        f"{(last_retr_rmse if last_retr_rmse is not None else float('nan')):.2E}")
            history.append({"epoch": int(epoch),
                            "train_rmse_std": float(train_rmse),
                            "val_rmse_std": float(val_rmse),
                            "retr_rmse_std": (None if last_retr_rmse is None
                                              else float(last_retr_rmse))})

            # Track & save per-trial best into trial_dir only
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                # checkpoint(s)
                torch.save({"model_state_dict": model.state_dict()}, trial_net_pt)
                # optional duplicate name for older helpers
                torch.save({"model_state_dict": model.state_dict()}, trial_net_prof_pt)

                best_hparams = {
                    "name": "optuna",
                    "best_params": {
                        "hyps.lr": float(cfg.hyps.lr),
                        "hyps.wd": float(cfg.hyps.wd),
                        "hyps.embed_dim": int(cfg.hyps.embed_dim),
                        **({"hyps.nhead": int(cfg.hyps.nhead)} if "nhead" in cfg.hyps else {}),
                        **({"hyps.nlayers": int(cfg.hyps.nlayers)} if "nlayers" in cfg.hyps else {}),
                        # The covariate set is part of the architecture, not a detail:
                        # it fixes the width of the auxiliary encoder and the size of its
                        # embedding tables. checkpoints.py reads this key to rebuild the
                        # right model, and without it a covariate-trained checkpoint is
                        # loaded into a four-input network and half its parameters are
                        # reported missing.
                        "covariates": cov_spec.name,
                    },
                    "best_value": float(val_rmse),  # objective
                }
                with open(trial_hparams, "w") as f:
                    yaml.safe_dump(best_hparams, f, sort_keys=False)

                with open(trial_metrics, "w") as f:
                    json.dump(
                        {
                            "val_rmse": float(val_rmse),
                            "retr_rmse": float(last_retr_rmse) if last_retr_rmse is not None else None,
                            "timestamp": int(datetime.now().timestamp()),
                        },
                        f,
                        indent=2,
                    )

                print(f"[INFO] Saved best model to: {trial_net_pt}")
                print(f"[INFO] Saved best hparams to: {trial_hparams}")

            # per-epoch logging
            mlflow.log_metric("val_rmse_epoch", float(val_rmse), step=epoch)
            if last_retr_rmse is not None:
                mlflow.log_metric("retr_rmse_epoch", float(last_retr_rmse), step=epoch)

        # end epochs

        with open(Path(trial_dir) / "history.json", "w") as f:
            json.dump({"study": "det", "target": "xco2",
                       "epochs": int(cfg.hyps.epochs),
                       "metric": "RMSE, standardized units",
                       "covariates": cov_spec.name,
                       "n_params": int(n_param),
                       "best_val_rmse_std": float(best_val_rmse),
                       "history": history}, f, indent=2)

        # Log final/best numbers
        mlflow.log_metric("val_rmse", float(best_val_rmse))
        if last_retr_rmse is not None:
            mlflow.log_metric("retr_rmse", float(last_retr_rmse))

    # After the run, maybe promote to global_best/
    _maybe_promote_to_global_best(trial_dir, best_val_rmse, weights_root)

    # Return objective for Hydra Optuna sweeper (direction=minimize)
    return float(best_val_rmse)


if __name__ == "__main__":
    main()

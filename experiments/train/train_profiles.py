"""train_profiles.py
Train the CO2 profile regressor (supports target slicing like last-4).
Author(s): Alejandro Calle-Saldarriaga
"""
import logging
logger = logging.getLogger(__name__)

import sys
from pathlib import Path
import os, json, re, shutil
from collections.abc import Sequence
import yaml
from datetime import datetime
from hydra.core.hydra_config import HydraConfig

import hydra
import mlflow
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

os.environ.setdefault("MLFLOW_TRACKING_URI", "file:///workspace/mlruns")
try:
    import mlflow
    mlflow.set_tracking_uri("file:///workspace/mlruns")
except Exception as _e:
    pass  # ok if mlflow is not used/installed

DEFAULT_WEIGHTS_DIR = Path("/workspace/project/experiments/weights/full_regressor_profiles")


def _weights_dir(cfg) -> Path:
    return Path(getattr(cfg, "weights_dir", DEFAULT_WEIGHTS_DIR))


def _global_best_dir(cfg) -> Path:
    """Where the cross-trial best checkpoint lives.

    This used to be a single hardcoded path shared by every task. Because the tracked
    metric is RMSE over whatever `target_indices` selects, a last-4 sweep (4 dims) and a
    full-profile sweep (20 dims) were writing mutually incomparable scores to the same
    file, so one could overwrite the other's checkpoint. It is now scoped to the run's
    own weights_dir, and can be overridden explicitly with cfg.global_best_dir.
    """
    explicit = getattr(cfg, "global_best_dir", None)
    return Path(explicit) if explicit else _weights_dir(cfg) / "global_best"

# make sure code/ is on path when running from experiments/train/
sys.path.append("/workspace/project/code")

from data.loading.spectral_loader import train_val_loader_profiles, _load_profile_stats
from data.loading.covariates import get_spec, cat_cardinalities, load_covariate_stats
from utils.schedule import cosine_with_warmup, clip_gradients, describe as describe_sched
from modeling.full_regressor_profiles import FullRegressorProfiles
from modeling.losses import level_weights, weighted_mse
from utils.targets import resolve_target_indices as _resolve_target_indices


def resolve_target_indices(cfg, D_full=20):
    """Which profile levels this run predicts. Lookup here, resolution in utils.targets.

    This function and its counterpart in train_nf_profiles.py were maintained separately
    and drifted twice, in opposite directions: the list branch was dead here (det_last1
    trained all twenty levels) and the string branch was incomplete there (`str` is a
    Sequence, so `all` was iterated character by character and every nf_profiles run died
    on `int('a')`). They now share one implementation; only the lookup differs, because
    Hydra keeps this at the top level and the flow trainers keep it under `model`.
    """
    return _resolve_target_indices(getattr(cfg, "target_indices", "all"), D_full)


def _require_consistent_out_dim(cfg, idxs, out_dim) -> None:
    """Stop if the head width and the target slice disagree.

    `target_indices` decides what is predicted and `model.out_dim` decides how wide
    the head is, and nothing tied them together. det_last1 was configured with
    target_indices=[19] and out_dim=1, ran with all twenty levels and a 20-wide head,
    trained for 25 minutes, and wrote a checkpoint indistinguishable from
    det_profiles into the surface task's weights directory. The cost of not noticing
    was two failed scoring jobs and a table row nobody could produce; the cost of
    noticing is this function, at second zero.

    Only checks when the config states a width. Configs that leave `model.out_dim`
    to the default are saying "follow the targets", which is what already happens.
    """
    declared = None
    try:
        declared = cfg["model"]["out_dim"]
    except Exception:
        return
    if declared is None:
        return
    if int(declared) != int(out_dim):
        raise SystemExit(
            f"config error: model.out_dim={int(declared)} but target_indices "
            f"resolved to {out_dim} level(s) {idxs}. These must agree -- the head "
            f"width and the target slice are the same decision. Check that the "
            f"experiment config in this code tarball is the one you meant: a stale "
            f"tarball is the usual cause, since Hydra merges whatever file is there."
        )


def _extract_hparams(cfg, target_indices, out_dim):
    # Pull out everything you’ll want to reconstruct the model
    def _pick(path, default=None):
        cur = cfg
        try:
            for k in path.split("."):
                cur = cur[k]
            return cur
        except Exception:
            return default

    return {
        "embed_dim": _pick("model.embed_dim", _pick("hyps.embed_dim", 256)),
        "nhead":     _pick("model.nhead",     _pick("hyps.nhead", 8)),
        "nlayers":   _pick("model.nlayers",   _pick("hyps.nlayers", 1)),
        "aux_dim":   _pick("model.aux_dim",   _pick("hyps.aux_dim", 32)),
        "lr":        _pick("hyps.lr", 3e-4),
        "wd":        _pick("hyps.wd", 3e-4),
        "seed":      _pick("hyps.seed", _pick("data.seed", 42)),
        # NEW:
        "target_indices": list(target_indices),
        "out_dim": int(out_dim),
        # The covariate set is part of the model's identity: two checkpoints with the
        # same architecture and different auxiliary inputs are not interchangeable, and
        # predict.py has to rebuild the right one.
        "covariates": _pick("covariates", "legacy"),
        "select_metric": _pick("select_metric", "std"),
        "loss_weight": _pick("loss_weight", "std"),
    }

def rmse_from_batches(sumsq: float, count: int) -> float:
    return (sumsq / count) ** 0.5

@torch.no_grad()
def evaluate(model, dl, device, idxs, sigma_ppm=None):
    """Validation RMSE in standardized units and, when sigma is known, in ppm.

    The ppm figure is the one Tables 2 and 3 report. Returning only the standardized one
    -- which is what this function used to do -- means checkpoint selection optimizes a
    differently weighted average of the per-level errors than the number in the paper.
    """
    model.eval()
    se_sum = 0.0
    se_retr_sum = 0.0
    se_ppm = 0.0
    se_retr_ppm = 0.0
    n = 0

    w = None
    if sigma_ppm is not None:
        w = torch.tensor(np.asarray(sigma_ppm, dtype=np.float64)[list(idxs)] * 1e6,
                         dtype=torch.float32, device=device) ** 2

    for (bands, aux_cont, aux_cat), y, y_retr in dl:
        # move to device
        bands = {k: (b.to(device), c.to(device)) for k, (b, c) in bands.items()}
        aux_cont = aux_cont.to(device)
        aux_cat = [t.to(device) for t in aux_cat]
        y = y.to(device)
        y_retr = y_retr.to(device)
        y_retr = y_retr[:, idxs]

        y_hat = model((bands, aux_cont, aux_cat))  # [B, D’]
        y_tgt = y[:, idxs]

        d2 = (y_hat - y_tgt) ** 2
        d2r = (y_retr - y_tgt) ** 2
        se_sum += d2.sum().item()
        se_retr_sum += d2r.sum().item()
        if w is not None:
            se_ppm += (d2 * w).sum().item()
            se_retr_ppm += (d2r * w).sum().item()
        n += y_tgt.numel()

    out = {
        "rmse_std": (se_sum / n) ** 0.5,
        "retr_rmse_std": (se_retr_sum / n) ** 0.5,
    }
    if w is not None:
        out["rmse_ppm"] = (se_ppm / n) ** 0.5
        out["retr_rmse_ppm"] = (se_retr_ppm / n) ** 0.5
    return out

def _maybe_update_global_best(best_metric_this_trial: float, best_state_dict: dict, cfg, target_indices, out_dim) -> bool:
    global_best_dir = _global_best_dir(cfg)
    global_best_dir.mkdir(parents=True, exist_ok=True)
    score_file = global_best_dir / "best_metric.txt"

    prev = float("inf")
    if score_file.exists():
        try:
            prev = float(score_file.read_text().strip())
        except Exception:
            pass

    if best_metric_this_trial < prev:
        # write weights
        torch.save(best_state_dict, global_best_dir / "net_profiles.pt")
        # write metric
        score_file.write_text(f"{best_metric_this_trial:.12e}\n")
        # write hparams (include target info)
        with open(global_best_dir / "best_hparams.yaml", "w") as f:
            yaml.safe_dump(_extract_hparams(cfg, target_indices, out_dim), f)
        print(f"[INFO] New GLOBAL BEST (RMSE={best_metric_this_trial:.6e}) -> {global_best_dir}")
        return True
    return False

def repo_root() -> Path:
    # experiments/train/train_profiles.py -> experiments -> project root
    return Path(__file__).resolve().parents[2]

def best_runs_root() -> Path:
    return repo_root() / "log" / "best_runs"

def save_global_best_checkpoint(
    model: torch.nn.Module,
    cfg,
    val_rmse: float,
    step_counter: int,
    epoch: int,
    run_id: str | None,
    global_weights_dir: Path,
    target_indices,
    out_dim: int,
):
    """
    - versioned:   log/best_runs/best_ckpt/{seed}/step_{step}.pkl
    - marker:      BEST.json + best.pkl in the same dir
    - global copy: /workspace/.../net_profiles.pt (only if globally better)
    """
    seed = int(cfg.hyps.seed)
    ckpt_root = best_runs_root() / "best_ckpt" / str(seed)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # Save a rich state (state_dict + config + meta)
    state = {
        "state_dict": model.state_dict(),
        "config": {
            "embed_dim": cfg.model.embed_dim,
            "nhead": cfg.model.nhead,
            "nlayers": cfg.model.nlayers,
            "aux_dim": cfg.model.aux_dim,
            "out_dim": int(out_dim),
            "target_indices": list(target_indices),
        },
        "seed": seed,
        "val_rmse": float(val_rmse),
        "step": int(step_counter),
        "epoch": int(epoch),
        "run_id": run_id or "",
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    versioned_ckpt = ckpt_root / f"step_{step_counter}.pkl"
    torch.save(state, versioned_ckpt)

    # Update BEST.json and best.pkl only if better
    marker = ckpt_root / "BEST.json"
    prev = {"val_rmse": float("inf")}
    if marker.exists():
        try:
            prev = json.loads(marker.read_text())
        except Exception:
            pass

    if state["val_rmse"] < float(prev.get("val_rmse", float("inf"))):
        marker.write_text(json.dumps(
            {"val_rmse": state["val_rmse"], "step": state["step"], "epoch": state["epoch"],
             "run_id": state["run_id"], "ckpt": str(versioned_ckpt)}, indent=2))
        torch.save(state, ckpt_root / "best.pkl")  # convenience pointer

        # also keep the simple global copy (state_dict only)
        global_weights_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), global_weights_dir / "net_profiles.pt")

def record_trial_metadata(run_id: str | None, cfg, val_rmse: float, target_indices, out_dim: int):
    runs_dir = best_runs_root() / "runs"
    trial_dir = runs_dir / (run_id or "no-mlflow")
    trial_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": run_id or "",
        "val_rmse": float(val_rmse),
        "seed": int(cfg.hyps.seed),
        "model": {
            "embed_dim": cfg.model.embed_dim,
            "nhead": cfg.model.nhead,
            "nlayers": cfg.model.nlayers,
            "aux_dim": cfg.model.aux_dim,
            "out_dim": int(out_dim),
            "target_indices": list(target_indices),
        },
        "hydra_output_dir": os.getcwd(),  # Hydra sets CWD per-trial
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    (trial_dir / "summary.json").write_text(json.dumps(meta, indent=2))


def move_to_device(batch, device=None):
    """Matches the (bands dict, aux_cont, aux_cat) structure produced by profiles loaders."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bands, aux_cont, aux_cat = batch
    bands = {k: (v[0].to(device), v[1].to(device)) for k, v in bands.items()}
    aux_cont = aux_cont.to(device)
    aux_cat = [t.to(device) for t in aux_cat]
    return (bands, aux_cont, aux_cat)


@hydra.main(version_base=None, config_path="conf", config_name="config_profiles")
def main(cfg):
    torch.manual_seed(cfg.hyps.seed)

    # ---------- target indices ----------
    idxs = resolve_target_indices(cfg, D_full=20)
    out_dim = len(idxs)
    logger.info(f"Training on target_indices={idxs} (out_dim={out_dim})")
    _require_consistent_out_dim(cfg, idxs, out_dim)

    # ---------- data ----------
    stats_dir = cfg.data.get("stats_dir", None)
    # data.split_csv selects a reference-sounding-disjoint split (built by
    # make_group_split.py); leaving it unset keeps the original replicate-level split.
    split_csv = cfg.data.get("split_csv", None)
    if split_csv is not None:
        logger.info(f"Using group-disjoint split from {split_csv}")

    # cfg.covariates is a preset name (see code/data/loading/covariates.py) or an inline
    # {continuous, categorical} mapping. Absent -> the published four inputs, encoded
    # exactly as before.
    cov_spec = get_spec(cfg.get("covariates", None))
    logger.info("covariates: %s (tier=%s, %d continuous, %d categorical)",
                cov_spec.name, cov_spec.tier, cov_spec.n_continuous, cov_spec.n_categorical)

    # Cluster-balanced training. data.balance_by names a column of the split table to
    # equalise (use "clst"); data.balance_alpha tempers it, 0 = off, 1 = full inverse
    # frequency, 0.5 = square root. Unset -> uniform shuffling, i.e. published behaviour.
    balance_by = cfg.data.get("balance_by", None)
    balance_alpha = float(cfg.data.get("balance_alpha", 0.5))
    if balance_by:
        logger.info("cluster-balanced sampling on '%s' with alpha=%.2f",
                    balance_by, balance_alpha)
    train_dl, val_dl = train_val_loader_profiles(
        root=cfg.data.files,
        annotations_filename=cfg.data.annotations,
        train_batch_size=cfg.data.train_batch_size,
        val_batch_size=cfg.data.val_batch_size,
        seed=cfg.data.seed,
        stats_dir=Path(stats_dir) if stats_dir else None,
        split_csv=split_csv,
        cov_spec=cov_spec,
        balance_by=balance_by,
        balance_alpha=balance_alpha,
    )

    # ---------- units ----------
    # Per-level sigma of the training targets: needed to report validation RMSE in ppm,
    # and to weight the loss the same way the reported metric does.
    sigma = None
    if stats_dir is not None:
        try:
            _, sigma = _load_profile_stats(Path(stats_dir))
        except FileNotFoundError:
            logger.warning("no profile_sigma.npy under %s; ppm metrics unavailable", stats_dir)
    select_metric = str(cfg.get("select_metric", "std"))
    loss_weight = str(cfg.get("loss_weight", "std"))
    if select_metric == "ppm" and sigma is None:
        raise ValueError("select_metric=ppm needs data.stats_dir to point at the split's "
                         "profile_sigma.npy")
    w_loss = None
    if loss_weight != "std":
        if sigma is None:
            raise ValueError("loss_weight=ppm needs data.stats_dir")
        w_loss = level_weights(sigma, loss_weight, idxs).to(device)
        logger.info("per-level loss weights (%s): %s", loss_weight,
                    np.array2string(w_loss.cpu().numpy(), precision=3))

    # ---------- model ----------
    # override out_dim at instantiate-time so head matches sliced targets
    model_kwargs = dict(out_dim=out_dim)
    if not cov_spec.legacy:
        cov_stats = load_covariate_stats(Path(stats_dir) if stats_dir else None)
        model_kwargs["n_cont"] = cov_spec.n_continuous
        model_kwargs["cat_cardinalities"] = list(cat_cardinalities(cov_spec, cov_stats))
    model = hydra.utils.instantiate(cfg.model, **model_kwargs)
    logger.info(f"model being trained: {type(model).__name__}")
    if torch.cuda.is_available():
        model = model.cuda()
    
    # warm up lazy params
    with torch.no_grad():
        model.eval()
        warm_batch = next(iter(train_dl))
        x_warm, y_warm, _ = warm_batch
        # keep it tiny to avoid OOM
        # take just the first item in the batch and keep batch dim
        bands, aux_cont, aux_cat = x_warm
        bands_1 = {k: (v[0][:1].to(device), v[1][:1].to(device)) for k, v in bands.items()}
        aux_cont_1 = aux_cont[:1].to(device)
        aux_cat_1  = [t[:1].to(device) for t in aux_cat]
        _ = model((bands_1, aux_cont_1, aux_cat_1))

    n_param = sum(p.numel() for p in model.parameters())
    logger.info(f"# params: {n_param:,}")
    model.train()

    # ---------- optim/loss ----------
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.hyps.lr,
        weight_decay=cfg.hyps.wd
    )
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
    # w_loss is None for the published unweighted MSE; see code/modeling/losses.py for
    # why the profile tasks want the ppm weighting instead.
    def loss_fn(y_hat, y_tgt):
        return weighted_mse(y_hat, y_tgt, w_loss)

    if torch.cuda.is_available():
        model = model.cuda()

    best_rmse = float("inf")
    best_state = None
    # One record per epoch -> weights_dir/history.json. See train.py for why.
    history = []
    weights_dir = _weights_dir(cfg)
    weights_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"weights_dir={weights_dir}  global_best_dir={_global_best_dir(cfg)}")

    with mlflow.start_run():
        mlflow.log_params(dict(
            epochs=cfg.hyps.epochs, lr=cfg.hyps.lr, wd=cfg.hyps.wd,
            warmup_epochs=warmup_epochs, eta_min_frac=eta_min_frac,
            grad_clip=grad_clip if grad_clip is not None else 0.0,
            seed=cfg.hyps.seed, embed_dim=cfg.model.embed_dim,
            nlayers=cfg.model.nlayers, nhead=cfg.model.nhead,
            out_dim=out_dim, target_indices=str(idxs),
            covariates=cov_spec.name, covariate_tier=cov_spec.tier,
            select_metric=select_metric, loss_weight=loss_weight,
            balance_by=balance_by or "none",
            balance_alpha=balance_alpha if balance_by else 0.0,
        ))
        step_counter = 0

        for epoch in range(cfg.hyps.epochs):
            # ---- train ----
            model.train()
            # The epoch's mean training loss, for history.json below: a validation curve
            # alone cannot show whether the gap to training is widening, which is the
            # over-fitting diagnostic.
            tr_loss_accum, n_tr_batches = 0.0, 0
            for x, y, _ in train_dl:
                step_counter += 1
                x = move_to_device(x, device)
                y = y.to(device)
                # slice targets to the subset
                opt.zero_grad(set_to_none=True)
                y_hat = model(x)                # [B, D’]
                y_tgt = y[:, idxs]
                loss = loss_fn(y_hat, y_tgt)        # MSE across selected dims
                loss.backward()
                clip_gradients(model.parameters(), grad_clip)
                opt.step()
                tr_loss_accum += float(loss.detach())
                n_tr_batches += 1
            sched.step()
            train_loss = tr_loss_accum / max(n_tr_batches, 1)

            # ---- validate (RMSE over selected dims) ----
            model.eval()
            metrics = evaluate(model, val_dl, device, idxs, sigma_ppm=sigma)
            val_rmse = metrics["rmse_ppm"] if select_metric == "ppm" else metrics["rmse_std"]
            msg = (f"epoch={epoch}  val_RMSE[std]={metrics['rmse_std']:.4f}  "
                   f"retr_RMSE[std]={metrics['retr_rmse_std']:.4f}")
            if "rmse_ppm" in metrics:
                msg += (f"  val_RMSE[ppm]={metrics['rmse_ppm']:.4f}  "
                        f"retr_RMSE[ppm]={metrics['retr_rmse_ppm']:.4f}")
            msg += f"  (selecting on {select_metric})"
            logger.info("epoch=%d  train_loss=%.6f", epoch, train_loss)
            logger.info(msg)
            history.append(dict(epoch=int(epoch), train_loss=float(train_loss),
                                **{k: float(v) for k, v in metrics.items()}))
            for k, v in metrics.items():
                mlflow.log_metric(f"val_{k}", float(v), step=epoch)
            run = mlflow.active_run()
            run_id = run.info.run_id if run else None
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                save_global_best_checkpoint(
                    model=model,
                    cfg=cfg,
                    val_rmse=best_rmse,
                    step_counter=step_counter,
                    epoch=epoch,
                    run_id=run_id,
                    global_weights_dir=weights_dir,
                    target_indices=idxs,
                    out_dim=out_dim,
                )
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                # save the versioned ckpt as an artifact if needed; path is printed in save_global_best_checkpoint

        # final logging & global best marker
        # `val_rmse` is whatever select_metric chose, and best_metric.txt holds the same
        # number -- so a sweep must not mix select_metric values within one weights_dir.
        mlflow.log_metric("val_rmse", best_rmse)
        mlflow.log_metric("val_rmse_selected_on_" + select_metric, best_rmse)
        with open(weights_dir / "history.json", "w") as f:
            json.dump({"study": "det", "target_indices": [int(i) for i in idxs],
                       "epochs": int(cfg.hyps.epochs),
                       "metric": f"RMSE, selected on {select_metric}",
                       "covariates": cov_spec.name,
                       "loss": "MSE on the selected levels",
                       "best_val_rmse": float(best_rmse),
                       "history": history}, f, indent=2)
        record_trial_metadata(run_id if 'run_id' in locals() else None, cfg, best_rmse, idxs, out_dim)
        if best_state is not None:
            _maybe_update_global_best(best_rmse, best_state, cfg, idxs, out_dim)
    return best_rmse


if __name__ == "__main__":
    main()

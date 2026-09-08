"""dump_checkpoint_config.py
Recover the configuration actually used to train a saved checkpoint.

Motivation: the hyperparameters recorded in the repository configs, in
weights/*/global_best/best_hparams.yaml, and in Table A1 of the manuscript do not all
agree. The checkpoints are the authoritative record, so this script reads them directly:
it prints any stored `cfg`/hparams payload and, independently, infers what it can from
the tensor shapes in the state dict.

What is recoverable from shapes alone:
    embed_dim E, transformer depth L, output dim D, flow depth, flow hidden width,
    number of spline bins.
What is NOT recoverable from shapes:
    number of attention heads H (nn.MultiheadAttention packs in_proj as [3E, E]
    regardless of H), tail_bound, learning rate, weight decay.
Those must come from a stored cfg; the script says so explicitly rather than guessing.

Usage:
    python3 dump_checkpoint_config.py weights/flow_regressor_profiles/best_flow_profiles.pt
    python3 dump_checkpoint_config.py weights/*/**/*.pt --json summary.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import torch

_UNKNOWN = "not recoverable from state_dict (needs stored cfg)"


def _flatten_tensors(obj: Any, prefix: str = "") -> Dict[str, torch.Tensor]:
    """Every tensor in a nested dict, keyed by its dotted path."""
    out: Dict[str, torch.Tensor] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, torch.Tensor):
                out[f"{prefix}{k}"] = v
            elif isinstance(v, dict):
                out.update(_flatten_tensors(v, f"{prefix}{k}."))
    return out


def _find_state_dict(obj: Any) -> Optional[Dict[str, torch.Tensor]]:
    """Locate the tensor dict inside the several checkpoint layouts used in this repo.

    Recurses, because it has to. train_nf.py saves
    `{"state_dict": {"backbone": {...}, "nf": {...}}, "extra": {...}}` -- a dict of dicts
    of tensors. The previous version looked exactly one level down, found that
    `obj["state_dict"]`'s values were dicts rather than tensors, and gave up with
    "no tensor state_dict found". That is why the scalar flow is the one checkpoint the
    Table A1 audit could not settle, and why xco2.ipynb ends up carrying its architecture
    as hand-typed literals. The information was in the file the whole time.

    Keys come back dotted (`backbone.encoder...`, `nf.flow...`), which the shape inference
    below matches on with `re.search`/`endswith` rather than `startswith`, so prefixes are
    harmless.
    """
    if not isinstance(obj, dict):
        return None
    for key in ("nf", "state_dict", "model_state_dict", "backbone", "feat"):
        if key in obj and isinstance(obj[key], dict):
            found = _flatten_tensors(obj[key])
            if found:
                return found
    found = _flatten_tensors({k: v for k, v in obj.items() if k not in ("cfg", "extra")})
    return found or None


def _stored_cfg(obj: Any) -> Optional[dict]:
    if isinstance(obj, dict):
        for key in ("cfg", "config"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return None


def infer_backbone(sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Infer FullRegressor(Profiles) architecture from parameter shapes."""
    out: Dict[str, Any] = {}

    # AuxEncoder.encoder: Linear(aux_dim -> E)
    for key in sd:
        if key.endswith("aux_encoder.encoder.weight"):
            E, aux_dim = sd[key].shape
            out["embed_dim"] = int(E)
            out["aux_dim"] = int(aux_dim)
            break

    # AttentionFusion: nn.Sequential of TransformerEncoderLayer -> fusion.layers.{i}.*
    depths = {
        int(m.group(1))
        for k in sd
        if (m := re.search(r"fusion\.layers\.(\d+)\.", k))
    }
    if depths:
        out["nlayers"] = max(depths) + 1

    # Prediction head: last Linear of `predictor` gives out_dim
    head = sorted(
        (int(m.group(1)), k)
        for k in sd
        if (m := re.search(r"predictor\.(\d+)\.weight$", k))
    )
    if head:
        out["out_dim"] = int(sd[head[-1][1]].shape[0])

    out["nhead"] = _UNKNOWN
    return out


def infer_flow(sd: Dict[str, torch.Tensor], out_dim: Optional[int] = None) -> Dict[str, Any]:
    """Infer coupling/autoregressive spline flow shape from parameter names."""
    out: Dict[str, Any] = {}

    layer_ids = {
        int(m.group(1))
        for k in sd
        if (m := re.search(r"flow\._transform\._transforms\.(\d+)\.", k))
    }
    if not layer_ids:
        return out

    # transform_net_create_fn builds a ResidualNet per coupling layer; its final linear
    # maps hidden_features -> transformed_dims * (3*bins + 2)
    finals = {}
    for k, v in sd.items():
        m = re.search(r"flow\._transform\._transforms\.(\d+)\..*final_layer\.weight$", k)
        if m:
            finals[int(m.group(1))] = tuple(v.shape)

    out["n_coupling_layers"] = len(finals) if finals else _UNKNOWN
    if finals:
        any_id = sorted(finals)[0]
        out_features, hidden_features = finals[any_id]
        out["hidden_features"] = int(hidden_features)
        out["_final_layer_out_features"] = int(out_features)

        # Bins, solved rather than left as a puzzle for the reader.
        #
        # For a rational-quadratic spline with tails="linear", nflows parameterizes each
        # transformed dimension with 3K-1 numbers: K widths, K heights, K-1 interior
        # derivatives. (An earlier version of this comment said 3K+2, which is the
        # unbounded-tails count; with linear tails it is wrong, and it does not divide
        # the shapes these checkpoints actually have.)
        #
        # An alternating coupling mask over D dimensions transforms D//2 of them, so
        #     out_features = (D // 2) * (3K - 1).
        # Checked against both profile checkpoints: 290 = 10*(3*10-1) for D=20, K=10, and
        # 46 = 2*(3*8-1) for D=4, K=8. The scalar flow is autoregressive rather than
        # coupling, so every dimension is transformed and D//2 becomes D.
        if out_dim:
            transformed = out_dim if out_dim == 1 else out_dim // 2
            k, rem = divmod(out_features // transformed + 1, 3)
            if rem == 0 and transformed * (3 * k - 1) == out_features:
                out["bins"] = int(k)
            else:
                out["bins"] = (
                    f"no integer K solves out_features = {transformed}*(3K-1); "
                    f"out_features={out_features}. Check the mask or the tail type."
                )
        else:
            out["bins"] = (
                "pass --out-dim to solve out_features = transformed_dims * (3*bins - 1); "
                f"out_features={out_features}"
            )
    out["tail_bound"] = _UNKNOWN
    return out


def _load(path: Path) -> Any:
    # These checkpoints carry plain dicts alongside tensors, so weights_only must be off;
    # the kwarg only exists on newer torch, hence the fallback.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def describe(path: Path) -> Dict[str, Any]:
    obj = _load(path)
    info: Dict[str, Any] = {"path": str(path)}

    cfg = _stored_cfg(obj)
    if cfg is not None:
        info["stored_cfg"] = cfg
    if isinstance(obj, dict):
        if "extra" in obj:
            info["extra"] = obj["extra"]
        info["top_level_keys"] = sorted(k for k in obj if not isinstance(obj[k], torch.Tensor))

    sd = _find_state_dict(obj)
    if sd is None:
        info["error"] = "no tensor state_dict found"
        return info

    info["n_tensors"] = len(sd)
    info["n_parameters"] = int(sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor)))
    backbone = infer_backbone(sd)
    info["inferred_backbone"] = backbone

    # The flow's target dimension, needed to solve for the bin count. The stored cfg is
    # authoritative where it exists; otherwise fall back to the backbone head's width.
    # The scalar flow has neither -- its backbone is built with out_dim=1 but only the
    # encoder is saved -- so treat a flow with no recoverable out_dim as 1-dimensional
    # only when the transform is autoregressive, which is what train_nf.py builds.
    out_dim = None
    if isinstance(cfg, dict):
        out_dim = cfg.get("out_dim") or cfg.get("model", {}).get("out_dim")
    if out_dim is None:
        out_dim = backbone.get("out_dim") if isinstance(backbone.get("out_dim"), int) else None
    if out_dim is None and any("autoregressive_net" in k for k in sd):
        out_dim = 1

    flow = infer_flow(sd, out_dim=out_dim)
    if flow:
        info["inferred_flow"] = flow
        if out_dim is not None:
            flow["_assumed_out_dim"] = int(out_dim)
    return info


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+", type=Path)
    p.add_argument("--json", type=Path, default=None, help="also write results as JSON")
    args = p.parse_args()

    results = []
    n_missing = 0
    for path in args.checkpoints:
        if not path.exists():
            print(f"\n=== {path} ===\n  MISSING")
            n_missing += 1
            continue
        print(f"\n=== {path} ===")
        try:
            info = describe(path)
        except Exception as exc:  # a corrupt or unexpected checkpoint should not abort the sweep
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            results.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        results.append(info)
        for key, value in info.items():
            if key == "path":
                continue
            if isinstance(value, dict):
                print(f"  {key}:")
                for k2, v2 in value.items():
                    print(f"    {k2}: {v2}")
            else:
                print(f"  {key}: {value}")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nWrote {args.json}")

    print(
        "\nNote: attention-head count, tail bound, learning rate and weight decay cannot "
        "be read off tensor shapes. Take them from `stored_cfg` where present; otherwise "
        "they must come from the training logs (mlflow / Optuna study)."
    )

    # Every path missing means the checkpoints were never unpacked, not that the audit
    # found nothing. Writing `[]` and exiting 0 made that look like a script bug for long
    # enough to be worth a nonzero status: on CHTC the .pt files are excluded from
    # oco2_code.tar.gz and have to come from the Zenodo weights archive
    # (`./stage_data.sh weights`, then chtc/audit.sub, which does the unpacking).
    if n_missing and not results:
        print(
            f"\nERROR: all {n_missing} checkpoint paths are missing, so nothing was "
            "audited.\n       The published .pt files are not in the code tarball. On "
            "CHTC use chtc/audit.sub,\n       which stages and unpacks "
            "oco2_model_weights.tar.gz for you."
        )
        return 2
    if n_missing:
        print(f"\nWARNING: {n_missing} of {len(args.checkpoints)} paths were missing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

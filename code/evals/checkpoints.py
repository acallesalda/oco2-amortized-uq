"""checkpoints.py

Rebuild a trained model from what is on disk, for all six runs in the DAG.

Every scoring notebook opened its checkpoints by hand, and every one of them hardcoded at
least one architecture constant that the checkpoint did not record -- which is how
`xco2.ipynb` came to contain `nflow_cfg = {"hidden": 384, "layers": 5, "bins": 9,
"tail_bound": 6.3108919641}` typed out as literals. If any of those had been wrong the
state dict would still have loaded partially and produced numbers.

The rule here: read the architecture from the checkpoint when it is recorded there,
otherwise from the run's own config file, and fail loudly when neither has it. Nothing is
guessed and nothing is loaded with `strict=False` unless the mismatch is understood and
named.

The group-split checkpoints written by the current training scripts all record their own
configuration -- `train_nf.py` stores `extra.backbone_cfg`/`extra.nflow_cfg`,
`train_nf_profiles.py` and `train_nf_profiles_4d.py` store `cfg`, and the deterministic
trainers write `best_hparams.yaml` next to the weights -- so the fallbacks below matter
only for the *published* checkpoints, whose payloads predate that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------
# Where each job leaves its output.
#
# Keyed by the DAG node name, relative to a weights root (experiments/weights/<split>/).
# Kept in one place so that a path change in a *_grouped.yaml has exactly one echo here.
# ---------------------------------------------------------------------------------
# The candidate lists carry two layouts. The first entries are where the group-split
# training jobs write; the trailing ones are where the PUBLISHED Zenodo weights archive
# actually puts things, which chtc/audit.sub enumerated on 2026-08-14 and which is not the
# same shape:
#
#     full_regressor/net.pt                                  (no global_best/)
#     full_regressor_profiles/global_best/net_profiles_old.pt (only the *old* one)
#     full_regressor_profiles_last4_bo/net_profiles.pt        (note the _bo suffix)
#     flow_regressor_nflow/best_flow_nflows.pt
#     flow_regressor_profiles/best_flow_profiles.pt
#     flow_regressor_profiles_nf4_bo_warm_v1/best_nf4.pt
#
# `net_profiles_old.pt` is the LEGACY architecture -- pass legacy=True (predict.py's
# --legacy-backbone) when evaluating it, which is what full_profiles.ipynb did.
LAYOUT: Dict[str, Dict[str, Any]] = {
    "det_xco2": {
        "ckpt": ["full_regressor/global_best/net.pt", "full_regressor/net.pt"],
        "hparams": ["full_regressor/global_best/best_hparams.yaml"],
        "out_dim": 1,
    },
    "det_profiles": {
        "ckpt": ["full_regressor_profiles/net_profiles.pt",
                 "full_regressor_profiles/global_best/net_profiles.pt",
                 "full_regressor_profiles/global_best/net_profiles_old.pt"],
        "hparams": ["full_regressor_profiles/global_best/best_hparams.yaml"],
        "out_dim": 20,
    },
    "det_last4": {
        "ckpt": ["full_regressor_profiles_last4/net_profiles.pt",
                 "full_regressor_profiles_last4/global_best/net_profiles.pt",
                 "full_regressor_profiles_last4_bo/net_profiles.pt"],
        "hparams": ["full_regressor_profiles_last4/global_best/best_hparams.yaml",
                    "full_regressor_profiles_last4_bo/best_hparams.yaml"],
        "out_dim": 4,
    },
    "nf_xco2": {
        "ckpt": ["flow_regressor_nflow/best_flow_nflows.pt"],
        "hparams": ["flow_regressor_nflow/best_flow_nflows_config.yaml"],
        "out_dim": 1,
    },
    "nf_profiles": {
        "ckpt": ["flow_regressor_profiles/best_flow_profiles.pt"],
        "hparams": [],
        "out_dim": 20,
    },
    "nf_last4": {
        "ckpt": ["flow_regressor_profiles_nf4/best_nf4.pt",
                 "flow_regressor_profiles_nf4_bo_warm_v1/best_nf4.pt"],
        "hparams": [],
        "out_dim": 4,
    },
    # The surface-only (level 20) scalar task. Same shape as the XCO2 row of Table 1 --
    # one number per sounding, with an ACOS posterior variance to compare against -- but
    # the quantity is the CO2 concentration in the bottom layer rather than the column
    # average, which is where the spectrum carries the least information and where ACOS
    # is weakest.
    #
    # The flow here is `train_nf.py` in its `surface` target mode, so the checkpoint has
    # the SCALAR flow's payload shape (state_dict/{backbone,nf}) and not the profile
    # flow's. That is not a naming accident: a PRQ *coupling* stack cannot transform a
    # one-dimensional variable at all -- the alternating mask leaves the single
    # coordinate untransformed -- so the 1-d flow has to be the autoregressive spline
    # stack in train_nf.py. `load_flow` dispatches on the payload, not on the job name.
    "det_last1": {
        "ckpt": ["full_regressor_profiles_last1/net_profiles.pt",
                 "full_regressor_profiles_last1/global_best/net_profiles.pt"],
        "hparams": ["full_regressor_profiles_last1/global_best/best_hparams.yaml",
                    "full_regressor_profiles_last1/best_hparams.yaml"],
        "out_dim": 1,
    },
    "nf_last1": {
        "ckpt": ["flow_regressor_nflow_surface/best_flow_nflows.pt"],
        "hparams": ["flow_regressor_nflow_surface/best_flow_nflows_config.yaml"],
        "out_dim": 1,
    },
}


def resolve(weights_root: Path | str, job: str, kind: str = "ckpt") -> Optional[Path]:
    """First existing candidate path for `job`, or None."""
    root = Path(weights_root)
    for rel in LAYOUT[job].get(kind, []):
        p = root / rel
        if p.exists():
            return p
    return None


def _ckpt_out_dim(path: Path) -> Optional[int]:
    """Output width of a stored deterministic checkpoint, or None if unreadable.

    Read from the last two-dimensional weight of the prediction head, which is the
    only part of these models whose width is the task rather than a hyperparameter.
    """
    try:
        sd = _state_dict(torch.load(path, map_location="cpu"))
    except Exception:  # a payload this helper does not understand is not fatal
        return None
    two_d = [(k, v) for k, v in sd.items()
             if k.endswith(".weight") and torch.is_tensor(v) and v.ndim == 2]
    if not two_d:
        return None
    head = [kv for kv in two_d if "predictor" in kv[0] or "head" in kv[0]] or two_d
    return int(head[-1][1].shape[0])


def require(weights_root: Path | str, job: str) -> Path:
    """The checkpoint for `job`: the first candidate that exists AND fits.

    Taking the first that merely exists was enough while one file could be in one
    place. It stopped being enough for `det_last1`: both
    full_regressor_profiles_last1/net_profiles.pt and its global_best/ copy can be
    present, train_profiles.py writes them at different moments, and eval_last1 died
    in load_state_dict with a 20-output head against a 1-output model, seconds after
    the datasets were built, with a torch shape error as the only diagnostic.
    LAYOUT already knows how many outputs each job has, so use it: prefer a candidate
    whose stored head matches, and when none does, say so with the widths found
    rather than leaving it to a stack trace.
    """
    root = Path(weights_root)
    present = [root / rel for rel in LAYOUT[job].get("ckpt", []) if (root / rel).exists()]
    if not present:
        cands = "\n  ".join(str(root / r) for r in LAYOUT[job]["ckpt"])
        raise FileNotFoundError(
            f"no checkpoint for '{job}' under {weights_root}. Looked for:\n  {cands}\n"
            "Has the training job finished, and was its result archive unpacked?"
        )
    want = LAYOUT[job].get("out_dim")
    if want is None or len(present) == 1:
        return present[0]
    widths = [(p, _ckpt_out_dim(p)) for p in present]
    for p, w in widths:
        if w == int(want):
            if p != present[0]:
                logger.warning(
                    "[%s] %s has out_dim=%s, not %s; using %s instead.",
                    job, present[0].name, widths[0][1], want, p)
            return p
    logger.warning(
        "[%s] no candidate checkpoint has out_dim=%s. Found: %s. Loading %s anyway; "
        "a size mismatch in load_state_dict below is this.", job, want,
        ", ".join(f"{p.name}={w}" for p, w in widths), present[0])
    return present[0]


def _strip_module_prefix(sd: dict) -> dict:
    if any(k.startswith("module.") for k in sd):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _state_dict(blob) -> dict:
    if isinstance(blob, dict):
        for key in ("model_state_dict", "state_dict"):
            inner = blob.get(key)
            if isinstance(inner, dict) and all(torch.is_tensor(v) for v in inner.values()):
                return _strip_module_prefix(inner)
    return _strip_module_prefix(blob)


def _read_yaml(path: Optional[Path]) -> dict:
    if path is None or not Path(path).exists():
        return {}
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def _arch_from_hparams(hp: dict, defaults: dict) -> dict:
    """Normalize the two hparam dialects the trainers write.

    `train.py` writes `{"best_params": {"hyps.embed_dim": ...}}`; `train_profiles.py`
    writes flat `{"embed_dim": ...}`.
    """
    bp = hp.get("best_params", hp)
    out = dict(defaults)
    for key in ("embed_dim", "nhead", "nlayers", "aux_dim", "out_dim"):
        for candidate in (key, f"hyps.{key}", f"model.{key}"):
            if candidate in bp and bp[candidate] is not None:
                out[key] = int(bp[candidate])
                break
    return out


def _covariates_from_hparams(hp: dict) -> str:
    """Which covariate set the checkpoint was trained with.

    Absent -> "legacy", which is what every checkpoint written before covariates were
    configurable used. Getting this wrong does not raise: the auxiliary encoder simply has
    different parameter shapes, and load_state_dict would report them as missing -- so it
    is read from the checkpoint's own hyperparameters rather than assumed by the caller.
    """
    bp = hp.get("best_params", hp)
    for candidate in ("covariates", "hyps.covariates", "model.covariates"):
        if candidate in bp and bp[candidate]:
            return str(bp[candidate])
    return "legacy"


def _hparams_beside(ckpt: Path) -> Optional[Path]:
    """The hyperparameter file belonging to an explicitly named checkpoint.

    LAYOUT cannot resolve these. A covariate run's weights directory is named after its
    preset -- `full_regressor_scene_prior_best/` -- which is a runtime choice, not a
    constant, so the only reliable way to find its `best_hparams.yaml` is to look beside
    the checkpoint. train.py writes both into `<weights>/global_best/`; train_profiles.py
    additionally drops a copy of the weights one level up, hence the third candidate.
    """
    d = ckpt.parent
    for cand in (d / "best_hparams.yaml",
                 d / "global_best" / "best_hparams.yaml",
                 d.parent / "global_best" / "best_hparams.yaml"):
        if cand.exists():
            return cand
    return None


def _aux_encoder_kwargs(cov_name: Optional[str], cov_stats: Optional[dict]):
    """`(kwargs, spec)` for the auxiliary encoder of a covariate-trained network.

    Legacy checkpoints take no keyword arguments at all -- the encoder is the hardcoded
    four-input one -- so an empty dict is the correct answer there, not a default.
    """
    from data.loading.covariates import cat_cardinalities, get_spec

    spec = get_spec(cov_name or "legacy")
    if spec.legacy:
        return {}, spec
    return {
        "n_cont": spec.n_continuous,
        "cat_cardinalities": list(cat_cardinalities(spec, cov_stats)),
    }, spec


@dataclass
class LoadedModel:
    """A model ready for evaluation, plus the provenance to put in the artifact."""

    kind: str                       # "deterministic" | "flow"
    module: nn.Module
    ckpt_path: Path
    arch: Dict[str, Any]
    out_dim: int
    extra: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "path": str(self.ckpt_path),
            "arch": self.arch,
            "out_dim": self.out_dim,
            "extra": {k: v for k, v in self.extra.items() if not isinstance(v, dict)},
        }


# ---------------------------------------------------------------------------------
# Deterministic regressors
# ---------------------------------------------------------------------------------
def load_deterministic(
    weights_root: Path | str,
    job: str,
    device: torch.device,
    warm_x_tuple=None,
    arch_override: Optional[dict] = None,
    legacy: bool = False,
    cov_stats: Optional[dict] = None,
    ckpt_path: Path | str | None = None,
    covariates: Optional[str] = None,
) -> LoadedModel:
    """Load a deterministic regressor: `det_xco2`, `det_profiles`, `det_last4`
    or `det_last1`.

    `legacy=True` selects `FullRegressorProfilesLegacy`, which is needed only for the
    published `net_profiles_old.pt` behind Table 2; group-split checkpoints are all
    written by the current class.

    `ckpt_path` names a checkpoint outright and bypasses LAYOUT. It is how a model whose
    weights directory is parameterized -- by a covariate preset, or by having come out of
    the Bayesian optimization -- gets scored: `job` then says only which task it is (and
    so how many outputs it has), not where it lives. `covariates` is a fallback for the
    preset when the checkpoint has no `best_hparams.yaml` beside it; what the checkpoint
    itself records always wins, because getting this wrong changes tensor shapes.
    """
    from modeling.full_regressor import FullRegressor
    from modeling.full_regressor_profiles import FullRegressorProfiles

    if ckpt_path is not None:
        ckpt = Path(ckpt_path)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"[{job}] no checkpoint at the path given explicitly: {ckpt}. "
                "Check the weights subdirectory the training job wrote -- it is named "
                "after the covariate preset, e.g. full_regressor_scene_prior_best/."
            )
        hp_path = _hparams_beside(ckpt)
    else:
        ckpt = require(weights_root, job)
        hp_path = resolve(weights_root, job, "hparams")
    out_dim = int(LAYOUT[job]["out_dim"])
    hp = _read_yaml(hp_path)
    arch = _arch_from_hparams(
        hp, {"embed_dim": 256, "nhead": 8, "nlayers": 1, "aux_dim": 32, "out_dim": out_dim}
    )
    if arch_override:
        arch.update({k: int(v) for k, v in arch_override.items()})
    arch["out_dim"] = out_dim

    # The auxiliary encoder's shape follows the covariate set the checkpoint was trained
    # with, which the trainer records in best_hparams.yaml.
    # An explicitly named checkpoint may have no hparams file beside it, in which case
    # _covariates_from_hparams answers "legacy" for a network that is nothing of the
    # kind. Fall back to the caller's preset there, and only there.
    cov_name = _covariates_from_hparams(hp)
    if not hp and covariates:
        logger.warning(
            "[%s] no best_hparams.yaml beside %s; taking the covariate preset from the "
            "caller (%s). If that is not what the checkpoint was trained with, the load "
            "below fails on shapes rather than quietly mis-scoring.",
            job, ckpt, covariates,
        )
        cov_name = covariates
    elif covariates and cov_name != covariates:
        logger.warning(
            "[%s] the checkpoint says covariates=%s but --covariates said %s. The "
            "checkpoint wins for the network's shape; the data loader will use the "
            "other, and the two have to agree.",
            job, cov_name, covariates,
        )
    aux_kwargs, cov_spec = _aux_encoder_kwargs(cov_name, cov_stats)
    arch["covariates"] = cov_spec.name

    if job == "det_xco2":
        # FullRegressor has a fixed scalar head and takes no out_dim.
        model = FullRegressor(
            aux_dim=arch["aux_dim"], embed_dim=arch["embed_dim"],
            nhead=arch["nhead"], nlayers=arch["nlayers"], **aux_kwargs,
        ).to(device)
    elif legacy:
        from modeling.legacy_full_regressor_profiles import FullRegressorProfilesLegacy
        model = FullRegressorProfilesLegacy(
            aux_dim=arch["aux_dim"], embed_dim=arch["embed_dim"], nhead=arch["nhead"],
            nlayers=arch["nlayers"], out_dim=out_dim,
        ).to(device)
    else:
        model = FullRegressorProfiles(
            aux_dim=arch["aux_dim"], embed_dim=arch["embed_dim"], nhead=arch["nhead"],
            nlayers=arch["nlayers"], out_dim=out_dim, **aux_kwargs,
        ).to(device)

    if warm_x_tuple is not None:
        with torch.no_grad():
            model(warm_x_tuple)         # initialize any lazy modules before loading

    sd = _state_dict(torch.load(ckpt, map_location="cpu"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        # Not fatal for the legacy variant, which is a different class by construction,
        # but always worth printing: a silently half-loaded network still produces a table.
        logger.warning(
            "[%s] state_dict mismatch loading %s\n  missing:    %s\n  unexpected: %s",
            job, ckpt, list(missing)[:8], list(unexpected)[:8],
        )
        if not legacy and len(missing) > 0:
            raise RuntimeError(
                f"[{job}] {len(missing)} parameters were not found in {ckpt}. The "
                f"architecture in best_hparams.yaml ({arch}) does not match the "
                "checkpoint; evaluating this would report a partly random network."
            )
    model.eval()
    return LoadedModel("deterministic", model, ckpt, arch, out_dim,
                       extra={"legacy": legacy, "covariates": cov_spec.name,
                              "hparams_file": str(hp_path)})


# ---------------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------------
def _nfhead_class():
    """`NFHead` as `train_nf.py` defines it.

    Imported rather than re-declared: the scalar flow's shape is a function of that file
    and of nothing else, and a second copy here would be free to drift out of step with
    the checkpoints it is meant to open. `experiments` resolves as a namespace package
    when the project root is on PYTHONPATH, which `job_common.sh` guarantees.
    """
    try:
        from experiments.train.train_nf import NFHead
    except ImportError as exc:      # pragma: no cover - environment problem, not logic
        raise ImportError(
            "could not import experiments.train.train_nf, which defines the scalar "
            "flow's architecture. Add the project root to PYTHONPATH "
            "(job_common.sh exports it as $PROJECT)."
        ) from exc
    return NFHead


class ScalarFlow(nn.Module):
    """`train_nf.py`'s pairing of a backbone encoder with a 1-d `NFHead`.

    The context definition below is `train_nf.get_context` verbatim; if that changes,
    this must change with it or every scalar-flow number silently shifts.
    """

    def __init__(self, backbone: nn.Module, nf: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.nf = nf
        self.out_dim = 1

    def context(self, x_tuple) -> torch.Tensor:
        z = self.backbone.encoder(x_tuple)
        return z.flatten(1) if z.dim() > 2 else z

    def log_prob(self, x_tuple, y: torch.Tensor) -> torch.Tensor:
        return self.nf.log_prob(y.reshape(-1), self.context(x_tuple))

    @torch.no_grad()
    def sample(self, x_tuple, n: int = 100) -> torch.Tensor:
        ctx = self.context(x_tuple)
        s = self.nf.sample(n=n, context=ctx)                      # [n,B]
        # Asserted rather than assumed. nflows draws as [B,n,1] and NFHead transposes it;
        # if that ever drifts back, the arrays still concatenate whenever the batch is
        # full and the table comes out silently transposed -- samples of sounding i read
        # as sounding j. A wrong number that runs is worse than a crash, so: crash.
        if s.shape[:2] != (n, ctx.shape[0]):
            raise RuntimeError(
                f"NFHead.sample returned {tuple(s.shape)}; expected "
                f"({n}, {ctx.shape[0]}) = (n_samples, batch). The sample and batch axes "
                "are transposed -- see NFHead.sample in experiments/train/train_nf.py."
            )
        return s.unsqueeze(-1)                                    # [n,B,1]

    @torch.no_grad()
    def latent(self, x_tuple, y: torch.Tensor) -> torch.Tensor:
        ctx = self.context(x_tuple)
        z, _ = self.nf.flow._transform(y.reshape(-1, 1), context=ctx)
        return z


def load_flow(
    weights_root: Path | str,
    job: str,
    device: torch.device,
    warm_x_tuple,
    arch_override: Optional[dict] = None,
    ckpt_path: Path | str | None = None,
    cov_stats: Optional[dict] = None,
    covariates: Optional[str] = None,
) -> LoadedModel:
    """Load a flow: `nf_xco2`, `nf_profiles`, `nf_last4` or `nf_last1`.

    `ckpt_path`, `cov_stats` and `covariates` mean what they do in `load_deterministic`.
    The flow's context backbone is a `FullRegressorProfiles` like any other and needs the
    same auxiliary-encoder shape -- and unlike the deterministic path, the scalar flow
    loads its backbone with `strict=False`, so a covariate flow built without them would
    have loaded to a partly random encoder and still produced a table.
    """
    if ckpt_path is None:
        ckpt_path = require(weights_root, job)
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"[{job}] no flow checkpoint at {ckpt_path}.")
    payload = torch.load(ckpt_path, map_location="cpu")
    out_dim = int(LAYOUT[job]["out_dim"])

    # Which of the two trainers wrote this, read off the payload rather than off the job
    # name. train_nf.py saves {"state_dict": {"backbone": ..., "nf": ...}, "extra": ...};
    # the profile trainers save {"nf": ..., "feat": ..., "cfg": ...} at the top level. Two
    # jobs now use the first shape (nf_xco2 and nf_last1, which is the same script in its
    # `surface` target mode), and keying off the name meant adding a task was editing an
    # `if` in a loader.
    inner = payload.get("state_dict") if isinstance(payload, dict) else None
    is_scalar_payload = isinstance(inner, dict) and {"backbone", "nf"} <= set(inner)
    if is_scalar_payload:
        return _load_scalar_flow(payload, ckpt_path, job, weights_root, device,
                                 warm_x_tuple, arch_override, cov_stats, covariates)
    return _load_profile_flow(payload, ckpt_path, job, out_dim, device, warm_x_tuple,
                              arch_override, cov_stats, covariates)


def _load_scalar_flow(payload, ckpt_path, job, weights_root, device, warm_x_tuple,
                      arch_override, cov_stats=None, covariates=None):
    from modeling.full_regressor_profiles import FullRegressorProfiles

    NFHead = _nfhead_class()
    state = payload.get("state_dict", {})
    if "backbone" not in state or "nf" not in state:
        raise ValueError(
            f"{ckpt_path} does not look like a train_nf.py checkpoint "
            f"(top-level keys: {list(payload)})"
        )

    extra = payload.get("extra", {}) or {}
    side = _read_yaml(resolve(weights_root, job, "hparams")
                      or (ckpt_path.parent / "best_flow_nflows_config.yaml"))
    backbone_cfg = extra.get("backbone_cfg") or side.get("backbone_cfg")
    nflow_cfg = extra.get("nflow_cfg") or side.get("nflow_cfg")
    if arch_override:
        backbone_cfg = {**(backbone_cfg or {}), **arch_override.get("backbone", {})}
        nflow_cfg = {**(nflow_cfg or {}), **arch_override.get("nflow", {})}
    if not backbone_cfg or not nflow_cfg:
        raise ValueError(
            f"{ckpt_path} records no architecture (`extra.backbone_cfg` / "
            "`extra.nflow_cfg`), and no best_flow_nflows_config.yaml sits next to it. "
            "This is the case for the published scalar-flow checkpoint. Pass the "
            "architecture explicitly with --nf-arch <yaml>; do not guess, because a "
            "wrong flow shape still loads partially and still produces a number."
        )

    # The covariate set is a property of the checkpoint, not of the caller: it sets the
    # auxiliary encoder's parameter shapes. train_nf.py records it in `extra`; the side
    # config carries it for runs that predate that.
    aux_kwargs, cov_spec = _aux_encoder_kwargs(
        extra.get("covariates") or side.get("covariates") or covariates, cov_stats
    )
    backbone = FullRegressorProfiles(
        aux_dim=int(backbone_cfg["aux_dim"]), embed_dim=int(backbone_cfg["embed_dim"]),
        nhead=int(backbone_cfg["nhead"]), nlayers=int(backbone_cfg["nlayers"]), out_dim=1,
        **aux_kwargs,
    ).to(device)
    with torch.no_grad():
        ctx_dim = backbone.encoder(warm_x_tuple)
        ctx_dim = int((ctx_dim.flatten(1) if ctx_dim.dim() > 2 else ctx_dim).shape[-1])

    nf = NFHead(
        context_dim=ctx_dim, num_layers=int(nflow_cfg["layers"]),
        hidden_features=int(nflow_cfg["hidden"]), num_bins=int(nflow_cfg["bins"]),
        tail_bound=float(nflow_cfg["tail_bound"]),
    ).to(device)

    # strict=False is deliberate and long-standing: the saved backbone carries a head
    # this evaluation does not use. It is also silent, so the one mismatch that must not
    # pass -- an auxiliary encoder that did not load -- is checked for by name.
    missing, unexpected = backbone.load_state_dict(state["backbone"], strict=False)
    if missing or unexpected:
        logger.warning(
            "[%s] backbone state_dict: %d missing, %d unexpected; "
            "missing: %s; unexpected: %s",
            job, len(missing), len(unexpected), list(missing)[:8], list(unexpected)[:8],
        )
    aux_missing = [k for k in missing if "aux" in k]
    if aux_missing:
        raise RuntimeError(
            f"[{job}] the auxiliary encoder did not load from {ckpt_path}: "
            f"{aux_missing[:8]}. The flow was trained with covariates={cov_spec.name} "
            "and the encoder built here has different shapes. Scoring this would report "
            "a network whose covariate branch is random."
        )
    nf.load_state_dict(state["nf"], strict=True)
    model = ScalarFlow(backbone, nf).to(device).eval()
    return LoadedModel(
        "flow", model, ckpt_path,
        {"backbone": backbone_cfg, "nflow": nflow_cfg, "context_dim": ctx_dim,
         "covariates": cov_spec.name}, 1,
        extra={k: v for k, v in extra.items() if k in ("val_nll", "epochs", "lr", "wd")},
    )


def _load_profile_flow(payload, ckpt_path, job, out_dim, device, warm_x_tuple,
                       arch_override, cov_stats=None, covariates=None):
    from modeling.full_regressor_profiles import FullRegressorProfiles
    from modeling.flow_regressor_profiles_nf import FeatureNetProfiles, NFProfiles

    cfg = payload.get("cfg", {}) or {}
    if arch_override:
        cfg = {**cfg, **arch_override}

    # nf_profiles stores a flat cfg ({layers, hidden, bins, tail_bound, out_dim});
    # nf_profiles_4d stores the whole run config ({model: {...}, nflow: {...}}).
    mcfg = cfg.get("model", {})
    ncfg = cfg.get("nflow", {})
    arch = {
        "aux_dim": int(mcfg.get("aux_dim", 32)),
        "embed_dim": int(mcfg.get("embed_dim", 256)),
        "nhead": int(mcfg.get("nhead", 8)),
        "nlayers": int(mcfg.get("nlayers", 1)),
        "out_dim": int(cfg.get("out_dim", mcfg.get("out_dim", out_dim))),
    }
    flow_arch = {
        "layers": int(ncfg.get("num_layers", cfg.get("layers", 0)) or 0),
        "hidden": int(ncfg.get("hidden_features", cfg.get("hidden", 0)) or 0),
        "bins": int(ncfg.get("num_bins", cfg.get("bins", 0)) or 0),
        "tail_bound": float(ncfg.get("tail_bound", cfg.get("tail_bound", 0.0)) or 0.0),
        "standardize_y": bool(
            cfg.get("standardize_y", cfg.get("task", {}).get("standardize_y", False))
        ),
        # Spline clamps. They change no tensor shape, so the wrong value loads cleanly and
        # silently returns a different log-density; nf_profiles_grouped.yaml sets
        # min_derivative to 0.01, ten times the class default. Checked in both dialects,
        # and absence is worth a warning rather than a shrug.
        "min_bin_width": float(ncfg.get("min_bin_width", cfg.get("min_bin_width", 5e-3))),
        "min_bin_height": float(ncfg.get("min_bin_height", cfg.get("min_bin_height", 5e-3))),
        "min_derivative": float(ncfg.get("min_derivative", cfg.get("min_derivative", 1e-3))),
    }
    if not any(k in ncfg or k in cfg for k in
               ("min_bin_width", "min_bin_height", "min_derivative")):
        logger.warning(
            "%s records no spline clamps; falling back to the NFProfiles defaults "
            "(%.3g / %.3g / %.3g). If the run's config set different ones the "
            "log-densities will be wrong -- pass them with --nf-arch.",
            ckpt_path, flow_arch["min_bin_width"], flow_arch["min_bin_height"],
            flow_arch["min_derivative"],
        )
    if not all(flow_arch[k] for k in ("layers", "hidden", "bins", "tail_bound")):
        raise ValueError(
            f"{ckpt_path} does not record a complete flow architecture: {flow_arch}. "
            "Pass --nf-arch <yaml> with layers/hidden/bins/tail_bound."
        )
    if arch["out_dim"] != out_dim:
        raise ValueError(
            f"[{job}] checkpoint says out_dim={arch['out_dim']} but this task is "
            f"{out_dim}-dimensional. Wrong checkpoint for the job."
        )

    aux_kwargs, cov_spec = _aux_encoder_kwargs(cfg.get("covariates") or covariates,
                                               cov_stats)
    backbone = FullRegressorProfiles(
        aux_dim=arch["aux_dim"], embed_dim=arch["embed_dim"], nhead=arch["nhead"],
        nlayers=arch["nlayers"], out_dim=out_dim, **aux_kwargs,
    ).to(device)
    feat = FeatureNetProfiles(backbone).to(device)
    ctx_dim = feat.infer_context_dim(warm_x_tuple)

    nf = NFProfiles(
        feature_net=feat, context_dim=ctx_dim, out_dim=out_dim,
        num_layers=flow_arch["layers"], hidden_features=flow_arch["hidden"],
        num_bins=flow_arch["bins"], tail_bound=flow_arch["tail_bound"],
        standardize_y=flow_arch["standardize_y"],
        min_bin_width=flow_arch["min_bin_width"],
        min_bin_height=flow_arch["min_bin_height"],
        min_derivative=flow_arch["min_derivative"],
    ).to(device)

    nf.load_state_dict(payload["nf"], strict=True)
    if "feat" in payload:
        feat.load_state_dict(payload["feat"], strict=True)
    if "backbone" in payload:
        backbone.load_state_dict(payload["backbone"], strict=False)
    nf.eval()

    return LoadedModel(
        "flow", nf, ckpt_path,
        {**arch, "nflow": flow_arch, "context_dim": ctx_dim,
         "covariates": cov_spec.name}, out_dim,
        extra=payload.get("extra", {}) or {},
    )

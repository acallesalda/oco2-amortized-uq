#!/usr/bin/env python3
"""bo_propose.py
The Bayesian-optimization driver: read the trials that have finished, propose the next
batch.

    python3 chtc/bo_propose.py --study det_profiles --round 0 --batch 5 \
        --results-dir bo_results --out-dir bo_params

Why a driver script and not `optuna.optimize`
---------------------------------------------
Optuna's usual loop owns the process that trains the model. On CHTC the training runs in
a GPU job on a machine with no shared filesystem and no database, so the loop has to be
turned inside out: this script is the *ask*, each GPU job is one objective evaluation, and
the *tell* happens on the next invocation by replaying the finished trials into a fresh
in-memory study. Given the same history and seed it proposes the same batch, so the
search is reproducible from the record rather than from a live process.

Five trials are proposed at a time so a round is one wave of GPU jobs rather than 25
sequential ones. `constant_liar=True` is what makes that safe: without it TPE, asked five
times with nothing told in between, returns five nearly identical points and the round is
wasted.

Round 0 always contains the current configuration, enqueued explicitly. Every search then
contains its own baseline, and "BO helped" is a comparison inside one study rather than
against a number from a different code version.

Failed and missing trials are recorded as pruned, not as a bad value: a job that died in
setup says nothing about its hyperparameters, and telling TPE that it scored infinity
would push the search away from a perfectly good region.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------
# Search spaces
#
# Every entry is (kind, *args) and maps to exactly one Hydra override, so the space and
# the command line cannot drift apart. `override` names the Hydra key.
#
# Deliberately absent:
#   covariates      the scientific comparison, not a nuisance parameter. Tuning over it
#                   would make the ablation incoherent -- fix the covariate set, then
#                   search within it.
#   epochs          trials run at BO_EPOCHS (20); the winner is retrained at full length.
#   select_metric   always the metric the paper reports, or the search optimizes one
#                   quantity and the table shows another.
# ---------------------------------------------------------------------------------
SPACES: Dict[str, Dict[str, tuple]] = {
    "det_xco2": {
        "hyps.lr":              ("logfloat", 3e-5, 5e-3),
        "hyps.wd":              ("logfloat", 1e-6, 5e-3),
        "hyps.embed_dim":       ("categorical", [128, 192, 256, 320, 384, 512]),
        "hyps.nhead":           ("categorical", [4, 8]),
        "hyps.nlayers":         ("int", 1, 4),
        "data.train_batch_size": ("categorical", [128, 256, 512]),
    },
    "det_profiles": {
        "hyps.lr":              ("logfloat", 3e-5, 5e-3),
        "hyps.wd":              ("logfloat", 1e-6, 5e-3),
        "hyps.embed_dim":       ("categorical", [128, 192, 256, 320, 384, 512]),
        "hyps.nhead":           ("categorical", [4, 8]),
        "hyps.nlayers":         ("int", 1, 4),
        "data.train_batch_size": ("categorical", [128, 256, 512]),
        # Only meaningful for D > 1: see code/modeling/losses.py.
        "loss_weight":          ("categorical", ["std", "ppm"]),
    },
}
SPACES["det_last4"] = dict(SPACES["det_profiles"])

# ---------------------------------------------------------------------------------
# The flows.
#
# Same design as the deterministic studies -- fixed covariate set, epochs held out of the
# space, one metric -- with two differences that are the point of running them.
#
#   * The objective is validation NLL, not RMSE. It is what the flow is trained on and
#     what the paper's uncertainty claims rest on; a flow tuned for RMSE is a flow tuned
#     to ignore its own spread.
#   * The backbone is NOT warm-started, and IS searched. Every published flow started from
#     a deterministic checkpoint, which had two consequences: the density was never asked
#     whether it was any good on its own terms, and the backbone's width and depth were
#     not free parameters -- they had to match whatever checkpoint was being loaded, or
#     the state dict would not fit. Training from scratch removes that constraint, so
#     embed_dim / nhead / nlayers join the space here and the six studies become six
#     searches of the same kind rather than three models and three heads.
#
# The keys are dotted paths into the YAML the trainer reads, not Hydra overrides, because
# the flow trainers take --exp_cfg and --set. Each of the three uses a different name for
# the same quantity -- `layers` / `num_layers`, `hidden` / `hidden_features`, `wd` /
# `weight_decay`, and the scalar flow keeps its backbone one level deeper under
# `model_cfg_obj` -- which is why these are spelled out per study rather than shared.
#
# `hidden` is an int range and not a set of round numbers on purpose: the published
# nf_last4 width is 597, which no sensible categorical list would have contained, and a
# baseline outside its own distribution cannot be enqueued as trial 0.
#
# Ten dimensions on 25 trials is a coarse search and is meant to be reported as one. What
# makes it worth running anyway is that trial 0 is the published configuration, so the
# study cannot do worse than report "the published architecture was the best of the 25".
# ---------------------------------------------------------------------------------
SPACES["nf_xco2"] = {
    "model_cfg_obj.full_backbone.embed_dim": ("categorical", [128, 192, 256, 320, 384, 512]),
    "model_cfg_obj.full_backbone.nhead":     ("categorical", [4, 8]),
    "model_cfg_obj.full_backbone.nlayers":   ("int", 1, 4),
    "model_cfg_obj.nflow.layers":      ("int", 3, 12),
    "model_cfg_obj.nflow.hidden":      ("int", 128, 768),
    "model_cfg_obj.nflow.bins":        ("int", 4, 12),
    "model_cfg_obj.nflow.tail_bound":  ("float", 3.0, 12.0),
    "optim.lr":                        ("logfloat", 1e-4, 5e-3),
    "optim.wd":                        ("logfloat", 1e-6, 5e-3),
    "data.train_batch_size":           ("categorical", [128, 256, 512]),
}
SPACES["nf_profiles"] = {
    "model.embed_dim":       ("categorical", [128, 192, 256, 320, 384, 512]),
    "model.nhead":           ("categorical", [4, 8]),
    "model.nlayers":         ("int", 1, 4),
    "nflow.layers":          ("int", 3, 12),
    "nflow.hidden":          ("int", 128, 768),
    "nflow.bins":            ("int", 4, 12),
    "nflow.tail_bound":      ("float", 3.0, 12.0),
    "optim.lr":              ("logfloat", 1e-4, 5e-3),
    "optim.wd":              ("logfloat", 1e-6, 5e-3),
    "data.train_batch_size": ("categorical", [128, 256, 512]),
}
SPACES["nf_last4"] = {
    "model.embed_dim":       ("categorical", [128, 192, 256, 320, 384, 512]),
    "model.nhead":           ("categorical", [4, 8]),
    "model.nlayers":         ("int", 1, 4),
    "nflow.num_layers":      ("int", 3, 12),
    "nflow.hidden_features": ("int", 128, 768),
    "nflow.num_bins":        ("int", 4, 12),
    "nflow.tail_bound":      ("float", 3.0, 12.0),
    "optim.lr":              ("logfloat", 1e-4, 5e-3),
    "optim.weight_decay":    ("logfloat", 1e-6, 5e-3),
    "data.train_batch_size": ("categorical", [128, 256, 512]),
}

# Studies whose overrides are `--set` dotted paths rather than Hydra `key=value`, and
# whose objective is minimized validation NLL rather than validation RMSE. bo_trial.sh
# reads the same list; keep them in step.
FLOW_STUDIES = ("nf_xco2", "nf_profiles", "nf_last4")

# The configuration each study starts from -- the *_cov_grouped.yaml values, which are the
# published hyperparameters. Trial r0/t0 is exactly this.
BASELINES: Dict[str, dict] = {
    "det_xco2": {
        "hyps.lr": 6.471796937675714e-05, "hyps.wd": 4.772666391106017e-06,
        "hyps.embed_dim": 512, "hyps.nhead": 4, "hyps.nlayers": 1,
        "data.train_batch_size": 512,
    },
    "det_profiles": {
        "hyps.lr": 8.0e-4, "hyps.wd": 9.1e-4,
        "hyps.embed_dim": 256, "hyps.nhead": 8, "hyps.nlayers": 1,
        "data.train_batch_size": 512, "loss_weight": "ppm",
    },
    "det_last4": {
        "hyps.lr": 1.95e-4, "hyps.wd": 4.76e-4,
        "hyps.embed_dim": 320, "hyps.nhead": 4, "hyps.nlayers": 2,
        "data.train_batch_size": 512, "loss_weight": "ppm",
    },
    # The published flow architectures, from the *_cov_grouped.yaml files -- which took
    # them from the released checkpoints' own stored configs. Trial r0/t0 is exactly this,
    # so every flow study contains the published flow as its own baseline, and the
    # difference between it and the winner is the search.
    #
    # It is not, however, the published MODEL: these run from scratch, so r0/t0 is the
    # published architecture without its warm start. That is the honest baseline for a
    # from-scratch search, and it means a flow study answers two questions at once -- what
    # the search bought (winner vs r0/t0, inside this study) and what the warm start was
    # worth (r0/t0 vs the published nf_*_cov run, across studies).
    #
    # The backbone dimensions are the ones the published flow's backbone had, which for
    # nf_profiles and nf_last4 is the deterministic model each warm-started from.
    "nf_xco2": {
        "model_cfg_obj.full_backbone.embed_dim": 256,
        "model_cfg_obj.full_backbone.nhead": 8,
        "model_cfg_obj.full_backbone.nlayers": 1,
        "model_cfg_obj.nflow.layers": 5,
        "model_cfg_obj.nflow.hidden": 384,
        "model_cfg_obj.nflow.bins": 9,
        "model_cfg_obj.nflow.tail_bound": 6.3108919641,
        "optim.lr": 4.533555957493404e-04,
        "optim.wd": 5.387680934037755e-04,
        "data.train_batch_size": 128,
    },
    "nf_profiles": {
        "model.embed_dim": 256,
        "model.nhead": 8,
        "model.nlayers": 1,
        "nflow.layers": 9,
        "nflow.hidden": 384,
        "nflow.bins": 10,
        "nflow.tail_bound": 8.391013429624302,
        "optim.lr": 5.0e-4,
        "optim.wd": 1.0e-5,
        "data.train_batch_size": 128,
    },
    "nf_last4": {
        # 320/4/2, matching det_last4 -- see the correction note in
        # nf_profiles4_cov_grouped.yaml about the 256/8/3 the checkpoint claimed.
        "model.embed_dim": 320,
        "model.nhead": 4,
        "model.nlayers": 2,
        "nflow.num_layers": 11,
        "nflow.hidden_features": 597,
        "nflow.num_bins": 8,
        "nflow.tail_bound": 5.05952066599221,
        "optim.lr": 2.0934846711727584e-4,
        "optim.weight_decay": 1.759837373984636e-6,
        "data.train_batch_size": 128,
    },
}

# nhead must divide embed_dim. Every width above is divisible by 4 and by 8, so the
# constraint never binds -- but it is checked rather than assumed, because a width added
# later that violates it would fail inside nn.TransformerEncoderLayer after the data has
# been unpacked.
def _check_widths() -> None:
    for study, space in SPACES.items():
        # Every study searches the backbone now, but under three different key names --
        # hyps.embed_dim, model.embed_dim, model_cfg_obj.full_backbone.embed_dim -- so the
        # pair is found by suffix rather than by a hardcoded name. A space that stopped
        # searching the width would then silently skip the check, which is why the
        # `if not width` below is a hard error and not a `continue`.
        width = next((k for k in space if k.endswith("embed_dim")), None)
        head = next((k for k in space if k.endswith("nhead")), None)
        if not width or not head:
            raise ValueError(
                f"{study}: the space has {'no embed_dim' if not width else 'no nhead'}. "
                "Both are searched in every study; if that changed deliberately, this "
                "check has to change with it.")
        bad = [(w, h) for w in space[width][1] for h in space[head][1] if w % h]
        if bad:
            raise ValueError(f"{study}: embed_dim not divisible by nhead for {bad}")


def _check_baselines() -> None:
    """Trial 0 must be the baseline, whole.

    `enqueue_trial` accepts a partial dict and samples whatever is missing, so a baseline
    that has fallen behind its space does not raise -- it produces a round-0 trial 0 that
    is the published configuration in the parameters it happens to name and a random draw
    in the rest. Every "the search improved on the published hyperparameters" claim in the
    paper rests on that trial being what it says it is, and nothing downstream can tell
    the difference. Hence a check, at import, in the process that enqueues it.
    """
    for study, space in SPACES.items():
        base = BASELINES.get(study)
        if base is None:
            raise ValueError(f"{study}: no baseline configuration")
        missing = sorted(set(space) - set(base))
        extra = sorted(set(base) - set(space))
        if missing or extra:
            raise ValueError(
                f"{study}: the baseline and the search space disagree. "
                f"missing from the baseline: {missing or 'none'}; "
                f"in the baseline but not searched: {extra or 'none'}.")
        for name, value in base.items():
            spec = space[name]
            if spec[0] == "categorical":
                if value not in spec[1]:
                    raise ValueError(
                        f"{study}: baseline {name}={value!r} is not one of {spec[1]}. "
                        "A categorical value outside its own distribution cannot be "
                        "enqueued as trial 0.")
            elif not (spec[1] <= value <= spec[2]):
                raise ValueError(
                    f"{study}: baseline {name}={value!r} is outside the searched range "
                    f"[{spec[1]}, {spec[2]}].")


def _distribution(spec):
    """Build an Optuna distribution, across the 2.x / 3.x API break.

    Optuna renamed the distribution classes in 3.0: `UniformDistribution`,
    `LogUniformDistribution` and `IntUniformDistribution` became `FloatDistribution(...,
    log=)` and `IntDistribution`. The old names still exist in 3.x as deprecated aliases,
    the new ones do not exist in 2.x -- so the new API is tried first and the old one is
    the fallback. requirements.txt does not pin optuna, and the container was built before
    this script existed, which is how a search that ran locally on 4.9 died on the access
    point's image with "module 'optuna.distributions' has no attribute
    'FloatDistribution'".
    """
    import optuna.distributions as od
    kind = spec[0]
    if kind in ("logfloat", "float"):
        log = kind == "logfloat"
        if hasattr(od, "FloatDistribution"):
            return od.FloatDistribution(spec[1], spec[2], log=log)
        legacy = od.LogUniformDistribution if log else od.UniformDistribution
        return legacy(spec[1], spec[2])
    if kind == "int":
        if hasattr(od, "IntDistribution"):
            return od.IntDistribution(spec[1], spec[2])
        return od.IntUniformDistribution(spec[1], spec[2])
    if kind == "categorical":
        return od.CategoricalDistribution(spec[1])
    raise ValueError(f"unknown parameter kind '{kind}'")


def _suggest(trial, name: str, spec):
    kind = spec[0]
    if kind in ("logfloat", "float"):
        log = kind == "logfloat"
        if hasattr(trial, "suggest_float"):
            return trial.suggest_float(name, spec[1], spec[2], log=log)
        # Optuna < 2.0
        legacy = trial.suggest_loguniform if log else trial.suggest_uniform
        return legacy(name, spec[1], spec[2])
    if kind == "int":
        return trial.suggest_int(name, spec[1], spec[2])
    if kind == "categorical":
        return trial.suggest_categorical(name, spec[1])
    raise ValueError(f"unknown parameter kind '{kind}'")


def read_history(results_dir: Path, study: str) -> List[dict]:
    """Every finished trial record for this study, oldest round first.

    A record is written by bo_trial.sh before training starts and rewritten when it ends,
    so a record with `value: null` is a trial that started and did not finish.
    """
    records = []
    for path in sorted(results_dir.glob(f"bo_{study}_r*_t*.json")):
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("unreadable trial record %s -- treating as failed", path)
            continue
        rec.setdefault("source", str(path))
        records.append(rec)
    records.sort(key=lambda r: (int(r.get("round", 0)), int(r.get("trial", 0))))
    return records


def round_seed(seed: int, rnd: int) -> int:
    """Sampler seed for one round.

    It must depend on the round. TPE's quasi-random startup phase draws from a seeded
    RNG that is reset every time the study is rebuilt, so a fixed seed makes round 1
    re-propose round 0's points verbatim -- four wasted GPU jobs out of twenty-five, and
    the symptom (a leaderboard with duplicate rows) looks like bad luck rather than a bug.
    Proposals stay reproducible: they are a function of (history, round, seed).
    """
    return int(seed) + 1000 * int(rnd)


def build_study(study: str, history: List[dict], seed: int, n_startup: int):
    import optuna
    from optuna.trial import TrialState, create_trial

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # `multivariate` and `constant_liar` are flagged experimental by Optuna and print a
    # warning on every construction; both are load-bearing here (see the module docstring)
    # and the warning is noise in a batch log.
    import warnings
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    space = SPACES[study]
    dists = {k: _distribution(v) for k, v in space.items()}

    # `constant_liar` needs optuna >= 2.8 and `multivariate` >= 2.2. Both are load-bearing
    # -- without constant_liar a batch of five comes back as five copies of the same point
    # and the round is wasted -- so a version that cannot do it is worth failing on
    # loudly, here, rather than producing a search that silently evaluates one point five
    # times.
    logger.info("optuna %s", optuna.__version__)
    try:
        sampler = optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=n_startup, constant_liar=True, multivariate=True,
        )
    except TypeError as exc:
        raise SystemExit(
            f"optuna {optuna.__version__} does not support constant_liar/multivariate "
            f"({exc}). A batched ask() without constant_liar returns near-identical "
            f"points, so this would burn a round of GPU jobs on one configuration. "
            f"Pin optuna>=2.8 in chtc/requirements.txt and rebuild the container, or run "
            f"the search with --batch 1.")
    st = optuna.create_study(direction="minimize", sampler=sampler,
                             study_name=f"oco2_{study}")

    n_ok = n_failed = 0
    for rec in history:
        params = {k: v for k, v in (rec.get("params") or {}).items() if k in dists}
        if set(params) != set(dists):
            logger.warning("trial %s/%s has parameters that do not match the space "
                           "(%s); skipping it", rec.get("round"), rec.get("trial"),
                           sorted(set(dists) ^ set(params)))
            continue
        value = rec.get("value")
        if value is None or not isinstance(value, (int, float)):
            st.add_trial(create_trial(params=params, distributions=dists,
                                      state=TrialState.PRUNED))
            n_failed += 1
        else:
            st.add_trial(create_trial(params=params, distributions=dists,
                                      value=float(value)))
            n_ok += 1
    logger.info("replayed %d completed and %d failed trials", n_ok, n_failed)
    return st, dists


def load_extra_baseline(path, study: str) -> dict:
    """Another run's winner, as a second enqueued trial. `{}` if it cannot be used.

    The rerun on the replicate split enqueues the group split's winner here. That costs
    one trial in twenty-five and answers a question the search cannot otherwise answer:
    the winner-vs-winner comparison across splits confounds "the split" with "the search
    picked something else", while this trial is the *same configuration* on both
    partitions, scored by both validations. One graph, two comparisons.

    Fail-soft on purpose. A missing or malformed file costs the extra trial and nothing
    else -- the search still contains its own baseline and still runs. A search that
    refuses to start because an optional file did not transfer would be a worse trade.
    """
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        logger.warning("extra baseline %s does not exist; continuing without it", path)
        return {}
    try:
        params = json.loads(path.read_text())["params"]
    except Exception as exc:                                  # noqa: BLE001
        logger.warning("could not read an extra baseline from %s (%s); continuing "
                       "without it", path, exc)
        return {}

    # Validated against this study's space, not trusted. The two runs share SPACES, so a
    # mismatch means the file is from a different study or a different version of the
    # search -- either way it would enqueue a trial that is not the model it claims.
    space = SPACES[study]
    if set(params) != set(space):
        logger.warning("extra baseline %s does not match %s's search space "
                       "(missing %s, extra %s); continuing without it", path, study,
                       sorted(set(space) - set(params)), sorted(set(params) - set(space)))
        return {}
    logger.info("extra baseline for %s from %s: %s", study, path.name, params)
    return params


def propose(st, study: str, rnd: int, batch: int, history: List[dict],
            extra_baseline: dict | None = None) -> List[dict]:
    space = SPACES[study]
    out = []
    if rnd == 0 and not history:
        # The baseline goes first, and is not sampled.
        st.enqueue_trial(BASELINES[study])
        if extra_baseline:
            st.enqueue_trial(extra_baseline)
    for _ in range(batch):
        trial = st.ask()
        params = {name: _suggest(trial, name, spec) for name, spec in space.items()}
        out.append(params)
    return out


def to_overrides(params: dict) -> List[str]:
    """Hydra command-line overrides, in a stable order."""
    def fmt(v):
        if isinstance(v, bool):
            return str(v).lower()
        if isinstance(v, float):
            return f"{v:.10g}"
        return str(v)
    return [f"{k}={fmt(v)}" for k, v in sorted(params.items())]


def leaderboard(history: List[dict]) -> str:
    done = [r for r in history if isinstance(r.get("value"), (int, float))]
    done.sort(key=lambda r: r["value"])
    lines = [f"{'rank':>4s} {'round':>5s} {'trial':>5s} {'val':>12s}  params"]
    for i, r in enumerate(done, 1):
        lines.append(f"{i:4d} {r.get('round'):>5} {r.get('trial'):>5} "
                     f"{r['value']:12.6f}  {' '.join(to_overrides(r.get('params') or {}))}")
    n_failed = len(history) - len(done)
    lines.append(f"({len(done)} completed, {n_failed} failed or unfinished)")
    return "\n".join(lines)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True, choices=sorted(SPACES))
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--results-dir", type=Path, default=Path("bo_results"),
                    help="directory of bo_<study>_r*_t*.json records from earlier rounds")
    ap.add_argument("--out-dir", type=Path, default=Path("bo_params"),
                    help="where the proposed parameter files are written")
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--n-startup", type=int, default=8,
                    help="trials sampled quasi-randomly before the TPE model is used")
    ap.add_argument("--extra-baseline", type=Path, default=None,
                    help="a bo_<study>_best.json from another run, enqueued as a second "
                         "trial in round 0. The rerun uses it to put the group split's "
                         "winner into the replicate split's search, so the pair contains "
                         "one identical configuration trained on both partitions. "
                         "Missing or mismatched: warned about and skipped.")
    ap.add_argument("--summary-only", action="store_true",
                    help="print the leaderboard and exit without proposing")
    args = ap.parse_args(argv)

    _check_widths()
    _check_baselines()
    history = read_history(args.results_dir, args.study)
    logger.info("study=%s round=%d: %d trial records on disk",
                args.study, args.round, len(history))

    if args.summary_only:
        print(leaderboard(history))
        return 0

    already = [r for r in history if int(r.get("round", -1)) == args.round]
    if already:
        logger.warning("round %d already has %d records; proposing anyway would "
                       "duplicate them", args.round, len(already))

    st, _ = build_study(args.study, history, round_seed(args.seed, args.round),
                        args.n_startup)
    extra = load_extra_baseline(args.extra_baseline, args.study)
    batch = propose(st, args.study, args.round, args.batch, history, extra)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for j, params in enumerate(batch):
        rec = {
            "study": args.study, "round": args.round, "trial": j,
            "params": params, "overrides": to_overrides(params),
            "seed": args.seed, "round_seed": round_seed(args.seed, args.round),
            "n_prior_trials": len(history),
        }
        path = args.out_dir / f"bo_{args.study}_r{args.round}_t{j}.params.json"
        path.write_text(json.dumps(rec, indent=2))
        print(f"{path.name}: {' '.join(rec['overrides'])}")

    if history:
        print()
        print(leaderboard(history))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

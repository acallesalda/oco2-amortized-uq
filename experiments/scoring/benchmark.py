#!/usr/bin/env python3
"""benchmark.py -- test-time cost of the amortized retrievals, on one GPU, with provenance.

A timing table is worth nothing without the runtime hardware, the batch size, the
numerical precision, the warm-up, the repetitions, and a statement of whether
preprocessing and posterior sampling are inside the timed region. This script measures the
first six and records the machine it measured them on. It also does not imply a
same-hardware comparison against the operational retrieval: no ACOS timing was produced
here.

**No ACOS row is ever written here, by design.** The operational figure quoted in the
manuscript comes from the group that produced the simulated ensemble, on their own
hardware, and is a different model of computation -- an iterative full-physics inversion
against a single amortized forward pass. It belongs in the caption as an order of
magnitude, not in this table as a row.

What is timed, per task and per method:

    point estimate        one forward pass of the deterministic network
    Laplace predictive    `posterior_mean_cov`, i.e. S draws of the last-layer weights,
                          the head re-run for each, and the sample moments
    normalizing flow      `flow.sample`, i.e. N draws from the conditional density

and each of those under two scopes, because "is preprocessing included" has no single
right answer and a reader is entitled to both:

    model        the inputs are already resident on the GPU. This is the cost of the
                 predictive itself and is the number that compares methods.
    end-to-end   the DataLoader is inside the timed region, so per-sounding
                 normalization, collation and the host-to-device copy are counted. This
                 is the number that describes a deployment, and on small batches it is
                 dominated by the loader rather than by the network.

Neither scope includes reading the L1B files or building the covariate table, and neither
includes fitting the Laplace approximation, which happens once, offline, before any
sounding is retrieved. Both facts are recorded in the provenance file rather than left to
be inferred.

    python3 experiments/scoring/benchmark.py --task xco2 \
        --split-csv  scratch/simulated/files/splits/group_split_v1_cov.csv \
        --stats-dir  experiments/pipelines/stats/group_split_v1 \
        --weights-root experiments/weights/group_split_v1 \
        --det-ckpt <...> --flow-ckpt <...> --covariates scene_prior \
        --out experiments/scoring/tables

Appends to `benchmark.csv` so that one job can loop over the three tasks and produce one
table from one device; writes `benchmark_provenance.json` and a pasteable
`benchmark.tex`. Run it on CHTC with `chtc/benchmark.sub`, which pins a single GPU model
so the rows are comparable with each other.

Author(s): Alejandro Calle-Saldarriaga
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
for p in (str(_ROOT / "code"), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from data.loading import spectral_loader as SL              # noqa: E402
from evals import checkpoints as ck                         # noqa: E402
from evals import laplace_last_layer as lap                 # noqa: E402
from evals.artifacts import _git_commit                     # noqa: E402
from predict import TASKS, build_data, target_stats         # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
logger = logging.getLogger("benchmark")

METHOD_ORDER = ["Point estimate", "Laplace", "Normalizing flow"]


# ---------------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------------
def provenance(args, device: torch.device) -> dict:
    """Everything a reader needs about the machine, plus what the timed region contains.

    Written before the timing runs and again after, so a job that dies half way still
    leaves a record of what it was measuring.
    """
    d = {
        "git_commit": _git_commit(_ROOT),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "condor_slot": os.environ.get("_CONDOR_SLOT", ""),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cpu_model": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "dataloader_workers": args.num_workers,
        # fp32 throughout: no autocast, no half precision, and TF32 explicitly pinned
        # rather than inherited, since the torch default has changed between versions and
        # a reader cannot otherwise know which kernels ran.
        "precision": "float32",
        "autocast": False,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "batch_sizes": args.batch_sizes,
        "warmup_passes": args.warmup,
        "repeats": args.repeats,
        "batches_timed": args.batches,
        "laplace_weight_samples": args.laplace_samples,
        "flow_samples": args.nf_samples,
        "timed_region": {
            "model": "predictive computation only; inputs already resident on the device",
            "end_to_end": "DataLoader iteration, host-to-device copy and the predictive",
            "excluded_from_both": [
                "reading L1B/simulated files and building the covariate table",
                "fitting the Laplace approximation (once, offline)",
                "loading checkpoints",
                "writing results",
            ],
        },
        "acos_row": "deliberately absent; no ACOS timing was produced by us",
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        d.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": f"{props.major}.{props.minor}",
            "gpu_memory_gb": round(props.total_memory / 1024 ** 3, 1),
            "gpu_multiprocessors": props.multi_processor_count,
            "driver": os.environ.get("CUDA_DRIVER_VERSION", ""),
        })
    else:
        d["gpu_name"] = "none (CPU run -- not reportable)"
    return d


# ---------------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------------
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_pass(pass_fn, n_soundings: int, device, warmup: int, repeats: int) -> dict:
    """Median ms per sounding over `repeats` timed passes, after `warmup` discarded ones.

    A "pass" is `--batches` batches, so the warm-up is that many forward calls and not a
    single one -- enough for cuDNN's autotuner to settle and for the allocator to stop
    growing, which is where most of the first pass goes.

    The median rather than the mean: on a shared cluster one pass in five lands on a slot
    that is doing something else, and that pass is not information about the model. The
    min and max across repeats are reported beside it so a reader can see whether the
    median means anything.
    """
    for _ in range(max(warmup, 1)):
        pass_fn()
    _sync(device)

    per_pass = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        pass_fn()
        _sync(device)
        per_pass.append(time.perf_counter() - t0)
    med = statistics.median(per_pass)
    return {
        "n_soundings": n_soundings,
        "ms_per_sounding": 1000.0 * med / n_soundings,
        "ms_per_sounding_min": 1000.0 * min(per_pass) / n_soundings,
        "ms_per_sounding_max": 1000.0 * max(per_pass) / n_soundings,
        "soundings_per_sec": n_soundings / med,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--out", default=str(_ROOT / "experiments/scoring/tables"))
    ap.add_argument("--files-root", default=str(_ROOT / "scratch/simulated/files"))
    ap.add_argument("--annotations", default="aux_data.csv")
    ap.add_argument("--split-csv", default=None)
    ap.add_argument("--stats-dir", default=None)
    ap.add_argument("--weights-root",
                    default=str(_ROOT / "experiments/weights/group_split_v1"))
    ap.add_argument("--covariates", default=None)
    ap.add_argument("--det-ckpt", default=None)
    ap.add_argument("--flow-ckpt", default=None)
    ap.add_argument("--nf-arch", default=None)
    ap.add_argument("--legacy-backbone", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-size", type=float, default=0.9)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 256],
                    help="batch 1 is the latency of a single sounding; 256 is the "
                         "throughput configuration the manuscript reports")
    ap.add_argument("--batches", type=int, default=8,
                    help="batches timed per pass. Kept small on purpose: this is a cost "
                         "measurement, not an evaluation, and a long pass on a shared "
                         "cluster measures the cluster")
    ap.add_argument("--warmup", type=int, default=1,
                    help="discarded passes before timing; a pass is --batches batches")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--laplace-samples", type=int, default=256)
    ap.add_argument("--laplace-fit-subset", type=int, default=2000,
                    help="the fit is offline and is NOT timed; a smaller subset than "
                         "predict.py's only makes the job shorter, and the predictive "
                         "being timed has the same shape and cost either way")
    ap.add_argument("--nf-samples", type=int, default=512)
    ap.add_argument("--tf32", choices=["on", "off"], default="off",
                    help="pinned rather than inherited, and recorded either way")
    ap.add_argument("--skip-laplace", action="store_true")
    ap.add_argument("--skip-flow", action="store_true")
    args = ap.parse_args(argv)

    spec = TASKS[args.task]
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = (args.tf32 == "on")
    torch.backends.cudnn.allow_tf32 = (args.tf32 == "on")

    if device.type != "cuda":
        logger.warning("no GPU: these timings are not the ones to report.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = provenance(args, device)
    (out_dir / "benchmark_provenance.json").write_text(
        json.dumps(prov, indent=2), encoding="utf-8")
    logger.info("device: %s", prov.get("gpu_name"))

    test_ds, train_ds, _val_ds, _frame, _sdc, split_name = build_data(args, spec)
    y_mu, _y_sigma = target_stats(args, spec)
    dim = len(y_mu)
    cov_stats = SL.load_covariate_stats(Path(args.stats_dir) if args.stats_dir else None)

    warm_loader = torch.utils.data.DataLoader(test_ds, batch_size=max(args.batch_sizes),
                                              shuffle=False, num_workers=0)
    warm_x, _, _ = lap.move_batch(next(iter(warm_loader)), device)

    # -- models, loaded exactly as predict.py loads them ---------------------------
    calls = {}
    det = ck.load_deterministic(
        args.weights_root, spec["det"], device, warm_x_tuple=warm_x,
        legacy=args.legacy_backbone, cov_stats=cov_stats,
        ckpt_path=args.det_ckpt, covariates=args.covariates,
    )
    net = det.module.eval()
    calls["Point estimate"] = lambda x: net(x)

    if not args.skip_laplace:
        fit_levels = spec.get("model_levels") or spec["levels"]
        y_idx = None if fit_levels is None else [l - 1 for l in fit_levels]
        t0 = time.perf_counter()
        fit = lap.fit_laplace(
            det.module,
            lap.build_laplace_loader(train_ds, batch_size=128,
                                     n_subset=args.laplace_fit_subset,
                                     seed=args.seed, target_idx=y_idx),
            device, mode=spec["laplace_mode"], hessian_structure=spec["hessian"],
        )
        # Recorded, not timed: it is a one-off cost paid before deployment, and saying so
        # is part of what the table has to state.
        prov["laplace_fit_seconds_not_timed"] = round(time.perf_counter() - t0, 1)
        calls["Laplace"] = lambda x: lap.posterior_mean_cov(
            fit, x, n_samples=args.laplace_samples)

    if not args.skip_flow:
        arch = None
        if args.nf_arch:
            with open(args.nf_arch) as fh:
                arch = yaml.safe_load(fh) or {}
        flow = ck.load_flow(args.weights_root, spec["flow"], device, warm_x_tuple=warm_x,
                            arch_override=arch, ckpt_path=args.flow_ckpt,
                            cov_stats=cov_stats, covariates=args.covariates).module.eval()
        calls["Normalizing flow"] = lambda x: flow.sample(x, n=args.nf_samples)

    # -- the measurement -----------------------------------------------------------
    rows = []
    with torch.no_grad():
        for bs in args.batch_sizes:
            loader = torch.utils.data.DataLoader(
                test_ds, batch_size=bs, shuffle=False,
                num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            )
            # Resident batches for the model scope. Materialized once and reused by every
            # method at this batch size, so the three methods are timed on identical
            # inputs and the loader cannot drift between them.
            resident = []
            for i, b in enumerate(loader):
                if i >= args.batches:
                    break
                x, y, _ = lap.move_batch(b, device)
                resident.append((x, y))
            if not resident:
                logger.warning("batch size %d: no batches; skipped.", bs)
                continue

            n_soundings = sum(int(y.shape[0]) for _x, y in resident)

            for name in [m for m in METHOD_ORDER if m in calls]:
                fn = calls[name]

                # Scope 1: the predictive alone, on batches already on the device.
                r = time_pass(lambda _fn=fn: [_fn(x) for x, _y in resident],
                              n_soundings, device, args.warmup, args.repeats)
                rows.append(dict(task=args.task, split=split_name, dim=dim, method=name,
                                 scope="model", batch_size=bs,
                                 n_batches=len(resident), **r))
                logger.info("%-16s bs=%-4d model      %8.3f ms/sounding",
                            name, bs, r["ms_per_sounding"])

                # Scope 2: the loader inside the timed region. A fresh iterator every
                # pass, because a DataLoader is not restartable and worker start-up is
                # part of what this scope is measuring. The same first `--batches`
                # batches, in the same order, so the two scopes cover the same soundings.
                def e2e(_fn=fn):
                    for j, b in enumerate(loader):
                        if j >= args.batches:
                            break
                        x, _y, _ = lap.move_batch(b, device)
                        _fn(x)

                r2 = time_pass(e2e, n_soundings, device, min(args.warmup, 1),
                               args.repeats)
                rows.append(dict(task=args.task, split=split_name, dim=dim, method=name,
                                 scope="end_to_end", batch_size=bs,
                                 n_batches=len(resident), **r2))
                logger.info("%-16s bs=%-4d end-to-end %8.3f ms/sounding",
                            name, bs, r2["ms_per_sounding"])

    # -- output --------------------------------------------------------------------
    csv_path = out_dir / "benchmark.csv"
    cols = ["task", "split", "dim", "method", "scope", "batch_size", "n_soundings",
            "n_batches", "ms_per_sounding", "ms_per_sounding_min", "ms_per_sounding_max",
            "soundings_per_sec"]
    header = not csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as fh:
        if header:
            fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    (out_dir / "benchmark_provenance.json").write_text(
        json.dumps(prov, indent=2), encoding="utf-8")
    write_tex(csv_path, out_dir / "benchmark.tex", prov)
    write_tex_batches(csv_path, out_dir / "benchmark_batch.tex", prov)
    logger.info("wrote %s (+ .tex, provenance)", csv_path)
    return 0


def _fmt(v: float) -> str:
    """Three significant figures across five orders of magnitude, without exponents."""
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def write_tex(csv_path: Path, tex_path: Path, prov: dict) -> None:
    """The manuscript table: one row per method per task, at the reported batch size.

    Rebuilt from the whole CSV every time, so a job that appends a third task produces a
    complete table rather than a third of one.
    """
    import csv as _csv
    with open(csv_path, encoding="utf-8") as fh:
        rows = [r for r in _csv.DictReader(fh)]
    if not rows:
        return
    bs = max(int(r["batch_size"]) for r in rows)
    lines = [
        "% generated by experiments/scoring/benchmark.py -- do not edit by hand",
        f"% device: {prov.get('gpu_name', '?')}, {prov.get('precision')}, "
        f"TF32 matmul {prov.get('tf32_matmul')}, torch {prov.get('torch')}",
        f"% batch size {bs}, {prov.get('warmup_passes')} warm-up pass(es), "
        f"{prov.get('repeats')} repeats, median reported",
        "% 'model' excludes the data loader; 'end-to-end' includes it. Neither includes "
        "the offline Laplace fit.",
        "% No ACOS row: no ACOS timing was produced on this hardware.",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Task & Method & \multicolumn{2}{c}{ms per sounding} \\",
        r"\cmidrule(lr){3-4}",
        r" & & model & end-to-end \\",
        r"\midrule",
    ]
    label = {"xco2": r"Scalar $X\rm{CO}_2$", "profiles": r"$20$-level profile",
             "last4": r"Near-surface ($17$--$20$)"}
    for task in ("xco2", "profiles", "last4"):
        sub = [r for r in rows if r["task"] == task and int(r["batch_size"]) == bs]
        if not sub:
            continue
        for method in METHOD_ORDER:
            m = {r["scope"]: r for r in sub if r["method"] == method}
            if not m:
                continue
            def cell(scope):
                v = m.get(scope)
                return "--" if v is None else f"{float(v['ms_per_sounding']):.3f}"
            lines.append(f"{label.get(task, task)} & {method} & "
                         f"{cell('model')} & {cell('end_to_end')} \\\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())


def write_tex_batches(csv_path: Path, tex_path: Path, prov: dict,
                      task: str = "profiles", scope: str = "model") -> None:
    """Cost against batch size for one task -- the table the manuscript should carry.

    Batch size is not a nuisance parameter here, it is the finding. Two of the three
    methods cost a fixed amount per *batch*, so their per-sounding cost falls as 1/B and
    the ordering between methods depends on which regime you are in; the flow's draws are
    per-sounding work and stop amortizing. A table at one batch size hides all of that,
    and it is exactly the comparison this table must not imply.

    One scope only, `model` by default: the cost of the predictive itself. The end-to-end
    figures stay in the CSV and belong in a sentence rather than in a table -- most of what
    they add is a per-sounding `.npy` read off the execute node's scratch disk, which
    describes our storage layout and not the retrieval.
    """
    import csv as _csv
    with open(csv_path, encoding="utf-8") as fh:
        rows = [r for r in _csv.DictReader(fh) if r["task"] == task]
    if not rows:
        return
    sizes = sorted({int(r["batch_size"]) for r in rows})
    lines = [
        "% generated by experiments/scoring/benchmark.py -- do not edit by hand",
        f"% task: {task}, scope: {scope} -- the predictive computation. Reading and",
        "% preparing the inputs is excluded; benchmark.csv carries the end-to-end figures.",
        f"% device: {prov.get('gpu_name', '?')}, {prov.get('precision')}, "
        f"TF32 matmul {prov.get('tf32_matmul')}, torch {prov.get('torch')}",
        f"% {prov.get('warmup_passes')} warm-up pass(es), {prov.get('repeats')} repeats of "
        f"{prov.get('batches_timed')} batches, median reported",
        r"\begin{tabular}{l" + "r" * len(sizes) + "}",
        r"\toprule",
        r" & \multicolumn{%d}{c}{ms per sounding} \\" % len(sizes),
        r"\cmidrule(lr){2-%d}" % (len(sizes) + 1),
        "Method & " + " & ".join(f"batch {b}" for b in sizes) + r" \\",
        r"\midrule",
    ]
    for method in METHOD_ORDER:
        cells = []
        for b in sizes:
            hit = [r for r in rows if r["method"] == method
                   and int(r["batch_size"]) == b and r["scope"] == scope]
            cells.append(_fmt(float(hit[0]["ms_per_sounding"])) if hit else "--")
        lines.append(f"{method} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

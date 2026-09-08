#!/usr/bin/env python3
"""spectral_attribution.py -- which spectral channels the retrieval actually uses.

One question the manuscript does not otherwise answer: *what features in the network allow
for retrieval of XCO2 from spectra?* It is a fair question, and the honest reply is that
neither the Laplace approximation nor the normalizing flow answers it -- both are
statements about the shape of the predictive distribution, not about the mapping. This
script is the smallest thing that does answer it, and it answers it in the input basis
(which wavelengths matter) rather than in the parameter basis (which units matter), because
the input basis is the one with a physical interpretation: CO2 absorption lines have known
positions, and a retrieval that reads them must be sensitive to them.

Three measurements, in increasing order of how much they can be trusted:

1. **Gradient saliency.** |d yhat / d r_c| for every channel c of the normalized radiance,
   averaged over the scored soundings. Cheap (one backward pass) and local: it says what
   the network responds to at the spectra it was given, and nothing about what it would do
   elsewhere.

2. **Mean-spectrum occlusion, per band.** Replace one band's radiance with the *training*
   mean spectrum -- not with zeros, which is off-distribution in a way no trained network
   is obliged to handle sensibly -- and re-score. The band-level version has one
   configuration that matters more than the rest: with all three bands replaced, the model
   is reduced to its auxiliary covariates, so its RMSE is the no-spectrum reference
   computed *inside the fitted model* rather than by a separately fitted baseline. That
   number is directly comparable to `generalization_diagnostics.py`'s ridge.

3. **Windowed occlusion.** The same substitution over a sliding window of `--window`
   channels, which gives an importance curve against wavelength. This is the measurement
   the physics can check, via the statistic in section 4 below.

**The statistic that makes it an answer and not a picture.** Absorption features are where
the mean spectrum falls below its own local continuum, and the continuum is recoverable
from the mean spectrum alone (a rolling upper quantile) with no line list. So for every
window we have two numbers -- its mean absorption depth and its occlusion importance -- and
their rank correlation tests, without naming a single line, whether the network's
sensitivity is concentrated on the absorption features or spread over the continuum. A
positive correlation in the two CO2 bands is the direct answer; a null
result would be worth reporting too, and would say the skill is carried by band-integrated
radiance, i.e. by something closer to a photometric than a spectroscopic measurement.

What this is NOT. It is not mechanistic interpretability: it says nothing about what the
intermediate layers represent, and a channel with a large gradient is not a channel the
network "uses" in any causal sense that survives correlated inputs. Spectral channels are
strongly correlated with their neighbours, so per-channel attributions divide the credit
for a shared signal arbitrarily, and only the windowed, occlusion-based curve should be
quoted at fine scale.

    python3 experiments/scoring/spectral_attribution.py --task xco2 \
        --split-csv  scratch/simulated/files/splits/group_split_v1.csv \
        --stats-dir  experiments/pipelines/stats/group_split_v1 \
        --weights-root experiments/weights/group_split_v1 \
        --covariates scene_prior \
        --det-ckpt ... --flow-ckpt ... \
        --out experiments/scoring/tables --fig latex/figures

Every output carries the task in its name, because the three tasks share one --out
directory and `fetch_results.sh --attr` unpacks all three archives into it -- without
the suffix the last one fetched would silently overwrite the other two.

Writes `attribution_channels_<task>.csv` (per channel: wavelength, mean radiance,
continuum, absorption depth, saliency), `attribution_bands_<task>.csv` (the five
band-level occlusions with bootstrap-over-scene standard errors),
`attribution_windows_<task>.csv` (the sweep),
`attribution_summary_<task>.{md,json}` (the correlations and headline numbers) and
`spectral_attribution_<task>.png`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats as sps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
for p in (str(_ROOT / "code"), str(_ROOT), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from data.loading import spectral_loader as SL              # noqa: E402
from evals import checkpoints as ck                         # noqa: E402
from evals import laplace_last_layer as lap                 # noqa: E402
from evals.tables import md_table                          # noqa: E402

# predict.py owns the task table, the dataset construction and the standardization
# constants. Importing them rather than restating them is what keeps a number here
# comparable with a cell of Tables 1-3: the same rows, in the same order, standardized by
# the same constants.
from predict import TASKS, build_data, target_stats         # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
logger = logging.getLogger("attribution")

BANDS = ("b1", "b2", "b3")
BAND_LABEL = {"b1": r"O$_2$ A-band", "b2": r"weak CO$_2$", "b3": r"strong CO$_2$"}
RAD = 1        # channel index of the normalized radiance; channel 0 is wavelength


# --------------------------------------------------------------------------------------
# The spectra, once, on the device
# --------------------------------------------------------------------------------------
def cache_partition(ds, device, batch_size, num_workers, limit=None):
    """Every scored sounding's inputs, resident on `device`.

    The windowed sweep is one forward pass per window and there are tens of windows; at
    that point the cost of the experiment is the dataloader, not the network. The whole
    scored partition is about 110 MB in float32 for 4,489 soundings (3 bands x 2 channels x
    1024 x 4 bytes), which fits beside the model on any GPU this pipeline runs on, so it is
    read once and never again. `--limit` exists for the smoke test.
    """
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,        # order defines alignment
        num_workers=num_workers, pin_memory=True,
    )
    out = []
    for i, batch in enumerate(loader):
        if limit is not None and i >= limit:
            break
        x, y, retr = lap.move_batch(batch, device)
        out.append((x, y, retr))
    n = sum(int(y.shape[0]) for _, y, _ in out)
    logger.info("cached %d soundings in %d batches on %s", n, len(out), device)
    return out


def n_real_channels(batches):
    """Unpadded length per band, read off the wavelength channel.

    `add_scaling_channels` pads on the right with zeros in both channels and no real
    channel has wavelength 0, so the count of nonzero wavelengths is the count of real
    channels: 835 / 652 / 811 for the three bands.
    """
    bands0 = batches[0][0][0]
    return {b: int((bands0[b][0][0, 0] != 0).sum().item()) for b in BANDS}


def mean_spectrum(train_ds, device, batch_size, num_workers, max_batches=None):
    """Per-band, per-channel mean of the normalized radiance over TRAINING rows.

    The training partition, deliberately: the substitution baseline is part of the
    measuring instrument, and estimating it on the rows being scored would make the
    occlusion depend on the test set. `max_batches` caps it -- the mean of 4,000 spectra is
    indistinguishable from the mean of 36,000 for this purpose, and the sweep's cost does
    not depend on it either way. The mean wavelength is accumulated at the same time and is
    the figure's abscissa.
    """
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    acc = {b: None for b in BANDS}
    wl = {b: None for b in BANDS}
    n = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, _, _ = lap.move_batch(batch, device)
        bands = x[0]
        for b in BANDS:
            t = bands[b][0]                              # [B, 2, L]
            s = t[:, RAD, :].sum(dim=0).double()
            w = t[:, 0, :].sum(dim=0).double()
            acc[b] = s if acc[b] is None else acc[b] + s
            wl[b] = w if wl[b] is None else wl[b] + w
        n += int(bands[BANDS[0]][0].shape[0])
    logger.info("mean spectrum from %d training soundings", n)
    return ({b: (acc[b] / n).float() for b in BANDS},
            {b: (wl[b] / n).float() for b in BANDS})


def occlude(bands, mean_rad, spec):
    """`bands` with the radiance channel replaced by the mean over the named windows.

    `spec` is a list of `(band, lo, hi)`. The wavelength channel and the two scale
    constants are left exactly as they are, so the network still knows which band it is
    looking at and on what absolute radiance scale; only the shape of the spectrum within
    the window is removed. Returns a new dict -- the cached tensors are never written to.
    """
    if not spec:
        return bands
    out = {b: (t.clone(), c) for b, (t, c) in bands.items()}
    for b, lo, hi in spec:
        out[b][0][:, RAD, lo:hi] = mean_rad[b][lo:hi].unsqueeze(0)
    return out


# --------------------------------------------------------------------------------------
# Forward passes
# --------------------------------------------------------------------------------------
@torch.no_grad()
def score_occlusion(batches, det, flow, mean_rad, spec, y_idx, nf_samples=0):
    """Per-sounding squared error for the deterministic model, and flow NLL, under `spec`.

    Per-sounding arrays are returned so a bootstrap over reference soundings can be taken
    outside. The flow's log-density needs only a forward pass, so it is computed for every
    configuration; its point estimate needs samples and is computed only when
    `nf_samples > 0`.
    """
    se_det, nll_flow, se_flow = [], [], []
    for x, y, _ in batches:
        bands, aux_c, aux_k = x
        xo = (occlude(bands, mean_rad, spec), aux_c, aux_k)
        yt = y if y_idx is None else y.index_select(1, y_idx)
        if det is not None:
            pred = det(xo)
            se_det.append(((pred - yt) ** 2).mean(dim=1).cpu().numpy())
        if flow is not None:
            nll_flow.append((-flow.log_prob(xo, yt)).cpu().numpy())
            if nf_samples:
                s = flow.sample(xo, n=nf_samples)          # [S, B, D]
                se_flow.append(((s.mean(dim=0) - yt) ** 2).mean(dim=1).cpu().numpy())

    def pack(v):
        return np.concatenate(v) if v else None

    return pack(se_det), pack(nll_flow), pack(se_flow)


def saliency(batches, model, y_idx, kind, out_index=None):
    """Mean |gradient| and mean signed gradient of the output w.r.t. each radiance channel.

    `kind="det"` differentiates the predicted mean (averaged over the task's outputs, or
    output `out_index` if given); `kind="flow"` differentiates the log-density at the true
    state, which is the flow's analogue -- the flow has no scalar output to differentiate,
    and log p(y|x) is the quantity it is trained on.

    Both channels of the band tensor are rebuilt from a detached copy, so the graph reaches
    the radiance channel alone and the wavelength channel cannot pick up a gradient it has
    no business having.

    The parameters are taken out of the graph first. Nothing here steps an optimizer, so
    their gradients are pure cost -- a second copy of the model, accumulated over every
    batch -- and switching them off also makes it impossible for this diagnostic to leave a
    fitted model in a different state than it found it.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    abs_acc = {b: None for b in BANDS}
    sgn_acc = {b: None for b in BANDS}
    n = 0
    for x, y, _ in batches:
        bands, aux_c, aux_k = x
        leaves, rebuilt = {}, {}
        for b, (t, c) in bands.items():
            wl = t[:, 0:1, :].detach()
            rad = t[:, RAD:RAD + 1, :].detach().clone().requires_grad_(True)
            leaves[b] = rad
            rebuilt[b] = (torch.cat([wl, rad], dim=1), c)
        xo = (rebuilt, aux_c, aux_k)
        yt = y if y_idx is None else y.index_select(1, y_idx)
        if kind == "det":
            pred = model(xo)
            target = pred[:, out_index] if out_index is not None else pred.mean(dim=1)
        else:
            target = model.log_prob(xo, yt)
        target.sum().backward()
        for b in BANDS:
            g = leaves[b].grad.detach()[:, 0, :]           # [B, L]
            a, s = g.abs().sum(dim=0).double(), g.sum(dim=0).double()
            abs_acc[b] = a if abs_acc[b] is None else abs_acc[b] + a
            sgn_acc[b] = s if sgn_acc[b] is None else sgn_acc[b] + s
        n += int(yt.shape[0])
    return ({b: (abs_acc[b] / n).cpu().numpy() for b in BANDS},
            {b: (sgn_acc[b] / n).cpu().numpy() for b in BANDS})


# --------------------------------------------------------------------------------------
# Absorption depth, with no line list
# --------------------------------------------------------------------------------------
def continuum(rad, width, q=0.9):
    """Local continuum of a mean spectrum: a rolling upper quantile.

    An absorption spectrum sits below its continuum everywhere and touches it between
    lines, so an upper quantile over a window several lines wide estimates the continuum
    without knowing where the lines are. `q < 1` rather than the rolling maximum because
    the maximum of a window is biased upward by its noisiest channel.
    """
    s = pd.Series(np.asarray(rad, dtype=np.float64))
    c = s.rolling(width, center=True, min_periods=max(3, width // 4)).quantile(q)
    return c.bfill().ffill().to_numpy()


def window_edges(n, width, stride):
    return [(lo, min(lo + width, n)) for lo in range(0, max(n - 1, 1), stride)]


def bootstrap_mean(v, groups, n_boot, seed, base=None):
    """Mean of `v`, its grouped standard error, and the paired standard error of the delta.

    The flow's NLL, which used to carry a plain per-sounding standard error of the mean.
    That is the wrong resampling unit for the same reason it is wrong for the RMSE -- the
    100 replicates of a reference sounding are not 100 independent draws -- so it reported
    an interval several times too tight. `base` makes the delta paired, as in
    `bootstrap_rmse`.
    """
    m = float(np.mean(v))
    nan = float("nan")
    if groups is None or n_boot <= 0:
        return m, nan, nan
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    d_draws = np.empty(n_boot) if base is not None else None
    for k in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        draws[k] = np.mean(v[rows])
        if d_draws is not None:
            d_draws[k] = draws[k] - np.mean(base[rows])
    return (m, float(np.std(draws, ddof=1)),
            nan if d_draws is None else float(np.std(d_draws, ddof=1)))


def bootstrap_rmse(se, groups, ppm, n_boot, seed, base_se=None):
    """RMSE in ppm, its standard error, and the standard error of the DELTA from `base_se`.

    Resampling whole reference soundings, as everywhere else in the paper: replicates of one
    reference sounding share a state and a geometry, so resampling soundings would
    understate the error of any quantity aggregated over them.

    `base_se` is the intact model's per-sounding squared error. The quantity the table
    actually reports is the *difference* of two RMSEs measured on the same soundings under
    the same model, and a difference of paired quantities has a much smaller standard error
    than either term: the scene-to-scene variation that dominates the RMSE's own error bar
    is common to both and cancels. Bootstrapping the difference on the same resample is what
    makes that cancellation happen, and without it a real 0.19 ppm cost looks insignificant
    against a 0.31 ppm marginal error bar. The two errors answer different questions and
    both are reported; `d_*_se` is the one to quote beside a `d_*` column.
    """
    rmse = float(np.sqrt(np.mean(se)) * ppm)
    nan = float("nan")
    if groups is None or n_boot <= 0:
        return rmse, nan, nan
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    d_draws = np.empty(n_boot) if base_se is not None else None
    for k in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        draws[k] = np.sqrt(np.mean(se[rows])) * ppm
        if d_draws is not None:
            d_draws[k] = draws[k] - np.sqrt(np.mean(base_se[rows])) * ppm
    return (rmse, float(np.std(draws, ddof=1)),
            nan if d_draws is None else float(np.std(d_draws, ddof=1)))


# --------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="xco2", choices=sorted(TASKS))
    ap.add_argument("--out", default=str(_ROOT / "experiments/scoring/tables"))
    ap.add_argument("--fig", default=None, help="directory for the figure; omit to skip")

    ap.add_argument("--files-root", default=str(_ROOT / "scratch/simulated/files"))
    ap.add_argument("--annotations", default="aux_data.csv")
    ap.add_argument("--split-csv", default=None)
    ap.add_argument("--stats-dir", default=None)
    ap.add_argument("--weights-root", default=str(_ROOT / "experiments/weights/group_split_v1"))
    ap.add_argument("--covariates", default=None,
                    help="covariate preset the checkpoints were trained with; must match")
    ap.add_argument("--det-ckpt", default=None)
    ap.add_argument("--flow-ckpt", default=None)
    ap.add_argument("--legacy-backbone", action="store_true")
    ap.add_argument("--nf-arch", default=None,
                    help="YAML overriding the flow architecture, for checkpoints that do "
                         "not record their own (the published scalar flow). The search "
                         "winners record theirs, so this is normally unset.")
    ap.add_argument("--skip-flow", action="store_true")
    ap.add_argument("--skip-det", action="store_true")

    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-size", type=float, default=0.9)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-partition", default="test", choices=["test", "train", "val"])
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N batches of the scored partition; smoke tests only")

    ap.add_argument("--window", type=int, default=32,
                    help="channels per occlusion window. 32 is about 0.47 nm in the "
                         "A-band and 1.3 nm in the strong CO2 band -- several lines wide, "
                         "which is the scale at which the curve is interpretable")
    ap.add_argument("--stride", type=int, default=None,
                    help="window stride (default: --window, i.e. non-overlapping)")
    ap.add_argument("--continuum-width", type=int, default=None,
                    help="rolling window for the continuum estimate (default: 8x --window)")
    ap.add_argument("--continuum-quantile", type=float, default=0.9)
    ap.add_argument("--mean-batches", type=int, default=32,
                    help="training batches averaged for the substitution spectrum")
    ap.add_argument("--nf-samples", type=int, default=256,
                    help="flow draws for the band-level point estimate; 0 skips it")
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--grad-output", type=int, default=None,
                    help="differentiate this output index instead of the mean over outputs")
    ap.add_argument("--skip-sweep", action="store_true",
                    help="band-level occlusion and saliency only")
    ap.add_argument("--from-tables", action="store_true",
                    help="redraw the figure from attribution_channels_<task>.csv and "
                         "attribution_windows_<task>.csv in --out and exit. No GPU, no "
                         "checkpoint, no data: for a caption or a colour, so that "
                         "rebuild.sh can redraw it like every other figure.")
    args = ap.parse_args(argv)

    spec = TASKS[args.task]
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    stride = args.stride or args.window
    cwidth = args.continuum_width or 8 * args.window
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_tables:
        if not args.fig:
            raise SystemExit("--from-tables redraws the figure and needs --fig.")
        chan_csv = out_dir / f"attribution_channels_{args.task}.csv"
        win_csv = out_dir / f"attribution_windows_{args.task}.csv"
        if not chan_csv.exists():
            raise SystemExit(
                f"{chan_csv} does not exist. --from-tables redraws a figure from a run "
                f"that already happened; run the job first (chtc/attribution.dag).")
        chan_df = pd.read_csv(chan_csv)
        win_df = pd.read_csv(win_csv) if win_csv.exists() else pd.DataFrame()
        if win_df.empty:
            logger.warning("%s is missing or empty: the occlusion curve will be absent "
                           "and the figure will carry saliency alone.", win_csv)
        split = "unknown split"
        summary_json = out_dir / f"attribution_summary_{args.task}.json"
        if summary_json.exists():
            with open(summary_json) as fh:
                split = json.load(fh).get("split", split)
        draw_figure(args, chan_df, win_df, split)
        return 0

    if device.type != "cuda":
        logger.warning("running on %s; use --limit and --skip-sweep for a smoke test.",
                       device)

    # predict.py's build_data reads these attributes and no others.
    da = SimpleNamespace(
        files_root=args.files_root, annotations=args.annotations,
        split_csv=args.split_csv, stats_dir=args.stats_dir,
        covariates=args.covariates, eval_partition=args.eval_partition,
        seed=args.seed, train_size=args.train_size,
    )
    test_ds, train_ds, _val_ds, frame, _src, split_name = build_data(da, spec)

    _y_mu, y_sigma = target_stats(args, spec)
    ppm = float(np.mean(np.asarray(y_sigma, dtype=np.float64))) * 1e6
    fit_levels = spec.get("model_levels") or spec["levels"]
    y_idx = None if fit_levels is None else torch.as_tensor(
        [lv - 1 for lv in fit_levels], device=device, dtype=torch.long)

    batches = cache_partition(test_ds, device, args.batch_size, args.num_workers,
                              limit=args.limit)
    n_scored = sum(int(y.shape[0]) for _, y, _ in batches)
    groups = None
    if {"clst", "ref"}.issubset(frame.columns):
        g = (frame["clst"].astype(str) + "/" + frame["ref"].astype(str)).to_numpy()
        groups = g[:n_scored]
    mean_rad, mean_wl = mean_spectrum(train_ds, device, args.batch_size,
                                      args.num_workers, max_batches=args.mean_batches)
    nreal = n_real_channels(batches)
    logger.info("real channels: %s", nreal)

    cov_stats = SL.load_covariate_stats(Path(args.stats_dir) if args.stats_dir else None)
    warm_x = batches[0][0]

    det = None
    if not args.skip_det:
        d = ck.load_deterministic(
            args.weights_root, spec["det"], device, warm_x_tuple=warm_x,
            legacy=args.legacy_backbone, cov_stats=cov_stats,
            ckpt_path=args.det_ckpt, covariates=args.covariates,
        )
        logger.info("deterministic: %s", d.describe())
        det = d.module.eval()
    nf_arch_override = None
    if args.nf_arch:
        with open(args.nf_arch) as fh:
            nf_arch_override = yaml.safe_load(fh) or {}
    flow = None
    if not args.skip_flow:
        f = ck.load_flow(args.weights_root, spec["flow"], device, warm_x_tuple=warm_x,
                         arch_override=nf_arch_override,
                         ckpt_path=args.flow_ckpt, cov_stats=cov_stats,
                         covariates=args.covariates)
        logger.info("flow: %s", f.describe())
        flow = f.module.eval()

    # ---- 1. band-level occlusion, including the covariates-only configuration ---------
    configs = [("intact", [])]
    for b in BANDS:
        configs.append((f"occlude {b}", [(b, 0, nreal[b])]))
    configs.append(("occlude all three", [(b, 0, nreal[b]) for b in BANDS]))

    # The intact run's per-sounding arrays are kept, not just its scalars: every delta in
    # the table is paired with them, and a paired standard error needs both arrays on the
    # same bootstrap resample. Three arrays of 4,489 floats.
    rows, base, base_arr = [], {}, {}
    for name, sp in configs:
        se_d, nll_f, se_f = score_occlusion(batches, det, flow, mean_rad, sp, y_idx,
                                            nf_samples=args.nf_samples)
        rec = {"task": args.task, "split": split_name, "config": name, "n": n_scored,
               "n_scenes": int(0 if groups is None else np.unique(groups).size)}
        if se_d is not None:
            r, se, dse = bootstrap_rmse(se_d, groups, ppm, args.n_boot, args.seed,
                                        base_se=base_arr.get("det"))
            rec.update(det_rmse_ppm=r, det_rmse_se=se)
            if name != "intact":
                rec["d_det_rmse_ppm_se"] = dse
        if nll_f is not None:
            m, se, dse = bootstrap_mean(nll_f, groups, args.n_boot, args.seed,
                                        base=base_arr.get("nll"))
            rec.update(flow_nll_std=m, flow_nll_se=se)
            if name != "intact":
                rec["d_flow_nll_std_se"] = dse
        if se_f is not None:
            r, se, dse = bootstrap_rmse(se_f, groups, ppm, args.n_boot, args.seed,
                                        base_se=base_arr.get("flow"))
            rec.update(flow_rmse_ppm=r, flow_rmse_se=se)
            if name != "intact":
                rec["d_flow_rmse_ppm_se"] = dse
        if name == "intact":
            base = dict(rec)
            base_arr = {"det": se_d, "nll": nll_f, "flow": se_f}
        for k in ("det_rmse_ppm", "flow_rmse_ppm", "flow_nll_std"):
            if k in rec and k in base:
                rec["d_" + k] = rec[k] - base[k]
        rows.append(rec)
        logger.info("%-20s %s", name, "  ".join(
            f"{k}={v:.4f}" for k, v in rec.items() if isinstance(v, float)))
    bands_df = pd.DataFrame(rows)
    bands_df.to_csv(out_dir / f"attribution_bands_{args.task}.csv", index=False)

    # ---- 2. saliency and the absorption-depth axis -----------------------------------
    sal = {}
    if det is not None:
        sal["det"] = saliency(batches, det, y_idx, "det", out_index=args.grad_output)
    if flow is not None:
        sal["flow"] = saliency(batches, flow, y_idx, "flow")
    chan = []
    for b in BANDS:
        L = nreal[b]
        rad = mean_rad[b][:L].cpu().numpy().astype(np.float64)
        cont = continuum(rad, cwidth, args.continuum_quantile)
        rec = {"band": b, "channel": np.arange(L),
               "wavelength_um": mean_wl[b][:L].cpu().numpy().astype(np.float64),
               "mean_radiance_norm": rad, "continuum": cont,
               "absorption_depth": 1.0 - rad / np.maximum(cont, 1e-12)}
        for who, (a, s) in sal.items():
            # ppm per unit normalized radiance for the deterministic model; the flow's is
            # nats per unit normalized radiance and is not converted.
            k = ppm if who == "det" else 1.0
            rec[f"{who}_abs_grad"] = a[b][:L] * k
            rec[f"{who}_signed_grad"] = s[b][:L] * k
        chan.append(pd.DataFrame(rec))
    chan_df = pd.concat(chan, ignore_index=True)
    chan_df.to_csv(out_dir / f"attribution_channels_{args.task}.csv", index=False)

    # ---- 3. the sweep ----------------------------------------------------------------
    win_df = pd.DataFrame()
    if not args.skip_sweep:
        wrows = []
        for b in BANDS:
            edges = window_edges(nreal[b], args.window, stride)
            logger.info("sweeping %s: %d windows of %d channels", b, len(edges), args.window)
            for lo, hi in edges:
                se_d, nll_f, _ = score_occlusion(batches, det, flow, mean_rad,
                                                 [(b, lo, hi)], y_idx, nf_samples=0)
                sub = chan_df[(chan_df.band == b) & (chan_df.channel >= lo)
                              & (chan_df.channel < hi)]
                rec = {"band": b, "lo": lo, "hi": hi,
                       "wavelength_um": float(sub.wavelength_um.mean()),
                       "absorption_depth": float(sub.absorption_depth.mean())}
                if se_d is not None:
                    rec["det_rmse_ppm"] = float(np.sqrt(np.mean(se_d)) * ppm)
                    rec["d_det_rmse_ppm"] = rec["det_rmse_ppm"] - base["det_rmse_ppm"]
                if nll_f is not None:
                    rec["flow_nll_std"] = float(np.mean(nll_f))
                    rec["d_flow_nll_std"] = rec["flow_nll_std"] - base["flow_nll_std"]
                for who in sal:
                    rec[f"{who}_abs_grad"] = float(sub[f"{who}_abs_grad"].mean())
                wrows.append(rec)
        win_df = pd.DataFrame(wrows)
        win_df.to_csv(out_dir / f"attribution_windows_{args.task}.csv", index=False)

    # ---- 4. the correlations ---------------------------------------------------------
    summary = {"task": args.task, "split": split_name, "n": n_scored,
               "n_scenes": int(0 if groups is None else np.unique(groups).size),
               "ppm_per_std": ppm, "window": args.window, "stride": stride,
               "continuum_width": cwidth, "continuum_quantile": args.continuum_quantile,
               "real_channels": nreal, "bands": rows, "correlations": []}
    sources = [(win_df, "window"), (chan_df, "channel")]
    for b in BANDS:
        for frame_, unit in sources:
            if frame_.empty:
                continue
            s = frame_[frame_.band == b]
            for col in ("d_det_rmse_ppm", "d_flow_nll_std", "det_abs_grad", "flow_abs_grad"):
                if col not in s.columns:
                    continue
                v, dep = s[col].to_numpy(), s["absorption_depth"].to_numpy()
                ok = np.isfinite(v) & np.isfinite(dep)
                if ok.sum() < 5:
                    continue
                rho, p = sps.spearmanr(dep[ok], v[ok])
                summary["correlations"].append(
                    {"band": b, "quantity": col, "unit": unit, "n": int(ok.sum()),
                     "spearman_rho": float(rho), "p": float(p)})
    with open(out_dir / f"attribution_summary_{args.task}.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    # ---- 5. the figure, BEFORE the prose ---------------------------------------------
    #
    # Order matters here and it cost three GPU jobs to learn. Everything above this point
    # is expensive; everything below is formatting. On 4 Sep 2026 all three attribution
    # jobs died on `DataFrame.to_markdown` -- tabulate is not in the container -- after the
    # sweep had finished, and took the figure down with a summary table. The CSVs survived
    # because they were written first. So: cheap-and-fragile last, and `md_table` cannot
    # raise on a missing dependency any more (evals/tables.py).
    draw_figure(args, chan_df, win_df, split_name)

    md = [f"# Spectral attribution -- {args.task}, {split_name}", "",
          f"{n_scored} soundings from {summary['n_scenes']} reference soundings; "
          f"{ppm:.4f} ppm per standardized unit.", "",
          "## Band-level occlusion (radiance replaced by the training mean spectrum)", "",
          md_table(bands_df), "",
          "The `occlude all three` row is the fitted model reduced to its auxiliary "
          "covariates: the no-spectrum reference computed inside the model, comparable "
          "with the ridge of generalization_diagnostics.py.", "",
          "## Rank correlation of importance with absorption depth", "",
          md_table(pd.DataFrame(summary["correlations"])), ""]
    (out_dir / f"attribution_summary_{args.task}.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("wrote %s", out_dir / f"attribution_summary_{args.task}.md")
    return 0


def draw_figure(args, chan_df, win_df, split_name):
    """Mean spectrum and continuum on top, saliency and the occlusion curve underneath.

    Split out of `main` so `--from-tables` can redraw it from the two CSVs with no GPU,
    no checkpoint and no data: a caption change should not cost a queue slot, which is the
    same rule the rest of `rebuild.sh` follows.
    """
    if args.fig:
        plt.rcParams.update({
            "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
            "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
            "figure.dpi": 120, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.25,
        })
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex="col")
        for j, b in enumerate(BANDS):
            s = chan_df[chan_df.band == b]
            ax = axes[0, j]
            ax.plot(s.wavelength_um, s.mean_radiance_norm, lw=0.7, color="0.35",
                    label="mean spectrum")
            ax.plot(s.wavelength_um, s.continuum, lw=0.9, color="tab:orange",
                    label="continuum")
            ax.set_title(f"{BAND_LABEL[b]}  ({len(s)} channels)")
            if j == 0:
                ax.set_ylabel("normalized radiance")
                ax.legend(loc="lower left", framealpha=0.9)
            ax = axes[1, j]
            if "det_abs_grad" in s:
                ax.plot(s.wavelength_um, s.det_abs_grad, lw=0.6, color="tab:blue",
                        alpha=0.8)
            if not win_df.empty and "d_det_rmse_ppm" in win_df:
                w = win_df[win_df.band == b]
                tw = ax.twinx()
                tw.step(w.wavelength_um, w.d_det_rmse_ppm, where="mid", lw=1.4,
                        color="tab:red")
                tw.grid(False)
                if j == 2:
                    tw.set_ylabel(r"occlusion $\Delta$RMSE (ppm)", color="tab:red")
            ax.set_xlabel(r"wavelength ($\mu$m)")
            if j == 0:
                ax.set_ylabel(r"mean $|\partial \hat{y}/\partial r|$ (ppm)",
                              color="tab:blue")
        fig.suptitle("Where the retrieval reads the spectrum -- "
                     f"{args.task}, {split_name}")
        fig.tight_layout()
        fig_dir = Path(args.fig)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"spectral_attribution_{args.task}.{ext}",
                        bbox_inches="tight")
        plt.close(fig)
        logger.info("wrote %s", fig_dir / f"spectral_attribution_{args.task}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

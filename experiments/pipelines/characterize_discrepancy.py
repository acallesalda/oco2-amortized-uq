"""characterize_discrepancy.py
Recover and characterise the model-discrepancy term delta from the simulated OCO-2 files.

The simulation's model discrepancy needs a description: what delta is, what space it lives in, whether it should be
indexed by location, which moment of the cluster distribution it modifies, and how its
parameters were estimated given its dimension. None of that is in the manuscript, and the
published description (Braverman et al. 2021, SIAM/ASA JUQ 9(3), Section 4.1.4) documents
the *method* rather than the settings used for this particular ensemble. It can, however,
be measured directly from the delivered files, because they ship the noiseless forward
model evaluation alongside the simulated observation:

    SpectralParameters/measured_radiance        = F(x_true; b) + delta + epsilon
    SpectralParameters/true_noiseless_radiance  = F(x_true; b)
    SpectralParameters/measured_radiance_uncert = sd(epsilon), per channel

so  r := measured - true_noiseless  is the realised  delta + epsilon  for every replicate,
and dividing by the reported noise sd puts it in units where pure instrument noise has
unit variance. Everything below is a statement about r.

What the script reports, per reference sounding and aggregated per cluster:

  * block structure -- how many *distinct* draws of delta there are. On the file committed
    to this repo, delta is constant within blocks of 10 consecutive replicates (the
    within-block sd of the whitened residual is 0.998, i.e. pure noise), so one reference
    sounding carries 10 draws of delta, not 100. This is checked, not assumed.
  * the mean  mu_delta,  in units of the instrument noise and as a fraction of radiance,
    per band -- i.e. how much of a systematic bias delta imposes.
  * the covariance  Sigma_delta:  eigen-spectrum, effective rank, and how much of it is
    carried by the leading direction. Braverman et al. retain only leading principal
    components, so a low effective rank is expected and is worth reporting.
  * whether mu_delta and Sigma_delta differ across clusters (which
    moments the clustering modifies), and whether the same draws recur across reference
    soundings inside a cluster.
  * whether delta is correlated with the state x (the design can be read as a sensitivity
    analysis in x; if delta is drawn independently of x, that reading is answered by a
    number rather than by an argument).
  * the fraction of the perturbation the retrieval absorbs into x_hat, and a decomposition
    of the ACOS XCO2 error into a delta-driven part and a state/noise-driven part.

Usage:
    python3 characterize_discrepancy.py --soundings /workspace/data/oco2/simulated/og_data \
        --json discrepancy.json --csv discrepancy_by_sounding.csv
    python3 characterize_discrepancy.py            # single committed test file, quick check
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

DEFAULT_H5 = Path(__file__).resolve().parents[2] / (
    "code/test/data/lnd_nadir_202002_clst01_fp_aggregate_retrieval_ref46.h5"
)

FILL = -9998.0
# Candidate numbers of replicates sharing one draw of delta. 1 means "redrawn every
# replicate"; 100 means "one draw for the whole reference sounding".
BLOCK_CANDIDATES = (1, 2, 4, 5, 10, 20, 25, 50, 100)
BAND_NAMES = ("o2_a", "weak_co2", "strong_co2")
# Band membership is decided from the wavelength axis, never from hardcoded indices: the
# number of good channels per band varies between files.
BAND_LIMITS = ((0.0, 1.0), (1.0, 1.9), (1.9, 3.0))


# --------------------------------------------------------------------------- reading


def _prop(path: Path, prop: str) -> Optional[int]:
    m = re.search(rf"{prop}([0-9]+)", path.stem)
    return int(m.group(1)) if m else None


def read_file(path: Path) -> Optional[Dict[str, Any]]:
    """Load one reference sounding and return the whitened discrepancy realisations."""
    with h5py.File(path, "r") as f:
        need = ("SpectralParameters/measured_radiance",
                "SpectralParameters/true_noiseless_radiance",
                "SpectralParameters/measured_radiance_uncert",
                "SpectralParameters/wavelength")
        if any(k not in f for k in need):
            return None
        meas = f["SpectralParameters/measured_radiance"][:].astype(np.float64)
        true0 = f["SpectralParameters/true_noiseless_radiance"][:].astype(np.float64)
        unc = f["SpectralParameters/measured_radiance_uncert"][:].astype(np.float64)
        wl = f["SpectralParameters/wavelength"][:].astype(np.float64)
        out: Dict[str, Any] = {}
        for key, dest in (("SpectralParameters/modeled_radiance", "modeled"),
                          ("StateVector/true_state_vector", "x_true"),
                          ("StateVector/true_xco2", "xco2_true"),
                          ("StateVector/mixture_component", "mixture"),
                          ("RetrievalResults/retrieved_xco2", "xco2_ret"),
                          ("RetrievalResults/retrieved_xco2_variance", "xco2_var"),
                          ("RetrievalResults/outcome_flag", "outcome"),
                          ("RetrievalResults/xco2_quality_flag", "qflag")):
            out[dest] = f[key][:] if key in f else None

    valid = ((wl[0] > 0) & np.all(meas > FILL, axis=0)
             & np.all(true0 > FILL, axis=0) & np.all(unc > 0, axis=0))
    if valid.sum() < 100:
        return None
    w = wl[0][valid]
    out.update(
        path=path, cluster=_prop(path, "clst"), ref=_prop(path, "ref"),
        wavelength=w, valid=valid,
        # r / sigma_eps: pure instrument noise has unit variance here
        z=(meas[:, valid] - true0[:, valid]) / unc[:, valid],
        radiance=true0[:, valid], sigma=unc[:, valid],
        bands=[(w >= lo) & (w < hi) for lo, hi in BAND_LIMITS],
    )
    if out["modeled"] is not None:
        out["z_resid"] = (meas[:, valid] - out["modeled"][:, valid]) / unc[:, valid]
    return out


# ------------------------------------------------------------------- block structure


def detect_block_size(z: np.ndarray) -> Tuple[int, Dict[int, float]]:
    """Smallest block of consecutive replicates within which delta is constant.

    If delta is constant inside a block, then removing the block mean leaves only
    epsilon, whose whitened sd is 1 by construction. The statistic is therefore the
    median over channels of the within-block sd; the answer is the *largest* block for
    which it is still ~1 (larger blocks start absorbing genuine variation in delta).
    """
    n = z.shape[0]
    scores: Dict[int, float] = {}
    for b in BLOCK_CANDIDATES:
        if b > n or n % b:
            continue
        if b == 1:
            # a per-replicate draw cannot be separated from noise by this test; the
            # comparison value is the ungrouped sd, which exceeds 1 when delta varies
            scores[b] = float(np.median(z.std(axis=0, ddof=1)))
            continue
        zb = z.reshape(n // b, b, -1)
        within = zb - zb.mean(axis=1, keepdims=True)
        sd = within.std(axis=(0, 1), ddof=0) * np.sqrt(b / (b - 1))
        scores[b] = float(np.median(sd))
    consistent = [b for b, s in scores.items() if b > 1 and abs(s - 1.0) < 0.05]
    best = max(consistent) if consistent else 1
    return best, scores


def delta_draws(z: np.ndarray, block: int) -> np.ndarray:
    """One row per distinct draw of delta, in units of the instrument noise sd."""
    if block <= 1:
        return z.copy()
    return z.reshape(z.shape[0] // block, block, -1).mean(axis=1)


# ------------------------------------------------------------------------ statistics


def band_stats(v: np.ndarray, bands: List[np.ndarray], radiance: np.ndarray,
               sigma: np.ndarray) -> Dict[str, Dict[str, float]]:
    """RMS of a whitened spectrum, per band, also expressed as a fraction of radiance."""
    out = {}
    for name, m in zip(BAND_NAMES, bands):
        if not m.any():
            continue
        rms = float(np.sqrt((v[m] ** 2).mean()))
        out[name] = {
            "n_channels": int(m.sum()),
            "rms_over_sigma_eps": rms,
            "mean_over_sigma_eps": float(v[m].mean()),
            "rms_over_radiance_pct": float(
                np.sqrt(((v[m] * sigma[m]) ** 2).mean()) / radiance[m].mean() * 100),
        }
    return out


def eigen_summary(draws: np.ndarray) -> Dict[str, Any]:
    """Eigen-spectrum of the covariance of the delta draws (centred within the group)."""
    d = draws - draws.mean(axis=0, keepdims=True)
    k = d.shape[0]
    if k < 3:
        return {}
    g = d @ d.T / d.shape[1]
    ev = np.clip(np.linalg.eigvalsh(g)[::-1], 0, None)[: k - 1]
    tot = ev.sum()
    share = ev / tot if tot > 0 else ev
    cum = np.cumsum(share)
    return {
        "n_draws": int(k),
        "eigenvalues_top10": [float(x) for x in ev[:10]],
        "variance_share_top5": [float(x) for x in share[:5]],
        "n_pc_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
        "n_pc_for_99pct": int(np.searchsorted(cum, 0.99) + 1),
        # eigenvalues are in units of the noise variance, so >1 means "above the noise"
        "n_eigenvalues_above_noise": int((ev > 1.25).sum()),
    }


def smoothness(v: np.ndarray, bands: List[np.ndarray]) -> Dict[str, float]:
    out = {}
    for name, m in zip(BAND_NAMES, bands):
        x = v[m]
        if x.size > 30:
            out[f"{name}_lag1_autocorr"] = float(np.corrcoef(x[:-1], x[1:])[0, 1])
            out[f"{name}_lag5_autocorr"] = float(np.corrcoef(x[:-5], x[5:])[0, 1])
    return out


def _standardise(a: np.ndarray) -> np.ndarray:
    sd = a.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    return (a - a.mean(axis=0)) / sd


def state_dependence(draws: np.ndarray, x_true: Optional[np.ndarray],
                     block: int) -> Tuple[Dict[str, Any], Optional[np.ndarray],
                                          Optional[np.ndarray]]:
    """Is delta drawn independently of the state?

    Projects each draw on its leading direction and correlates the resulting score with
    the block-mean of every state element. With only ~10 draws per reference sounding the
    per-file correlations are almost pure sampling noise -- the largest of 49 of them is
    routinely ~0.8 under independence -- so the per-file numbers are reported next to a
    permutation reference, and the standardised score and state are also returned so the
    caller can pool them over all reference soundings into one powerful test.
    """
    if x_true is None or draws.shape[0] < 5:
        return {}, None, None
    d = draws - draws.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(d, full_matrices=False)
    score = u[:, 0] * s[0]
    n = draws.shape[0]
    xb = np.asarray(x_true, dtype=np.float64)
    xb = xb.reshape(n, block, -1).mean(axis=1) if block > 1 else xb

    # Correlations as one matmul of standardised columns, so the permutation null below
    # costs a single (200 x n) @ (n x 49) product rather than 200*49 calls.
    xs = _standardise(xb)
    keep = xb.std(axis=0, ddof=1) > 0
    xs[:, ~keep] = 0.0
    ss = _standardise(score.reshape(-1, 1)).ravel()
    cors = np.abs(ss @ xs) / (n - 1)
    rng = np.random.default_rng(0)
    perm = np.stack([rng.permutation(ss) for _ in range(200)])
    null_max = np.abs(perm @ xs).max(axis=1) / (n - 1)
    info = {
        "max_abs_corr_delta_score_vs_state": float(cors.max()),
        "argmax_state_element": int(cors.argmax()) + 1,
        "mean_abs_corr": float(cors.mean()),
        "permutation_null_max_abs_corr_median": float(np.median(null_max)),
        "permutation_null_max_abs_corr_p95": float(np.percentile(null_max, 95)),
        "exceeds_permutation_null": bool(cors.max() > np.percentile(null_max, 95)),
    }
    return info, cors * np.sign(ss @ xs), n


def _null_median_abs_corr(n: float, size: int = 20000) -> float:
    """Median |r| for a sample correlation from n independent points, under the null."""
    from scipy import stats
    t = stats.t.rvs(n - 2, size=size, random_state=0)
    return float(np.median(np.abs(t / np.sqrt(t ** 2 + n - 2))))


def pooled_state_dependence(cors: List[np.ndarray], n_draws: List[int]) -> Dict[str, Any]:
    """Combine the per-reference-sounding correlations into one test.

    The leading direction of the discrepancy is only defined up to sign, and the sign an
    SVD returns is arbitrary, so the correlations cannot simply be averaged across files.
    They are combined instead through their two-sided p-values (Fisher's method), which is
    invariant to that sign. The null is that delta was drawn independently of the state
    *within* a reference sounding; every between-sounding effect has already been removed
    by working inside each file.
    """
    if not cors:
        return {}
    from scipy import stats

    ncol = min(c.size for c in cors)
    r = np.stack([c[:ncol] for c in cors])            # (n_files, n_state_elements)
    n = np.asarray(n_draws, dtype=float)[:, None]
    r = np.clip(r, -0.999999, 0.999999)
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    p = np.clip(2 * stats.t.sf(np.abs(t), n - 2), 1e-300, 1.0)
    chi2 = -2 * np.log(p).sum(axis=0)
    p_comb = stats.chi2.sf(chi2, 2 * r.shape[0])
    j = int(np.argmin(p_comb))
    alpha = 0.05 / ncol                                # Bonferroni over state elements
    return {
        "n_files": int(r.shape[0]),
        "n_pooled_delta_draws": int(np.sum(n_draws)),
        "n_state_elements": int(ncol),
        "method": "Fisher combination of per-reference-sounding correlation p-values",
        "min_combined_p_value": float(p_comb[j]),
        "argmin_state_element": j + 1,
        "bonferroni_threshold": float(alpha),
        "median_abs_corr": float(np.median(np.abs(r))),
        "expected_median_abs_corr_under_null": float(_null_median_abs_corr(n.mean())),
        "delta_appears_independent_of_state": bool(p_comb[j] > alpha),
    }


def error_decomposition(xco2_true, xco2_ret, outcome, block: int) -> Dict[str, Any]:
    """Split the ACOS XCO2 error into a delta-driven and a state/noise-driven part.

    Replicates sharing a draw of delta form a block. The spread of the block means is
    what delta contributes; the spread inside a block is what the state and the noise
    contribute. Reported in ppm.
    """
    if xco2_true is None or xco2_ret is None or block <= 1:
        return {}
    err = (np.asarray(xco2_ret, float) - np.asarray(xco2_true, float)) * 1e6
    ok = np.ones_like(err, dtype=bool) if outcome is None else (np.asarray(outcome) <= 2)
    n = err.size
    eb = err.reshape(n // block, block)
    ok_b = ok.reshape(n // block, block)
    means, withins = [], []
    for row, m in zip(eb, ok_b):
        if m.sum() >= 3:
            means.append(row[m].mean())
            withins.append(row[m].std(ddof=1))
    if len(means) < 3:
        return {}
    means = np.asarray(means)
    within = float(np.mean(withins))
    between = float(means.std(ddof=1))
    # remove the part of the between-block spread that is just within-block sampling
    between_corrected = float(np.sqrt(max(between ** 2 - within ** 2 / block, 0.0)))
    return {
        "mean_error_ppm": float(err[ok].mean()),
        "rmse_ppm": float(np.sqrt((err[ok] ** 2).mean())),
        "sd_between_delta_blocks_ppm": between,
        "sd_between_delta_blocks_corrected_ppm": between_corrected,
        "sd_within_delta_block_ppm": within,
        "share_of_error_variance_from_delta": float(
            between_corrected ** 2 / (between_corrected ** 2 + within ** 2))
        if (between_corrected or within) else None,
    }


def absorption(z: np.ndarray, z_resid: Optional[np.ndarray],
               bands: List[np.ndarray]) -> Dict[str, float]:
    """Share of the perturbation energy the retrieval moves into x_hat rather than
    leaving in the converged spectral residual."""
    if z_resid is None:
        return {}
    out = {}
    for name, m in zip(BAND_NAMES, bands):
        before = float((z[:, m] ** 2).mean())
        after = float((z_resid[:, m] ** 2).mean())
        out[f"{name}_chi2_per_channel_before"] = before
        out[f"{name}_chi2_per_channel_after"] = after
        out[f"{name}_absorbed_fraction"] = float(1 - after / before) if before else None
    return out


# ------------------------------------------------------------------------ per-file


def analyse_file(d: Dict[str, Any]) -> Dict[str, Any]:
    z, bands = d["z"], d["bands"]
    block, block_scores = detect_block_size(z)
    draws = delta_draws(z, block)
    mu = draws.mean(axis=0)
    radiance, sigma = d["radiance"].mean(axis=0), d["sigma"].mean(axis=0)

    sd_info, sd_cors, sd_n = state_dependence(draws, d.get("x_true"), block)
    res: Dict[str, Any] = {
        "file": d["path"].name,
        "cluster": d["cluster"],
        "reference_sounding": d["ref"],
        "n_replicates": int(z.shape[0]),
        "n_valid_channels": int(z.shape[1]),
        "replicates_per_delta_draw": block,
        "n_distinct_delta_draws": int(draws.shape[0]),
        "block_detection_within_sd": {str(k): round(v, 4) for k, v in block_scores.items()},
        "mu_delta": band_stats(mu, bands, radiance, sigma),
        "mu_delta_rms_over_sigma_eps_all_bands": float(np.sqrt((mu ** 2).mean())),
        "mu_delta_smoothness": smoothness(mu, bands),
        "sigma_delta": eigen_summary(draws),
        "state_dependence": sd_info,
        "retrieval_absorption": absorption(z, d.get("z_resid"), bands),
        "xco2_error": error_decomposition(d.get("xco2_true"), d.get("xco2_ret"),
                                          d.get("outcome"), block),
    }
    sd_tot = z.std(axis=0, ddof=1)
    implied = np.sqrt(np.maximum(sd_tot ** 2 - 1.0, 0.0))
    res["sd_delta_over_sigma_eps"] = {
        name: float(np.median(implied[m])) for name, m in zip(BAND_NAMES, bands) if m.any()
    }
    if d.get("qflag") is not None:
        res["frac_xco2_quality_flag_good"] = float((np.asarray(d["qflag"]) == 0).mean())
    if d.get("outcome") is not None:
        res["frac_converged"] = float((np.asarray(d["outcome"]) <= 2).mean())
    res["_mu"] = mu           # kept for cluster aggregation, stripped before writing
    res["_draws"] = draws
    res["_valid"] = d["valid"]
    res["_sd_cors"] = sd_cors
    res["_sd_n"] = sd_n
    return res


# ---------------------------------------------------------------------- per-cluster


def aggregate_cluster(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pool reference soundings in one cluster.

    Files can mask out different channels, so pooling is done on the intersection of the
    valid masks; the number of shared channels is reported so a small intersection is
    visible rather than silent.
    """
    shared = np.logical_and.reduce([r["_valid"] for r in rows])
    idx = [np.flatnonzero(r["_valid"]) for r in rows]
    keep = [np.isin(i, np.flatnonzero(shared)) for i in idx]
    mus = np.stack([r["_mu"][k] for r, k in zip(rows, keep)])
    all_draws = np.concatenate([r["_draws"][:, k] for r, k in zip(rows, keep)], axis=0)

    # Is mu_delta the same for every reference sounding in the cluster, or does each
    # reference sounding get its own? Compare the spread of the per-sounding means with
    # the Monte-Carlo error of a single one of those means.
    n_draws = np.array([r["_draws"].shape[0] for r in rows])
    mc_error = float(np.sqrt((all_draws.var(axis=0, ddof=1) / n_draws.mean()).mean()))
    if len(rows) > 1:
        per_sounding_spread = float(np.sqrt((mus.var(axis=0, ddof=1)).mean()))
        corr = np.corrcoef(mus)
        off = corr[np.triu_indices_from(corr, 1)]
    else:
        per_sounding_spread = float("nan")
        off = np.empty(0)

    return {
        "cluster": rows[0]["cluster"],
        "n_reference_soundings": len(rows),
        "n_shared_channels": int(shared.sum()),
        "n_delta_draws_total": int(all_draws.shape[0]),
        "mu_delta_rms_over_sigma_eps": float(np.sqrt((all_draws.mean(axis=0) ** 2).mean())),
        "mu_delta_spread_across_reference_soundings": (
            per_sounding_spread if np.isfinite(per_sounding_spread) else None),
        "mu_delta_monte_carlo_error": mc_error,
        "mu_delta_is_sounding_specific": (
            bool(per_sounding_spread > 2 * mc_error)
            if np.isfinite(per_sounding_spread) else None),
        "corr_between_reference_sounding_means": {
            "median": float(np.median(off)) if off.size else None,
            "min": float(off.min()) if off.size else None,
            "max": float(off.max()) if off.size else None,
        },
        "sigma_delta_pooled": eigen_summary(all_draws),
    }


def compare_clusters(clusters: List[Dict[str, Any]],
                     rows_by_cluster: Dict[Any, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Does the discrepancy distribution actually differ between clusters?"""
    keys = sorted(rows_by_cluster)
    if len(keys) < 2:
        return {}
    shared = np.logical_and.reduce(
        [r["_valid"] for rows in rows_by_cluster.values() for r in rows])
    mus, spreads = [], []
    for k in keys:
        rows = rows_by_cluster[k]
        sel = [np.isin(np.flatnonzero(r["_valid"]), np.flatnonzero(shared)) for r in rows]
        draws = np.concatenate([r["_draws"][:, s] for r, s in zip(rows, sel)], axis=0)
        mus.append(draws.mean(axis=0))
        spreads.append(np.sqrt((draws.var(axis=0, ddof=1) / draws.shape[0]).mean()))
    mus = np.stack(mus)
    corr = np.corrcoef(mus)
    off = corr[np.triu_indices_from(corr, 1)]
    between = float(np.sqrt(mus.var(axis=0, ddof=1).mean()))
    return {
        "clusters": [int(k) if k is not None else None for k in keys],
        "n_shared_channels": int(shared.sum()),
        "between_cluster_sd_of_mu_delta": between,
        "within_cluster_monte_carlo_error": float(np.mean(spreads)),
        "mu_delta_differs_between_clusters": bool(between > 2 * float(np.mean(spreads))),
        "corr_between_cluster_mu_delta": {
            "median": float(np.median(off)), "min": float(off.min()), "max": float(off.max()),
        },
        "note": "mu_delta compared on the channel intersection of every file; "
                "wavelength grids differ slightly between reference soundings, so "
                "channel-index alignment is approximate across soundings.",
    }


# ----------------------------------------------------------------------------- main


def iter_h5(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.h5") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--soundings", type=Path, default=None,
                    help="root of the raw simulated data (cluster directories of .h5)")
    ap.add_argument("--h5", type=Path, default=None, help="a single file to analyse")
    ap.add_argument("--json", type=Path, default=None, help="write the full report here")
    ap.add_argument("--csv", type=Path, default=None,
                    help="write one row per reference sounding here")
    ap.add_argument("--limit", type=int, default=None,
                    help="analyse at most this many files (smoke test)")
    args = ap.parse_args()

    if args.soundings:
        paths = iter_h5(args.soundings)
    elif args.h5:
        paths = [args.h5]
    else:
        paths = [DEFAULT_H5]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit("no .h5 files found")

    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for i, p in enumerate(paths, 1):
        try:
            d = read_file(p)
        except OSError as exc:
            skipped.append(f"{p.name}: {exc}")
            continue
        if d is None:
            skipped.append(f"{p.name}: missing true_noiseless_radiance or too few channels")
            continue
        rows.append(analyse_file(d))
        if i % 25 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)} files", flush=True)

    if not rows:
        raise SystemExit("nothing analysed; "
                         + ("; ".join(skipped[:5]) if skipped else ""))

    by_cluster: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        by_cluster.setdefault(r["cluster"], []).append(r)
    clusters = [aggregate_cluster(v) for _, v in sorted(by_cluster.items(),
                                                        key=lambda kv: (kv[0] is None, kv[0]))]

    blocks = sorted({r["replicates_per_delta_draw"] for r in rows})
    report = {
        "n_files": len(rows),
        "n_skipped": len(skipped),
        "skipped": skipped[:20],
        "replicates_per_delta_draw_observed": blocks,
        "total_distinct_delta_draws": int(sum(r["n_distinct_delta_draws"] for r in rows)),
        "total_replicates": int(sum(r["n_replicates"] for r in rows)),
        "per_cluster": clusters,
        "cluster_comparison": compare_clusters(clusters, by_cluster),
        "pooled_state_dependence": pooled_state_dependence(
            [r["_sd_cors"] for r in rows if r["_sd_cors"] is not None],
            [r["_sd_n"] for r in rows if r["_sd_cors"] is not None]),
        "per_reference_sounding": [
            {k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
    }

    print("\n=== summary ===")
    print(f"files analysed          : {len(rows)}")
    print(f"replicates per delta draw: {blocks}  "
          f"(1 would mean delta is redrawn for every replicate)")
    print(f"distinct delta draws     : {report['total_distinct_delta_draws']} "
          f"across {report['total_replicates']} replicates")
    for c in clusters:
        print(f"  cluster {c['cluster']}: {c['n_reference_soundings']} ref soundings, "
              f"RMS(mu_delta)/sigma_eps = {c['mu_delta_rms_over_sigma_eps']:.3f}, "
              f"Sigma_delta 90% in {c['sigma_delta_pooled'].get('n_pc_for_90pct')} PC, "
              f"mu_delta sounding-specific: {c['mu_delta_is_sounding_specific']}")
    cc = report["cluster_comparison"]
    if cc:
        print(f"mu_delta differs between clusters: {cc['mu_delta_differs_between_clusters']} "
              f"(between-cluster sd {cc['between_cluster_sd_of_mu_delta']:.3f} vs "
              f"MC error {cc['within_cluster_monte_carlo_error']:.3f})")
    ps = report["pooled_state_dependence"]
    if ps:
        print(f"delta independent of the state: {ps['delta_appears_independent_of_state']} "
              f"(smallest combined p over {ps['n_state_elements']} state elements = "
              f"{ps['min_combined_p_value']:.3g} vs Bonferroni threshold "
              f"{ps['bonferroni_threshold']:.3g}; {ps['n_pooled_delta_draws']} draws)")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    if args.csv:
        import csv as _csv
        flat = []
        for r in rows:
            row = {
                "cluster": r["cluster"], "reference_sounding": r["reference_sounding"],
                "replicates_per_delta_draw": r["replicates_per_delta_draw"],
                "n_distinct_delta_draws": r["n_distinct_delta_draws"],
                "mu_delta_rms_over_sigma_eps": r["mu_delta_rms_over_sigma_eps_all_bands"],
                "frac_converged": r.get("frac_converged"),
                "frac_quality_good": r.get("frac_xco2_quality_flag_good"),
            }
            for b in BAND_NAMES:
                row[f"sd_delta_over_sigma_{b}"] = r["sd_delta_over_sigma_eps"].get(b)
                row[f"mu_delta_pct_radiance_{b}"] = r["mu_delta"].get(b, {}).get(
                    "rms_over_radiance_pct")
                row[f"absorbed_{b}"] = r["retrieval_absorption"].get(f"{b}_absorbed_fraction")
            row.update({f"xco2_{k}": v for k, v in r["xco2_error"].items()})
            flat.append(row)
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            wtr = _csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
            wtr.writeheader()
            wtr.writerows(flat)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()

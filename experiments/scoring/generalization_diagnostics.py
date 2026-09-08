"""generalization_diagnostics.py

The generalization diagnostics that do not need a checkpoint: everything that can be
computed from the split table alone. No GPU, no artifacts, ~1 minute.

Three blocks, written as CSV and as a markdown table:

1. **Partition inventory and difficulty.** Reference soundings, soundings and clusters in
   each partition, plus the variance structure of the truth (total / between reference
   sounding / within reference sounding) in each. The gap between a metric on training
   rows and the same metric on held-out rows is only an overfitting measurement if the
   two partitions are equally hard; this block is that control.

2. **Standardization constants.** The mu/sigma that `compute_split_stats.py` writes,
   recomputed here from the training rows, so the validation figures of
   Table~\ref{tab:model_search} -- which are in standardized units -- can be quoted in
   ppm beside the test figures of Table~\ref{tab:splitcompare}.

3. **No-spectrum baselines.** What a model that never looks at a spectrum achieves on the
   held-out reference soundings, fit on the training reference soundings only: the
   training mean, the a-priori column, the a-priori column recalibrated by least squares,
   and the auxiliary covariate vector the network is conditioned on (`scene_prior`), as a
   ridge and as a gradient-boosted tree, each also given the two Gaussian predictive
   densities -- diagonal and correlated -- that make its NLL comparable with the paper's. Every auxiliary input is a property of the
   reference sounding rather than of the individual sounding, so each of these baselines
   is constant across a scene's replicates and cannot beat the within-scene floor. That
   is what makes them the right control: a network that only exploited scene-level
   shortcuts (latitude, geometry, the prior) could not do better than the best of them,
   and could not get below the floor at all.

Metrics follow the paper's conventions: RMSE and MAE in ppm, the placement / within-scene
decomposition of MSE, and bootstrap standard errors resampling reference soundings.

Usage:
    python3 experiments/scoring/generalization_diagnostics.py \
        --split-csv group_split_v1_cov.csv \
        --out experiments/scoring/tables
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PPM = 1.0e6

XCO2_COL = "true_xco2"
NEAR_COLS = [f"true_state_vector_{i:02d}" for i in range(17, 21)]
PROFILE_COLS = [f"true_state_vector_{i:02d}" for i in range(1, 21)]

# The `scene_prior` preset of code/data/loading/covariates.py -- the auxiliary inputs the
# revised models are conditioned on, and nothing else.
AUX_CONT = ["SlrZen_fix", "sec_sza", "vza_deg", "sec_vza", "airmass",
            "prior_psfc_pa", "XCO2Prior", "AODPrior1", "AODPrior2"]
AUX_CAT = ["Footprint_fix", "AerType1_fix", "AerType2_fix"]
GEO_CONT = ["Latitude", "Longitude"]          # the `scene_geo` ablation adds these

ID_COLS = ["clst", "ref", "rep", "split"]


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def _decompose(err: np.ndarray, groups: np.ndarray) -> dict:
    """MSE = mean(e_s^2) + mean((e - e_s)^2), exactly, with e_s the group mean error.

    `err` is [n] or [n, d]; for [n, d] the decomposition is pooled over the d columns,
    which is the paper's `pooled` convention.
    """
    err = np.asarray(err, dtype=np.float64)
    if err.ndim == 1:
        err = err[:, None]
    scene_mean = pd.DataFrame(err).groupby(groups).transform("mean").to_numpy()
    placement = float(np.mean(scene_mean ** 2))
    within = float(np.mean((err - scene_mean) ** 2))
    return dict(rmse=float(np.sqrt(np.mean(err ** 2))),
                mae=float(np.mean(np.abs(err))),
                placement=float(np.sqrt(placement)),
                within=float(np.sqrt(within)),
                bias=float(np.mean(err)))


def _floor(truth: np.ndarray, groups: np.ndarray) -> float:
    """The within-scene spread of the truth: the error of a scene-mean oracle.

    No predictor that is constant inside a reference sounding can beat this, so
    `within / floor < 1` is the signature of per-sounding (i.e. spectral) information.
    """
    truth = np.asarray(truth, dtype=np.float64)
    if truth.ndim == 1:
        truth = truth[:, None]
    scene_mean = pd.DataFrame(truth).groupby(groups).transform("mean").to_numpy()
    return float(np.sqrt(np.mean((truth - scene_mean) ** 2)))


def _gaussian_scores(err: np.ndarray, cov: np.ndarray, sigma_std: np.ndarray) -> dict:
    """NLL (standardized units, per sounding) and 95% coverage for N(pred, cov).

    `cov` is [d, d] in ppm^2 -- one homoscedastic predictive covariance, estimated out of
    fold on training reference soundings -- and `sigma_std` is the standardization
    constant of each output, so that the NLL is `metrics.mvn_nll` on standardized
    coordinates: 0.5 * (d log 2pi + log|C| + z'z), the same quantity as the paper's
    `nll_std` column.

    Both a diagonal and a full covariance are worth scoring, and the difference between
    them is not small on a profile task: the four near-surface levels are strongly
    correlated, so a diagonal Gaussian is a strawman that any joint model beats without
    reading a spectrum. `full` says which one this is.
    """
    err = np.asarray(err, dtype=np.float64)
    if err.ndim == 1:
        err = err[:, None]
    cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    sigma_std = np.atleast_1d(np.asarray(sigma_std, dtype=np.float64))
    d = err.shape[1]
    # standardized coordinates: e / sigma, C / (sigma sigma')
    e = err / sigma_std
    C = cov / np.outer(sigma_std, sigma_std)
    L = np.linalg.cholesky(C)
    z = np.linalg.solve(L, e.T).T
    nll = 0.5 * (d * np.log(2.0 * np.pi) + 2.0 * np.sum(np.log(np.diag(L)))
                 + np.sum(z ** 2, axis=1))
    sd = np.sqrt(np.diag(cov))
    return dict(nll_std=float(nll.mean()),
                cov95_marginal=float(np.mean(np.abs(err / sd) <= 1.959963985)),
                mean_pred_sd=float(sd.mean()))


def _oof_cov(Xtr, ytr, groups_tr, fit, n_folds=5, seed=0):
    """Out-of-fold residual covariance, folds = disjoint reference soundings.

    Fitting the width on in-sample residuals would hand the baseline an advantage the
    trained models do not get, so the width is estimated on training reference soundings
    the baseline's own fit has not seen.
    """
    ytr = np.asarray(ytr, dtype=np.float64)
    single = ytr.ndim == 1
    Y = ytr[:, None] if single else ytr
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups_tr)
    fold_of = {g: int(i % n_folds) for i, g in enumerate(rng.permutation(uniq))}
    resid = np.zeros_like(Y)
    for k in range(n_folds):
        te = np.array([fold_of[g] == k for g in groups_tr])
        tr = ~te
        if te.sum() == 0:
            continue
        pred = fit(Xtr[tr], Y[tr], Xtr[te])
        resid[te] = np.asarray(pred, dtype=np.float64).reshape(te.sum(), Y.shape[1]) - Y[te]
    full = np.cov(resid, rowvar=False, bias=True).reshape(Y.shape[1], Y.shape[1])
    return np.diag(np.diag(full)), full


def _bootstrap(err: np.ndarray, groups: np.ndarray, n_boot: int, seed: int) -> dict:
    """Standard errors of RMSE and MAE, resampling reference soundings with replacement."""
    err = np.asarray(err, dtype=np.float64)
    if err.ndim == 1:
        err = err[:, None]
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in uniq}
    rmse, mae = [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([rows[g] for g in pick])
        e = err[idx]
        rmse.append(np.sqrt(np.mean(e ** 2)))
        mae.append(np.mean(np.abs(e)))
    return dict(rmse_se=float(np.std(rmse, ddof=1)), mae_se=float(np.std(mae, ddof=1)))


# --------------------------------------------------------------------------------------
# design matrices
# --------------------------------------------------------------------------------------
def _design(train: pd.DataFrame, other: pd.DataFrame, cont, cat):
    """Standardized continuous block plus one-hot categoricals, fit on `train` only."""
    Xc_tr = train[cont].astype("float64").to_numpy()
    Xc_ot = other[cont].astype("float64").to_numpy()
    mu, sd = Xc_tr.mean(0), Xc_tr.std(0)
    sd[sd == 0] = 1.0
    blocks_tr, blocks_ot = [(Xc_tr - mu) / sd], [(Xc_ot - mu) / sd]
    for c in cat:
        levels = sorted(train[c].astype(str).unique())
        lut = {lv: i for i, lv in enumerate(levels)}
        for frame, out in ((train, blocks_tr), (other, blocks_ot)):
            codes = frame[c].astype(str).map(lut).to_numpy()
            oh = np.zeros((len(frame), len(levels)), dtype=np.float64)
            seen = ~pd.isna(codes)
            oh[np.flatnonzero(seen), codes[seen].astype(int)] = 1.0
            out.append(oh)
    return np.hstack(blocks_tr), np.hstack(blocks_ot)


def _ridge(Xtr, ytr, Xte, alpha=1.0):
    """Closed-form ridge with an unpenalized intercept; ytr may be [n] or [n, d]."""
    ytr = np.asarray(ytr, dtype=np.float64)
    single = ytr.ndim == 1
    if single:
        ytr = ytr[:, None]
    xm, ym = Xtr.mean(0), ytr.mean(0)
    Xc, Yc = Xtr - xm, ytr - ym
    A = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
    W = np.linalg.solve(A, Xc.T @ Yc)
    pred = (Xte - xm) @ W + ym
    return pred[:, 0] if single else pred


def _gbt(Xtr, ytr, Xte, seed=0):
    """Gradient-boosted trees, one model per output column. Optional dependency."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:
        return None
    ytr = np.asarray(ytr, dtype=np.float64)
    single = ytr.ndim == 1
    if single:
        ytr = ytr[:, None]
    out = np.zeros((Xte.shape[0], ytr.shape[1]), dtype=np.float64)
    for j in range(ytr.shape[1]):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                          max_depth=None, random_state=seed)
        m.fit(Xtr, ytr[:, j])
        out[:, j] = m.predict(Xte)
    return out[:, 0] if single else out


# --------------------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------------------
def inventory(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, sub in df.groupby("split", sort=False):
        g = (sub["clst"].astype(str) + "/" + sub["ref"].astype(str)).to_numpy()
        xco2 = sub[XCO2_COL].to_numpy(dtype=np.float64) * PPM
        near = sub[NEAR_COLS].to_numpy(dtype=np.float64) * PPM
        rows.append(dict(
            partition=name,
            reference_soundings=int(pd.unique(g).size),
            soundings=int(len(sub)),
            clusters=int(sub["clst"].nunique()),
            replicates_per_ref=float(len(sub) / pd.unique(g).size),
            xco2_sd=float(xco2.std(ddof=0)),
            xco2_between=float(np.sqrt(max(xco2.std(ddof=0) ** 2 - _floor(xco2, g) ** 2, 0.0))),
            xco2_within=_floor(xco2, g),
            near_sd=float(near.std(ddof=0)),
            near_within=_floor(near, g),
            xco2_mean=float(xco2.mean()),
            near_mean=float(near.mean()),
        ))
    return pd.DataFrame(rows)


def standardization(df: pd.DataFrame) -> dict:
    tr = df[df["split"] == "train"]
    xco2 = tr[XCO2_COL].to_numpy(dtype=np.float64)
    prof = tr[PROFILE_COLS].to_numpy(dtype=np.float64)
    return dict(
        n_train_rows=int(len(tr)),
        xco2_mu_ppm=float(xco2.mean() * PPM),
        xco2_sigma_ppm=float(xco2.std(ddof=0) * PPM),
        profile_sigma_ppm=[float(v) for v in prof.std(axis=0, ddof=0) * PPM],
        near_sigma_ppm=[float(v) for v in prof[:, 16:20].std(axis=0, ddof=0) * PPM],
    )


def baselines(df: pd.DataFrame, eval_split: str, n_boot: int, seed: int) -> pd.DataFrame:
    train = df[df["split"] == "train"].reset_index(drop=True)
    ev = df[df["split"] == eval_split].reset_index(drop=True)
    groups = (ev["clst"].astype(str) + "/" + ev["ref"].astype(str)).to_numpy()
    groups_tr = (train["clst"].astype(str) + "/" + train["ref"].astype(str)).to_numpy()

    prof_sigma = train[PROFILE_COLS].to_numpy(dtype=np.float64).std(axis=0, ddof=0) * PPM
    tasks = {
        "xco2": (train[XCO2_COL].to_numpy(dtype=np.float64) * PPM,
                 ev[XCO2_COL].to_numpy(dtype=np.float64) * PPM,
                 np.array([train[XCO2_COL].to_numpy(dtype=np.float64).std(ddof=0) * PPM])),
        "last4": (train[NEAR_COLS].to_numpy(dtype=np.float64) * PPM,
                  ev[NEAR_COLS].to_numpy(dtype=np.float64) * PPM,
                  prof_sigma[16:20]),
    }

    X_aux_tr, X_aux_ev = _design(train, ev, AUX_CONT, AUX_CAT)
    X_geo_tr, X_geo_ev = _design(train, ev, AUX_CONT + GEO_CONT, AUX_CAT)
    X_pri_tr, X_pri_ev = _design(train, ev, ["XCO2Prior"], [])

    # (name, training design, eval design, fitter). A `None` fitter means the predictor
    # is not fit at all, so its width is the root mean square of its training residuals.
    ridge_w = lambda a: (lambda Xa, ya, Xb: _ridge(Xa, ya, Xb, alpha=a))
    specs = [
        ("training mean", np.zeros((len(train), 1)), np.zeros((len(ev), 1)), ridge_w(1e-6)),
        ("a-priori column, recalibrated", X_pri_tr, X_pri_ev, ridge_w(1e-6)),
        ("auxiliary inputs, ridge", X_aux_tr, X_aux_ev, ridge_w(10.0)),
        ("auxiliary inputs, gradient-boosted trees", X_aux_tr, X_aux_ev,
         (lambda Xa, ya, Xb: _gbt(Xa, ya, Xb, seed=seed))),
        ("auxiliary inputs + lat/lon, ridge", X_geo_tr, X_geo_ev, ridge_w(10.0)),
        ("auxiliary inputs + lat/lon, gradient-boosted trees", X_geo_tr, X_geo_ev,
         (lambda Xa, ya, Xb: _gbt(Xa, ya, Xb, seed=seed))),
    ]

    rows = []
    for task, (ytr, yev, sigma_std) in tasks.items():
        floor = _floor(yev, groups)
        entries = []

        if task == "xco2":
            # the a-priori column itself, no fitting at all
            pri_ev = ev["XCO2Prior"].to_numpy(dtype=np.float64)
            pri_tr = train["XCO2Prior"].to_numpy(dtype=np.float64)
            c = np.array([[np.mean((pri_tr - ytr) ** 2)]])
            entries.append(("a-priori column", pri_ev, c, c))

        for name, Xa, Xb, fit in specs:
            if fit is None:
                continue
            pred = fit(Xa, ytr, Xb)
            if pred is None:                      # sklearn missing; skip that row
                continue
            diag, full = _oof_cov(Xa, ytr, groups_tr, fit, seed=seed)
            entries.append((name, pred, diag, full))

        for name, pred, diag, full in entries:
            pred = np.asarray(pred, dtype=np.float64)
            err = (pred.reshape(yev.shape) if pred.size == yev.size else pred) - yev
            m = _decompose(err, groups)
            m.update(_bootstrap(err, groups, n_boot=n_boot, seed=seed))
            g_diag = _gaussian_scores(err, diag, sigma_std)
            g_full = _gaussian_scores(err, full, sigma_std)
            m.update(g_diag)
            # The joint Gaussian with the residual correlations kept. On the scalar task
            # it is the same number; on the four near-surface levels it is 4-5 nats
            # better, and it is the version the flow's joint NLL has to beat.
            m["nll_std_correlated"] = g_full["nll_std"]
            rows.append(dict(task=task, eval_split=eval_split, baseline=name,
                             n=int(len(ev)),
                             reference_soundings=int(pd.unique(groups).size),
                             floor=floor, within_over_floor=m["within"] / floor,
                             rmse_over_floor=m["rmse"] / floor, **m))
    return pd.DataFrame(rows)


def markdown(inv: pd.DataFrame, std: dict, base: pd.DataFrame) -> str:
    L = ["# Generalization diagnostics from the split table", "",
         "Computed by `experiments/scoring/generalization_diagnostics.py`; no checkpoint, "
         "no GPU. Every figure in ppm.", "", "## 1. Partitions and their difficulty", ""]
    cols = ["partition", "reference_soundings", "soundings", "clusters",
            "xco2_sd", "xco2_between", "xco2_within", "near_sd", "near_within"]
    L.append("| " + " | ".join(cols) + " |")
    L.append("|" + "---|" * len(cols))
    for _, r in inv.iterrows():
        L.append("| " + " | ".join(
            f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    L += ["", "## 2. Standardization constants, from training rows only", "",
          f"- XCO2: mu = {std['xco2_mu_ppm']:.3f} ppm, sigma = {std['xco2_sigma_ppm']:.3f} ppm",
          f"- near-surface sigma (levels 17-20): "
          + ", ".join(f"{v:.2f}" for v in std["near_sigma_ppm"]) + " ppm",
          f"- training rows: {std['n_train_rows']}",
          "",
          "A validation figure in standardized units multiplies by sigma to become ppm.",
          "", "## 3. What a model with no spectrum achieves on held-out scenes", ""]
    cols = ["task", "eval_split", "baseline", "rmse", "rmse_se", "mae", "bias",
            "placement", "within", "floor", "within_over_floor",
            "nll_std", "nll_std_correlated", "cov95_marginal", "mean_pred_sd"]
    L.append("| " + " | ".join(cols) + " |")
    L.append("|" + "---|" * len(cols))
    for _, r in base.iterrows():
        L.append("| " + " | ".join(
            f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    L += ["", "`within_over_floor` is the within-scene error divided by the within-scene "
          "spread of the truth. It is 1.0 for every predictor that is constant inside a "
          "reference sounding, which every baseline here is; a trained model gets below it "
          "only by reading the individual spectrum.",
          "",
          "`nll_std` scores each baseline as a Gaussian centred on its prediction with a "
          "diagonal, homoscedastic covariance estimated out of fold on training reference "
          "soundings; `nll_std_correlated` keeps that covariance's off-diagonal. They are "
          "the same number on the scalar task (d = 1). On the four near-surface levels the "
          "diagonal version is a strawman -- the levels are strongly correlated, so any "
          "joint model beats it without reading a spectrum -- and the correlated one is "
          "what a joint predictive density has to beat.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("experiments/scoring/tables"))
    ap.add_argument("--eval-splits", nargs="+", default=["test", "val"])
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # The split table is a Zenodo item, not a repository file: a checkout that has only
    # downloaded the scoring artifacts does not have it, and that is a normal state rather
    # than a broken one. Saying so beats a pandas traceback -- and it used to be one,
    # which also took the three readers after this one down with it.
    if not args.split_csv.is_file():
        raise SystemExit("\n".join([
            f"{args.split_csv}: not found.",
            "  The split table is not in the repository (69 MB). Fetch it with",
            "      ./scripts/get_data.sh splits",
            "  or point --split-csv at a table built by",
            "  experiments/pipelines/make_group_split.py.",
        ]))

    need = set(ID_COLS + [XCO2_COL] + PROFILE_COLS + AUX_CONT + AUX_CAT + GEO_CONT)
    head = pd.read_csv(args.split_csv, nrows=0)
    missing = need - set(head.columns)
    if missing:
        raise KeyError(f"{args.split_csv} is missing columns: {sorted(missing)}")
    df = pd.read_csv(args.split_csv, usecols=sorted(need))

    args.out.mkdir(parents=True, exist_ok=True)
    inv = inventory(df)
    std = standardization(df)
    base = pd.concat([baselines(df, s, args.n_boot, args.seed) for s in args.eval_splits],
                     ignore_index=True)

    inv.to_csv(args.out / "generalization_partitions.csv", index=False)
    base.to_csv(args.out / "generalization_baselines.csv", index=False)
    (args.out / "generalization_standardization.json").write_text(json.dumps(std, indent=2))
    (args.out / "generalization_diagnostics.md").write_text(markdown(inv, std, base))
    print(markdown(inv, std, base))


if __name__ == "__main__":
    main()

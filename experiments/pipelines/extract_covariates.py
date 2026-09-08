"""extract_covariates.py
Build a per-sounding covariate table from the raw .h5 soundings.

Why this exists
---------------
`clean_data.py` carries four variables into `aux_data.csv` (Footprint, SlrZen, AerType1,
AerType2, plus lat/lon as diagnostics) and drops everything else the delivered files
contain. Several of the things it drops are genuine *pre-retrieval* covariates -- known
before ACOS runs, and known operationally -- and the manuscript claims two of them
(viewing geometry, prior surface pressure) are already inputs. They are not, but they are
recoverable:

  * `FilterVariables/airmass` is constant across the 100 replicates of a reference
    sounding, i.e. it is pure observing geometry: airmass = 1/cos(SZA) + 1/cos(VZA).
    With SlrZen known, VZA follows.
  * `retrieved_psfc - 100 * delta_psfc` is constant across the 100 replicates to within
    2 mPa, so it is the meteorological *prior* surface pressure ACOS was given, not a
    retrieved quantity.

This script writes one row per sounding with every candidate covariate, each tagged in the
sidecar JSON by when it becomes available (`prior`, `l1b`, `acos`, `simulation`), so that
an experiment can never quietly condition on something a real retrieval would not have.

It does NOT touch the .npy spectra: adding covariates costs one small CSV and a merge into
the split table (`augment_split.py`), not a reprocessing run.

Usage
-----
    python3 experiments/pipelines/extract_covariates.py \
        --raw-root /workspace/data/oco2/simulated/og_data \
        --out      /workspace/project/scratch/simulated/files/covariates_v1.csv.gz

    # verify the row alignment against the published annotations
    python3 experiments/pipelines/extract_covariates.py ... \
        --verify-against /workspace/project/scratch/simulated/files/train/aux_data.csv \
        --verify-against /workspace/project/scratch/simulated/files/test/aux_data.csv

    # inventory only: which variables are per-scene, which vary per replicate
    python3 experiments/pipelines/extract_covariates.py --raw-root ... --inventory-only
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# IDs must match oco2_utils.add_uniq_id exactly, or the merge in augment_split.py silently
# drops rows: clst%04d_ref%04d_rep%04d, replicate index 1-based in file order.
_PROP_RE = "{prop}([0-9]+)"

# ---------------------------------------------------------------------------------
# What we pull out of each file.
#
# availability:
#   prior      -- known before the retrieval runs (geometry, meteorology, aerosol priors)
#   l1b        -- delivered with the radiances
#   acos       -- an output of the operational retrieval; using it makes the model a
#                 post-processor of ACOS rather than an independent retrieval
#   simulation -- an artefact of how the ensemble was built; never available anywhere
# ---------------------------------------------------------------------------------
SCALARS = {
    # group/name                                   availability
    "FilterVariables/airmass":                     "prior",
    "FilterVariables/windspeed":                   "prior",
    "FilterVariables/albedo_blended":              "acos",
    "FilterVariables/albedo_strong_co2":           "acos",
    "FilterVariables/albedo_weak_co2":             "acos",
    "FilterVariables/aod_dust":                    "acos",
    "FilterVariables/aod_ice":                     "acos",
    "FilterVariables/aod_sulfate":                 "acos",
    "FilterVariables/co2_grad_del":                "acos",
    "FilterVariables/delta_psfc":                  "acos",
    "FilterVariables/dof_co2":                     "acos",
    "FilterVariables/log_dws":                     "acos",
    "FilterVariables/reduced_chi_squared_o2":      "acos",
    "SpectralParameters/reduced_chi_squared_o2":   "acos",
    "SpectralParameters/reduced_chi_squared_weak_co2":   "acos",
    "SpectralParameters/reduced_chi_squared_strong_co2": "acos",
    "RetrievalResults/iterations":                 "acos",
    "RetrievalResults/diverging_steps":            "acos",
    "RetrievalResults/outcome_flag":               "acos",
    "RetrievalResults/xco2_quality_flag":          "acos",
    "RetrievalResults/retrieved_xco2":             "acos",
    "RetrievalResults/retrieved_xco2_variance":    "acos",
    "RetrievalResults/retrieved_xco2_variance_noise":  "acos",
    "RetrievalResults/retrieved_xco2_variance_smooth": "acos",
    "RetrievalResults/retrieved_xco2_variance_interf": "acos",
    "StateVector/mixture_component":               "simulation",
}

# Elements of the 49-d retrieved state vector worth carrying: everything ACOS retrieves
# that is NOT the CO2 profile. These are the "interferents" whose errors alias into XCO2,
# and they are the natural feature set for an ACOS-informed (post-processing) variant.
# 1-based indices into RetrievalResults/retrieved_state_vector.
RETRIEVED_STATE_ELEMENTS = {
    21: "retr_h2o_scale",
    22: "retr_psfc_pa",
    23: "retr_temp_offset_k",
    39: "retr_brdf_o2_1",
    40: "retr_brdf_o2_2",
    41: "retr_brdf_wco2_1",
    42: "retr_brdf_wco2_2",
    43: "retr_brdf_sco2_1",
    44: "retr_brdf_sco2_2",
    45: "retr_disp_o2",
    46: "retr_disp_wco2",
    47: "retr_disp_sco2",
    48: "retr_fluor_1",
    49: "retr_fluor_2",
}

# Element 22 of the 49-element state vector, 1-based: surface pressure in Pascals. Used
# only to recover prior_psfc_pa on a delivery that ships no delta_psfc; see read_one_file.
TRUE_PSFC_ELEMENT = 22

BANDS = (("b1", 0.0, 1.2), ("b2", 1.2, 1.8), ("b3", 1.8, 9.9))

# Covariates encoded as levels rather than numbers. Kept in step with
# augment_split.KNOWN_CATEGORICAL, which is what the loader actually uses; this copy only
# labels the inventory, and duplicating six names is cheaper than importing a script that
# lives outside the package tree.
CATEGORICAL = {"Footprint", "AerType1", "AerType2", "Region",
               # The corrected copies are categorical too. augment_split.py already knows
               # this (its KNOWN_CATEGORICAL lists them, and the encoder builds embedding
               # tables from that), so leaving them out here mislabelled only the printed
               # inventory -- but an inventory that calls Footprint_fix continuous is
               # exactly the kind of thing that gets believed later.
               "Footprint_fix", "AerType1_fix", "AerType2_fix",
               "outcome_flag", "xco2_quality_flag", "mixture_component"}

# A covariate whose within-scene spread is this small relative to its magnitude is
# constant to float32 precision, and is therefore a property of the scene rather than of
# the individual sounding. Prior surface pressure, reconstructed as
# retrieved_psfc - 100*delta_psfc, lands at ~1e-8 relative -- which is what identifies it
# as the met-analysis prior and not a retrieval output.
CONSTANT_REL_TOL = 1e-6

FILL = -9999.0

# Soundings whose whole band is fill, counted rather than warned about, per band.
_EMPTY_BANDS: Dict[str, int] = {}


def get_property(path: Path, prop: str = "ref") -> int:
    return int(re.search(_PROP_RE.format(prop=prop), path.stem).group(1))


def _clean(a: np.ndarray, dset: Optional[h5py.Dataset] = None) -> np.ndarray:
    """Fill values -> NaN, so a missing entry cannot masquerade as a number in a regression.

    -9999 is the declared fill on the floating-point datasets and is the only one February
    ever needed, because clean_data.py dropped the replicates whose truth was fill before
    the covariates were read. The February *extension* ships those replicates instead
    (`outcome_flag == 5`), and `iterations` and `diverging_steps` declare a fill of **-99**,
    not -9999 -- so a -9999-only test lets 311 of them through as the number -99, which is
    a real value for neither. Where the dataset is available its own `_FillValue` is
    honoured; the -9999 test stays as the fallback for the arrays passed in bare.
    """
    a = np.asarray(a, dtype=np.float64)
    bad = np.isclose(a, FILL)
    if dset is not None:
        for attr in ("_FillValue", "missing_value"):
            v = dset.attrs.get(attr)
            if v is not None:
                bad |= np.isclose(a, float(np.asarray(v).ravel()[0]))
    return np.where(bad, np.nan, a)


def read_one_file(path: Path, clst: int, ref: int,
                  recover_prior_psfc: bool = False,
                  rep_offset: int = 0) -> pd.DataFrame:
    """Every candidate covariate for the replicates of one reference sounding.

    `rep_offset` must equal the one `prepare_month.py` was run with, or the merge in
    `augment_split.py` -- which is keyed on ID -- drops every row of this delivery
    silently. See prepare_month.read_reference for why the offset exists at all.
    """
    out: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        n = f["RetrievalHeader/sounding_id"].shape[0]
        out["sounding_id"] = f["RetrievalHeader/sounding_id"][:]

        for key, _avail in SCALARS.items():
            if key not in f:
                continue
            name = key.split("/")[-1]
            if key.startswith("SpectralParameters/"):
                name = "sp_" + name          # collides with the FilterVariables copy
            out[name] = _clean(f[key][:], f[key])

        sv = _clean(f["RetrievalResults/retrieved_state_vector"][:])
        for idx, name in RETRIEVED_STATE_ELEMENTS.items():
            out[name] = sv[:, idx - 1]

        # ---- derived, pre-retrieval -------------------------------------------------
        # ACOS's a-priori surface pressure. delta_psfc is (retrieved - prior) in hPa;
        # the reconstruction is constant across replicates to ~2 mPa, which is what
        # identifies it as the met-analysis prior rather than a retrieval output.
        out["prior_psfc_reconstructed"] = np.zeros(n)
        if "delta_psfc" in out:
            out["prior_psfc_pa"] = out["retr_psfc_pa"] - 100.0 * out["delta_psfc"]
        elif recover_prior_psfc:
            # The February extension does not ship delta_psfc, and prior_psfc_pa is a
            # continuous input of every non-legacy preset in covariates.py. The truth is
            # drawn around the meteorological prior, so the replicate mean of the true
            # surface pressure estimates it with Monte Carlo error only: checked against
            # the delivered value on the 34 scenes present in both February deliveries,
            # the recovery is unbiased to 0.4 Pa with an RMSE of 50 Pa against a
            # scene-to-scene spread of 15.9 kPa (survey_extension.py).
            #
            # It is a *scene-level* summary of the truth, identical for every replicate,
            # exactly as the delivered quantity is -- but it is still built from the
            # truth, so it is flagged in `prior_psfc_reconstructed` and must not be used
            # for a scene that is being held out. Off by default for that reason.
            psfc = _clean(f["StateVector/true_state_vector"][:, TRUE_PSFC_ELEMENT - 1])
            out["prior_psfc_pa"] = np.full(n, np.nanmean(psfc))
            out["prior_psfc_reconstructed"] = np.ones(n)

        # Pressure weighting function: XCO2 = h . profile, exactly (checked to float32
        # precision). Determined by the pressure grid, hence a prior quantity.
        if "RetrievalResults/xco2_pressure_weighting_function" in f:
            pwf = _clean(f["RetrievalResults/xco2_pressure_weighting_function"][:])
            for j in range(pwf.shape[1]):
                out[f"pwf_{j + 1:02d}"] = pwf[:, j]

        # ---- derived, L1B -----------------------------------------------------------
        rad = _clean(f["SpectralParameters/measured_radiance"][:])
        unc = _clean(f["SpectralParameters/measured_radiance_uncert"][:])
        wl = _clean(f["SpectralParameters/wavelength"][:])
        with np.errstate(invalid="ignore", divide="ignore"):
            snr = rad / unc
        for name, lo, hi in BANDS:
            m = (wl[0] > lo) & (wl[0] < hi) & np.isfinite(wl[0])
            if not m.any():
                continue
            # Some soundings have an entire band filled with the fill value, and
            # nanmedian/nanmax of an all-NaN slice warns once per call -- thousands of
            # identical RuntimeWarnings that bury the two findings this script exists to
            # report. NaN is the right answer for those rows (augment_split imputes them
            # at the training median), so the warning is suppressed and counted instead.
            band_nan = int(np.isnan(rad[:, m]).all(axis=1).sum())
            if band_nan:
                _EMPTY_BANDS[name] = _EMPTY_BANDS.get(name, 0) + band_nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[f"snr_{name}"] = np.nanmedian(snr[:, m], axis=1)
                out[f"continuum_{name}"] = np.nanmax(rad[:, m], axis=1)

    df = pd.DataFrame(out)
    rep = np.arange(1, n + 1) + rep_offset
    df.insert(0, "ID", [f"clst{clst:04d}_ref{ref:04d}_rep{i:04d}" for i in rep])
    df.insert(1, "clst", clst)
    df.insert(2, "ref", ref)
    df.insert(3, "rep", rep)
    return df


# The cluster metadata columns worth carrying, beyond the four the pipeline already reads.
# XCO2Prior is the ACOS a-priori column amount for the scene: prior tier, known before the
# retrieval runs, and the single piece of information ACOS has that the network was never
# given. AODPrior1/2 are the prior aerosol optical depths that go with AerType1/2.
# SoundingID, Orbit and Cluster are identifiers -- carried for provenance, never as inputs.
CSV_METADATA_KEEP = ("SlrZen", "Footprint", "AerType1", "AerType2", "Latitude", "Longitude",
                     "XCO2Prior", "AODPrior1", "AODPrior2", "Region",
                     "SoundingID", "Orbit", "Cluster")

# The metadata columns the published aux_data.csv also contains, and which therefore need
# a suffix to sit beside the corrected ones without a collision.
CORRECTED_SUFFIX = "_fix"
PUBLISHED_ALSO_HAS = ("SlrZen", "Footprint", "AerType1", "AerType2")

PAIRINGS = ("sorted_ref", "iterdir", "sorted_name")


def read_cluster(cluster_dir: Path, pairing: str = "sorted_ref",
                 recover_prior_psfc: bool = False,
                 rep_offset: int = 0) -> pd.DataFrame:
    """One cluster: the .h5 contents, plus the cluster metadata under a chosen pairing.

    Which row of the cluster's .csv belongs to which .h5 file is the whole question here.
    `oco2_utils.get_cluster` answers it with `Path.iterdir()` order, which is neither
    sorted nor stable, and `diagnose_metadata_pairing.py` shows what that costs: the
    published pairing puts 49.0 % of soundings at sec(VZA) < 1, which is geometrically
    impossible and statistically indistinguishable from a random permutation (null 46.6 -
    49.7 %). Sorting the files by reference-sounding number puts it at **0.000 in all nine
    clusters** -- 451 independent checks, none failing. That is the pairing the metadata
    was written in.

    So `sorted_ref` is the default and is what the corrected columns use. `iterdir` is kept
    only to reproduce the published (scrambled) assignment on demand.

    Nothing read out of the .h5 is affected by any of this: airmass, the filter variables,
    the retrieved state and the radiances are keyed by the cluster and reference numbers in
    the *filename*, exactly as `add_uniq_id` builds the ID.
    """
    if pairing not in PAIRINGS:
        raise ValueError(f"unknown pairing '{pairing}' (expected one of {PAIRINGS})")
    clst = get_property(cluster_dir, "clst") if re.search(r"clst[0-9]+", cluster_dir.name)         else int(re.search(r"([0-9]+)", cluster_dir.name).group(1))

    meta = None
    for p in cluster_dir.iterdir():
        if p.suffix == ".csv":
            meta = pd.read_csv(str(p))
    if meta is None:
        raise FileNotFoundError(f"no metadata .csv in {cluster_dir}")

    files = [f for f in cluster_dir.iterdir() if f.suffix != ".csv"]
    if pairing == "sorted_ref":
        files = sorted(files, key=lambda f: get_property(f, "ref"))
    elif pairing == "sorted_name":
        files = sorted(files, key=lambda f: f.name)
    # `iterdir` leaves the order alone, which is the point of it.

    if len(files) != len(meta):
        logger.warning("cluster %d: %d .h5 files but %d metadata rows; the pairing cannot "
                       "be right for every file", clst, len(files), len(meta))

    frames: List[pd.DataFrame] = []
    for i, sounding in enumerate(files):
        ref = get_property(sounding, "ref")
        df = read_one_file(sounding, clst, ref, recover_prior_psfc, rep_offset)
        row = meta.iloc[i] if i < len(meta) else None
        for col in CSV_METADATA_KEEP:
            if col not in meta.columns:
                continue
            name = col + CORRECTED_SUFFIX if col in PUBLISHED_ALSO_HAS else col
            df[name] = row[col] if row is not None else np.nan
        df["meta_row"] = i
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def attach_published_metadata(df: pd.DataFrame, aux_paths: List[Path]) -> pd.DataFrame:
    """Carry the PUBLISHED scene metadata alongside the corrected one, keyed by ID.

    These are the values every released checkpoint was trained on. They are also, as
    diagnose_metadata_pairing.py shows, attached to the wrong scenes -- so they are kept
    under their plain names purely so `covariates=legacy` reproduces the published models
    and the `_base2` control means what it says. Anything new should use the *_fix columns.
    """
    aux = pd.concat([pd.read_csv(q) for q in aux_paths], ignore_index=True)
    if "ID" not in aux.columns:
        raise SystemExit(f"{aux_paths[0]} has no ID column")
    have = [c for c in PUBLISHED_ALSO_HAS if c in aux.columns]
    out = df.merge(aux[["ID"] + have], on="ID", how="left", validate="one_to_one")
    n_unmatched = int(out[have[0]].isna().sum()) if have else len(out)
    if n_unmatched:
        logger.warning("%d of %d soundings have no row in the published annotations. "
                       "clean_data.py drops soundings with missing radiances, so a small "
                       "number is expected; a large number is not.", n_unmatched, len(out))
    return out


def compare_metadata_sources(df: pd.DataFrame) -> Dict[str, object]:
    """How far the published assignment is from the corrected one, column by column."""
    report: Dict[str, object] = {}
    for col in PUBLISHED_ALSO_HAS:
        a, b = col + CORRECTED_SUFFIX, col
        if a not in df.columns or b not in df.columns:
            continue
        x, y = df[a], df[b]
        both = x.notna() & y.notna()
        if not both.any():
            continue
        if x.dtype.kind in "ifc" and y.dtype.kind in "ifc":
            same = np.isclose(x[both].astype(float), y[both].astype(float),
                              rtol=1e-5, atol=1e-8)
        else:
            same = x[both].astype(str).to_numpy() == y[both].astype(str).to_numpy()
        report[col] = {"n_compared": int(both.sum()),
                       "n_disagree": int((~same).sum()),
                       "fraction_agree": float(same.mean())}
    return report


def airmass_feasibility(df: pd.DataFrame, sza_col: str) -> Dict[str, float]:
    """Fraction of soundings for which sec(VZA) = airmass - sec(SZA) is >= 1.

    The one physical check available on the pairing. airmass comes from the .h5 and is
    right whatever the pairing does, so an SZA attached to the wrong sounding violates it
    for about half the soundings -- which is what the published assignment does.
    """
    if sza_col not in df.columns or "airmass" not in df.columns:
        return {}
    sec_sza = 1.0 / np.cos(np.radians(df[sza_col].to_numpy(dtype=float)))
    sec_vza = df["airmass"].to_numpy(dtype=float) - sec_sza
    # A row with no airmass is not a failed check, it is an absent one -- the failed
    # retrievals, which the February extension ships and every earlier delivery dropped.
    # The fraction is over the rows the check can actually see; `n_unresolvable` says how
    # many it could not, so a delivery that is mostly fill cannot report 100 % of nothing.
    resolvable = np.isfinite(sec_vza)
    ok = resolvable & (sec_vza >= 1.0)
    n_res = int(resolvable.sum())
    frac = float(ok.sum() / n_res) if n_res else float("nan")
    return {"n": int(len(sec_vza)), "n_resolvable": n_res,
            "n_unresolvable": int(len(sec_vza) - n_res),
            "fraction_feasible": frac, "fraction_infeasible": 1.0 - frac}


def check_metadata_alignment(df: pd.DataFrame) -> Dict[str, object]:
    """Is the i-th file in directory order really the i-th row of the cluster metadata?

    `Path.iterdir()` returns directory order, which is not sorted. If the metadata .csv is
    ordered by reference sounding and the directory is not, every scene in the published
    aux_data.csv carries another scene's geometry. Nothing downstream would notice.
    """
    per_cluster = {}
    ok = True
    for clst, sub in df.groupby("clst"):
        pairs = sub.drop_duplicates(subset=["ref"])[["meta_row", "ref"]]
        monotone = bool((np.diff(pairs.sort_values("meta_row")["ref"].to_numpy()) > 0).all())
        per_cluster[int(clst)] = {
            "n_reference_soundings": int(len(pairs)),
            "directory_order_is_ref_sorted": monotone,
            "first_pairs": pairs.head(5).to_dict("records"),
        }
        ok = ok and monotone
    return {"all_clusters_ref_sorted": bool(ok), "per_cluster": per_cluster}


def verify_against(df: pd.DataFrame, aux_paths: List[Path]) -> Dict[str, object]:
    """Compare the metadata we re-derived with what the published aux_data.csv holds."""
    aux = pd.concat([pd.read_csv(p) for p in aux_paths], ignore_index=True)
    merged = df.merge(aux, on="ID", how="inner", suffixes=("", "_published"))
    report: Dict[str, object] = {
        "n_rows_extracted": int(len(df)),
        "n_rows_published": int(len(aux)),
        "n_matched_ids": int(len(merged)),
    }
    for col in ("SlrZen", "Footprint", "AerType1", "AerType2", "Latitude", "Longitude"):
        pub = f"{col}_published"
        if col not in merged.columns or pub not in merged.columns:
            continue
        a, b = merged[col], merged[pub]
        if a.dtype.kind in "fc":
            agree = np.isclose(a.astype(float), b.astype(float),
                               rtol=1e-5, atol=1e-8, equal_nan=True)
        else:
            agree = a.astype(str).to_numpy() == b.astype(str).to_numpy()
        report[f"agree_{col}"] = float(np.mean(agree))
    return report


def inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Per-scene or per-replicate? The one fact that decides how a covariate may be used.

    A covariate constant across the 100 replicates of a reference sounding describes the
    *scene*. One that varies within a scene is either an L1B quantity (noise, radiance) or
    an output of the retrieval, which responds to the state and the discrepancy draw.
    """
    rows = []
    grouped = df.groupby(["clst", "ref"])
    avail = {k.split("/")[-1]: v for k, v in SCALARS.items()}
    for col in df.columns:
        if col in ("ID", "clst", "ref", "rep", "meta_row", "sounding_id"):
            continue
        s = df[col]
        if s.dtype.kind not in "ifb" or col in CATEGORICAL:
            rows.append(dict(covariate=col, kind="categorical",
                             n_unique=int(s.nunique()),
                             within_scene_max_unique=int(grouped[col].nunique().max()),
                             constant_within_scene=bool(grouped[col].nunique().max() <= 1),
                             availability=avail.get(col, "prior")))
            continue
        nun = grouped[col].nunique()
        with np.errstate(invalid="ignore"):
            within = grouped[col].std().mean()
            between = grouped[col].mean().std()
            scale = max(abs(float(np.nanmean(s))), 1e-30)
        rel = float(within) / scale if np.isfinite(within) else np.nan
        rows.append(dict(
            covariate=col, kind="continuous", n_unique=int(s.nunique()),
            within_scene_max_unique=int(nun.max()),
            # Exact equality is the wrong test for a float32 quantity reconstructed by
            # arithmetic: prior_psfc_pa varies by 8 mPa in 99 kPa across a scene and would
            # read as "varies per sounding", which is the opposite of what it is.
            constant_within_scene=bool(np.isfinite(rel) and rel < CONSTANT_REL_TOL),
            sd_within_scene=float(within) if np.isfinite(within) else np.nan,
            rel_sd_within_scene=rel,
            sd_between_scenes=float(between) if np.isfinite(between) else np.nan,
            availability=avail.get(col, "prior"),
        ))
    inv = pd.DataFrame(rows)
    # Derived columns are not in SCALARS; label them here rather than guess above.
    derived = {"prior_psfc_pa": "prior", "vza_deg": "prior", "sza_deg": "prior",
               # 1 where prior_psfc_pa was rebuilt from the truth rather than delivered.
               # "simulation" so that a run which conditions on it cannot claim otherwise.
               "prior_psfc_reconstructed": "simulation"}
    for c in inv["covariate"]:
        if c.startswith("pwf_"):
            derived[c] = "prior"
        elif c.startswith(("snr_", "continuum_")):
            derived[c] = "l1b"
        elif c.startswith("retr_"):
            derived[c] = "acos"
    inv["availability"] = [derived.get(c, a) for c, a in
                           zip(inv["covariate"], inv["availability"])]
    return inv.sort_values(["availability", "covariate"]).reset_index(drop=True)


def add_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Recover the viewing zenith angle from airmass and the solar zenith angle.

    airmass = sec(SZA) + sec(VZA). SlrZen comes from the cluster metadata; anything the
    inversion cannot resolve (sec(VZA) < 1, i.e. airmass below sec(SZA) + 1) is left NaN
    rather than clipped, so a bad assumption shows up as missing data instead of a
    plausible angle.
    """
    sza_col = "SlrZen" + CORRECTED_SUFFIX
    if sza_col not in df.columns or "airmass" not in df.columns:
        logger.warning("%s or airmass absent; skipping viewing-geometry recovery", sza_col)
        return df
    sza = df[sza_col].to_numpy(dtype=float)
    am = df["airmass"].to_numpy(dtype=float)
    sec_sza = 1.0 / np.cos(np.radians(sza))
    sec_vza = am - sec_sza
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_vza = 1.0 / sec_vza
    bad = ~((sec_vza >= 1.0) & np.isfinite(sec_vza))
    cos_vza = np.where(bad, np.nan, cos_vza)
    df["sza_deg"] = sza
    df["vza_deg"] = np.degrees(np.arccos(np.clip(cos_vza, -1.0, 1.0)))
    df["sec_sza"] = sec_sza
    df["sec_vza"] = sec_vza

    # This doubles as an independent test of the scene metadata alignment, and it is the
    # only one available. `--verify-against` compares this table with aux_data.csv, but
    # both come from the same file-to-metadata pairing, so it cannot detect a pairing that
    # is wrong in the same way twice. The airmass check can: airmass is read from the .h5
    # and SlrZen from the cluster .csv, so if the two are being matched up wrongly, some
    # scenes get sec(VZA) = airmass - sec(SZA) < 1, which no geometry produces.
    # Two different things live in `bad`, and printing them as one number is what made a
    # clean pairing look like a broken one on the February extension. A row whose airmass
    # is FILL cannot be checked -- there is nothing to compare SlrZen against, and the row
    # is a failed retrieval that gets dropped before training anyway. A row with a finite
    # airmass that is too small is the real alarm, because no geometry produces it.
    #
    # The extension is the first delivery to ship its failed retrievals rather than drop
    # them: 311 of 24,000 rows carry airmass = -9999, and they were being counted as
    # "geometrically impossible" and reported as a metadata-pairing failure. Per scene, on
    # the usable rows, that delivery is 0.000 infeasible against a 0.491 random-permutation
    # null in both clusters.
    unresolvable = ~np.isfinite(sec_vza)
    n_missing = int(unresolvable.sum())
    n_bad = int((bad & ~unresolvable).sum())
    if n_missing:
        logger.info(
            "%d of %d soundings (%.1f%%) have no airmass, so sec(VZA) cannot be formed "
            "for them and vza_deg is NaN. These are the fill-valued rows -- failed "
            "retrievals -- and they carry no spectrum either, so they are dropped before "
            "training. They are NOT evidence about the metadata pairing.",
            n_missing, len(df), 100.0 * n_missing / max(len(df), 1))
    if n_bad:
        logger.warning(
            "%d of %d soundings (%.1f%%) give sec(VZA) = airmass - sec(SZA) < 1, which is "
            "geometrically impossible. Either the cluster metadata is paired with the "
            "wrong .h5 files, or `airmass` is not sec(SZA)+sec(VZA) in this ensemble. "
            "vza_deg is NaN for those rows.", n_bad, len(df), 100.0 * n_bad / len(df))
    vza = df["vza_deg"].dropna()
    if len(vza):
        logger.info("recovered viewing zenith angle: min %.2f, median %.2f, max %.2f deg "
                    "(%d/%d resolved)", vza.min(), vza.median(), vza.max(), len(vza), len(df))
    return df


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-root", type=Path,
                   default=Path("/workspace/data/oco2/simulated/og_data"),
                   help="directory of cluster subdirectories of .h5 soundings")
    p.add_argument("--out", type=Path,
                   default=Path("/workspace/project/scratch/simulated/files/covariates_v1.csv.gz"))
    p.add_argument("--rep-offset", type=int, default=0,
                   help="add this to every replicate index in the ID. MUST equal the "
                        "value prepare_month.py was run with for the same delivery, or "
                        "augment_split.py's ID-keyed merge drops every row.")
    p.add_argument("--recover-prior-psfc", action="store_true",
                   help="if the delivery ships no FilterVariables/delta_psfc (the February "
                        "extension does not), rebuild prior_psfc_pa as the replicate mean "
                        "of the true surface pressure and set prior_psfc_reconstructed=1. "
                        "Unbiased to 0.4 Pa, RMSE 50 Pa, against a 15.9 kPa scene-to-scene "
                        "spread -- but built from the truth, so never for a held-out scene.")
    p.add_argument("--pairing", default="sorted_ref", choices=list(PAIRINGS),
                   help="how to match the cluster .csv rows to the .h5 files. "
                        "'sorted_ref' is the only one consistent with the airmass "
                        "geometry (0 %% impossible, against a ~49 %% random-permutation "
                        "null) and is what the corrected *_fix columns use. 'iterdir' "
                        "reproduces the scrambled published assignment.")
    p.add_argument("--metadata-from", type=Path, action="append", default=[],
                   help="published aux_data.csv to take the scene metadata (SlrZen, "
                        "Footprint, AerType1/2, Latitude, Longitude) from, keyed by ID. "
                        "STRONGLY RECOMMENDED, and used by chtc/covariates.sh: the "
                        "cluster-.csv pairing is directory-order dependent and does not "
                        "reproduce. May be given more than once (train and test).")
    p.add_argument("--verify-against", type=Path, action="append", default=[],
                   help="published aux_data.csv to cross-check the metadata alignment "
                        "against; may be given more than once (train and test)")
    p.add_argument("--inventory-only", action="store_true",
                   help="print the availability inventory and exit without writing")
    p.add_argument("--limit-clusters", type=int, default=0,
                   help="read only the first N clusters (smoke test)")
    args = p.parse_args()

    if not args.raw_root.is_dir():
        raise SystemExit(f"raw root {args.raw_root} does not exist")

    clusters = sorted(d for d in args.raw_root.iterdir() if d.is_dir())
    if args.limit_clusters:
        clusters = clusters[: args.limit_clusters]
    logger.info("reading %d clusters from %s", len(clusters), args.raw_root)

    frames = []
    for d in clusters:
        logger.info("  %s", d.name)
        frames.append(read_cluster(d, args.pairing, args.recover_prior_psfc,
                                   args.rep_offset))
    df = pd.concat(frames, ignore_index=True)

    align = check_metadata_alignment(df)
    if args.metadata_from:
        # The published values are carried too, under their plain names, so the split
        # table's own columns keep their meaning and `covariates=legacy` still reproduces
        # the released models exactly. The corrected values live beside them as *_fix.
        df = attach_published_metadata(df, args.metadata_from)
        source_report = compare_metadata_sources(df)
    else:
        logger.warning(
            "no --metadata-from: the published (scrambled) metadata will not be in the "
            "table, so the *_base2 control cannot be run from it.")
        source_report = {}

    feasibility = {
        f"corrected ({args.pairing})": airmass_feasibility(df, "SlrZen" + CORRECTED_SUFFIX),
        "published aux_data.csv": airmass_feasibility(df, "SlrZen"),
    }
    df = add_geometry(df)
    logger.info("extracted %d rows x %d columns", len(df), df.shape[1])
    if _EMPTY_BANDS:
        logger.info("soundings with an entirely missing band (SNR/continuum left NaN): %s",
                    _EMPTY_BANDS)

    inv = inventory(df)
    print("\n=== covariate inventory ===")
    with pd.option_context("display.width", 200, "display.max_rows", 300):
        print(inv.to_string(index=False))

    align = check_metadata_alignment(df)
    print("\n=== metadata alignment ===")
    print(f"directory order matches metadata row order in every cluster: "
          f"{align['all_clusters_ref_sorted']}")
    if not align["all_clusters_ref_sorted"]:
        print("  WARNING: at least one cluster's .h5 files are not in reference-sounding")
        print("  order. oco2_utils.get_cluster pairs the i-th file with the i-th metadata")
        print("  row regardless, so the published per-scene metadata may be misassigned.")
        print("  This script reproduces that pairing; see the summary JSON for details.")

    if source_report:
        print("\n--- cluster .csv pairing vs the published aux_data.csv ---")
        for col, r in source_report.items():
            print(f"  {col:12s} agree on {r['fraction_agree']:6.1%} of "
                  f"{r['n_compared']} soundings ({r['n_disagree']} disagree)")
        print("  The table carries BOTH: the published values under their plain names")
        print("  (so covariates=legacy still reproduces the released models) and the")
        print("  corrected ones as *_fix. The airmass check below says which is right.")

    print("\n--- airmass consistency: is sec(VZA) = airmass - sec(SZA) >= 1 ? ---")
    for name, r in feasibility.items():
        if not r:
            continue
        miss = r.get("n_unresolvable", 0)
        note = (f"  ({miss} of {r['n']} have no airmass and cannot be checked)"
                if miss else "")
        print(f"  {name:22s} feasible for {r['fraction_feasible']:6.1%} of "
              f"{r.get('n_resolvable', r['n'])} checkable soundings{note}")
    print("  A pairing that is right should be feasible for ~100 %. Read these against the")
    print("  random-permutation null in diagnose_metadata_pairing.py before concluding")
    print("  anything: without it, a 50 % failure rate is not evidence of much.")

    ver = {}
    if args.verify_against:
        ver = verify_against(df, args.verify_against)
        print("\n=== verification against published aux_data.csv ===")
        for k, v in ver.items():
            print(f"  {k}: {v}")

    if args.inventory_only:
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    inv_path = args.out.with_suffix("").with_suffix(".inventory.csv")
    inv.to_csv(inv_path, index=False)

    meta = {
        "out": str(args.out),
        "raw_root": str(args.raw_root),
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_clusters": len(clusters),
        "n_reference_soundings": int(df.drop_duplicates(["clst", "ref"]).shape[0]),
        "availability": {r.covariate: r.availability for r in inv.itertuples()},
        "metadata_alignment": align,
        "metadata_source": ("published aux_data.csv" if args.metadata_from
                            else "cluster .csv, directory order"),
        "metadata_source_comparison": source_report,
        "airmass_feasibility": feasibility,
        "empty_bands": dict(_EMPTY_BANDS),
        "verification": ver,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path = args.out.with_suffix("").with_suffix(".summary.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    print(f"\nWrote {args.out}  ({len(df)} rows, {df.shape[1]} columns)")
    print(f"Wrote {inv_path}")
    print(f"Wrote {meta_path}")
    print("\nNext:  python3 experiments/pipelines/augment_split.py "
          f"--split-csv <group_split_v1.csv> --covariates {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

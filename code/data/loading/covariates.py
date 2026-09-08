"""covariates.py
Which auxiliary covariates a model sees, and how they are encoded.

The published models take four auxiliary inputs -- SlrZen, Footprint, AerType1, AerType2 --
hardcoded into `SpectralDataset.load_aux` and into `AuxEmbedding`'s three `nn.Embedding`
layers. This module replaces the hardcoding with a declared *specification*, so that a
covariate set is a config choice and an ablation is a one-line change.

Availability, and why it is tracked here rather than left to the reader
----------------------------------------------------------------------
A covariate is only admissible if a real retrieval would have it. Four tiers, following
`extract_covariates.py`:

  prior       geometry and meteorology known before the retrieval is run (solar and
              viewing zenith angle, prior surface pressure, prior aerosol types,
              footprint, location)
  l1b         delivered with the radiances (per-channel noise, continuum level)
  acos        an output of the operational retrieval -- admissible, but a model that uses
              it is a *post-processor* of ACOS, not an independent retrieval, and must be
              reported as such
  simulation  an artefact of how the ensemble was built (SOM cluster label, mixture
              component). Never available operationally, and the cluster label is derived
              from retrieved states, so it also leaks the target. Not usable.

The presets below never mix a tier silently: `SPEC_TIER` records the highest tier each one
reaches, and `CovariateSpec.tier` is written into every checkpoint's hyperparameters.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# The published categorical encoding, kept verbatim so `legacy` reproduces the released
# checkpoints exactly.
LEGACY_AEROSOL_VOCAB = {"DU": 0, "SO": 1, "SS": 2, "OC": 3, "BC": 4}


@dataclass(frozen=True)
class CovariateSpec:
    """Declares the auxiliary inputs of a model.

    `legacy=True` selects the original hardcoded encoding (SlrZen/90 - 0.5, Footprint-1,
    two aerosol-type vocabularies) and the original `AuxEmbedding` module, so published
    checkpoints load unchanged. Every other spec standardizes continuous covariates with
    training-set statistics and builds categorical vocabularies from the training rows.
    """
    name: str
    continuous: Tuple[str, ...] = ()
    categorical: Tuple[str, ...] = ()
    legacy: bool = False
    tier: str = "prior"

    @property
    def n_continuous(self) -> int:
        return len(self.continuous)

    @property
    def n_categorical(self) -> int:
        return len(self.categorical)

    @property
    def columns(self) -> Tuple[str, ...]:
        return tuple(self.continuous) + tuple(self.categorical)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "continuous": list(self.continuous),
            "categorical": list(self.categorical),
            "legacy": self.legacy,
            "tier": self.tier,
        }


# ---------------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------------
# `legacy` is the published model. `legacy_v2` is the same four covariates through the new
# encoder: it exists so that "the covariate set changed" can be separated from "the
# encoding changed" -- without it, every comparison against `legacy` confounds the two.
# Two families, and the difference between them is a data bug, not a modelling choice.
#
# The scene metadata in the published aux_data.csv is attached to the WRONG soundings.
# oco2_utils.get_cluster pairs the i-th .h5 file in Path.iterdir() order with row i of the
# cluster's .csv, and diagnose_metadata_pairing.py shows that assignment puts 49.0 % of
# soundings at sec(VZA) < 1 -- geometrically impossible, and indistinguishable from a
# random permutation (null 46.6-49.7 %). Sorting the files by reference-sounding number
# gives 0.000 in all nine clusters, 451 checks with no failures.
#
# So: the plain names (SlrZen, Footprint, AerType1, AerType2) are the published, scrambled
# values, kept so the released checkpoints still reproduce. The *_fix names are the
# corrected ones. Everything new should use the corrected family.
_LEGACY = CovariateSpec(
    name="legacy",
    continuous=("SlrZen",),
    categorical=("Footprint", "AerType1", "AerType2"),
    legacy=True,
    tier="prior",
)

# The same four published values through the new encoder: separates "the covariate set
# changed" from "the encoding changed".
_LEGACY_V2 = CovariateSpec(
    name="legacy_v2",
    continuous=("SlrZen",),
    categorical=("Footprint", "AerType1", "AerType2"),
    tier="prior",
)

# The same four, correctly assigned. Against legacy_v2 this isolates one thing: what the
# metadata bug was costing. Nothing else changes.
_LEGACY_FIX = CovariateSpec(
    name="legacy_fix",
    continuous=("SlrZen_fix",),
    categorical=("Footprint_fix", "AerType1_fix", "AerType2_fix"),
    tier="prior",
)

# Observing geometry and the meteorological prior. sec_sza / sec_vza are what the radiative
# transfer depends on (path length); airmass is their sum and is what the file reports.
# All three are computed from the corrected SlrZen.
_SCENE = CovariateSpec(
    name="scene",
    continuous=("SlrZen_fix", "sec_sza", "vza_deg", "sec_vza", "airmass", "prior_psfc_pa"),
    categorical=("Footprint_fix", "AerType1_fix", "AerType2_fix"),
    tier="prior",
)

# Adds the priors ACOS itself was given: the a-priori column amount and the two aerosol
# optical depths. These are in the cluster metadata and have never been read by the
# pipeline -- META_COLUMNS does not list them. XCO2Prior in particular is the piece of
# information the physical retrieval has and the network never did.
_SCENE_PRIOR = CovariateSpec(
    name="scene_prior",
    continuous=_SCENE.continuous + ("XCO2Prior", "AODPrior1", "AODPrior2"),
    categorical=_SCENE.categorical,
    tier="prior",
)

# Adds location. Operationally available, and ACOS uses it through its CO2 prior -- but it
# is also the input most easily read as climatology rather than retrieval, so it is a
# separate preset and is always reported next to scene_prior.
_SCENE_GEO = CovariateSpec(
    name="scene_geo",
    continuous=_SCENE_PRIOR.continuous + ("Latitude", "Longitude"),
    categorical=_SCENE_PRIOR.categorical,
    tier="prior",
)

# Adds per-band signal-to-noise and continuum level from L1B.
_L1B = ("snr_b1", "snr_b2", "snr_b3", "continuum_b1", "continuum_b2", "continuum_b3")

_SCENE_L1B = CovariateSpec(
    name="scene_l1b",
    continuous=_SCENE_PRIOR.continuous + _L1B,
    categorical=_SCENE_PRIOR.categorical,
    tier="l1b",
)

_SCENE_GEO_L1B = CovariateSpec(
    name="scene_geo_l1b",
    continuous=_SCENE_GEO.continuous + _L1B,
    categorical=_SCENE_GEO.categorical,
    tier="l1b",
)

# The ACOS-informed variant: everything above plus the operational retrieval's own output.
# A post-processor of ACOS, not an independent retrieval: it cannot claim ACOS's runtime
# and must be labelled as such wherever it is reported.
_ACOS_INFORMED = CovariateSpec(
    name="acos_informed",
    continuous=_SCENE_L1B.continuous + (
        "retrieved_xco2", "retrieved_xco2_variance",
        "retrieved_xco2_variance_noise", "retrieved_xco2_variance_smooth",
        "retrieved_xco2_variance_interf",
        "delta_psfc", "dof_co2", "log_dws", "co2_grad_del",
        "albedo_blended", "albedo_weak_co2", "albedo_strong_co2",
        "aod_dust", "aod_sulfate", "aod_ice",
        "sp_reduced_chi_squared_o2", "sp_reduced_chi_squared_weak_co2",
        "sp_reduced_chi_squared_strong_co2",
        "retr_h2o_scale", "retr_psfc_pa", "retr_temp_offset_k",
        "retr_brdf_o2_1", "retr_brdf_o2_2", "retr_brdf_wco2_1", "retr_brdf_wco2_2",
        "retr_brdf_sco2_1", "retr_brdf_sco2_2", "retr_fluor_1", "retr_fluor_2",
    ),
    categorical=_SCENE_L1B.categorical + ("outcome_flag", "xco2_quality_flag"),
    tier="acos",
)

_ACOS_INFORMED_GEO = CovariateSpec(
    name="acos_informed_geo",
    continuous=_ACOS_INFORMED.continuous + ("Latitude", "Longitude"),
    categorical=_ACOS_INFORMED.categorical,
    tier="acos",
)

PRESETS: Dict[str, CovariateSpec] = {
    s.name: s for s in (
        _LEGACY, _LEGACY_V2, _LEGACY_FIX, _SCENE, _SCENE_PRIOR, _SCENE_GEO,
        _SCENE_L1B, _SCENE_GEO_L1B, _ACOS_INFORMED, _ACOS_INFORMED_GEO,
    )
}


def get_spec(spec) -> CovariateSpec:
    """Resolve a preset name, a mapping, or a CovariateSpec into a CovariateSpec."""
    if spec is None:
        return _LEGACY
    if isinstance(spec, CovariateSpec):
        return spec
    if isinstance(spec, str):
        if spec not in PRESETS:
            raise KeyError(f"unknown covariate preset '{spec}'. "
                           f"Available: {sorted(PRESETS)}")
        return PRESETS[spec]
    # An omegaconf DictConfig or a plain dict.
    d = dict(spec)
    name = str(d.get("name", "custom"))
    if set(d) == {"name"} or (len(d) == 1 and "name" in d):
        return get_spec(name)
    return CovariateSpec(
        name=name,
        continuous=tuple(d.get("continuous", ()) or ()),
        categorical=tuple(d.get("categorical", ()) or ()),
        legacy=bool(d.get("legacy", False)),
        tier=str(d.get("tier", "prior")),
    )


# ---------------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------------
class CovariateEncoder:
    """Turns the covariate columns of an annotations frame into model inputs.

    Encoding happens once, at construction: continuous covariates become one `[N, C]`
    float32 array and categorical ones a `[N, K]` int64 array, so `__getitem__` is two
    array reads rather than `len(spec.columns)` pandas scalar lookups.

    Statistics come from `covariate_stats.json` (written by `augment_split.py` from the
    training rows only). A missing stats file is an error for any non-legacy spec: falling
    back to statistics of the frame in hand would standardize the validation and test
    partitions with their own means.
    """

    def __init__(self, frame: pd.DataFrame, spec: CovariateSpec,
                 stats: Optional[dict] = None):
        self.spec = spec
        self.legacy = spec.legacy
        n = len(frame)

        missing = [c for c in spec.columns if c not in frame.columns]
        if missing:
            raise KeyError(
                f"covariate spec '{spec.name}' needs columns absent from the annotations "
                f"table: {missing}. Build an augmented split table with "
                f"experiments/pipelines/augment_split.py -- and note that Latitude and "
                f"Longitude are not in the published aux_data.csv at all: they need "
                f"chtc/reprocess.sh, whose metadata pairing is itself the open question "
                f"in experiments/pipelines/diagnose_metadata_pairing.py.")

        if self.legacy:
            self.cont = np.empty((n, 1), dtype=np.float32)
            self.cont[:, 0] = frame["SlrZen"].to_numpy(dtype=np.float32) / 90.0 - 0.5
            cats = [
                frame["Footprint"].to_numpy(dtype=np.int64) - 1,
                np.array([LEGACY_AEROSOL_VOCAB[v] for v in frame["AerType1"]], dtype=np.int64),
                np.array([LEGACY_AEROSOL_VOCAB[v] for v in frame["AerType2"]], dtype=np.int64),
            ]
            self.cat = np.stack(cats, axis=1) if cats else np.zeros((n, 0), np.int64)
            self.cat_cardinalities = (8, 5, 5)
            return

        if stats is None:
            raise ValueError(
                f"covariate spec '{spec.name}' requires covariate_stats.json; pass "
                f"stats_dir to the loader so it is read from the split's own statistics.")
        table = stats["stats"] if "stats" in stats else stats

        cont = np.empty((n, spec.n_continuous), dtype=np.float32)
        for j, col in enumerate(spec.continuous):
            st = table.get(col)
            if st is None:
                raise KeyError(f"no training statistics for continuous covariate '{col}'")
            if st["kind"] != "continuous":
                raise ValueError(f"'{col}' is categorical in covariate_stats.json but the "
                                 f"spec lists it as continuous")
            v = frame[col].to_numpy(dtype=np.float64)
            v = np.where(np.isfinite(v), v, st.get("impute", st["mean"]))
            cont[:, j] = (v - st["mean"]) / st["std"]
        self.cont = cont

        cats = []
        cards = []
        for col in spec.categorical:
            st = table.get(col)
            if st is None:
                raise KeyError(f"no training statistics for categorical covariate '{col}'")
            if st["kind"] != "categorical":
                raise ValueError(f"'{col}' is continuous in covariate_stats.json but the "
                                 f"spec lists it as categorical")
            lookup = {lvl: i for i, lvl in enumerate(st["levels"])}
            unknown = st["unknown_index"]
            codes = np.fromiter(
                (lookup.get(str(v), unknown) for v in frame[col].to_numpy()),
                dtype=np.int64, count=n)
            cats.append(codes)
            cards.append(int(st["cardinality"]))
        self.cat = np.stack(cats, axis=1) if cats else np.zeros((n, 0), dtype=np.int64)
        self.cat_cardinalities = tuple(cards)

    def __call__(self, idx: int):
        cont = torch.from_numpy(self.cont[idx])
        cat = [torch.tensor([int(v)], dtype=torch.long) for v in self.cat[idx]]
        return cont, cat


def model_aux_kwargs(spec, stats: Optional[dict] = None) -> dict:
    """Keyword arguments that give a regressor the auxiliary encoder this spec implies.

    `FullRegressor`, `FullRegressorProfiles` and `FullEncoder` all take `n_cont` and
    `cat_cardinalities`, and all treat both being absent as "the published four inputs".
    Five call sites were assembling that dict by hand -- the two Hydra trainers, the three
    flow trainers and the checkpoint loader -- which is five places for the legacy branch
    to be got subtly wrong.

    Returns an empty dict for a legacy spec, so `Model(**model_aux_kwargs(spec, stats))`
    is correct for every spec without a conditional at the call site.
    """
    spec = get_spec(spec)
    if spec.legacy:
        return {}
    return {"n_cont": spec.n_continuous,
            "cat_cardinalities": list(cat_cardinalities(spec, stats))}


def load_covariate_stats(stats_dir: Optional[Path]) -> Optional[dict]:
    """Read `covariate_stats.json` from a split's statistics directory, if present."""
    if stats_dir is None:
        return None
    path = Path(stats_dir) / "covariate_stats.json"
    if not path.exists():
        return None
    with open(path, "r") as fh:
        return json.load(fh)


def cat_cardinalities(spec: CovariateSpec, stats: Optional[dict]) -> Tuple[int, ...]:
    """Embedding table sizes for a spec, without needing an annotations frame.

    Needed at model construction time, which happens before the first batch is drawn.
    """
    if spec.legacy:
        return (8, 5, 5)
    if stats is None:
        raise ValueError(f"spec '{spec.name}' needs covariate_stats.json to size its "
                         f"embedding tables")
    table = stats["stats"] if "stats" in stats else stats
    out = []
    for col in spec.categorical:
        st = table.get(col)
        if st is None:
            raise KeyError(f"no training statistics for categorical covariate '{col}'")
        out.append(int(st["cardinality"]))
    return tuple(out)

#!/usr/bin/env python3
"""make_input_table.py
The definitive input table, generated from the loader rather than written by hand.

    python3 experiments/pipelines/make_input_table.py \
        --split-table group_split_v1_cov.csv \
        --out experiments/scoring/tables

Writes four files into `--out`:

    inputs_spectral.tex   the spectral inputs and their transformation order
    inputs_aux.tex        one row per auxiliary variable
    inputs.md             both tables in markdown
    inputs.json           the same content as data, for anything downstream

Why this is a script and not a table in the manuscript source
------------------------------------------------------------
The paper needs one definitive input table giving every variable, dimension, unit or
category vocabulary, source, availability at inference time, and transformation order".
The manuscript has been wrong about its own inputs three times in three places, so the
table has to be derived from the code that builds the tensors:

  * membership and continuous/categorical status come from `PRESETS` in
    `code/data/loading/covariates.py` -- the same objects the dataset constructs its
    encoder from, so a preset edit shows up here;
  * the encoding conventions (which statistics, the reserved unknown level, the imputation
    rule) come from `build_stats` in `augment_split.py` -- the same function whose output
    the encoder reads at training time;
  * the ranges and level vocabularies are computed from the TRAINING rows of the split
    table, which is where the encoder's statistics come from as well;
  * the spectral rows are the constants in `SpectralDataset.load_spectra` /
    `add_scaling_channels` and the padding widths in `spectral_loader.py`.

Anything this file states that the code does not do is a bug in this file.

The per-band channel counts cannot be read out of the loader (they are a property of the
delivered spectra, fixed by `clean_data.py`'s all-fill mask), so they are constants here,
measured once on the ensemble and checked by `prepare_month.py` on every new month:
835 / 652 / 811 = 2298 kept of 3048 delivered.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
for p in (str(_ROOT / "code"), str(_ROOT), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from data.loading.covariates import PRESETS, get_spec          # noqa: E402
from augment_split import build_stats                          # noqa: E402

logger = logging.getLogger("make_input_table")

# The two presets the manuscript reports: what the released models saw, and what the
# revision's models see. `scene_geo` is the ablation, and gets a column only in markdown.
PUBLISHED = "legacy"
REVISED = "scene_prior"
ABLATION = "scene_geo"

# ---------------------------------------------------------------------------------
# Spectral inputs. Constants from SpectralDataset.load_spectra, add_scaling_channels and
# the b*_transform pads in spectral_loader.py.
# ---------------------------------------------------------------------------------
BAND_ROWS = [
    # name, symbol, channels kept, wavelength range, what it constrains
    ("O$_2$ A-band", r"$\by^{(1)}$", 835, "0.7592--0.7714",
     "surface pressure, cloud and aerosol scattering"),
    ("Weak CO$_2$", r"$\by^{(2)}$", 652, "1.5980--1.6178",
     "the CO$_2$ column"),
    ("Strong CO$_2$", r"$\by^{(3)}$", 811, "2.0477--2.0798",
     "CO$_2$, water vapour and aerosol"),
]
N_CHANNELS_RAW = 3048
N_CHANNELS_KEPT = 2298
PAD_WIDTH = 1024


def spectral_rows() -> List[dict]:
    """One row per model-visible spectral quantity, in transformation order."""
    rows = []
    for name, sym, n, wl, what in BAND_ROWS:
        rows.append(dict(
            group="spectral",
            variable=f"{name}: wavelength channel",
            symbol=f"{sym}, ch.~1",
            dim=f"{n} (padded to {PAD_WIDTH})",
            unit=rf"$\mu$m, {wl}",
            source="L1B (delivered with the radiance)",
            available="yes",
            transform="none; zero-padded on the right",
            note=what,
        ))
        rows.append(dict(
            group="spectral",
            variable=f"{name}: radiance channel",
            symbol=f"{sym}, ch.~2",
            dim=f"{n} (padded to {PAD_WIDTH})",
            unit="L1B radiance scale (see caption)",
            source="L1B (simulated)",
            available="yes",
            transform=r"divided by that band's own maximum; zero-padded on the right",
            note=what,
        ))
        rows.append(dict(
            group="spectral",
            variable=f"{name}: scale constants",
            symbol=r"$\bc^{(j)}$",
            dim="2",
            unit="dimensionless",
            source="derived from the band maximum",
            available="yes",
            transform=(r"$(\lfloor\log_{10} m\rfloor,\; m\,10^{-1-\lfloor\log_{10} "
                       r"m\rfloor})$ for band maximum $m$"),
            note="restores the absolute radiance scale exactly",
        ))
    return rows


# ---------------------------------------------------------------------------------
# Auxiliary inputs. One descriptor per column that any reported preset can name.
# `tier` repeats CovariateSpec.tier's vocabulary and is checked against it below.
# ---------------------------------------------------------------------------------
D = dict  # brevity in the table below


AUX: Dict[str, dict] = {
    # --- the published four, as released (scrambled scene pairing) -------------------
    "SlrZen": D(label="Solar zenith angle", symbol=r"$\theta_0$", unit="degrees",
                source="cluster metadata (published pairing)", tier="prior",
                transform=r"$\theta_0/90 - 0.5$",
                note="published models only; attached to the wrong scene (Sec.~[bug])"),
    "Footprint": D(label="Footprint index", symbol="--", unit="8 levels, 1--8",
                   source="cluster metadata (published pairing)", tier="prior",
                   transform="integer $-1$, then an 8-row embedding",
                   note="published models only; attached to the wrong scene"),
    "AerType1": D(label="Prior aerosol type 1", symbol="--",
                  unit="5 levels: DU, SO, SS, OC, BC",
                  source="cluster metadata (published pairing)", tier="prior",
                  transform="fixed vocabulary, then a 5-row embedding",
                  note="published models only; attached to the wrong scene"),
    "AerType2": D(label="Prior aerosol type 2", symbol="--",
                  unit="5 levels: DU, SO, SS, OC, BC",
                  source="cluster metadata (published pairing)", tier="prior",
                  transform="fixed vocabulary, then a 5-row embedding",
                  note="published models only; attached to the wrong scene"),
    # --- the corrected family --------------------------------------------------------
    "SlrZen_fix": D(label="Solar zenith angle", symbol=r"$\theta_0$", unit="degrees",
                    source="cluster metadata (corrected pairing)", tier="prior",
                    transform="standardized",
                    note="sets the solar path length, $\\sec\\theta_0$"),
    "sec_sza": D(label="Solar air mass", symbol=r"$\sec\theta_0$", unit="dimensionless",
                 source="derived from the solar zenith angle", tier="prior",
                 transform="standardized",
                 note="what the radiative transfer depends on, rather than the angle"),
    "vza_deg": D(label="Viewing zenith angle", symbol=r"$\theta$", unit="degrees",
                 source=r"derived: $\sec\theta = \mathrm{airmass} - \sec\theta_0$",
                 tier="prior", transform="standardized",
                 note="nadir throughout this ensemble; see caption"),
    "sec_vza": D(label="Viewing air mass", symbol=r"$\sec\theta$", unit="dimensionless",
                 source="derived from air mass and the solar zenith angle", tier="prior",
                 transform="standardized", note="1.00001--1.00022 over the ensemble"),
    "airmass": D(label="Total air mass", symbol="--", unit="dimensionless",
                 source="delivered per sounding; pure observing geometry", tier="prior",
                 transform="standardized",
                 note=r"$\sec\theta_0 + \sec\theta$; constant across a scene's replicates"),
    "prior_psfc_pa": D(label="Prior surface pressure", symbol=r"$p_{s}^{a}$", unit="Pa",
                       source="meteorological prior, recovered as retrieved $-$ offset",
                       tier="prior", transform="standardized",
                       note="the a-priori surface pressure the retrieval was given"),
    "Footprint_fix": D(label="Footprint index", symbol="--", unit="8 levels, 1--8",
                       source="cluster metadata (corrected pairing)", tier="prior",
                       transform="vocabulary from training rows, then an embedding",
                       note="which of the 8 cross-track detector elements observed"),
    "AerType1_fix": D(label="Prior aerosol type 1", symbol="--",
                      unit="vocabulary from training rows",
                      source="cluster metadata (corrected pairing)", tier="prior",
                      transform="vocabulary from training rows, then an embedding",
                      note="selects the first variable aerosol species of the state vector"),
    "AerType2_fix": D(label="Prior aerosol type 2", symbol="--",
                      unit="vocabulary from training rows",
                      source="cluster metadata (corrected pairing)", tier="prior",
                      transform="vocabulary from training rows, then an embedding",
                      note="selects the second variable aerosol species"),
    "XCO2Prior": D(label="Prior XCO$_2$", symbol=r"$X^{a}$", unit="ppm",
                   source="cluster metadata; the retrieval's own a priori column",
                   tier="prior", transform="standardized",
                   note="available to the full-physics retrieval, and to ours"),
    "AODPrior1": D(label="Prior AOD, species 1", symbol=r"$\log\tau^{a}_1$",
                   unit="natural log of optical depth",
                   source="cluster metadata", tier="prior", transform="standardized",
                   note="the aerosol amount that goes with AerType1"),
    "AODPrior2": D(label="Prior AOD, species 2", symbol=r"$\log\tau^{a}_2$",
                   unit="natural log of optical depth",
                   source="cluster metadata", tier="prior", transform="standardized",
                   note="the aerosol amount that goes with AerType2"),
    "Latitude": D(label="Latitude", symbol="--", unit="degrees north",
                  source="cluster metadata", tier="prior", transform="standardized",
                  note="ablation only; not an input to the reported models"),
    "Longitude": D(label="Longitude", symbol="--", unit="degrees east",
                   source="cluster metadata", tier="prior", transform="standardized",
                   note="ablation only; not an input to the reported models"),
}

# The legacy path does not build its vocabularies from the data: `CovariateEncoder` maps
# Footprint by `int - 1` into an 8-row table and both aerosol types through
# LEGACY_AEROSOL_VOCAB into 5-row tables, whatever the training rows contain. So the
# observed level counts are not what those models encode, and printing them would say the
# published encoding reserves an unseen slot, which it does not -- an unseen level there is
# a KeyError, not a fallback.
LEGACY_VOCAB = {
    "Footprint": ("8 levels, 1--8 (fixed)", "no reserved level; an unseen value is an error"),
    "AerType1": ("5 levels: BC, DU, OC, SO, SS (fixed)",
                 "no reserved level; an unseen value is an error"),
    "AerType2": ("5 levels: BC, DU, OC, SO, SS (fixed)",
                 "no reserved level; an unseen value is an error"),
}

TIER_LABEL = {
    "prior": "known before the retrieval runs",
    "l1b": "delivered with the radiances",
    "acos": "an output of the full-physics retrieval",
    "simulation": "an artefact of the ensemble; never available",
}


# ---------------------------------------------------------------------------------
def _num(v: float) -> str:
    """Enough digits to be informative for every column in the table.

    `%.4g` prints sec(VZA)'s range as "1 to 1", which is the one thing that column is
    there to say something about, and surface pressure as "5.542e+04", which nobody reads
    as 554 hPa by accident but nobody reads as 554 hPa on purpose either.
    """
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}".replace(",", r"\,")
    if a >= 1:
        return f"{v:.6g}" if a < 1.001 or abs(round(v) - v) > 1e-6 else f"{v:g}"
    return f"{v:.4g}"


def aux_rows(stats: Optional[dict], counts: Optional[pd.DataFrame]) -> List[dict]:
    """One row per auxiliary variable named by any reported preset."""
    specs = {name: get_spec(name) for name in (PUBLISHED, REVISED, ABLATION)}
    ordered: List[str] = []
    for name in (PUBLISHED, REVISED, ABLATION):
        for col in specs[name].continuous + specs[name].categorical:
            if col not in ordered:
                ordered.append(col)

    missing = [c for c in ordered if c not in AUX]
    if missing:
        raise KeyError(
            f"presets name columns this table has no descriptor for: {missing}. "
            f"Add them to AUX rather than dropping them -- an input the table omits is "
            f"exactly the failure this table exists to prevent.")

    rows = []
    for col in ordered:
        d = AUX[col]
        spec_names = [n for n, s in specs.items() if col in s.columns]
        kind = "categorical" if any(col in specs[n].categorical for n in spec_names) \
            else "continuous"
        st = (stats or {}).get(col)
        unit = d["unit"]
        rng = ""
        if col in LEGACY_VOCAB:
            unit, rng = LEGACY_VOCAB[col]
        elif st is not None and st["kind"] == "continuous":
            rng = f"{_num(st['min'])} to {_num(st['max'])}"
        elif st is not None:
            levels = st["levels"]
            unit = f"{len(levels)} levels: " + ", ".join(levels)
            rng = f"+1 reserved for an unseen level ({st['cardinality']} in all)"
        rows.append(dict(
            group="aux", column=col, variable=d["label"], symbol=d["symbol"],
            kind=kind, dim="1", unit=unit, range=rng,
            source=d["source"], tier=d["tier"],
            available=TIER_LABEL[d["tier"]],
            transform=d["transform"], note=d["note"],
            presets=spec_names,
            published=(PUBLISHED in spec_names), revised=(REVISED in spec_names),
            ablation=(ABLATION in spec_names),
        ))
    return rows


def compute_stats(table: Path) -> tuple:
    """Training-row statistics, by the same function the encoder's statistics come from."""
    want = sorted({c for name in (PUBLISHED, REVISED, ABLATION)
                   for c in get_spec(name).columns} | {"split", "clst", "ref"})
    have = set(pd.read_csv(table, nrows=0).columns)
    use = [c for c in want if c in have]
    absent = [c for c in want if c not in have]
    if absent:
        logger.warning("split table has no column(s) %s; their rows will carry no range",
                       absent)
    df = pd.read_csv(table, usecols=use)
    train = df[df["split"] == "train"] if "split" in df.columns else df
    cols = [c for c in use if c not in ("split", "clst", "ref")]
    stats = build_stats(train, cols)
    # build_stats reports the encoding parameters; the table also wants the observed range.
    for col, st in stats.items():
        if st["kind"] == "continuous":
            v = train[col].astype(float)
            st["min"], st["max"] = float(v.min()), float(v.max())
    n_scenes = (train[["clst", "ref"]].drop_duplicates().shape[0]
                if {"clst", "ref"} <= set(train.columns) else None)
    return stats, dict(n_train=int(len(train)), n_scenes=n_scenes,
                       n_total=int(len(df)))


# ---------------------------------------------------------------------------------
def _tex_escape(s: str) -> str:
    r"""Escape only what a descriptor can legally contain.

    Underscores are deliberately NOT escaped: every one in this file is inside math that
    was written on purpose ($\theta_0$, XCO$_2$), and escaping them prints the backslash.
    A descriptor needing a literal underscore should write `\_` itself.
    """
    return s.replace("%", r"\%").replace("&", r"\&")


def write_spectral_tex(rows: List[dict], path: Path) -> Path:
    lines = [
        r"\begin{table*}[htpb]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{llllp{0.34\linewidth}}",
        r"\toprule",
        r"Input & Symbol & Channels & Unit / range & Transformation \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(" & ".join([r["variable"], r["symbol"], r["dim"], r["unit"],
                                 r["transform"]]) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\textbf{Spectral inputs.} Of the $%d$ delivered channels, $%d$ carry "
        r"radiance in every sounding; the remainder are fill and are dropped once, from "
        r"the training data, and the same channel set is used thereafter. Each band is "
        r"presented to its own encoder as a two-channel signal, wavelength and normalized "
        r"radiance, together with the two constants that recover its maximum. "
        r"Transformation order is: drop fill channels, split by wavelength, divide each "
        r"band by its own maximum, form the two constants, zero-pad to $%d$. No "
        r"zero-mean, unit-variance standardization is applied to the spectra at any "
        r"point. Radiances are used on the scale delivered in the simulation files "
        r"(order $10^{19}$); since each band is divided by its own maximum, the scale "
        r"enters the network only through the two constants.}"
        % (N_CHANNELS_RAW, N_CHANNELS_KEPT, PAD_WIDTH),
        r"\label{tab:inputs_spectral}",
        r"\end{table*}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_aux_tex(rows: List[dict], meta: dict, path: Path) -> Path:
    lines = [
        r"\begin{table*}[htpb]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{llp{0.20\linewidth}p{0.17\linewidth}lcc}",
        r"\toprule",
        (r"Variable & Unit or levels & Range (training) & Source & Transformation "
         r"& P & R \\"),
        r"\midrule",
    ]
    for r in rows:
        mark = lambda b: r"$\checkmark$" if b else "--"          # noqa: E731
        lines.append(" & ".join([
            _tex_escape(r["variable"]),
            _tex_escape(r["unit"]),
            _tex_escape(r["range"]) or "--",
            _tex_escape(r["source"]),
            _tex_escape(r["transform"]),
            mark(r["published"]), mark(r["revised"]),
        ]) + r" \\")
    n_pub = sum(r["published"] for r in rows)
    n_rev = sum(r["revised"] for r in rows)
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{\textbf{Auxiliary inputs.} Every scalar the network receives beside "
         r"the spectra, with the transformation applied to it. Column P marks the $%d$ "
         r"inputs of the released models, column R the $%d$ of the models reported here. "
         r"Every variable listed is known before the full-physics retrieval is run: none "
         r"is an output of that retrieval, and none is a quantity the retrieval itself "
         r"estimates. Continuous variables are standardized with means and standard "
         r"deviations computed from the $%s$ training soundings alone; categorical "
         r"variables use a vocabulary built from the training rows with one further slot "
         r"reserved for a level not seen in training, and are mapped to $8$-dimensional "
         r"embeddings. Viewing zenith angle is retained for completeness: the ensemble is "
         r"nadir throughout, so it varies only between $0.25^\circ$ and $1.20^\circ$ and "
         r"$\sec\theta$ never departs from $1$ by more than $2.2\times10^{-4}$. Latitude "
         r"and longitude are inputs to neither model and appear only in the ablation of "
         r"Section~[ablation].}"
         % (n_pub, n_rev, f"{meta.get('n_train', 0):,}".replace(",", r"{,}"))),
        r"\label{tab:inputs_aux}",
        r"\end{table*}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_markdown(spec_rows, aux_rows_, meta, path: Path) -> Path:
    def strip(s: str) -> str:
        for a, b in ((r"$\mu$m", "um"), ("$", ""), (r"\by", "y"), (r"\bc", "c"),
                     (r"\theta_0", "SZA"), (r"\theta", "VZA"), (r"\sec", "sec "),
                     (r"\log", "log"), (r"\tau", "tau"), (r"\lfloor", ""),
                     (r"\rfloor", ""), (r"\,", " "), (r"\;", " "), ("^{a}", "_prior"),
                     ("_{s}", "_s"), (r"\rm", ""), ("--", "-"),
                     (r"\mathrm{airmass}", "airmass"), (r"\times", "x"),
                     (r"\checkmark", "x")):
            s = s.replace(a, b)
        return s

    out = [f"# Model inputs, generated from the loader",
           "",
           f"- training soundings: {meta.get('n_train')}"
           f" ({meta.get('n_scenes')} reference soundings)",
           f"- published preset `{PUBLISHED}`: "
           f"{sum(r['published'] for r in aux_rows_)} auxiliary inputs",
           f"- revision preset `{REVISED}`: "
           f"{sum(r['revised'] for r in aux_rows_)} auxiliary inputs",
           f"- ablation preset `{ABLATION}`: "
           f"{sum(r['ablation'] for r in aux_rows_)} auxiliary inputs",
           "",
           "## Spectral inputs", "",
           "| input | channels | unit / range | transformation |",
           "|---|---|---|---|"]
    for r in spec_rows:
        out.append(f"| {strip(r['variable'])} | {r['dim']} | {strip(r['unit'])} "
                   f"| {strip(r['transform'])} |")
    out += ["",
            f"Transformation order: drop the {N_CHANNELS_RAW - N_CHANNELS_KEPT} all-fill "
            f"channels, split by wavelength at 1.2 and 1.8 um, divide each band by its own "
            f"maximum, form the two constants, zero-pad to {PAD_WIDTH}. No zero-mean "
            f"unit-variance standardization is applied to the spectra.",
            "",
            "## Auxiliary inputs", "",
            "| column | variable | kind | unit / levels | range (training) | source "
            "| tier | transformation | published | revision | ablation |",
            "|---|---|---|---|---|---|---|---|:-:|:-:|:-:|"]
    for r in aux_rows_:
        mark = lambda b: "x" if b else ""                        # noqa: E731
        out.append("| `{c}` | {v} | {k} | {u} | {g} | {s} | {t} | {x} | {p} | {r} | {a} |"
                   .format(c=r["column"], v=strip(r["variable"]), k=r["kind"],
                           u=strip(r["unit"]), g=strip(r["range"]) or "-",
                           s=strip(r["source"]), t=r["tier"], x=strip(r["transform"]),
                           p=mark(r["published"]), r=mark(r["revised"]),
                           a=mark(r["ablation"])))
    out += ["", "## Notes", ""]
    for r in aux_rows_:
        out.append(f"- `{r['column']}` -- {strip(r['note'])}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-table", default="group_split_v1_cov.csv",
                    help="augmented split table written by augment_split.py")
    ap.add_argument("--out", default="experiments/scoring/tables")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = Path(args.split_table)
    if table.exists():
        stats, meta = compute_stats(table)
        logger.info("statistics from %d training soundings (%s reference soundings)",
                    meta["n_train"], meta["n_scenes"])
    else:
        logger.warning("%s not found; writing the schema without ranges or vocabularies",
                       table)
        stats, meta = None, {}

    srows = spectral_rows()
    arows = aux_rows(stats, None)

    written = [
        write_spectral_tex(srows, out_dir / "inputs_spectral.tex"),
        write_aux_tex(arows, meta, out_dir / "inputs_aux.tex"),
        write_markdown(srows, arows, meta, out_dir / "inputs.md"),
    ]
    payload = {"meta": meta, "published_preset": PUBLISHED, "revised_preset": REVISED,
               "ablation_preset": ABLATION, "spectral": srows, "auxiliary": arows,
               "n_channels_raw": N_CHANNELS_RAW, "n_channels_kept": N_CHANNELS_KEPT,
               "pad_width": PAD_WIDTH}
    (out_dir / "inputs.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    written.append(out_dir / "inputs.json")

    for p in written:
        print(f"wrote {p}")
    print(f"\npublished preset '{PUBLISHED}': "
          f"{sum(r['published'] for r in arows)} auxiliary inputs")
    print(f"revision preset  '{REVISED}': "
          f"{sum(r['revised'] for r in arows)} auxiliary inputs")
    print(f"ablation preset  '{ABLATION}': "
          f"{sum(r['ablation'] for r in arows)} auxiliary inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""make_hparam_table.py -- Appendix Table A1 (`tab:model_hyperparams`), generated.

The published Table A1 was typed by hand from the pre-revision checkpoints. Every model
in the paper was re-searched and re-trained with corrected inputs (twelve covariates, flows from
scratch), so every cell of it is wrong, and `experiments/dump_checkpoint_config.py` --
which is what produced the numbers the first time -- needs the checkpoints, which live on
CHTC. The winners are already on this laptop in a smaller and more authoritative form:
`hyperparameters/bo_<study>_best.json` -- the records `bo_select.py` wrote at
selection time, tracked in the repository.

    python3 experiments/scoring/make_hparam_table.py
    python3 experiments/scoring/make_hparam_table.py

Writes, under --out:

    model_hyperparams.tex   the drop-in replacement for Table A1
    model_search.tex        the search budget and what it bought over the published
                            configuration -- the "was the flow under-specified"
                            table, which is the same six JSON files read differently

Standard library only, so it runs on the CHTC access point as well as here. It reads the
selection records and nothing else: no GPU, no checkpoint, no artifact.

The three trainers speak three dialects of the same configuration -- `hyps.embed_dim`,
`model.embed_dim` and `model_cfg_obj.full_backbone.embed_dim` are one hyperparameter --
so the keys are matched by suffix rather than by name. A key that matches nothing is
reported rather than dropped: a silently missing hyperparameter is how the published
table came to disagree with the models it described.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

# What the study is, in the manuscript's language rather than the queue's. `D` is the
# output dimension and `flow` the density family; both are properties of the task and the
# trainer, not of the search, so they are declared here rather than parsed.
STUDIES = [
    ("det_xco2",     "Deterministic", r"XCO$_2$",      1,  "--"),
    ("det_profiles", "Deterministic", "Full profile",  20, "--"),
    ("det_last4",    "Deterministic", "Near-surface",  4,  "--"),
    ("nf_xco2",      "NF",            r"XCO$_2$",      1,  "AR RQ spline"),
    ("nf_profiles",  "NF",            "Full profile",  20, "Coupling RQ spline"),
    ("nf_last4",     "NF",            "Near-surface",  4,  "Coupling RQ spline"),
]

# Suffix -> column. Longest suffix wins, so `hidden_features` is not read as `features`
# and `num_layers` (the flow's depth) is never confused with `nlayers` (the Transformer's).
FIELDS = {
    "embed_dim":        "E",
    "nhead":            "H",
    "nlayers":          "L",
    "num_layers":       "flow_layers",
    ".layers":          "flow_layers",
    "hidden_features":  "hidden",
    ".hidden":          "hidden",
    "num_bins":         "bins",
    ".bins":            "bins",
    "tail_bound":       "tail",
    ".lr":              "lr",
    ".wd":              "wd",
    "weight_decay":     "wd",
    "train_batch_size": "batch",
    "loss_weight":      "loss_weight",
}


def collect(params: dict) -> tuple[dict, list[str]]:
    """`{column -> value}` plus the keys that matched no column."""
    out, unmatched = {}, []
    for key, value in params.items():
        hits = sorted((s for s in FIELDS if key.endswith(s)), key=len, reverse=True)
        if hits:
            out[FIELDS[hits[0]]] = value
        else:
            unmatched.append(key)
    return out, unmatched


def sci(x: float, digits: int = 2) -> str:
    """`1.06\\times10^{-4}`, in the form the published table used."""
    s = f"{float(x):.{digits}e}"
    mant, exp = s.split("e")
    return rf"{mant}\times10^{{{int(exp)}}}"


def cell(v, fmt=None) -> str:
    if v is None:
        return "--"
    return fmt(v) if fmt else str(v)


def hyperparams_table(records: dict, epochs: int) -> str:
    rows = []
    for study, family, target, dim, flow in STUDIES:
        rec = records.get(study)
        if rec is None:
            print(f"  WARNING: no record for {study}; its row is left as '--'.",
                  file=sys.stderr)
            p = {}
        else:
            p, unmatched = collect(rec.get("params", {}))
            for key in unmatched:
                print(f"  note: {study}: '{key}' has no column in the table.",
                      file=sys.stderr)
        det = family == "Deterministic"
        # The search also chose the coordinate system the deterministic loss is weighted
        # in, which the published table has no column for. It goes in the optimizer cell
        # rather than being dropped: it is the difference between minimizing squared error
        # in ppm and in standardized units, which for a profile target is a different
        # objective, not a different scale.
        loss = {"ppm": ", loss in ppm", "std": ", loss standardized"}.get(
            p.get("loss_weight"), "")
        rows.append(" & ".join([
            family, target, str(dim),
            cell(p.get("E")), cell(p.get("H")), cell(p.get("L")),
            flow,
            "--" if det else cell(p.get("flow_layers")),
            "--" if det else cell(p.get("hidden")),
            "--" if det else cell(p.get("bins")),
            "--" if det else cell(p.get("tail"), lambda v: f"{float(v):.2f}"),
            cell(p.get("batch")),
            (rf"lr $={sci(p['lr'])}$, wd $={sci(p['wd'])}${loss}"
             if "lr" in p and "wd" in p else "--"),
        ]) + r" \\")

    body = "\n\n".join(rows)
    return rf"""% generated by experiments/scoring/make_hparam_table.py -- do not edit by hand
% source: hyperparameters/bo_<study>_best.json, written by bo_select.py
\begin{{table*}}[htpb]
\centering
\caption{{Final model configurations. \(D\) is the output dimension, \(E\) the embedding
dimension, \(H\) the number of attention heads and \(L\) the number of Transformer encoder
layers; the flow columns give the number of spline transforms, the width of each
conditioner, the number of bins and the tail bound. Every configuration is the winner of a
$25$-trial Bayesian-optimization search over the covariate set of
Table~\ref{{tab:inputs_aux}}, retrained from scratch for {epochs} epochs at the selected
configuration. Deterministic regressors were selected on validation RMSE and the flows on
validation negative log-likelihood; no flow was warm-started from a deterministic
backbone, so the six searches are comparable. All models use AdamW with cosine-annealed
learning rate.}}
\label{{tab:model_hyperparams}}
\resizebox{{\textwidth}}{{!}}{{
\begin{{tabular}}{{llcccclcccccl}}
\toprule
Model & Target & \(D\) & \(E\) & \(H\) & \(L\) & Flow & Flow layers & Hidden & Bins
& Tail bound & Batch & Optimizer \\
\midrule

{body}

\bottomrule
\end{{tabular}}
}}
\end{{table*}}
"""


def search_table(records: dict, epochs: int) -> str:
    # The selection metric's coordinate system, from `BO_SELECT` in bo_studies.sh. It is
    # NOT the same for the three deterministic studies -- at D=1 the ppm and standardized
    # RMSE differ by a constant and select the same checkpoint, so det_xco2 selects on the
    # standardized one -- and a column mixing 0.55 with 11.7 and saying only "val RMSE"
    # invites the reader to compare them.
    UNITS = {
        "det_xco2": "val RMSE [std]", "det_profiles": "val RMSE [ppm]",
        "det_last4": "val RMSE [ppm]", "nf_xco2": "val NLL [std]",
        "nf_profiles": "val NLL [std]", "nf_last4": "val NLL [std]",
    }
    rows = []
    for study, family, target, _dim, _flow in STUDIES:
        rec = records.get(study)
        if rec is None:
            continue
        rows.append(" & ".join([
            f"{family}, {target}",
            UNITS.get(study, rec.get("metric_label", rec.get("metric", "--"))),
            f"{rec['baseline_value']:.4g}" if "baseline_value" in rec else "--",
            f"{rec['value']:.4g}",
            (f"{rec['improvement_abs']:.4g}" if "improvement_abs" in rec else "--"),
            f"{rec.get('n_completed', '--')}/{rec.get('n_trials', '--')}",
            f"{rec.get('round', '--')}",
        ]) + r" \\")
    body = "\n".join(rows)
    return rf"""% generated by experiments/scoring/make_hparam_table.py -- do not edit by hand
\begin{{table}}[htpb]
\centering
\small
\caption{{The hyperparameter search, and what it bought. ``Published'' is the
configuration of the released models re-trained on the corrected covariates, so the two
columns differ in the configuration alone; both metrics are computed on the validation
partition and lower is better. Each search ran $25$ trials in five sequential rounds of
five at $20$ epochs, and the winner was re-trained for {epochs} epochs. The round column
is the round the winner was proposed in.}}
\label{{tab:model_search}}
\begin{{tabular}}{{llrrrcc}}
\toprule
Model & Metric & Published & Searched & Gain & Trials & Round \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # hyperparameters/ is TRACKED; chtc/bo/results/ is what a fetch writes and is
    # gitignored. Defaulting to the tracked copy is what makes this table regenerable in a
    # checkout that has never touched the cluster -- which is every checkout but ours.
    _default_records = _ROOT / "hyperparameters"
    if not _default_records.is_dir():
        _default_records = _ROOT / "chtc" / "bo" / "results"
    ap.add_argument("--results", default=str(_default_records),
                    help="directory holding bo_<study>_best.json")
    ap.add_argument("--out", default=str(_ROOT / "experiments" / "scoring" / "tables"))
    ap.add_argument("--epochs", type=int, default=40,
                    help="epochs the winners were retrained for (chtc/bo/final.sub)")
    args = ap.parse_args(argv)

    res_dir = Path(args.results)
    records = {}
    for study, *_ in STUDIES:
        path = res_dir / f"bo_{study}_best.json"
        if not path.exists():
            print(f"  MISSING: {path}", file=sys.stderr)
            continue
        with path.open() as fh:
            records[study] = json.load(fh)

    if not records:
        print(f"ERROR: no bo_<study>_best.json under {res_dir}.\n"
              f"  ./chtc/fetch_results.sh --best   (they arrive with the archives)",
              file=sys.stderr)
        return 1

    # One covariate set for all six, or the table's caption is a lie. bo_studies.sh fixes
    # it deliberately (`BO_COVARIATES_DEFAULT`); this checks that the records agree.
    covs = {r.get("covariates") for r in records.values()}
    if len(covs) > 1:
        print(f"ERROR: the winners were searched over different covariate sets: {covs}. "
              f"They are not one table.", file=sys.stderr)
        return 2
    print(f"  {len(records)} of {len(STUDIES)} studies, covariates = {covs.pop()}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in (("model_hyperparams.tex", hyperparams_table(records, args.epochs)),
                       ("model_search.tex", search_table(records, args.epochs))):
        (out_dir / name).write_text(text, encoding="utf-8")
        print(f"  wrote {out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

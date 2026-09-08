#!/usr/bin/env python3
"""bo_select.py
Read a finished Bayesian-optimization study and write the configuration to retrain.

    python3 chtc/bo_select.py --study det_profiles --results-dir bo_results --out-dir .

Writes two files per study:

    bo_<study>_best.json          what bo_final.sh retrains: the winner's overrides
    bo_<study>_leaderboard.md     all 25 trials, ranked, with the baseline marked

It runs in two places and must behave identically in both: inside the container as the
study's select node, between the last wave of trials and the final retrain, and on a
laptop over records brought down by `fetch_results.sh --bo`. So it imports nothing but
the standard library -- the records ARE the study, and rebuilding an optuna object to
read a minimum out of 25 numbers would only add a dependency that can be missing.

What "the winner" means here
----------------------------
The minimum over the trials that completed. Failed trials are not zeros and not
infinities: a job that died in setup says nothing about its hyperparameters, so it is
reported as failed and otherwise ignored -- the same convention bo_propose.py used when
it told the sampler.

Round 0 trial 0 is the published configuration, enqueued by bo_propose.py so that every
search contains its own baseline. If it wins, that IS the result: the search did not beat
the published hyperparameters, this file says so, and bo_final.sh retrains the baseline.
A search that cannot lose is not a search.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

STUDIES = ("det_xco2", "det_profiles", "det_last4",
           "nf_xco2", "nf_profiles", "nf_last4")

# What the minimized number is, per study. The deterministic studies minimize validation
# RMSE; the flows minimize validation NLL, which is what they are trained on and what the
# paper's uncertainty claims rest on. A leaderboard that mixed the two would be nonsense,
# which is why the studies are separate and the label is carried here.
METRIC_LABEL = {"val_rmse": "val RMSE", "val_nll": "val NLL"}


def load(results_dir: Path, study: str) -> List[dict]:
    recs = []
    for path in sorted(results_dir.glob("bo_{}_r*_t*.json".format(study))):
        if path.name.endswith(".params.json"):
            continue
        try:
            recs.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            print("WARNING: {} is not readable JSON; skipping".format(path))
    recs.sort(key=lambda r: (int(r.get("round", 0)), int(r.get("trial", 0))))
    return recs


def completed(recs: List[dict]) -> List[dict]:
    return [r for r in recs if isinstance(r.get("value"), (int, float))]


def baseline_of(recs: List[dict]) -> Optional[dict]:
    for r in recs:
        if int(r.get("round", -1)) == 0 and int(r.get("trial", -1)) == 0:
            return r
    return None


def leaderboard_md(study: str, recs: List[dict], best: dict,
                   base: Optional[dict], label: str) -> str:
    done = sorted(completed(recs), key=lambda r: r["value"])
    failed = [r for r in recs if not isinstance(r.get("value"), (int, float))]

    lines = [
        "# {} -- Bayesian optimization".format(study),
        "",
        "{} trials, {} completed, {} failed. Objective: **{}**, minimized. "
        "Covariates: `{}`. Trial epochs: {}.".format(
            len(recs), len(done), len(failed), label,
            best.get("covariates"), best.get("epochs", "?")),
        "",
        "The baseline is round 0 trial 0, the published hyperparameters, enqueued in every "
        "study so the comparison is inside one study rather than across code versions. "
        "The objective is minimized, so a **negative** delta is an improvement on it.",
        "",
        "| rank | round | trial | {} | vs baseline | configuration |".format(label),
        "|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(done, 1):
        rel = ""
        if base is not None and isinstance(base.get("value"), (int, float)):
            b = float(base["value"])
            # A relative change is only meaningful for a positive quantity. NLL is a
            # log-density and routinely negative, so a percentage of it is a number
            # without a meaning; the absolute difference always has one.
            if label == "val RMSE" and b:
                rel = "{:+.2f} %".format(100.0 * (r["value"] - b) / b)
            else:
                rel = "{:+.4f}".format(r["value"] - b)
        mark = " **(baseline)**" if r is base else ""
        cfg = " ".join(r.get("overrides") or [])
        lines.append("| {} | {} | {} | {:.6f} | {} | `{}`{} |".format(
            i, r.get("round"), r.get("trial"), r["value"], rel, cfg, mark))

    if failed:
        lines += ["", "Failed or unfinished, recorded as pruned (which is why the "
                      "search continued):", ""]
        for r in failed:
            lines.append("- r{} t{}: exit_status={} status={}".format(
                r.get("round"), r.get("trial"), r.get("exit_status"), r.get("status")))
    return "\n".join(lines) + "\n"


def select(study: str, results_dir: Path, out_dir: Path) -> dict:
    recs = load(results_dir, study)
    if not recs:
        raise SystemExit(
            "no trial records for {} in {}. Inside a job this means the select node's "
            "transfer_input_files did not name them; on a laptop, run "
            "./chtc/fetch_results.sh --bo first.".format(study, results_dir))

    done = completed(recs)
    failed = [r for r in recs if not isinstance(r.get("value"), (int, float))]
    if not done:
        raise SystemExit(
            "every one of the {} {} trials failed; there is nothing to retrain. Read "
            "chtc/logs/bo_{}_r0_t0.err -- 25 trials failing the same way is one bug, "
            "not twenty-five.".format(len(recs), study, study))

    best = min(done, key=lambda r: r["value"])
    base = baseline_of(recs)
    metric = next((r.get("metric") for r in done if r.get("metric")), None)
    label = METRIC_LABEL.get(metric, "val NLL" if study.startswith("nf_") else "val RMSE")

    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "study": study,
        "covariates": best.get("covariates"),
        "metric": metric,
        "metric_label": label,
        "value": best["value"],
        "round": best.get("round"),
        "trial": best.get("trial"),
        "trial_epochs": best.get("epochs"),
        "params": best.get("params") or {},
        "overrides": best.get("overrides") or [],
        "n_trials": len(recs),
        "n_completed": len(done),
        "n_failed": len(failed),
        "selected_utc": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if base is not None and isinstance(base.get("value"), (int, float)):
        rec["baseline_value"] = base["value"]
        rec["baseline_overrides"] = base.get("overrides") or []
        rec["improvement_abs"] = base["value"] - best["value"]
        if label == "val RMSE" and base["value"]:
            rec["improvement_pct"] = 100.0 * (base["value"] - best["value"]) / base["value"]
        rec["search_beat_baseline"] = not (int(best.get("round", -1)) == 0
                                           and int(best.get("trial", -1)) == 0)
    else:
        rec["baseline_value"] = None
        rec["search_beat_baseline"] = None

    (out_dir / "bo_{}_best.json".format(study)).write_text(
        json.dumps(rec, indent=2), encoding="utf-8")
    (out_dir / "bo_{}_leaderboard.md".format(study)).write_text(
        leaderboard_md(study, recs, best, base, label), encoding="utf-8")

    # The same thing on stdout, because for the select node the .out file is what gets
    # read first when a number looks wrong.
    print("=== {}: {} trials, {} completed, {} failed ===".format(
        study, len(recs), len(done), len(failed)))
    print("{:>4s} {:>2s} {:>2s} {:>13s}  configuration".format("rank", "r", "t", label))
    for i, r in enumerate(sorted(done, key=lambda r: r["value"]), 1):
        print("{:4d} {:>2} {:>2} {:13.6f}  {}".format(
            i, r.get("round"), r.get("trial"), r["value"],
            " ".join(r.get("overrides") or [])))
    for r in failed:
        print("  --  r{} t{}: FAILED (exit_status={})".format(
            r.get("round"), r.get("trial"), r.get("exit_status")))
    print()
    print("winner   : r{} t{}  {} = {:.6f}".format(
        rec["round"], rec["trial"], label, rec["value"]))
    if rec.get("baseline_value") is not None:
        print("baseline : r0 t0 (the published hyperparameters)  {:.6f}".format(
            rec["baseline_value"]))
        if "improvement_pct" in rec:
            print("the search moved it by {:+.2f} % ({:+.6f})".format(
                rec["improvement_pct"], -rec["improvement_abs"]))
        else:
            print("the search moved it by {:+.6f}".format(-rec["improvement_abs"]))
        if not rec["search_beat_baseline"]:
            print("The search did not beat its own baseline. That is the result: report "
                  "it, and note that the retrained model IS the published configuration "
                  "at full epochs.")
    else:
        print("WARNING: round 0 trial 0 did not complete, so this study has no baseline "
              "to be compared against. The winner is still retrained; the comparison "
              "against the published hyperparameters is simply not available.")
    print()
    print("wrote {}".format(out_dir / "bo_{}_best.json".format(study)))
    print("wrote {}".format(out_dir / "bo_{}_leaderboard.md".format(study)))
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", action="append", choices=list(STUDIES),
                    help="may be given more than once; default is every study that has "
                         "records in --results-dir")
    ap.add_argument("--results-dir", type=Path, default=Path("bo_results"))
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    args = ap.parse_args(argv)

    studies = args.study
    if not studies:
        studies = [s for s in STUDIES
                   if any(args.results_dir.glob("bo_{}_r*_t*.json".format(s)))]
        if not studies:
            raise SystemExit("no trial records in {}".format(args.results_dir))

    # One study failing must not hide the other five: a select node runs per study, but
    # the laptop invocation is over all six at once and the useful output is the five
    # that worked.
    failures = []
    for i, study in enumerate(studies):
        if i:
            print()
        try:
            select(study, args.results_dir, args.out_dir)
        except SystemExit as exc:
            if len(studies) == 1:
                raise
            print("ERROR: {}: {}".format(study, exc))
            failures.append(study)
    if failures:
        print("\n{} study/studies produced no winner: {}".format(
            len(failures), " ".join(failures)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

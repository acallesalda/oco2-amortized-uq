"""collect_covariate_runs.py
Read the 21 covariate-comparison runs and say which won, in both units.

    python3 experiments/scoring/collect_covariate_runs.py

Standard library only -- no numpy, no pandas, no torch -- so it runs on the CHTC access
point, where the archives have just been unpacked and nothing numerical is installed.

Why it parses the training logs rather than `global_best/best_metric.txt`
------------------------------------------------------------------------
`best_metric.txt` holds whatever `select_metric` selected on, and the runs do not agree on
that: `_base2` selects on standardized RMSE (it is the control, reproducing the published
behaviour) while every other run selects on ppm. Comparing those files directly compares
two different quantities and the ranking is meaningless.

The per-epoch log line carries both:

    epoch=12  val_RMSE[std]=0.9013  retr_RMSE[std]=0.8874  val_RMSE[ppm]=6.4413  ...

so taking the minimum of each column over the run gives every model a number in each unit,
comparable across the whole table. For the scalar XCO2 task there is only one target, so
ppm is the standardized value times sigma * 1e6, an exact constant read from the split's
own `xco2_stats.json`.

`best_val` is the minimum over epochs of the *selected* metric, which is what the
checkpoint on disk actually is. The two columns are the minima of each metric taken
independently, and can therefore come from different epochs -- they are for ranking
configurations, not for reporting a checkpoint's score.

Everything here is validation, not test. That is deliberate: the covariate set is chosen
here, and the test set is spent once, afterwards, on the chosen one.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

# weights_dir names, as the *_cov_grouped.yaml configs interpolate them.
#   full_regressor_<covariates>                                    (xco2)
#   full_regressor_profiles_<covariates>_<loss_weight>             (20-level)
#   full_regressor_profiles_last4_<covariates>_<loss_weight>       (near-surface)
DIR_RE = re.compile(
    r"^full_regressor(?P<profiles>_profiles)?(?P<last4>_last4)?_(?P<rest>.+)$")

# One regex per field. A single pattern with an optional trailing group does not work:
# the lazy `.*?` between them lets the optional group match empty every time, so the ppm
# column silently comes out blank for every run.
VAL_STD_RE = re.compile(r"val_RMSE\[std\]=(?P<v>[0-9.eE+-]+)")
VAL_PPM_RE = re.compile(r"val_RMSE\[ppm\]=(?P<v>[0-9.eE+-]+)")
XCO2_EPOCH_RE = re.compile(r"val loss = (?P<std>[0-9.eE+-]+)")
# train.py reports the ACOS baseline as `retr_loss`, standardized. Without it the XCO2
# rows have no reference column and the comparison has to borrow ACOS's RMSE from a
# different partition -- which is exactly the kind of cross-partition comparison the rest
# of this file is careful to avoid.
XCO2_RETR_RE = re.compile(r"retr_loss = (?P<v>[0-9.eE+-]+)")
RETR_STD_RE = re.compile(r"retr_RMSE\[std\]=(?P<v>[0-9.eE+-]+)")
RETR_PPM_RE = re.compile(r"retr_RMSE\[ppm\]=(?P<v>[0-9.eE+-]+)")

# The order the runbook says to read them in. Anything not listed still prints, at the end.
VARIANT_ORDER = ["legacy_std", "legacy", "legacy_ppm", "legacy_v2_ppm", "legacy_v2",
                 "legacy_fix_ppm", "legacy_fix", "scene_ppm", "scene",
                 "scene_prior_ppm", "scene_prior", "scene_geo_ppm", "scene_geo",
                 "acos_informed_ppm", "acos_informed"]

TASK_LABEL = {"xco2": "XCO2 (D=1)", "profiles": "20-level profile",
              "last4": "near-surface, levels 17-20"}


def task_of(dirname: str) -> Optional[str]:
    m = DIR_RE.match(dirname)
    if not m:
        return None
    if m.group("last4"):
        return "last4"
    return "profiles" if m.group("profiles") else "xco2"


def variant_of(dirname: str) -> str:
    m = DIR_RE.match(dirname)
    return m.group("rest") if m else dirname


def scan_logs(run_root: Path, task: str, xco2_sigma_ppm: Optional[float]) -> Dict:
    """Best validation RMSE in each unit, over every epoch of every run in this directory."""
    out: Dict[str, Optional[float]] = {
        "std": None, "ppm": None, "retr_std": None, "retr_ppm": None,
        "n_epochs": 0, "n_runs": 0,
    }
    logs = sorted(run_root.glob("runs/*/*.log"))
    for log in logs:
        out["n_runs"] += 1
        for line in log.read_text(errors="replace").splitlines():
            if task == "xco2":
                m = XCO2_EPOCH_RE.search(line)
                if not m:
                    continue
                std = float(m.group("std"))
                ppm = std * xco2_sigma_ppm if xco2_sigma_ppm else None
                mr = XCO2_RETR_RE.search(line)
                if mr:
                    rstd = float(mr.group("v"))
                    out["retr_std"] = rstd
                    if xco2_sigma_ppm:
                        out["retr_ppm"] = rstd * xco2_sigma_ppm
            else:
                ms = VAL_STD_RE.search(line)
                if not ms:
                    continue
                std = float(ms.group("v"))
                mp = VAL_PPM_RE.search(line)
                ppm = float(mp.group("v")) if mp else None
            out["n_epochs"] += 1
            for key, val in (("std", std), ("ppm", ppm)):
                if val is not None and (out[key] is None or val < out[key]):
                    out[key] = val
            for key, rx in (("retr_std", RETR_STD_RE), ("retr_ppm", RETR_PPM_RE)):
                rm = rx.search(line)
                if rm:
                    out[key] = float(rm.group("v"))     # constant; last one wins
    return out


def selected_metric(run_root: Path) -> Dict:
    """What the checkpoint on disk was actually chosen by."""
    out: Dict[str, Optional[object]] = {"best_val": None, "select_metric": None}
    txt = run_root / "global_best" / "best_metric.txt"
    js = run_root / "global_best" / "metrics.json"
    if txt.exists():
        try:
            out["best_val"] = float(txt.read_text().strip())
        except ValueError:
            pass
    elif js.exists():
        try:
            out["best_val"] = float(json.loads(js.read_text())["val_rmse"])
        except Exception:
            pass
    hp = run_root / "global_best" / "best_hparams.yaml"
    if hp.exists():
        m = re.search(r"^select_metric:\s*(\S+)", hp.read_text(), re.M)
        if m:
            out["select_metric"] = m.group(1).strip("'\"")
    return out


def fmt(v, width=8, prec=4):
    return " " * width if v is None else f"{v:{width}.{prec}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights-root", type=Path,
                    default=Path("experiments/weights/group_split_v1"))
    ap.add_argument("--stats-dir", type=Path,
                    default=Path("experiments/pipelines/stats/group_split_v1"),
                    help="for xco2_stats.json, which converts the D=1 task to ppm")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    if not args.weights_root.is_dir():
        raise SystemExit(f"{args.weights_root} does not exist. Run "
                         f"./chtc/fetch_results.sh --cov first.")

    xco2_sigma_ppm = None
    xs = args.stats_dir / "xco2_stats.json"
    if xs.exists():
        xco2_sigma_ppm = float(json.loads(xs.read_text())["sigma"]) * 1e6
        print(f"XCO2 sigma from {xs}: {xco2_sigma_ppm:.4f} ppm "
              f"(standardized x this = ppm)")
    else:
        print(f"WARNING: {xs} not found; the XCO2 rows will have no ppm column.")

    rows: List[Dict] = []
    for d in sorted(args.weights_root.iterdir()):
        if not d.is_dir():
            continue
        task = task_of(d.name)
        if task is None:
            continue
        rec = {"dir": d.name, "task": task, "variant": variant_of(d.name)}
        rec.update(scan_logs(d, task, xco2_sigma_ppm))
        rec.update(selected_metric(d))
        if rec["n_epochs"] == 0 and rec["best_val"] is None:
            continue
        rows.append(rec)

    if not rows:
        raise SystemExit(f"no runs found under {args.weights_root}")

    def sort_key(r):
        v = r["variant"]
        return (VARIANT_ORDER.index(v) if v in VARIANT_ORDER else len(VARIANT_ORDER), v)

    for task in ("xco2", "profiles", "last4"):
        group = [r for r in rows if r["task"] == task]
        if not group:
            continue
        group.sort(key=sort_key)
        print(f"\n{'=' * 78}\n {TASK_LABEL[task]}\n{'=' * 78}")
        print(f"{'covariate set':22s} {'val RMSE':>10s} {'val RMSE':>10s} "
              f"{'ACOS':>9s} {'sel':>5s} {'ep':>4s}")
        print(f"{'':22s} {'[std]':>10s} {'[ppm]':>10s} {'[ppm]':>9s}")
        best_ppm = min((r["ppm"] for r in group if r["ppm"] is not None), default=None)
        best_std = min((r["std"] for r in group if r["std"] is not None), default=None)
        for r in group:
            star = ""
            if best_ppm is not None and r["ppm"] == best_ppm:
                star = "  <-- best"
            elif best_ppm is None and r["std"] == best_std:
                star = "  <-- best"
            print(f"{r['variant']:22s} {fmt(r['std'],10)} {fmt(r['ppm'],10)} "
                  f"{fmt(r['retr_ppm'],9,2)} {str(r['select_metric'] or '-'):>5s} "
                  f"{r['n_epochs']:4d}{star}")

        # The subtractions the runbook asks for, done here so nobody has to do them by eye.
        by = {r["variant"]: r for r in group}
        def diff(a, b, unit="ppm"):
            ra, rb = by.get(a), by.get(b)
            if not ra or not rb or ra[unit] is None or rb[unit] is None:
                return None
            return rb[unit] - ra[unit]           # negative = the second one is better
        suffix = "" if task == "xco2" else "_ppm"
        pairs = [
            ("encoder change alone", f"legacy_v2{suffix}",
             "legacy_ppm" if suffix else "legacy"),
            ("METADATA BUG", f"legacy_fix{suffix}", f"legacy_v2{suffix}"),
            ("+ geometry, prior psfc", f"scene{suffix}", f"legacy_fix{suffix}"),
            ("+ XCO2Prior, AOD priors", f"scene_prior{suffix}", f"scene{suffix}"),
            ("+ latitude, longitude", f"scene_geo{suffix}", f"scene_prior{suffix}"),
            ("+ ACOS output", f"acos_informed{suffix}", f"scene_prior{suffix}"),
        ]
        shown = False
        for label, new, old in pairs:
            d = diff(old, new)
            if d is None:
                continue
            if not shown:
                print(f"\n  {'change':26s} {'delta val RMSE [ppm]':>22s}")
                shown = True
            arrow = "better" if d < 0 else "worse"
            print(f"  {label:26s} {d:+22.4f}   {arrow}")
        if suffix:
            d = diff("legacy_std", "legacy_ppm", "ppm")
            if d is not None:
                print(f"  {'ppm loss + selection':26s} {d:+22.4f}   "
                      f"{'better' if d < 0 else 'worse'}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json_out}")

    print("\nAll numbers are VALIDATION, not test. Pick the covariate set here; spend the")
    print("test set once, afterwards, on the one that wins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

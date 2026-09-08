#!/usr/bin/env python3
"""Rank the sweep trials.

    ./fetch_results.sh --sweep        # unpack the archives into the checkout
    python3 chtc/collect_sweep.py

Reads experiments/weights/sweep/<base>/<trial>/trial.json and prints a table sorted by
validation RMSE, with the baseline (t00, the current *_grouped.yaml) marked.

The number to look at is not the winner but the SPREAD. If sixteen configurations all land
within the epoch-to-epoch noise of the baseline, retuning is not what is holding the model
back and the group-split result stands on its own. If the winner is clearly outside that
band, the published hyperparameters were the problem and the winner should be retrained at
full epochs before anything goes in a table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "experiments" / "weights" / "sweep"

# Epoch-to-epoch oscillation of val_RMSE observed on the group split, in standardized
# units. Differences smaller than this are not evidence of anything from a single run.
NOISE = 0.03


def main() -> int:
    if not SWEEP.exists():
        print(f"no sweep results under {SWEEP}. Run ./chtc/fetch_results.sh --sweep first.")
        return 1

    for base_dir in sorted(p for p in SWEEP.iterdir() if p.is_dir()):
        rows = []
        for f in sorted(base_dir.glob("*/trial.json")):
            try:
                rows.append(json.loads(f.read_text()))
            except Exception as exc:
                print(f"  ! unreadable {f}: {exc}", file=sys.stderr)
        rows = [r for r in rows if r.get("best_val_rmse") is not None]
        if not rows:
            print(f"\n=== {base_dir.name}: no completed trials")
            continue

        rows.sort(key=lambda r: r["best_val_rmse"])
        base = next((r for r in rows if r["trial"] == "t00"), None)

        print(f"\n=== {base_dir.name}  ({len(rows)} trials)")
        hdr = f"{'trial':6s} {'val_RMSE':>9s} {'vs t00':>8s}  overrides"
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            d = ""
            if base is not None:
                delta = r["best_val_rmse"] - base["best_val_rmse"]
                d = f"{delta:+.4f}"
            ov = " ".join(f"{k.split('.')[-1]}={v}" for k, v in r["overrides"].items())
            mark = "  <- baseline" if r["trial"] == "t00" else ""
            print(f"{r['trial']:6s} {r['best_val_rmse']:9.4f} {d:>8s}  {ov}{mark}")

        if base is not None:
            best = rows[0]
            gain = base["best_val_rmse"] - best["best_val_rmse"]
            print()
            if best["trial"] == "t00":
                print(f"  The baseline won. Retuning does not explain the group-split result.")
            elif gain < NOISE:
                print(f"  Best is {gain:.4f} better than the baseline, which is inside the "
                      f"±{NOISE} epoch-to-epoch noise.")
                print(f"  Treat that as 'no configuration in this grid helps', not as a win.")
            else:
                print(f"  Best is {gain:.4f} better than the baseline, outside the ±{NOISE} "
                      f"noise band.")
                print(f"  Retrain {best['trial']} at full epochs and with 3 seeds before "
                      f"putting it in a table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

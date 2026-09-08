#!/usr/bin/env python3
"""Turn a selection record in hyperparameters/ into command-line overrides.

    python3 scripts/overrides.py hyperparameters/bo_det_profiles_best.json
    data.train_batch_size=512 hyps.embed_dim=256 hyps.lr=0.0007991836574 ...

    # the deterministic trainers are Hydra, so the overrides go straight on the line:
    python3 experiments/train/train_profiles.py +experiment=full_regressor_profiles_cov_grouped \\
        covariates=scene_prior select_metric=ppm hyps.epochs=40 \\
        $(python3 scripts/overrides.py hyperparameters/bo_det_profiles_best.json)

    # the flow trainers take --set:
    python3 experiments/train/train_nf_profiles.py \\
        --exp_cfg code/shared_conf/experiment/nf_profiles_cov_grouped.yaml \\
        --set covariates=scene_prior epochs=40 \\
        $(python3 scripts/overrides.py hyperparameters/bo_nf_profiles_best.json)

The point of reading them from the record rather than typing them is that the model that
gets trained and the configuration that was selected are then the same object by
construction. The record also carries the value it achieved, the round and trial it came
from, and the baseline it was compared against -- `--show` prints all of that.

`--json` re-emits the parameters as a dict, for a driver that would rather not shell out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record", type=Path, help="a hyperparameters/bo_<study>_best.json")
    ap.add_argument("--show", action="store_true",
                    help="print the whole record in readable form instead of the overrides")
    ap.add_argument("--json", action="store_true",
                    help="print the parameters as a JSON object")
    args = ap.parse_args(argv)

    try:
        rec = json.loads(args.record.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"{args.record}: no such file. The six records live in hyperparameters/.")
    except json.JSONDecodeError as exc:
        sys.exit(f"{args.record}: not valid JSON ({exc}).")

    overrides = rec.get("overrides")
    if not overrides:
        sys.exit(f"{args.record}: carries no `overrides`, so it cannot be turned back "
                 f"into a run. Keys present: {sorted(rec)}")

    if args.show:
        print(f"study      : {rec.get('study')}")
        print(f"covariates : {rec.get('covariates')}")
        print(f"selected on: {rec.get('metric_label', rec.get('metric'))} = {rec.get('value')}")
        print(f"             round {rec.get('round')}, trial {rec.get('trial')}, "
              f"at {rec.get('trial_epochs')} epochs "
              f"({rec.get('n_completed')} of {rec.get('n_trials')} trials completed)")
        base = rec.get("baseline_value")
        if base is not None:
            better = rec.get("search_beat_baseline")
            print(f"baseline   : {base}  -> the search "
                  f"{'improved on' if better else 'did NOT improve on'} it")
        print("overrides  :")
        for o in overrides:
            print(f"    {o}")
        return 0

    if args.json:
        print(json.dumps(rec.get("params", {}), indent=2, sort_keys=True))
        return 0

    # One line, space separated, so `$(...)` in a shell expands to one argument each.
    # Nothing here needs quoting: every override is `key=value` with no whitespace, and
    # the trainers are given the epoch count and the covariate preset separately because
    # those are properties of the run rather than of the selected configuration.
    print(" ".join(overrides))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

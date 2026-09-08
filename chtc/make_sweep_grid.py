#!/usr/bin/env python3
"""Write the hyperparameter grids that sweep.sub queues from.

    python3 chtc/make_sweep_grid.py

Deterministic: the same seed produces the same 16 trials, and the files are committed, so
the search is part of the record rather than something that happened once on a Friday.

Why a random search and not BO. The published models were tuned with Optuna over ~100
trials *on the replicate split*. Reusing those winners on the reference-sounding-disjoint
split is what makes the current comparison attackable -- "you transferred someone else's
tuning and then reported that it did not work". A 16-point random search is not equivalent
to 100 BO trials, and should not be described as such; it is enough to answer the only
question that matters here, which is whether the group-split result survives retuning. Say
"a coarse random search over lr, weight decay, width and depth" in the paper, and give the
grid.

Trial t00 is always the current config, so every sweep contains its own baseline.
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N_TRIALS = 16
SEED = 20260814

# nhead must divide embed_dim; 4 and 8 divide all four widths below.
WIDTHS = [192, 256, 320, 384]
HEADS = [4, 8]
DEPTHS = [1, 2, 3]

BASELINES = {
    # base job -> the hyperparameters its *_grouped.yaml currently uses
    "det_profiles": dict(lr=8.0e-4, wd=9.1e-4, embed_dim=256, nhead=8, nlayers=1),
    "det_last4": dict(lr=1.95e-4, wd=4.76e-4, embed_dim=320, nhead=4, nlayers=2),
}


def grid(base: str, rng: np.random.Generator) -> list:
    rows = [("t00", BASELINES[base])]
    for i in range(1, N_TRIALS):
        rows.append((f"t{i:02d}", dict(
            lr=float(10 ** rng.uniform(np.log10(5e-5), np.log10(5e-3))),
            wd=float(10 ** rng.uniform(np.log10(1e-6), np.log10(3e-3))),
            embed_dim=int(rng.choice(WIDTHS)),
            nhead=int(rng.choice(HEADS)),
            nlayers=int(rng.choice(DEPTHS)),
        )))
    return rows


def main() -> int:
    out_dir = HERE / "sweeps"
    out_dir.mkdir(exist_ok=True)
    for base in BASELINES:
        rng = np.random.default_rng(SEED + sum(map(ord, base)))

        # NO COMMENTS AND NO BLANK LINES IN THIS FILE.
        #
        # `queue <vars> from <file>` does not skip them -- it makes a job per line. Three
        # header comments turned a 16-trial sweep into 19 jobs, three of which parsed as
        # base="# det_last4: 16 trials" and had output filenames containing spaces and a
        # '#'. Provenance goes in the .info sidecar instead.
        #
        # No space after the comma either. HTCondor does trim the fields, but the trial id
        # ends up in six filenames (log, out, err, the archive, and both remap sides), and
        # relying on that trimming for all of them is a bet with no upside.
        rows = grid(base, rng)
        lines = []
        for trial, hp in rows:
            ov = " ".join(f"hyps.{k}={v:.6g}" if isinstance(v, float) else f"hyps.{k}={v}"
                          for k, v in hp.items())
            lines.append(f"{base},{trial},{ov}")
        path = out_dir / f"{base}.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        info = out_dir / f"{base}.info"
        info.write_text(
            f"{base}: {N_TRIALS} trials, seed {SEED}.\n"
            f"Regenerate with `python3 chtc/make_sweep_grid.py`.\n"
            f"Columns in {base}.txt: base,trial,hydra-overrides -- one job per line,\n"
            f"no comments, because condor_submit queues comment lines as jobs.\n"
            f"t00 is the current *_grouped.yaml configuration: the sweep's own baseline.\n",
            encoding="utf-8")
        print(f"wrote {path}  ({len(lines)} lines = {len(lines)} jobs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

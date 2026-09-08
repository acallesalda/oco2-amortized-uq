"""prepare_month.py
Turn one raw simulation month into the spectra, targets and ACOS covariances that
`predict.py` reads -- so a model trained on February can be scored on March or July
without retraining anything.

Why this exists rather than re-running the February pipeline
-----------------------------------------------------------
February went through `extract_data_tree.py` (HDF5 -> zarr DataTree) and then
`clean_data.py` (DataTree -> .npy + aux_data.csv). That is two heavy passes and a large
intermediate, and it does one thing this script must not do: it decides which spectral
channels to keep *from the data in front of it* --

    missing = dt.map_over_datasets(get_missing, "measured_radiance")
    mask    = np.all(missing, axis=0)          # over February
    ... drop_along_coord("wavelength_index", mask)

-- so run against a new month it would choose that month's channels. If the two sets ever
differed, the network would be handed an input of a different width, and the failure mode
is not a crash, it is a number. Here the channel set is derived the same way and then
CHECKED against a February .npy; the job stops if they disagree.

Two fill conventions, both load-bearing
---------------------------------------
1. `_FillValue = -9999` is declared on most datasets. 750 of the 3048 spectral channels
   are -9999 in every sounding of every file -- the padding between the three bands -- and
   those are exactly the channels February dropped, leaving 2298.
2. The 49-element state vectors carry a *second*, undeclared sentinel of order -1e19, on
   about one sounding in a thousand. `np.isfinite` does not catch it and neither does the
   -9999 test. One of them turns a scene's RMSE into 5e8 ppm. CO2 is a mole fraction, so
   the test used here is physical: anything outside [1e-6, 1e-2] is not a measurement of
   it, and the sounding is dropped with a count in the summary.

What it does NOT produce
------------------------
The covariates. `extract_covariates.py` already derives them -- the corrected `*_fix`
metadata under the sorted-reference pairing, the geometry (`sec_sza`, `sec_vza`,
`vza_deg`), `prior_psfc_pa`, the priors -- and it is month-agnostic as written. Run it
against the same `--raw-root`; `make_month_split.py` merges the two.

Everything here is keyed by the cluster and reference numbers in the *filename*, which is
what `add_uniq_id` built the February IDs from, so nothing in this script depends on the
metadata pairing that the published preprocessing got wrong.

Usage:
    python3 prepare_month.py --raw-root <month>/og_data --month 202003 \\
        --files-root scratch/simulated/files \\
        --reference-npy scratch/simulated/files/train/clst0001_ref0001_rep0001.npy
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# IDs are written exactly as oco2_utils.add_uniq_id does: clst%04d_ref%04d_rep%04d.
_ID_FMT = "clst{clst:04d}_ref{ref:04d}_rep{rep:04d}"
_REF_RE = re.compile(r"ref(\d+)")
_CLST_RE = re.compile(r"[Cc]l(?:uste)?r?s?t?(\d+)")

FILL = -9999.0
N_LEVELS = 20
# A CO2 mole fraction. Anything outside this is a sentinel, whatever it claims to be.
VMR_LO, VMR_HI = 1e-6, 1e-2

TRUE_COLS = [f"true_state_vector_{i:02d}" for i in range(1, N_LEVELS + 1)]
RETR_COLS = [f"retrieved_state_vector_{i:02d}" for i in range(1, N_LEVELS + 1)]


def cluster_number(path: Path) -> int:
    m = _CLST_RE.search(path.name)
    if not m:
        raise ValueError(f"cannot read a cluster number from {path.name}")
    return int(m.group(1))


def reference_number(path: Path) -> int:
    m = _REF_RE.search(path.name)
    if not m:
        raise ValueError(f"cannot read a reference number from {path.name}")
    return int(m.group(1))


def valid_vmr(a: np.ndarray) -> np.ndarray:
    """Elementwise: is this a plausible CO2 mole fraction?"""
    a = np.asarray(a, dtype=np.float64)
    return np.isfinite(a) & (a > VMR_LO) & (a < VMR_HI)


def channel_mask(files: List[Path]) -> np.ndarray:
    """Channels to KEEP: those not fill-valued in every sounding of every file.

    The same rule clean_data.py applied to February, over the same kind of population.
    Computed here rather than hardcoded so that a month with a different instrument
    configuration would produce a different mask -- and then fail the check against the
    February reference, loudly, instead of being silently reshaped.
    """
    always_fill = None
    for path in files:
        with h5py.File(path, "r") as f:
            rad = f["SpectralParameters/measured_radiance"][:]
        m = np.isclose(rad, FILL).all(axis=0)
        always_fill = m if always_fill is None else (always_fill & m)
    return ~always_fill


def read_reference(path: Path, clst: int, ref: int, keep: np.ndarray, rep_offset: int = 0):
    """(aux rows, spectra [n, L, 2], covariances [n, 20, 20]) for one reference sounding.

    `rep_offset` shifts the replicate index in the ID. It exists for the February
    extension, which re-simulates 34 of February's own reference soundings with fresh
    realisations: those files carry `ref01..ref18` in clusters 8 and 9, exactly the numbers
    February already owns, so `clst0008_ref0001_rep0001` would name two different
    soundings. The scenes are the same scenes -- binding them to the same `ref` is what
    keeps a reference-disjoint split from leaking -- so it is the *replicate* index that
    has to move. Offset 100 puts the extension's 200 replicates at rep0101..rep0300,
    beyond February's rep0001..rep0100, and leaves the reference number alone.
    """
    with h5py.File(path, "r") as f:
        wl = f["SpectralParameters/wavelength"][:][:, keep].astype(np.float32)
        rad = f["SpectralParameters/measured_radiance"][:][:, keep].astype(np.float32)
        true_sv = f["StateVector/true_state_vector"][:, :N_LEVELS].astype(np.float64)
        retr_sv = f["RetrievalResults/retrieved_state_vector"][:, :N_LEVELS].astype(np.float64)
        true_xco2 = f["StateVector/true_xco2"][:].astype(np.float64)
        retr_xco2 = f["RetrievalResults/retrieved_xco2"][:].astype(np.float64)
        retr_var = f["RetrievalResults/retrieved_xco2_variance"][:].astype(np.float64)
        # The 20x20 CO2 block of the 49x49 posterior. save_data_with_covariance wrote
        # exactly this slice for February, and evals/acos.py reads it as <ID>_cov.npy.
        cov = f["RetrievalResults/retrieved_state_vector_covariance"][:, :N_LEVELS, :N_LEVELS]
        cov = cov.astype(np.float32)
        sid = f["RetrievalHeader/sounding_id"][:]

    n = len(true_xco2)
    # A row is usable if its truth is: that is what clean_data.py filtered on, and a row
    # with no truth cannot be scored. A bad *retrieval* is kept and NaN-ed -- the ACOS row
    # loses that sounding, every other row keeps it, and evals/acos.py already handles a
    # missing covariance that way.
    ok = valid_vmr(true_xco2) & valid_vmr(true_sv).all(axis=1)
    retr_bad = ~(valid_vmr(retr_xco2) & valid_vmr(retr_sv).all(axis=1))

    retr_xco2 = np.where(valid_vmr(retr_xco2), retr_xco2, np.nan)
    retr_sv = np.where(valid_vmr(retr_sv), retr_sv, np.nan)
    retr_var = np.where(np.isfinite(retr_var) & (retr_var > 0), retr_var, np.nan)

    aux = pd.DataFrame({
        "ID": [_ID_FMT.format(clst=clst, ref=ref, rep=i + 1 + rep_offset) for i in range(n)],
        "sounding_id": sid,
        "true_xco2": true_xco2,
        "retrieved_xco2": retr_xco2,
        "retrieved_xco2_variance": retr_var,
    })
    for j, c in enumerate(TRUE_COLS):
        aux[c] = true_sv[:, j]
    for j, c in enumerate(RETR_COLS):
        aux[c] = retr_sv[:, j]

    # clean_data.py fills the remaining radiance NaNs with 1.0 after dropping the empty
    # channels; a sounding whose whole spectrum is fill therefore reaches the network as a
    # constant. That is what February did, so it is what this does, and the count is
    # reported rather than silently absorbed.
    rad = np.where(np.isclose(rad, FILL) | ~np.isfinite(rad), 1.0, rad)
    wl = np.where(np.isclose(wl, FILL) | ~np.isfinite(wl), 1.0, wl)
    spectra = np.stack([wl, rad], axis=2)

    dead = int((rad == 1.0).all(axis=1).sum())
    return (aux.loc[ok].reset_index(drop=True), spectra[ok], cov[ok],
            dict(dropped=int((~ok).sum()), retr_bad=int(retr_bad[ok].sum()), dead=dead))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-root", type=Path, required=True,
                   help="directory holding the ClusterNN/ subdirectories")
    p.add_argument("--month", required=True, help="e.g. 202003; names the output directory")
    p.add_argument("--files-root", type=Path, required=True,
                   help="scratch/simulated/files; the month lands in <files-root>/<month>")
    p.add_argument("--reference-npy", type=Path, default=None,
                   help="any February .npy. Its channel count is checked against this "
                        "month's, because a model cannot be scored on a different input "
                        "layout and the failure is otherwise silent.")
    p.add_argument("--mask-sample", type=int, default=20,
                   help="how many files to derive the channel mask from. The empty "
                        "channels are an instrument constant, so a sample is enough; 0 "
                        "means every file.")
    p.add_argument("--rep-offset", type=int, default=0,
                   help="add this to every replicate index in the ID. 100 for the "
                        "February extension, whose re-simulated scenes share February's "
                        "cluster and reference numbers; see read_reference.")
    p.add_argument("--limit-clusters", type=int, default=0, help="smoke tests only")
    args = p.parse_args()

    out_dir = args.files_root / args.month
    out_dir.mkdir(parents=True, exist_ok=True)

    clusters = sorted((d for d in args.raw_root.iterdir() if d.is_dir()), key=cluster_number)
    if args.limit_clusters:
        clusters = clusters[:args.limit_clusters]
    if not clusters:
        raise SystemExit(f"no cluster directories under {args.raw_root}")
    all_files = [f for c in clusters for f in sorted(c.iterdir()) if f.suffix == ".h5"]

    sample = all_files if args.mask_sample <= 0 else all_files[:args.mask_sample]
    keep = channel_mask(sample)
    logger.info("channel mask from %d file(s): keeping %d of %d channels",
                len(sample), int(keep.sum()), len(keep))

    if args.reference_npy is not None:
        ref_arr = np.load(args.reference_npy)
        if int(ref_arr.shape[0]) != int(keep.sum()):
            raise SystemExit(
                f"February's spectra have {ref_arr.shape[0]} channels, this month's mask "
                f"keeps {int(keep.sum())}. The band encoders are built for February's "
                f"layout, so scoring this month would be scoring a different input. "
                f"Stopping rather than producing a number.")
        logger.info("channel count matches February (%d)", int(ref_arr.shape[0]))
    else:
        logger.warning("no --reference-npy: the channel layout is NOT being checked "
                       "against February. Pass one before trusting any number from this.")

    frames: List[pd.DataFrame] = []
    n_files = n_sound = 0
    tot = dict(dropped=0, retr_bad=0, dead=0)
    for cdir in clusters:
        clst = cluster_number(cdir)
        files = sorted((f for f in cdir.iterdir() if f.suffix == ".h5"), key=reference_number)
        for path in files:
            ref = reference_number(path)
            aux, spectra, cov, stats = read_reference(path, clst, ref, keep,
                                                      args.rep_offset)
            for i, sid in enumerate(aux["ID"]):
                np.save(out_dir / f"{sid}.npy", spectra[i])
                np.save(out_dir / f"{sid}_cov.npy", cov[i])
            for k in tot:
                tot[k] += stats[k]
            n_sound += len(aux)
            n_files += 1
            frames.append(aux)
        logger.info("cluster %d: %d reference soundings", clst, len(files))

    aux_all = pd.concat(frames, ignore_index=True)
    aux_all.to_csv(out_dir / "aux_data.csv", index=False)

    meta = {
        "month": args.month,
        "raw_root": str(args.raw_root),
        "rep_offset": int(args.rep_offset),
        "n_reference_soundings": n_files,
        "n_soundings": int(n_sound),
        "n_clusters": len(clusters),
        "spectral_channels": int(keep.sum()),
        "dropped_no_truth": tot["dropped"],
        "retrieval_unusable": tot["retr_bad"],
        "spectra_entirely_fill": tot["dead"],
        "true_xco2_mean_ppm": float(aux_all["true_xco2"].mean() * 1e6),
        "true_xco2_sd_ppm": float(aux_all["true_xco2"].std() * 1e6),
        "acos_bias_ppm": float((aux_all["retrieved_xco2"] - aux_all["true_xco2"]).mean() * 1e6),
    }
    (out_dir / "prepare_month.summary.json").write_text(json.dumps(meta, indent=2))

    print()
    print(json.dumps(meta, indent=2))
    print()
    print(f"wrote {2 * n_sound} .npy files and aux_data.csv to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

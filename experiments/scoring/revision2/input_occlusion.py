#!/usr/bin/env python3
"""input_occlusion.py -- occlusion of the climatological inputs (second-round revision, R2.2).

Produces the auxiliary-input rows and the "all three" row of Table 9 and all of Table 10.
Inside the fitted XCO2 models (Laplace backbone and normalizing flow), replace an input and
re-score:
  * the a priori column (XCO2Prior),
  * the observing geometry (SZA, sec SZA, VZA, sec VZA, airmass),
  * both together,
  * every continuous auxiliary input (adds prior surface pressure and the two AOD priors),
  * the three spectral bands together.
Each is replaced three ways:
  zero     the raw value 0 (0 ppm, 0 degrees, zero radiance);
  mean     the training-partition mean (Table 9, and the "overall mean" columns of Table 10);
  cluster  the training mean of the sounding's own geophysical cluster (Table 10).
Continuous auxiliary inputs are standardized with training statistics, so their training mean
is 0 in model units. Spectra are replaced in the radiance channel only; the wavelength channel
and the band scale constants are kept, as in spectral_attribution.py. Categorical inputs
(footprint, aerosol types) are left intact. The training means are computed over the full
training partition and cached next to --out.

The single-band rows of Table 9 are spectral_attribution.py's, unchanged from the first
revision. That script's substitution spectrum is the mean of the first 32 training batches
(`--mean-batches 32`), not of the full partition; the "all three" row was rerun here with the
full-partition mean, which raises its cost slightly and changes no conclusion.

Deltas carry paired bootstrap SEs resampling reference soundings. The flow's RMSE is that of
the mean of 256 posterior draws (--nf-samples), as in spectral_attribution.py.

Reference-disjoint (Table 9; Table 10 top block):
    python3 experiments/scoring/revision2/input_occlusion.py --setting ref_disjoint \
        --split-csv scratch/simulated/files/splits/group_split_v1_cov.csv \
        --stats-dir experiments/pipelines/stats/group_split_v1 \
        --weights-root experiments/weights/group_split_v1 --out occlusion/ref_disjoint.csv
Within-reference (Table 10 bottom block): the same with replicate_split_v1 in all three paths.

Needs a GPU, the published weights, the split tables and the processed spectra.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
ap.add_argument("--setting", required=True)
ap.add_argument("--split-csv", required=True)
ap.add_argument("--stats-dir", required=True)
ap.add_argument("--weights-root", required=True)
ap.add_argument("--files-root", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--nf-samples", type=int, default=256)
ap.add_argument("--n-boot", type=int, default=400)
ap.add_argument("--batch-size", type=int, default=256)
ap.add_argument("--num-workers", type=int, default=0)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)

REPO = Path(args.repo).resolve()
sys.path.insert(0, str(REPO / "experiments" / "scoring"))
import spectral_attribution as SA                            # noqa: E402  (sets up paths)
from data.loading import spectral_loader as SL               # noqa: E402
from data.loading.covariates import get_spec                 # noqa: E402
from evals import checkpoints as ck                          # noqa: E402
from predict import TASKS, build_data, target_stats          # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("input_occlusion")

COV = "scene_prior"
CONT = list(get_spec(COV).continuous)
GEOM = ["SlrZen_fix", "sec_sza", "vza_deg", "sec_vza", "airmass"]
GROUPS = [
    ("XCO2 prior", ["XCO2Prior"], False),
    ("geometry", GEOM, False),
    ("XCO2 prior + geometry", GEOM + ["XCO2Prior"], False),
    ("all continuous auxiliary", CONT, False),
    ("all three bands", [], True),
]
# zero: the raw value 0 (0 ppm, 0 degrees, zero radiance); mean: the training mean;
# cluster: the training mean of the sounding's own geophysical cluster.
MODES = ("zero", "mean", "cluster")
CONFIGS = [("intact", "none", [], False)] + [
    (f"{g} [{m}]", m, cols, bt) for g, cols, bt in GROUPS for m in MODES]

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda")
spec = TASKS["xco2"]
files_root = args.files_root or str(REPO / "scratch/simulated/files")

da = SimpleNamespace(files_root=files_root, annotations="aux_data.csv",
                     split_csv=args.split_csv, stats_dir=args.stats_dir,
                     covariates=COV, eval_partition="test", seed=args.seed, train_size=0.9)
test_ds, train_ds, _val, frame, _src, split_name = build_data(da, spec)
_mu, y_sigma = target_stats(da, spec)
ppm = float(np.mean(y_sigma)) * 1e6

batches = SA.cache_partition(test_ds, device, args.batch_size, args.num_workers,
                             limit=args.limit)
n = sum(int(y.shape[0]) for _, y, _ in batches)
frame = frame.iloc[:n].reset_index(drop=True)
groups = (frame["clst"].astype(str) + "/" + frame["ref"].astype(str)).to_numpy()

# Table 7 drops the one July sounding whose ACOS profile is non-physical from the point-error
# columns of all methods; apply the same rule everywhere (it removes nothing elsewhere).
sv = [f"retrieved_state_vector_{i:02d}" for i in range(1, 21)]
keep = np.ones(n, dtype=bool)
if set(sv).issubset(frame.columns):
    v = frame[sv].to_numpy(dtype=np.float64)
    keep = ~((v < 0).any(axis=1) | np.isnan(v).any(axis=1))
log.info("scored %d soundings, %d excluded from RMSE", n, int((~keep).sum()))

nreal = SA.n_real_channels(batches)
aux_c0 = batches[0][0][1]
assert aux_c0.shape[1] == len(CONT), (aux_c0.shape, CONT)
cov_stats = SL.load_covariate_stats(Path(args.stats_dir))
table = cov_stats["stats"] if "stats" in cov_stats else cov_stats

# ---- substitution values, from the TRAINING partition only -------------------------------
# one pass over every training sounding: overall and per-cluster means of the normalized
# radiance (per channel) and of the standardized continuous covariates.
train_frame = SL.load_split_table(args.split_csv, "train")
assert len(train_frame) == len(train_ds)
clusters = np.sort(train_frame["clst"].unique())
cidx = {c: i for i, c in enumerate(clusters)}
K, L = len(clusters), batches[0][0][0][SA.BANDS[0]][0].shape[-1]
# The training partition is a property of the split, not of the scored month, so the sums
# are cached per (stats dir, training IDs): March and July share February's.
import hashlib                                                # str hash() is salted per process
key = f"{Path(args.stats_dir).name}_{len(train_frame)}_" + hashlib.md5(
    ",".join(train_frame["ID"].astype(str)).encode()).hexdigest()[:8]
cache = Path(args.out).parent / f"train_means_{key}.pt"
if cache.exists():
    d = torch.load(cache, map_location=device)
    rad_sum, aux_sum, cnt = d["rad_sum"], d["aux_sum"], d["cnt"]
    log.info("training means loaded from %s", cache)
else:
    rad_sum = {b: torch.zeros(K, L, dtype=torch.float64, device=device) for b in SA.BANDS}
    aux_sum = torch.zeros(K, len(CONT), dtype=torch.float64, device=device)
    cnt = torch.zeros(K, dtype=torch.float64, device=device)
    tr_c = torch.as_tensor(train_frame["clst"].map(cidx).to_numpy(), device=device)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_workers)
    off = 0
    for batch in loader:
        (bands, aux_c, _), _y, _r = SA.lap.move_batch(batch, device)
        B = aux_c.shape[0]
        c = tr_c[off:off + B]
        off += B
        cnt.index_add_(0, c, torch.ones(B, dtype=torch.float64, device=device))
        aux_sum.index_add_(0, c, aux_c.double())
        for b in SA.BANDS:
            rad_sum[b].index_add_(0, c, bands[b][0][:, SA.RAD, :].double())
    assert off == len(train_frame)
    torch.save({"rad_sum": rad_sum, "aux_sum": aux_sum, "cnt": cnt}, cache)
    log.info("training means from %d soundings over %d clusters", off, K)
rad_clu = {b: (rad_sum[b] / cnt[:, None]).float() for b in SA.BANDS}
rad_all = {b: (rad_sum[b].sum(0) / cnt.sum()).float() for b in SA.BANDS}
aux_clu = (aux_sum / cnt[:, None]).float()
aux_zero = torch.tensor([(0.0 - table[c]["mean"]) / table[c]["std"] for c in CONT],
                        device=device, dtype=torch.float32)

# per-sounding cluster index of the scored rows, batch-aligned
te_clst = frame["clst"].to_numpy()
unseen = sorted(set(te_clst) - set(clusters))
if unseen:
    raise SystemExit(f"test clusters {unseen} have no training rows")
te_c = torch.as_tensor(pd.Series(te_clst).map(cidx).to_numpy(), device=device)


def substitute(bands, aux_c, c, mode, cols, bands_too):
    """Copies of the inputs with `cols` (and the spectra, if `bands_too`) replaced."""
    if mode == "none":
        return bands, aux_c
    if cols:
        idx = [CONT.index(k) for k in cols]
        aux_c = aux_c.clone()
        if mode == "zero":
            aux_c[:, idx] = aux_zero[idx]
        elif mode == "mean":
            aux_c[:, idx] = 0.0                      # training mean in standardized units
        else:
            aux_c[:, idx] = aux_clu[c][:, idx]
    if bands_too:
        out = {b: (t.clone(), s) for b, (t, s) in bands.items()}
        for b in SA.BANDS:
            hi = nreal[b]
            if mode == "zero":
                out[b][0][:, SA.RAD, :hi] = 0.0
            elif mode == "mean":
                out[b][0][:, SA.RAD, :hi] = rad_all[b][:hi]
            else:
                out[b][0][:, SA.RAD, :hi] = rad_clu[b][c][:, :hi]
        bands = out
    return bands, aux_c
warm = batches[0][0]
W = Path(args.weights_root) / "bo_final"
det = ck.load_deterministic(args.weights_root, spec["det"], device, warm_x_tuple=warm,
                            cov_stats=cov_stats, covariates=COV,
                            ckpt_path=W / "det_xco2/global_best/net.pt").module.eval()
flow = ck.load_flow(args.weights_root, spec["flow"], device, warm_x_tuple=warm,
                    cov_stats=cov_stats, covariates=COV,
                    ckpt_path=W / "nf_xco2/best_flow_nflows.pt").module.eval()


@torch.no_grad()
def score(mode, cols, bands_too):
    se_d, nll, se_f = [], [], []
    off = 0
    for x, y, _ in batches:
        bands, aux_c, aux_k = x
        c = te_c[off:off + y.shape[0]]
        off += y.shape[0]
        b2, a2 = substitute(bands, aux_c, c, mode, cols, bands_too)
        xo = (b2, a2, aux_k)
        se_d.append(((det(xo) - y) ** 2).mean(dim=1).cpu().numpy())
        nll.append((-flow.log_prob(xo, y)).cpu().numpy())
        s = flow.sample(xo, n=args.nf_samples)
        se_f.append(((s.mean(dim=0) - y) ** 2).mean(dim=1).cpu().numpy())
    return [np.concatenate(a)[keep] for a in (se_d, nll, se_f)]


g = groups[keep]
rows, base = [], None
for name, mode, cols, bands_too in CONFIGS:
    se_d, nll, se_f = score(mode, cols, bands_too)
    if base is None:
        base = (se_d, nll, se_f)
    rec = dict(setting=args.setting, split=split_name, config=name, mode=mode,
               n=int(keep.sum()), scenes=int(np.unique(g).size))
    for key, arr, b, kind in [("laplace_rmse", se_d, base[0], "rmse"),
                              ("flow_rmse", se_f, base[2], "rmse"),
                              ("flow_nll", nll, base[1], "mean")]:
        if kind == "rmse":
            m, se, dse = SA.bootstrap_rmse(arr, g, ppm, args.n_boot, args.seed, base_se=b)
            d = m - float(np.sqrt(np.mean(b)) * ppm)
        else:
            m, se, dse = SA.bootstrap_mean(arr, g, args.n_boot, args.seed, base=b)
            d = m - float(np.mean(b))
        rec.update({key: m, f"{key}_se": se, f"d_{key}": d, f"d_{key}_se": dse})
    rows.append(rec)
    log.info("%-26s Laplace %.3f (%+.3f)  flow %.3f (%+.3f)  NLL %.3f (%+.3f)", name,
             rec["laplace_rmse"], rec["d_laplace_rmse"], rec["flow_rmse"],
             rec["d_flow_rmse"], rec["flow_nll"], rec["d_flow_nll"])

pd.DataFrame(rows).to_csv(args.out, index=False)
log.info("wrote %s", args.out)

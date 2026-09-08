#!/usr/bin/env python3
"""predict.py -- the GPU half of scoring.

Runs one trained task over the held-out partition and writes every predictive quantity
for every test sounding to a single `.npz`. It computes no metric and draws no figure;
`score.py` and `figures.py` do that from the file this produces.

The split exists because the two halves have nothing in common operationally. This half
needs a GPU, the training data, the checkpoints and a batch slot, and takes tens of
minutes; the other half is seconds of numpy on a laptop and gets re-run every time a
caption changes. Merging them, as the notebooks did, means paying for the first every
time you want the second -- on a shared cluster, in queue time.

    python3 experiments/scoring/predict.py --task profiles \
        --split-csv  scratch/simulated/files/splits/group_split_v1.csv \
        --stats-dir  experiments/pipelines/stats/group_split_v1 \
        --weights-root experiments/weights/group_split_v1 \
        --out results/group_split_v1/profiles.npz

Omit `--split-csv` to evaluate on the original replicate-level split, which is how the
published Tables 1-3 were produced and what the revision reports as the secondary
"within-reference interpolation" result. Everything else, including the code path, stays
identical -- which is the point: the difference between the two numbers is then the split
and nothing else.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# The project puts `code/` on sys.path rather than installing it.
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
for p in (str(_ROOT / "code"), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from data.loading import spectral_loader as SL              # noqa: E402
from evals import acos as acos_mod                          # noqa: E402
from evals import checkpoints as ck                         # noqa: E402
from evals import gaussianize as gz                         # noqa: E402
from evals import laplace_last_layer as lap                 # noqa: E402
from evals.artifacts import Artifact, ArtifactMeta          # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
logger = logging.getLogger("predict")


# ---------------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------------
# `levels` are 1-based pressure levels of the 20-level target vector. The near-surface
# models emit 4 outputs natively (they were trained with target_indices=last4), so the
# slice applies to the *target*, never to the model output.
#
# `model_levels` is what the checkpoint emits. It differs from `levels` only for
# `last4_marginal`, where the 20-level models are evaluated on the near-surface levels --
# the "marginalized" rows of the published Table 3, which the native `last4` task cannot
# produce because it loads different checkpoints. There the model is fitted, calibrated
# and run over all 20 outputs exactly as in `profiles`, and the predictive is sliced at
# the very end; for a Gaussian that slice IS the marginal, exactly.
TASKS = {
    # `mc`, not `marglik`. The `marglik` path was chosen on the belief that the library's
    # own predictive supplies the observation-noise term for a regression Laplace, so that
    # `calibrate_aleatoric` would be double counting (see posterior_mean_cov's docstring).
    # It does not, and the artifacts say so: for every `xco2` run ever scored,
    # `lap_cov_std` is bit-identical to `lap_cov_epi_std`, the mean predictive standard
    # deviation is 0.127 in standardized units against a residual standard deviation of
    # 0.715, and `sigma_noise` is recorded as exactly 1.0 -- the constructor default,
    # because `optimize_prior_precision(method="marglik")` optimizes the prior precision
    # and nothing else. The scalar Laplace predictive was epistemic-only and 5.7x too
    # narrow, which is the whole of its 38.4% coverage at nominal 95%.
    #
    # `mc` calibrates the aleatoric diagonal on the validation partition, which is what
    # the other four tasks have always done and what makes Table 1 comparable with Tables
    # 2 and 3. Re-scoring is required for any xco2/last1 artifact produced before this.
    "xco2":     dict(det="det_xco2",     flow="nf_xco2",     levels=None,
                     laplace_mode="mc", hessian="full"),
    "profiles": dict(det="det_profiles", flow="nf_profiles", levels=list(range(1, 21)),
                     laplace_mode="mc", hessian="diag"),
    "last4":    dict(det="det_last4",    flow="nf_last4",    levels=[17, 18, 19, 20],
                     laplace_mode="mc", hessian="diag"),
    "last4_marginal": dict(
        det="det_profiles", flow="nf_profiles", levels=[17, 18, 19, 20],
        model_levels=list(range(1, 21)), laplace_mode="mc", hessian="diag",
        # Scored into table_last4 as a second block, so the native and marginalized rows
        # sit in one table the way the published one did.
        table_task="last4", split_suffix="__marginal",
    ),
    # The surface-only scalar task: the CO2 concentration of the bottom layer, level 20.
    #
    # It is the near-surface counterpart of `xco2`, and it exists for the same reason
    # that task does: with D=1 there is no joint-vs-marginal coverage question and no
    # near-singular covariance, so the comparison against ACOS is as clean as it gets --
    # one number, one variance, one NLL. Level 20 is also where the spectrum is least
    # informative and where ACOS's own posterior is widest, which makes it the scalar the
    # amortized retrieval has the best chance of winning outright.
    #
    # A full Hessian, matching `xco2` rather than `last4`, but `mc` for the aleatoric term
    # for the reason given above: at D=1 that term is a single scalar, and the
    # marginal-likelihood fit does not estimate it. No `last1` artifact was ever produced
    # (the job returned an empty archive twice), so nothing reported carries the defect
    # here -- but the entry mirrored `xco2`'s and would have.
    "last1": dict(det="det_last1", flow="nf_last1", levels=[20],
                  laplace_mode="mc", hessian="full"),
    # The 20-level models scored on level 20 alone -- the same marginalization
    # `last4_marginal` performs, one level further in. It answers whether a model trained
    # to predict the surface layer *only* beats one trained on the whole profile and then
    # asked about the surface, which is the obvious question as soon as the
    # native row appears. Rows land in table_last1 beside the native ones.
    "last1_marginal": dict(
        det="det_profiles", flow="nf_profiles", levels=[20],
        model_levels=list(range(1, 21)), laplace_mode="mc", hessian="diag",
        table_task="last1", split_suffix="__marginal",
    ),
}


def model_slice(spec):
    """Indices of `levels` within the model's own output, or None if they coincide."""
    ml = spec.get("model_levels")
    if ml is None or list(ml) == list(spec["levels"] or []):
        return None
    pos = {lv: i for i, lv in enumerate(ml)}
    return [pos[lv] for lv in spec["levels"]]


def build_data(args, spec):
    """Evaluation / train / validation datasets, plus the annotation frame for the rows scored.

    The frame is returned because ACOS's mean and covariance are read from it and from
    the files it names, and because its row order defines the alignment of every array in
    the artifact.

    The partition that gets scored is `--eval-partition` and is the test one by default.
    The other two returns are always the training and validation partitions whatever that
    is set to, because the Laplace approximation fits on the first and calibrates on the
    second; pointing `--eval-partition` at `train` therefore scores the rows the model was
    fit on, which is the generalization-gap measurement and is not a result
    about held-out data. `meta.notes.eval_partition` records which it was.
    """
    files_root = Path(args.files_root)
    stats_dir = Path(args.stats_dir) if args.stats_dir else None
    is_profiles = spec["levels"] is not None
    want = getattr(args, "eval_partition", "test") or "test"

    if args.split_csv:
        split_csv = Path(args.split_csv)
        build = SL.build_dataset_profiles_grouped if is_profiles else SL.build_dataset_grouped
        cov = getattr(args, "covariates", None)
        train_ds = build(files_root, split_csv, "train", stats_dir=stats_dir, cov_spec=cov)
        val_ds = build(files_root, split_csv, "val", stats_dir=stats_dir, cov_spec=cov)
        if want == "train":
            test_ds = train_ds
        elif want == "val":
            test_ds = val_ds
        else:
            # `want`, not "test". The two are the same for every partition the published
            # runs score, and they are NOT the same for `resim_test` -- the re-simulated
            # realisations of the four held-out tail scenes. Building "test" here while the
            # frame below is `want` would score February's 4,489 soundings and label them
            # with 788 other rows' IDs, truths and ACOS values: every array would have a
            # consistent shape and every metric would compute. The builders filter on the
            # string they are given, so any partition label in the table works.
            test_ds = build(files_root, split_csv, want, stats_dir=stats_dir, cov_spec=cov)
        frame = SL.load_split_table(split_csv, want)
        # Cheap, and the one thing that would make the above silent if it regressed.
        if len(test_ds) != len(frame):
            raise SystemExit(
                f"the '{want}' dataset has {len(test_ds)} soundings and its annotation "
                f"frame {len(frame)}. The frame defines the row alignment of every array "
                f"in the artifact, so these must be the same rows.")
        source_dir_col = acos_mod.SOURCE_DIR_COL
        split_name = split_csv.stem + ("" if want == "test" else f"+{want}")
    else:
        if want != "test":
            raise SystemExit(
                "--eval-partition is only defined under --split-csv: the replicate-level "
                "split has no train/val annotation frames of its own, its validation rows "
                "being a random_split of the training pool.")
        build = SL.build_dataset_profiles if is_profiles else SL.build_dataset
        test_ds = build(files_root / "test", args.annotations, stats_dir=stats_dir)
        pool = build(files_root / "train", args.annotations, stats_dir=stats_dir)
        # The published runs drew validation from the training pool with this generator;
        # reproduce it exactly so a Laplace refit sees the same held-out rows.
        g = torch.Generator().manual_seed(args.seed)
        n_train = int(args.train_size * len(pool))
        train_ds, val_ds = torch.utils.data.random_split(
            pool, [n_train, len(pool) - n_train], generator=g
        )
        frame = pd.read_csv(files_root / "test" / args.annotations)
        source_dir_col = None
        split_name = "replicate"

    logger.info("scored=%d (%s)  train=%d  val=%d  (split=%s)",
                len(test_ds), want, len(train_ds), len(val_ds), split_name)
    if want == "train":
        logger.warning("scoring the TRAINING partition: these rows were fitted, and for "
                       "the profile tasks the Laplace's marginal-likelihood/MC noise "
                       "estimate comes from them too. This artifact is the in-sample side "
                       "of a generalization gap, not a result.")
    if want == "val":
        logger.warning("scoring the VALIDATION partition: every checkpoint was selected on "
                       "these rows, and in `mc` mode the Laplace's aleatoric diagonal is "
                       "calibrated on them, so its intervals here are in-sample. Read the "
                       "flow's numbers and treat the Laplace's as optimistic.")
    if want.startswith("resim_"):
        n_scenes = (frame[["clst", "ref"]].drop_duplicates().shape[0]
                    if {"clst", "ref"} <= set(frame.columns) else None)
        logger.warning("scoring the %s partition: %d soundings on %s reference soundings, "
                       "held out of training and NOT part of the default test partition. "
                       "These are the February extension's fresh discrepancy draws for "
                       "scenes February already holds out, so this artifact is a SECOND "
                       "REALISATION of those scenes' numbers -- comparable with the same "
                       "scenes' rows in the test artifact, and not with the pooled test "
                       "number, which is 45 scenes rather than %s.",
                       want, len(frame), n_scenes, n_scenes)
    return test_ds, train_ds, val_ds, frame, source_dir_col, split_name


def target_stats(args, spec):
    """`(y_mu, y_sigma)` in mole fraction, as `[D]` arrays.

    These come from the split's own training rows (`compute_split_stats.py`). Reusing the
    published constants under a new split would standardize held-out soundings with
    statistics computed from them, which is precisely the leakage to avoid -- so a
    missing stats file is an error here, not a fallback.
    """
    stats_dir = Path(args.stats_dir) if args.stats_dir else None
    if spec["levels"] is None:
        mu, sigma = SL._load_scalar_stats(stats_dir)
        if args.split_csv and not (stats_dir and (stats_dir / "xco2_stats.json").exists()):
            raise FileNotFoundError(
                f"xco2_stats.json not found in {stats_dir}. Under a group split the "
                "scalar standardization must be recomputed from that split's training "
                "rows; run experiments/pipelines/compute_split_stats.py first."
            )
        return np.array([mu]), np.array([sigma])

    mu, sd = SL._load_profile_stats(stats_dir if stats_dir else SL._DEFAULT_STATS_DIR)
    idx0 = np.asarray([l - 1 for l in spec["levels"]])
    return np.asarray(mu, dtype=np.float64)[idx0], np.asarray(sd, dtype=np.float64)[idx0]


def _y_slice(y: torch.Tensor, spec) -> torch.Tensor:
    """Cut the 20-level target down to the task's levels."""
    if spec["levels"] is None or len(spec["levels"]) == y.shape[1]:
        return y
    idx = torch.as_tensor([l - 1 for l in spec["levels"]], device=y.device, dtype=torch.long)
    return y.index_select(1, idx)


# ---------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--out", required=True, help="path of the .npz to write")

    ap.add_argument("--files-root", default=str(_ROOT / "scratch/simulated/files"))
    ap.add_argument("--annotations", default="aux_data.csv")
    ap.add_argument("--split-csv", default=None,
                    help="group split table; omit for the original replicate split")
    ap.add_argument("--stats-dir", default=None,
                    help="directory holding profile_mu/sigma.npy and xco2_stats.json")
    ap.add_argument("--weights-root", default=str(_ROOT / "experiments/weights/group_split_v1"))
    ap.add_argument("--covariates", default=None,
                    help="covariate preset the checkpoints were trained with (see "
                         "code/data/loading/covariates.py). Must match, or the auxiliary "
                         "encoder will not load. Default: legacy.")

    # Where the checkpoints are, when LAYOUT cannot say. A covariate model's weights
    # directory is named after its preset (full_regressor_scene_prior_best/), and the
    # Bayesian optimization's winner adds another suffix on top of that -- neither is a
    # constant the layout table can hold, so the caller names the file. --task still
    # decides how many outputs the model has and which Laplace structure it gets.
    ap.add_argument("--det-ckpt", default=None,
                    help="explicit path to the deterministic checkpoint, instead of the "
                         "standard layout under --weights-root")
    ap.add_argument("--flow-ckpt", default=None,
                    help="explicit path to the flow checkpoint")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-size", type=float, default=0.9,
                    help="only used for the replicate split, where val is drawn at random")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ids", nargs="+", default=None, metavar="ID",
                    help="score only these sounding ids (the `ID` column of the split "
                         "table, which figures.py prints for the panel it drew). For "
                         "Figure 7's dense flow draws; the artifact is marked partial.")
    ap.add_argument("--ids-file", default=None,
                    help="one sounding id per line; combined with --ids")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N test batches; for smoke tests only")
    # `resim_test` belongs to the extension split only: its re-simulated copies of the
    # four held-out test scenes in clusters 8 and 9, never trained on and never part of the
    # default test partition (experiments/pipelines/make_ext_split.py, rule 2). Scoring it
    # answers whether a per-scene result survives a fresh set of discrepancy draws, which
    # is not a given: a scene's mean ACOS error moves by up to
    # 2.8 ppm between realisations of the same scene.
    ap.add_argument("--eval-partition", default="test",
                    choices=["test", "train", "val", "resim_test", "resim_val"],
                    help="which partition to score (default: test). `train` measures the "
                         "generalization gap and is a diagnostic, not a result; needs "
                         "--split-csv")
    ap.add_argument("--eval-subsample-refs", type=int, default=None, metavar="K",
                    help="score K reference soundings of the chosen partition instead of "
                         "all of them, drawn with --seed and all their replicates kept. "
                         "Use it to give a train-partition estimate the same number of "
                         "groups as the test partition, so the two bootstrap standard "
                         "errors are comparable")

    ap.add_argument("--laplace-mode", default=None, choices=["mc", "marglik"],
                    help="default: per task, matching how the published table was made")
    ap.add_argument("--laplace-hessian", default=None, choices=["diag", "full", "kron"])
    ap.add_argument("--laplace-fit-subset", type=int, default=6000)
    ap.add_argument("--laplace-fit-batch", type=int, default=128)
    ap.add_argument("--laplace-samples", type=int, default=256)
    ap.add_argument("--aleatoric-samples", type=int, default=256)
    ap.add_argument("--aleatoric-max-batches", type=int, default=None)

    ap.add_argument("--nf-samples", type=int, default=512)
    ap.add_argument("--sample-dtype", default="float32", choices=["float32", "float16"],
                    help="float16 halves the file; the samples are only used for "
                         "quantiles and moments, where it costs nothing")
    ap.add_argument("--nf-arch", default=None,
                    help="YAML overriding the flow architecture, for checkpoints that "
                         "do not record their own (the published scalar flow)")

    # The moment-matched Gaussian control (evals/gaussianize.py). Off by default, and it
    # adds to the pass rather than replacing anything: the `--nf-samples` draw happens
    # first and is untouched, so `nf_samples_std` is bit-identical to a run without this.
    ap.add_argument("--gaussianize", type=int, default=0, metavar="S",
                    help="stream S flow samples per sounding and record the moment-matched "
                         "Gaussian control at a doubling schedule of budgets. The samples "
                         "themselves are never stored; S can be far larger than "
                         "--nf-samples. 0 disables.")
    ap.add_argument("--gaussianize-chunk", type=int, default=1024,
                    help="samples per draw. The context/backbone forward is recomputed "
                         "once per chunk, so large is cheap and small is not.")
    ap.add_argument("--gaussianize-first", type=int, default=gz.STAGE_MIN,
                    help="smallest budget in the schedule; the default matches "
                         "--nf-samples so the first stage reproduces the published tables.")

    ap.add_argument("--legacy-backbone", action="store_true",
                    help="load the deterministic profile model as FullRegressorProfilesLegacy; "
                         "needed only for the published net_profiles_old.pt")
    ap.add_argument("--skip-laplace", action="store_true")
    ap.add_argument("--skip-flow", action="store_true")
    ap.add_argument("--skip-acos", action="store_true")
    args = ap.parse_args(argv)

    spec = TASKS[args.task]
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if device.type != "cuda":
        logger.warning("running on %s -- this is a CPU-hours-not-minutes proposition; "
                       "intended only for smoke tests with --limit.", device)

    test_ds, train_ds, val_ds, frame, source_dir_col, split_name = build_data(args, spec)

    # --eval-subsample-refs: keep K whole reference soundings of the scored partition.
    #
    # The training partition has 361 groups against the test partition's 45, so a
    # train-vs-test comparison of bootstrap standard errors is otherwise a comparison of
    # two different sample sizes. Drawn without replacement on --seed, all replicates of a
    # drawn group kept, so the subsample is the same kind of object as the test partition.
    if args.eval_subsample_refs:
        if not {"clst", "ref"}.issubset(frame.columns):
            raise SystemExit("--eval-subsample-refs needs the split table's clst/ref "
                             "columns, which only --split-csv provides.")
        key = frame["clst"].astype(str) + "/" + frame["ref"].astype(str)
        uniq = key.drop_duplicates().to_numpy()
        k = int(args.eval_subsample_refs)
        if k > uniq.size:
            raise SystemExit(f"--eval-subsample-refs {k} exceeds the {uniq.size} "
                             f"reference soundings in the {args.eval_partition} partition.")
        keep = set(np.random.default_rng(args.seed).choice(uniq, size=k, replace=False))
        pos = np.flatnonzero(key.isin(keep).to_numpy())
        test_ds = torch.utils.data.Subset(test_ds, pos.tolist())
        frame = frame.iloc[pos].reset_index(drop=True)
        logger.info("--eval-subsample-refs: %d of %d reference soundings, %d soundings",
                    k, uniq.size, len(frame))

    # --ids: score a named handful of soundings instead of the partition.
    #
    # Written for Figure 7, which needs many more flow draws than the 512 the full run
    # stores -- at 20,000 draws the profile artifact would be about three gigabytes for the
    # sake of six panels, while six soundings at 20,000 is ten megabytes. It subsets the
    # test rows only: the Laplace still fits on the training partition and calibrates on
    # the validation one, so the predictive is the same object the full artifact carries.
    #
    # The frame's row order defines the alignment of every array in the artifact, so the
    # dataset and the frame have to be subset with the same positions, in the same order.
    ids_want = list(args.ids or [])
    if args.ids_file:
        with open(args.ids_file) as fh:
            ids_want += [ln.strip() for ln in fh if ln.strip()]
    if ids_want:
        have = frame["ID"].astype(str)
        pos = [i for i, v in enumerate(have) if v in set(ids_want)]
        missing = set(ids_want) - set(have.iloc[pos]) if pos else set(ids_want)
        if missing:
            raise SystemExit(
                f"--ids: {sorted(missing)} are not in the test partition of "
                f"{split_name} ({len(frame)} rows). The ids are the `ID` column of the "
                f"split table and are what figures.py prints for the panel it drew.")
        test_ds = torch.utils.data.Subset(test_ds, pos)
        frame = frame.iloc[pos].reset_index(drop=True)
        logger.warning("--ids: scoring %d of the partition's soundings. This artifact is "
                       "a panel, NOT a result -- meta.notes.ids_subset marks it and "
                       "score.py must not be pointed at it.", len(pos))
    y_mu, y_sigma = target_stats(args, spec)
    dim = len(y_mu)
    m_idx_list = model_slice(spec)
    m_idx = None if m_idx_list is None else torch.as_tensor(
        m_idx_list, device=device, dtype=torch.long)
    logger.info("task=%s  D=%d  y_sigma=%s", args.task, dim, np.array2string(y_sigma, precision=3))
    if m_idx is not None:
        logger.info("marginalizing: models emit levels %s, scoring levels %s "
                    "(output positions %s)",
                    spec["model_levels"], spec["levels"], m_idx_list)

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,   # order defines alignment
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    warm_batch = next(iter(test_loader))
    warm_x, _, _ = lap.move_batch(warm_batch, device)

    # The auxiliary encoder's embedding tables are sized from the split's own covariate
    # statistics; a checkpoint trained with a non-legacy covariate set cannot be rebuilt
    # without them. Read once -- the flow's context backbone needs the same thing.
    cov_stats = SL.load_covariate_stats(Path(args.stats_dir) if args.stats_dir else None)

    nf_arch_override = None
    if args.nf_arch:
        with open(args.nf_arch) as fh:
            nf_arch_override = yaml.safe_load(fh) or {}

    # -- models ------------------------------------------------------------------
    fit = None
    det_desc = {}
    if not args.skip_laplace:
        det = ck.load_deterministic(
            args.weights_root, spec["det"], device,
            warm_x_tuple=warm_x, legacy=args.legacy_backbone,
            cov_stats=cov_stats,
            ckpt_path=args.det_ckpt,
            covariates=args.covariates,
        )
        det_desc = det.describe()
        logger.info("deterministic: %s", det_desc)
        # The loader's target is always the full 20-level vector for the profile tasks;
        # `y_idx` cuts it to the levels this model predicts. The models themselves emit
        # exactly those levels (det_last4 was trained with target_indices=last4), so the
        # model output is never sliced -- see the note in calibrate_aleatoric.
        #
        # `last4_marginal` is the exception: there the model emits 20 levels, and it is
        # fitted and calibrated against all 20 of them. Slicing to the near-surface levels
        # happens after the predictive is formed, below, because marginalizing a Gaussian
        # is taking a submatrix of the covariance -- not refitting on a subset, which
        # would be a different (and wrong) object.
        fit_levels = spec.get("model_levels") or spec["levels"]
        y_idx = None if fit_levels is None else [l - 1 for l in fit_levels]
        fit_loader = lap.build_laplace_loader(
            train_ds, batch_size=args.laplace_fit_batch,
            n_subset=args.laplace_fit_subset, seed=args.seed, target_idx=y_idx,
        )
        fit = lap.fit_laplace(
            det.module, fit_loader, device,
            mode=args.laplace_mode or spec["laplace_mode"],
            hessian_structure=args.laplace_hessian or spec["hessian"],
        )
        if fit.mode == "mc":
            logger.info("calibrating the aleatoric diagonal on the validation partition ...")
            lap.calibrate_aleatoric(
                fit, val_loader, device, n_samples=args.aleatoric_samples,
                max_batches=args.aleatoric_max_batches,
                y_idx=y_idx,           # cut the 20-level target down to this task
                target_idx=None,       # the model already emits this task's dimensions
            )
            logger.info("aleatoric variance (std units): %s",
                        np.array2string(fit.aleatoric_var.cpu().numpy(), precision=4))

    flow = None
    flow_desc = {}
    if not args.skip_flow:
        flow_model = ck.load_flow(args.weights_root, spec["flow"], device,
                                  warm_x_tuple=warm_x, arch_override=nf_arch_override,
                                  ckpt_path=args.flow_ckpt, cov_stats=cov_stats,
                                  covariates=args.covariates)
        flow = flow_model.module
        flow_desc = flow_model.describe()
        logger.info("flow: %s", flow_desc)

    # -- the pass over the test partition ----------------------------------------
    acc = {k: [] for k in ("y_std", "lap_mu_std", "lap_cov_epi_std", "nf_samples_std",
                           "nf_logprob_std", "nf_latent_std", "y_retrieved_std",
                           "nfg_mu_std", "nfg_var_std", "nfg_nll_std", "nfg_pit_std",
                           "nfg_cov_std", "nfg_skew_std", "nfg_exkurt_std")}
    gz_stages = (gz.stage_schedule(args.gaussianize, args.gaussianize_first)
                 if args.gaussianize > 0 else [])
    if gz_stages:
        logger.info("moment-matched Gaussian control: %d samples per sounding, "
                    "chunk %d, stages %s (plug-in penalty at the last stage ~%.4f nats)",
                    args.gaussianize, args.gaussianize_chunk, gz_stages,
                    gz.plugin_penalty(dim, gz_stages[-1]))
    sample_dtype = np.float16 if args.sample_dtype == "float16" else np.float32
    n_seen = 0

    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            if args.limit is not None and bi >= args.limit:
                logger.warning("--limit reached after %d batches; this artifact is a "
                               "smoke test, not a result.", bi)
                break
            x_tuple, y, retr = lap.move_batch(batch, device)
            y = _y_slice(y.reshape(y.shape[0], -1), spec)
            acc["y_std"].append(y.cpu().numpy().astype(np.float32))
            if retr is not None:
                acc["y_retrieved_std"].append(
                    _y_slice(retr.reshape(retr.shape[0], -1), spec).cpu().numpy().astype(np.float32)
                )

            if fit is not None:
                mean, cov = lap.posterior_mean_cov(fit, x_tuple, n_samples=args.laplace_samples)
                if m_idx is not None:
                    mean = mean.index_select(1, m_idx)
                    cov = cov.index_select(1, m_idx).index_select(2, m_idx)
                acc["lap_mu_std"].append(mean.cpu().numpy().astype(np.float32))
                acc["lap_cov_epi_std"].append(cov.cpu().numpy().astype(np.float32))

            if flow is not None:
                s = flow.sample(x_tuple, n=args.nf_samples)          # [S,B,D]
                if m_idx is not None:
                    s = s.index_select(2, m_idx)
                acc["nf_samples_std"].append(
                    s.permute(1, 0, 2).contiguous().cpu().numpy().astype(sample_dtype)
                )
                # A flow gives an exact JOINT density and no marginal one: integrating out
                # 16 levels has no closed form. So when the predictive is a marginal, the
                # flow's own log_prob is the wrong number and the latent z is the wrong
                # dimension, and both are omitted -- score.py then falls back to the KDE
                # over samples, which is what the published near-surface table used, and
                # records `nll_estimator` so the two rows are never confused.
                if m_idx is None:
                    acc["nf_logprob_std"].append(
                        flow.log_prob(x_tuple, y).cpu().numpy().astype(np.float64)
                    )
                    acc["nf_latent_std"].append(
                        flow.latent(x_tuple, y).cpu().numpy().astype(np.float32)
                    )

                # Runs after the stored draw, so the two never interfere: the samples in
                # the artifact are the ones a run without --gaussianize would have written.
                if gz_stages:
                    g = gz.stream_batch(flow, x_tuple, y, args.gaussianize,
                                        args.gaussianize_chunk, gz_stages, m_idx=m_idx)
                    for key, arr in (("nfg_mu_std", g["mu"]), ("nfg_var_std", g["var"]),
                                     ("nfg_nll_std", g["nll"]), ("nfg_pit_std", g["pit"]),
                                     ("nfg_cov_std", g["cov"]), ("nfg_skew_std", g["skew"]),
                                     ("nfg_exkurt_std", g["exkurt"])):
                        acc[key].append(arr)

            n_seen += int(y.shape[0])
            if bi % 20 == 0:
                logger.info("  %d / %d soundings", n_seen, len(test_ds))

    arrays = {k: np.concatenate(v, axis=0) for k, v in acc.items() if v}
    frame = frame.iloc[:n_seen].reset_index(drop=True)

    # Add the calibrated observation noise to the epistemic covariance, once, here, so
    # that `lap_cov_std` in the artifact is the predictive covariance a reader would
    # expect and the two terms remain separately inspectable.
    if fit is not None and fit.aleatoric_var is not None:
        aleat = fit.aleatoric_var.cpu().numpy().astype(np.float64)
        # Marginalize the aleatoric diagonal the same way the epistemic covariance was
        # marginalized in the loop above.
        #
        # `last4_marginal` fits and calibrates over all 20 levels -- that is what makes it
        # the marginal of the 20-level predictive rather than a model refitted on four
        # levels -- and then slices. The epistemic covariance was sliced inside the loop
        # (index_select on both axes); this term was not, so the artifact tried to add a
        # (20, 20) diagonal to a (N, 4, 4) covariance and died with
        #
        #   ValueError: operands could not be broadcast together with
        #               shapes (4489,4,4) (1,20,20)
        #
        # Slicing a diagonal is the same submatrix operation, so the marginal stays exact.
        # It is a ValueError rather than a wrong number only by luck: had the task sliced
        # to 20 of 20 levels, or to a leading block, this would have broadcast quietly.
        if m_idx_list is not None:
            aleat = aleat[np.asarray(m_idx_list)]
        arrays["lap_cov_std"] = (arrays["lap_cov_epi_std"].astype(np.float64)
                                 + np.diag(aleat)[None, :, :])
        arrays["lap_aleatoric_var_std"] = np.repeat(aleat[None, :], n_seen, axis=0)
    elif fit is not None:
        arrays["lap_cov_std"] = arrays["lap_cov_epi_std"].astype(np.float64)

    # A Laplace predictive that is far narrower than the residuals it is meant to describe
    # is a defect, not a result, and it is invisible in every downstream artifact: the
    # tables print a coverage, the figures draw a PIT histogram, and both look like
    # findings. This is what `marglik` mode did to the scalar task for the whole project
    # (the predictive carried no observation-noise term at all, 5.7x too narrow), so the
    # check is made here, once, where the covariance is assembled.
    if fit is not None and "lap_cov_std" in arrays and "y_std" in arrays:
        _y = np.atleast_2d(arrays["y_std"].astype(np.float64))
        _y = _y.T if _y.shape[0] == 1 and _y.shape[1] == len(arrays["lap_mu_std"]) else _y
        _mu = np.atleast_2d(arrays["lap_mu_std"].astype(np.float64))
        _mu = _mu.reshape(_y.shape)
        _v = arrays["lap_cov_std"]
        _v = np.einsum("nii->ni", _v) if _v.ndim == 3 else _v.reshape(_y.shape)
        _ratio = float(np.mean((_y - _mu) ** 2) / max(np.mean(_v), 1e-30))
        logger.info("Laplace predictive variance vs residual variance: %.2fx "
                    "(1 is calibrated in scale; aleatoric term %s)",
                    _ratio, "present" if fit.aleatoric_var is not None else "ABSENT")
        if _ratio > 4.0 or _ratio < 0.25:
            logger.warning(
                "the Laplace predictive is off by %.1fx in variance against the residuals "
                "of the partition it was just scored on. If the aleatoric term is ABSENT "
                "the mode is wrong for this task -- use --laplace-mode mc. Do not report "
                "the coverage or NLL of this row before resolving it.", _ratio)

    # -- ACOS --------------------------------------------------------------------
    if not args.skip_acos:
        if spec["levels"] is None:
            mu, cov, valid = acos_mod.load_xco2(frame, float(y_mu[0]), float(y_sigma[0]))
        else:
            mu, cov, valid = acos_mod.load_profiles(
                frame, args.files_root, y_mu, y_sigma,
                levels=spec["levels"], source_dir_col=source_dir_col,
            )
        arrays["acos_mu_std"] = mu.astype(np.float64)
        arrays["acos_cov_std"] = cov.astype(np.float64)
        arrays["acos_valid"] = valid

    # -- identifiers and diagnostics ---------------------------------------------
    arrays["ids"] = frame["ID"].to_numpy().astype("U")
    # Both spellings are tried: the pre-revision annotation files call these `cluster`
    # and `reference_sounding`, the group split table written by `make_group_split.py`
    # calls them `clst` and `ref`. Only the second pair exists in the revision's split
    # table, so with the old names alone every artifact of the revision carried no cluster
    # label, and no per-cluster breakdown was possible from the .npz alone.
    for col in ("split", "cluster", "reference_sounding", "clst", "ref", "rep",
                acos_mod.SOURCE_DIR_COL, "mixture_component", "dof_co2",
                "SlrZen", "Footprint", "AerType1", "AerType2", "Latitude", "Longitude"):
        if col in frame.columns:
            v = frame[col].to_numpy()
            arrays[col] = v.astype("U") if v.dtype == object else v

    meta = ArtifactMeta(
        # `last4_marginal` reports itself as task `last4` under a suffixed split name, so
        # score.py writes both sets of rows into one table_last4.{csv,tex} -- native and
        # marginalized side by side, which is what the published Table 3 compared.
        task=spec.get("table_task", args.task),
        split_name=split_name + spec.get("split_suffix", ""),
        n=n_seen,
        dim=dim,
        y_mu=[float(v) for v in y_mu],
        y_sigma=[float(v) for v in y_sigma],
        levels=list(spec["levels"] or []),
        n_nf_samples=0 if flow is None else int(args.nf_samples),
        device=str(device),
        checkpoints={"deterministic": det_desc, "flow": flow_desc},
        laplace=({} if fit is None else fit.describe()),
        nf={"sample_dtype": args.sample_dtype,
            "n_samples": 0 if flow is None else int(args.nf_samples),
            # The K axis of every nfg_* array. Without it the stage columns are
            # uninterpretable, and the 1/S extrapolation cannot be fitted at all.
            "gaussianize_stages": gz_stages,
            "gaussianize_chunk": int(args.gaussianize_chunk) if gz_stages else 0},
        notes={
            "files_root": str(args.files_root),
            "stats_dir": str(args.stats_dir),
            "weights_root": str(args.weights_root),
            "split_csv": str(args.split_csv) if args.split_csv else None,
            "seed": args.seed,
            # `partial` is what tells a reader this artifact is not the partition. An
            # --ids subset is partial in exactly the same sense as --limit, and more
            # dangerous, because it looks complete: every array has consistent shapes and
            # every metric would compute on six soundings without complaining.
            # A resim_* artifact is `partial` by construction: four reference soundings,
            # not forty-five. Saying so in the metadata is what stops its RMSE being read
            # off as a pooled test number by anything that reads artifacts generically.
            "partial": (args.limit is not None or bool(ids_want)
                        or args.eval_partition.startswith("resim_")
                        or (args.eval_partition == "test"
                            and bool(args.eval_subsample_refs))),
            "ids_subset": sorted(ids_want) if ids_want else None,
            "cli_task": args.task,
            "model_levels": list(spec.get("model_levels") or spec["levels"] or []),
            "marginalized": m_idx is not None,
            # Which partition these rows are. Anything other than `test` is a diagnostic:
            # `train` is the in-sample side of the generalization gap and must never be
            # quoted as a result.
            "eval_partition": args.eval_partition,
            "eval_subsample_refs": (None if not args.eval_subsample_refs
                                    else int(args.eval_subsample_refs)),
        },
    )

    art = Artifact(meta, arrays)
    out = art.save(args.out)
    logger.info("wrote %s (%.1f MB)", out, out.stat().st_size / 2**20)
    print(art.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

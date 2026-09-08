#!/usr/bin/env python3
"""amortization.py -- the offline-versus-online cost ledger, and where the two cross.

The speed claim in the paper is about online cost, and the obvious objection is about the
*accounting*: amortization can be read as only moving the computation from retrieval time
to training time, and retraining after a change to L1b calibration, geolocation or
absorption coefficients means the move has to be paid for again.

That is an arithmetic question, so this script does the arithmetic rather than asserting an
answer to it. The
two costs scale differently in the number of soundings $N$:

    full physics   C_F(N) = N * t_F            marginal, one core-second per sounding
    amortized      C_A(N) = T_0 + N * t_N      fixed T_0, then a network evaluation

and a fixed cost divided by a large $N$ is not a trade-off, it is a rounding error. The
break-even count

    N* = T_0 / (t_F - t_N)

is the only number that decides the question, and with $t_F$ a hundred and fifty seconds
and $t_N$ four milliseconds it is small enough that the *retraining* cadence has to be
absurd before the ledger turns over. The
same algebra answers the "but the operational pipeline starts a day earlier" version: with
a head start Delta and K cores against G GPUs the crossover is at

    T_x = T_0 + R_F*K*(T_0 + Delta) / (R_N*G - R_F*K),      R = 1/t

which for any K below t_F/t_N cores per GPU is T_0 plus a remainder measured in seconds.

**Nothing here is a controlled benchmark, and the script refuses to pretend otherwise.**
t_F defaults to the 150 s the manuscript quotes, which came from the group that produced
the simulation ensemble; it is the *smallest* figure available, and the literature reports
comparable or larger costs -- "about 5 min per sounding on a single processor", Taylor et
al. (2023), AMT 16, 3173-3209, section 3, and the 221 s implied by c_F below -- so the
default makes the comparison conservative rather than convenient. t_N is measured by
`benchmark.py` on one GPU. A core-second and a GPU-second are not the same unit, so every
row carries its unit, and the `--gpu-core-equivalence` grid reports the ratio again after
dividing the GPU by 100 or 1000 notional cores. The conclusion has to survive that haircut
to be worth writing down, and it does.

Where the numbers come from, in order of preference:

    T_0   1. `run_meta_*.json` inside the fetched job archives (`elapsed_seconds`, `gpu`),
             which is what the jobs themselves measured;
          2. the HTCondor event logs in `chtc/logs/` -- terminate minus execute;
          3. `condor_history -json` output, for a run whose logs were archived;
          4. `--assume-trial-seconds` / `--assume-final-seconds`, which mark every derived
             row as `assumed` in the provenance so an estimate cannot be mistaken for a
             measurement.

    t_N   `benchmark.csv` written by `experiments/scoring/benchmark.py`, selected by task /
          method / scope / batch size; or `--assume-ms-per-sounding`.

    N_sim the split table, i.e. how many simulated soundings the ensemble actually holds.

    c_F   Lamminpaa et al. (2025), AMT 18, 673-694, Table 2: one ReFRACtor RT evaluation is
          33.45 s and RT plus its Jacobian 55.26 s, on a 2023 MacBook Pro. Different
          implementation and hardware from the build that generated our ensemble, so it is
          an order-of-magnitude figure -- but it corroborates t_F, since one iteration costs
          an RT evaluation plus a Jacobian and Section 4.1 reports a median of four
          iterations: 4 x 55.26 = 221 s against the 300 s quoted by Taylor et al.

Usage -- on the access point, where the logs and archives are:

    python3 experiments/scoring/amortization.py \
        --condor-logs chtc/logs \
        --trial-archives /staging/$USER/oco2/results/bo25 \
        --benchmark-csv experiments/scoring/tables/benchmark.csv \
        --split-csv group_split_v1_cov.csv \
        --out experiments/scoring/tables --fig latex/figures

Writes `amortization.csv` (long form: quantity, value, unit, basis), `amortization.json`
(every input, its source, and the git commit), `amortization.tex` (the ledger as a table)
and optionally `amortization_crossover.png` (cumulative retrievals against wall clock, with
the crossover marked). Prints a markdown summary of the ledger.

Author(s): Alejandro Calle-Saldarriaga
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import statistics
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("amortization")

# The six Bayesian-optimization studies, and which task pair each belongs to. One study is
# one model, so "the cost of a model" is a study's total and "the cost of a task" is the
# deterministic study plus the flow study of the same task.
STUDIES = ("det_xco2", "det_profiles", "det_last4",
           "nf_xco2", "nf_profiles", "nf_last4")
TASK_PAIRS = {
    "xco2": ("det_xco2", "nf_xco2"),
    "profiles": ("det_profiles", "nf_profiles"),
    "last4": ("det_last4", "nf_last4"),
}

# "005 (6037497.000.000) 2026-08-14 16:41:14 Job terminated."  -- same shape status.py reads.
EVENT = re.compile(r"^(\d{3}) \((\d+)\.(\d+)\.(\d+)\) "
                   r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})")
TRIAL_NAME = re.compile(r"^bo_(?P<study>[a-z0-9_]+?)_r(?P<rnd>\d+)_t(?P<trial>\d+)$")
FINAL_NAME = re.compile(r"^(?P<study>[a-z0-9_]+)_final$")


# ----------------------------------------------------------------------------------------
# T_0: what the offline work actually cost
# ----------------------------------------------------------------------------------------
def classify(job: str):
    """('trial' | 'final' | None, study) from a job name."""
    m = TRIAL_NAME.match(job)
    if m and m.group("study") in STUDIES:
        return "trial", m.group("study")
    m = FINAL_NAME.match(job)
    if m and m.group("study") in STUDIES:
        return "final", m.group("study")
    return None, None


def _parse_when(text: str) -> datetime:
    """HTCondor writes local time, with or without the ISO 'T'. Durations are differences,
    so the missing zone offset cancels."""
    return datetime.strptime(text.replace("T", " "), "%Y-%m-%d %H:%M:%S")


def from_archives(root: Path) -> list:
    """`run_meta_<job>.json` out of every job archive under `root`, without unpacking it.

    This is the best source: `oco2_pack_results` writes it from inside the job, so
    `elapsed_seconds` is the job's own measurement and `gpu` is the card it ran on. Reading
    it straight out of the tarball means a 300 MB archive costs one seek.
    """
    # Keyed by job name, not appended: `results/` has held the same trial under two
    # namespaces (a re-searched study leaves `bo_det_xco2_r0_t0.tar.gz` in both), and
    # rglob finds both. Counting both doubled `det_*` to 50 trials and inflated the
    # program total by 40 %. The newest archive wins, the duplicate is reported rather
    # than silently dropped, and `source` carries the path relative to the root so the
    # two are distinguishable in `amortization_runs.csv`.
    by_job = {}
    duplicates = []
    for tar_path in sorted(root.rglob("*.tar.gz")):
        # Filter on the *name* before opening anything. `results/` also holds the eval and
        # month archives, and a few of those are gigabytes; `getnames()` on a .tar.gz reads
        # the whole file to walk the headers, so scanning them to find no run_meta of
        # interest would cost minutes. The BO archives are named after their jobs, which is
        # exactly what `classify` recognises.
        stem = tar_path.name[:-len(".tar.gz")]
        if classify(stem)[0] is None:
            continue
        try:
            with tarfile.open(tar_path, "r:gz") as tf:
                names = [n for n in tf.getnames()
                         if Path(n).name.startswith("run_meta_")
                         and n.endswith(".json")]
                for name in names:
                    fh = tf.extractfile(name)
                    if fh is None:
                        continue
                    meta = json.loads(fh.read().decode("utf-8", "replace"))
                    job = str(meta.get("job", Path(name).name[9:-5]))
                    kind, study = classify(job)
                    if kind is None:
                        continue
                    row = {
                        "job": job, "kind": kind, "study": study,
                        "seconds": float(meta.get("elapsed_seconds", 0.0)),
                        "gpu": meta.get("gpu") or "",
                        "exit_status": meta.get("exit_status"),
                        "source": f"run_meta:{tar_path.relative_to(root)}",
                        "mtime": tar_path.stat().st_mtime,
                        "finished": str(meta.get("finished_utc") or ""),
                    }
                    prev = by_job.get(job)
                    if prev is None:
                        by_job[job] = row
                    else:
                        # `finished_utc` is the job's own clock and survives being copied
                        # between filesystems; the file mtime is only the fallback.
                        newer = ((row["finished"], row["mtime"]) >
                                 (prev["finished"], prev["mtime"]))
                        loser, winner = (prev, row) if newer else (row, prev)
                        by_job[job] = winner
                        duplicates.append((job, loser["source"], winner["source"],
                                           loser["seconds"], winner["seconds"]))
        except (tarfile.TarError, OSError, ValueError, KeyError) as exc:
            logger.warning("%s: unreadable (%s)", tar_path.name, exc)

    if duplicates:
        logger.warning("%d job name(s) found in more than one place under %s. The newest "
                       "archive is charged and the older one ignored; if the older run was "
                       "a separate search you actually paid for, say so in the reply rather "
                       "than summing them here.", len(duplicates), root)
        for job, older, newer, s_old, s_new in duplicates[:10]:
            logger.warning("    %s: kept %s (%.1f min), ignored %s (%.1f min)",
                           job, newer, s_new / 60, older, s_old / 60)
        if len(duplicates) > 10:
            logger.warning("    ... and %d more; amortization_runs_duplicates.csv has "
                           "all of them", len(duplicates) - 10)
    return list(by_job.values()), duplicates


def from_condor_logs(root: Path) -> list:
    """Wall clock from the HTCondor event logs: last terminate minus first execute.

    A job that was evicted and restarted has more than one execute event; taking the first
    execute of the *last* cluster charges the run for the attempt that produced the result,
    which is the honest figure for a cost table.
    """
    # Recursive, because `chtc/status.py --archive` moves finished logs to
    # logs/archive/<utc>/ and a run collected weeks ago is exactly the one whose cost we
    # want. When the same job appears in more than one place, the latest terminate wins.
    by_job = {}
    for log_path in sorted(root.rglob("*.log")):
        job = log_path.stem
        kind, study = classify(job)
        if kind is None:
            continue
        clusters = {}
        order = []
        for line in log_path.read_text(errors="replace").splitlines():
            m = EVENT.match(line)
            if not m:
                continue
            code, cluster, _proc, _sub, when = m.groups()
            if cluster not in clusters:
                clusters[cluster] = {"exec": None, "term": None}
                order.append(cluster)
            rec = clusters[cluster]
            t = _parse_when(when)
            if code == "001" and rec["exec"] is None:
                rec["exec"] = t
            elif code == "005":
                rec["term"] = t
        if not order:
            continue
        rec = clusters[order[-1]]
        if rec["exec"] is None or rec["term"] is None:
            logger.warning("%s: no execute/terminate pair; skipped", log_path.name)
            continue
        row = {
            "job": job, "kind": kind, "study": study,
            "seconds": (rec["term"] - rec["exec"]).total_seconds(),
            "gpu": "", "exit_status": None,
            "source": f"condor_log:{log_path.relative_to(root)}",
            "start": rec["exec"].isoformat(), "end": rec["term"].isoformat(),
        }
        prev = by_job.get(job)
        if prev is None or row["end"] > prev["end"]:
            by_job[job] = row
    return list(by_job.values())


def from_history_json(path: Path) -> list:
    """`condor_history -json` output, for a run whose event logs were archived away.

    Wall clock is CompletionDate - JobCurrentStartDate when both are present, and
    RemoteWallClockTime otherwise; the ClassAd carries the job name only as the log file
    path, so that is what the job name is recovered from.
    """
    runs = []
    ads = json.loads(path.read_text(errors="replace"))
    for ad in ads:
        name = ""
        for key in ("UserLog", "Out", "Err"):
            if ad.get(key):
                name = Path(str(ad[key])).stem
                break
        kind, study = classify(name)
        if kind is None:
            continue
        start = ad.get("JobCurrentStartDate") or ad.get("JobStartDate")
        end = ad.get("CompletionDate")
        if start and end and float(end) > float(start):
            seconds = float(end) - float(start)
        else:
            seconds = float(ad.get("RemoteWallClockTime") or 0.0)
        if seconds <= 0:
            continue
        runs.append({
            "job": name, "kind": kind, "study": study, "seconds": seconds,
            "gpu": str(ad.get("GPUsDeviceName") or ad.get("CUDADeviceName") or ""),
            "exit_status": ad.get("ExitCode"),
            "source": f"condor_history:{path.name}",
        })
    return runs


def from_runs_csv(path: Path) -> tuple:
    """Re-read an `amortization_runs.csv` this script wrote earlier.

    The point is that the measurement happens where the logs are and the ledger is wanted
    where the paper is. One 12 kB file crosses that gap, so the derived rows can be rebuilt
    -- with different `--fp-seconds`, a different `t_N`, a different haircut -- without
    going back to the access point.

    It also de-duplicates, because the first run of this script did not: pointed at a
    `results/` tree holding the same trial under two namespaces, it counted both and
    reported 50 trials for a 25-trial study. Newest wins where the row carries
    `finished`; otherwise the *larger* wall clock wins, which is the conservative choice
    for a number being used to argue that our costs are small.
    """
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_job, duplicates = {}, []
    for r in rows:
        job = r.get("job", "")
        kind, study = classify(job)
        if kind is None:
            continue
        row = {
            "job": job, "kind": kind, "study": study,
            "seconds": float(r.get("seconds") or 0.0),
            "gpu": r.get("gpu") or "",
            "exit_status": r.get("exit_status") or None,
            "finished": r.get("finished") or "",
            "source": f"runs_csv:{path.name}:{r.get('source', '')}",
        }
        prev = by_job.get(job)
        if prev is None:
            by_job[job] = row
        else:
            if row["finished"] or prev["finished"]:
                newer = row["finished"] > prev["finished"]
            else:
                newer = row["seconds"] > prev["seconds"]
            loser, winner = (prev, row) if newer else (row, prev)
            by_job[job] = winner
            duplicates.append((job, loser["source"], winner["source"],
                               loser["seconds"], winner["seconds"]))
    if duplicates:
        logger.warning("%s: %d duplicated job name(s); charged the %s of each pair",
                       path.name, len(duplicates),
                       "newest" if any(d for d in rows if d.get("finished"))
                       else "longer")
    return list(by_job.values()), duplicates


def assumed_runs(trials_per_study: int, trial_seconds: float,
                 final_seconds: float) -> list:
    """The fallback. Every row it produces is labelled `assumed:` so that provenance keeps
    an estimate distinguishable from a measurement -- the whole point of this script is that
    the ledger is checkable, and an unlabelled guess would undo that."""
    runs = []
    for study in STUDIES:
        for i in range(trials_per_study):
            runs.append({"job": f"bo_{study}_r{i // 5}_t{i % 5}", "kind": "trial",
                         "study": study, "seconds": float(trial_seconds), "gpu": "",
                         "exit_status": None, "source": "assumed:--assume-trial-seconds"})
        runs.append({"job": f"{study}_final", "kind": "final", "study": study,
                     "seconds": float(final_seconds), "gpu": "",
                     "exit_status": None, "source": "assumed:--assume-final-seconds"})
    return runs


def collect_runs(args) -> tuple:
    """Runs from the best available source, plus a note on which one it was.

    Sources are not merged. A run that appears in both an archive and an event log would be
    counted twice, and the two disagree slightly anyway -- the archive measures the job's
    own runtime, the log measures what HTCondor charged for it, and the difference is file
    transfer. Preferring one keeps the ledger internally consistent.
    """
    if args.runs_csv and args.runs_csv.exists():
        runs, duplicates = from_runs_csv(args.runs_csv)
        if runs:
            note = (f"measured runs re-read from {args.runs_csv.name}"
                    + (f"; {len(duplicates)} duplicate job name(s) de-duplicated"
                       if duplicates else ""))
            return runs, note, duplicates
        logger.warning("%s carried no recognised BO jobs", args.runs_csv)
    if args.trial_archives and args.trial_archives.exists():
        runs, duplicates = from_archives(args.trial_archives)
        if runs:
            note = "run_meta inside job archives (the jobs' own measurement)"
            if duplicates:
                note += (f"; {len(duplicates)} duplicate job name(s) de-duplicated, "
                         "newest archive charged")
            return runs, note, duplicates
        logger.warning("%s held no run_meta_*.json", args.trial_archives)
    if args.history_json and args.history_json.exists():
        runs = from_history_json(args.history_json)
        if runs:
            return runs, f"condor_history dump {args.history_json.name}", []
    if args.condor_logs and args.condor_logs.exists():
        runs = from_condor_logs(args.condor_logs)
        if runs:
            return runs, "HTCondor event logs (terminate minus execute)", []
        logger.warning("%s held no recognised BO job logs", args.condor_logs)
    logger.warning("no measured training cost available -- falling back to --assume-*. "
                   "Every derived row will be marked `assumed` in the provenance.")
    return (assumed_runs(args.trials_per_study, args.assume_trial_seconds,
                         args.assume_final_seconds),
            "ASSUMED (no logs or archives found)", [])


def summarize_runs(runs: list) -> dict:
    """Per-study and program totals, in GPU-seconds."""
    per_study = {}
    for study in sorted({r["study"] for r in runs}):
        trials = [r["seconds"] for r in runs
                  if r["study"] == study and r["kind"] == "trial"]
        finals = [r["seconds"] for r in runs
                  if r["study"] == study and r["kind"] == "final"]
        per_study[study] = {
            "n_trials": len(trials),
            "trial_seconds_total": sum(trials),
            "trial_seconds_median": statistics.median(trials) if trials else 0.0,
            "final_seconds": sum(finals),
            "total_seconds": sum(trials) + sum(finals),
        }
    per_task = {
        task: sum(per_study.get(s, {}).get("total_seconds", 0.0) for s in pair)
        for task, pair in TASK_PAIRS.items()
    }
    totals = {
        "program_seconds": sum(v["total_seconds"] for v in per_study.values()),
        "n_runs": len(runs),
        "n_trials": sum(v["n_trials"] for v in per_study.values()),
        "median_model_seconds": (statistics.median(
            [v["total_seconds"] for v in per_study.values()]) if per_study else 0.0),
        "median_task_seconds": (statistics.median(list(per_task.values()))
                                if per_task else 0.0),
        "gpus_seen": sorted({r["gpu"] for r in runs if r.get("gpu")}),
    }
    return {"per_study": per_study, "per_task": per_task, "totals": totals}


# ----------------------------------------------------------------------------------------
# t_N: what one retrieval costs online
# ----------------------------------------------------------------------------------------
def inference_cost(args) -> tuple:
    """(seconds per sounding, description) from benchmark.csv, or from --assume.

    The default selection is the costliest honest one: the 20-level profile task, the flow,
    posterior sampling included, at the largest batch the benchmark ran. Quoting the point
    estimate's 8 microseconds would be true and useless -- nobody deploys the model without
    the posterior, and the ledger is about the deployed system.
    """
    path = args.benchmark_csv
    if path and path.exists():
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        want = [r for r in rows
                if r.get("task") == args.task
                and r.get("method") == args.method
                and r.get("scope") == args.scope]
        if want:
            if args.batch_size is not None:
                want = [r for r in want
                        if int(float(r["batch_size"])) == args.batch_size] or want
            # Largest batch size available, i.e. the deployment configuration.
            row = max(want, key=lambda r: int(float(r["batch_size"])))
            ms = float(row["ms_per_sounding"])
            desc = (f"benchmark.csv: task={row['task']} method={row['method']} "
                    f"scope={row['scope']} batch={row['batch_size']} "
                    f"({ms:.3f} ms/sounding)")
            return ms / 1000.0, desc
        logger.warning("%s has no row for task=%s method=%s scope=%s",
                       path, args.task, args.method, args.scope)
    logger.warning("using --assume-ms-per-sounding=%.4f; re-run benchmark.py and point "
                   "--benchmark-csv at its output to make this a measurement",
                   args.assume_ms_per_sounding)
    return (args.assume_ms_per_sounding / 1000.0,
            f"ASSUMED {args.assume_ms_per_sounding} ms/sounding "
            f"(--assume-ms-per-sounding)")


def ensemble_size(args) -> tuple:
    """How many simulated soundings the training ensemble holds, and how many reference
    soundings they came from. Regenerating the ensemble after a forward-model change is the
    one recurring cost that is paid in *core*-seconds rather than GPU-seconds, so it is the
    only part of the ledger that compares with t_F without a unit caveat."""
    if args.n_ensemble:
        return args.n_ensemble, None, "--n-ensemble"
    if args.split_csv and args.split_csv.exists():
        with open(args.split_csv, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        refs = None
        if rows and "clst" in rows[0] and "ref" in rows[0]:
            refs = len({(r["clst"], r["ref"]) for r in rows})
        return len(rows), refs, f"split table {args.split_csv.name}"
    return 0, None, "unavailable"


# ----------------------------------------------------------------------------------------
# The ledger
# ----------------------------------------------------------------------------------------
# Where t_F comes from depends on what the caller asked for, and getting this wrong is a
# mis-citation rather than a rounding error: 150 s is *our* figure, obtained by the UQ group
# at JPL on the runs that produced the ACOS comparator, and Taylor et al. (2023) quote twice
# that. Attributing the default 150 s to Taylor -- which every provenance string in this file
# used to do -- credits them with a number they did not report, and throws away the fact that
# the figure carrying the argument is the most conservative one available.
FP_SOURCES = (
    (150.0, "obtained by the UQ group at JPL on the runs that produced the ACOS L2FP "
            "comparator for this ensemble; the smallest figure available to us, and so "
            "the conservative choice. Corroborated by Taylor et al. (2023) AMT 16, "
            "3173-3209 sec. 3 ('about 5 min per sounding on a single processor', 2x) and "
            "by the 221 core-s implied by Lamminpaa et al. (2025) Table 2 at the median "
            "of four iterations (1.5x)"),
    (300.0, "Taylor et al. (2023) AMT 16, 3173-3209 sec. 3: 'about 5 min per sounding on "
            "a single processor'"),
    (221.04, "implied by Lamminpaa et al. (2025) AMT 18, 673-694 Table 2: 55.26 core-s "
             "per RT evaluation with Jacobian, at the median of four iterations of "
             "Section 4.1"),
)


def fp_provenance(t_f: float) -> str:
    """Basis string for t_F, keyed on the value actually used."""
    for known, note in FP_SOURCES:
        if abs(t_f - known) < 1e-6:
            return note
    return ("supplied by the caller via --fp-seconds; no published source is asserted for "
            "this value")


def fp_caption_clause(t_f: float) -> str:
    """How the generated caption should attribute t_F. The default 150 s is unpublished, so
    it gets a prose attribution rather than a key it has no right to."""
    if abs(t_f - 150.0) < 1e-6:
        return ("a single-processor figure obtained by the group that produced the "
                "simulation ensemble, and the smallest available to us: the literature "
                "reports comparable or larger costs "
                "\\citep{taylor2023evaluating, lamminpaa2025}")
    if abs(t_f - 300.0) < 1e-6:
        return ("a single-processor figure on unspecified hardware "
                "\\citep{taylor2023evaluating}")
    if abs(t_f - 221.04) < 1e-6:
        return ("implied by the radiative-transfer timings of "
                "\\citet{lamminpaa2025} at the median iteration count of Section "
                "\\ref{subsec:baselines}")
    return "supplied by the caller; no published source is asserted for it"


# ----------------------------------------------------------------------------------------
def build_ledger(args, cost: dict, t_n: float, t_n_desc: str,
                 n_sim: int, n_refs, sim_desc: str) -> tuple:
    """Every quantity the reply needs, as (quantity, value, unit, basis) rows."""
    t_f = args.fp_seconds
    rows = []

    def add(q, v, u, basis):
        rows.append({"quantity": q, "value": v, "unit": u, "basis": basis})

    # -- the two per-sounding costs -----------------------------------------------------
    add("t_F, full-physics retrieval", t_f, "core-s/sounding", fp_provenance(t_f))
    add("t_N, amortized predictive", t_n, "GPU-s/sounding", t_n_desc)
    ratio = t_f / t_n
    add("throughput ratio per device", ratio, "x",
        "t_F / t_N -- one core against one GPU, NOT a same-hardware benchmark")
    add("cores matched by one GPU", ratio, "cores/GPU",
        "the same number read the other way: how many single-core full-physics workers "
        "one GPU replaces")
    add("soundings per GPU-day", 86400.0 / t_n, "soundings/GPU-day", "86400 / t_N")

    # -- the one-time costs ---------------------------------------------------------------
    tot = cost["totals"]
    t0_model = tot["median_model_seconds"]
    t0_task = tot["median_task_seconds"]
    t0_prog = tot["program_seconds"]
    add("T_0, one model (search + retrain)", t0_model, "GPU-s",
        f"median over {len(cost['per_study'])} studies, {tot['n_trials']} trials total")
    add("T_0, one task (deterministic + flow)", t0_task, "GPU-s",
        "median over the three task pairs")
    add("T_0, whole program (six models)", t0_prog, "GPU-s",
        "every search trial and every retrain in the revision")
    for label, seconds in (("one model", t0_model), ("one task", t0_task),
                           ("whole program", t0_prog)):
        add(f"T_0, {label}", seconds / 3600.0, "GPU-h", "the row above, in hours")

    # Ensemble regeneration: the recurring cost that is *not* GPU time. c_F is bounded
    # above by t_F divided by the median iteration count of the operational retrieval,
    # because every iteration costs at least one forward evaluation.
    c_f = args.forward_seconds
    t_sim = n_sim * c_f
    add("N_sim, simulated soundings in the ensemble", n_sim, "soundings", sim_desc)
    if n_refs:
        add("reference soundings behind them", n_refs, "soundings", sim_desc)
    add("c_F, one forward-model (RT) evaluation", c_f, "core-s",
        "Lamminpaa et al. (2025) AMT 18, 673-694, Table 2: ReFRACtor RT, 2023 MacBook Pro")
    add("c_F + Jacobian, one inversion iteration", args.forward_jacobian_seconds, "core-s",
        "same table, RT + Jacobian column")
    add("T_sim, regenerate the training ensemble", t_sim, "core-s",
        "N_sim * c_F -- what a forward-model or spectroscopy change actually costs us")
    add("T_sim, regenerate the training ensemble", t_sim / 3600.0, "core-h",
        "the row above, in hours")

    # A consistency check between the two published figures, which come from different
    # papers, different hardware and different implementations: one iteration costs an RT
    # evaluation plus its Jacobian, and Section 4.1 reports a median of four iterations on
    # this ensemble. If the product lands near t_F, the two numbers corroborate each other
    # and neither is carrying the argument alone.
    implied = args.forward_jacobian_seconds * args.fp_iterations
    add("t_F implied by (RT + Jacobian) x median iterations", implied, "core-s/sounding",
        f"{args.forward_jacobian_seconds:g} s x {args.fp_iterations:g} iterations, against "
        f"the {t_f:g} s used as t_F above -- an independent corroboration, not an "
        "input to anything below")
    add("ratio of that to the quoted t_F", implied / t_f, "x",
        "how well the two published figures agree")

    # -- break-even ------------------------------------------------------------------------
    denom = t_f - t_n
    n_day_all = args.soundings_per_day
    n_day_qc = args.soundings_per_day * args.qc_fraction
    add("soundings collected per day", n_day_all, "soundings/day",
        "'yielding almost 1 million soundings each day' (Crisp et al. 2017, AMT 10, 59-81); "
        "'approximately 1 million soundings per day' (Taylor et al. 2016, AMT 9, 973-989)")
    add("soundings the full-physics retrieval is run on per day",
        n_day_all * args.prescreen_fraction, "soundings/day",
        f"{args.prescreen_fraction:.1%} of collected pass cloud screening (Taylor et al. "
        "2016: 20-25 %). This is the operational workload, and it is larger than the "
        "quality-passing count below")
    add("soundings passing quality control per day", n_day_qc, "soundings/day",
        f"{args.qc_fraction:.0%} of collected: Eldering et al. (2017) report 7-12 % "
        "sufficiently cloud- and aerosol-free to pass all quality tests")

    for label, t0, unit_note in (
            ("one model", t0_model, "GPU-s charged against core-s"),
            ("one task", t0_task, "GPU-s charged against core-s"),
            ("whole program", t0_prog, "GPU-s charged against core-s"),
            ("ensemble regeneration", t_sim, "core-s against core-s, unit-clean")):
        n_star = t0 / denom
        add(f"N*, break-even soundings ({label})", n_star, "soundings",
            f"T_0 / (t_F - t_N); {unit_note}")
        add(f"N* as a fraction of one day's QC soundings ({label})",
            n_star / n_day_qc, "days", "N* / (soundings passing QC per day)")
        add(f"N* as hours of the QC stream ({label})", 24.0 * n_star / n_day_qc, "h",
            "how long the mission takes to collect N* usable soundings")

    # -- the head-start question ----------------------------------------------------------
    # N_F(T) = (T + Delta) * K / t_F ;  N_A(T) = max(0, T - T_0) * G / t_N
    delta = args.head_start_hours * 3600.0
    r_f, r_n = 1.0 / t_f, 1.0 / t_n
    for k in args.cores:
        for t0_label, t0 in (("one model", t0_model), ("whole program", t0_prog)):
            num = r_f * k * (t0 + delta)
            den = r_n * args.gpus - r_f * k
            if den <= 0:
                add(f"crossover, {k} cores vs {args.gpus} GPU ({t0_label})",
                    float("inf"), "h",
                    "the CPU pool out-throughputs the GPU: never crosses")
                continue
            t_x = t0 + num / den
            add(f"crossover, {k} cores vs {args.gpus} GPU ({t0_label})", t_x / 3600.0, "h",
                f"T_0 + R_F*K*(T_0+Delta)/(R_N*G - R_F*K), head start "
                f"Delta = {args.head_start_hours:g} h")
            add(f"crossover minus T_0, {k} cores ({t0_label})", (t_x - t0) / 3600.0, "h",
                "how long after training ends the amortized pipeline is ahead")

    # -- keeping up with the stream --------------------------------------------------------
    for label, n_day in (("QC-passing", n_day_qc),
                         ("pre-screened", n_day_all * args.prescreen_fraction),
                         ("all collected", n_day_all)):
        add(f"cores needed to keep up with the {label} stream",
            n_day * t_f / 86400.0, "cores", "N_day * t_F / 86400")
        add(f"core-hours to retrieve one day, {label}", n_day * t_f / 3600.0, "core-h",
            "N_day * t_F")
        add(f"GPU-minutes to retrieve one day, {label}", n_day * t_n / 60.0, "GPU-min",
            "N_day * t_N")

    # -- retraining as a recurring charge --------------------------------------------------
    per_year = n_day_qc * 365.0
    add("QC-passing soundings per year", per_year, "soundings/year",
        "365 * the daily QC figure")
    for k in args.retrains_per_year:
        for t0_label, t0 in (("one model", t0_model), ("whole program", t0_prog)):
            surcharge = k * t0 / per_year
            add(f"amortized training surcharge, {k}x/year ({t0_label})",
                surcharge * 1000.0, "ms/sounding", "k * T_0 / (soundings per year)")
            add(f"total per-sounding cost, {k}x/year ({t0_label})",
                (t_n + surcharge) * 1000.0, "ms/sounding", "t_N + k*T_0/N_year")
            add(f"ratio to full physics, {k}x/year ({t0_label})",
                t_f / (t_n + surcharge), "x", "t_F / (t_N + surcharge)")
    add("most frequent retraining that still breaks even (whole program)",
        24.0 * (t0_prog / denom) / n_day_qc, "h of QC stream",
        "retrain every N* soundings and the ledger is still even")

    # -- the unit haircut -------------------------------------------------------------------
    # A GPU-second is not a core-second. Divide the GPU by H notional cores and see whether
    # the conclusion survives; if it does not at H = 100 the claim should not be made.
    for h in args.gpu_core_equivalence:
        add(f"throughput ratio, GPU charged as {h} cores", ratio / h, "x",
            "t_F / (H * t_N) -- hardware-agnostic haircut")
        add(f"N* whole program, GPU charged as {h} cores", h * t0_prog / denom,
            "soundings", "H * T_0 / (t_F - t_N)")
        add(f"N* whole program as hours of QC stream, GPU charged as {h} cores",
            24.0 * (h * t0_prog / denom) / n_day_qc, "h", "the row above, as stream time")
        # The two stress tests crossed: charge the GPU as H cores *and* retrain the whole
        # six-model program k times a year. This is the row that decides whether the reply
        # survives a sceptical reader who grants us nothing.
        for k in args.retrains_per_year + [365]:
            total = h * (t_n + k * t0_prog / per_year)
            add(f"total per-sounding cost, {k}x/year, GPU charged as {h} cores",
                total, "core-s/sounding", "H * (t_N + k*T_0/N_year)")
            add(f"ratio to full physics, {k}x/year, GPU charged as {h} cores",
                t_f / total, "x", "t_F / the row above")

    return rows, {"t_f": t_f, "t_n": t_n, "ratio": ratio, "t0_model": t0_model,
                  "t0_task": t0_task, "t0_prog": t0_prog, "t_sim": t_sim,
                  "n_day_qc": n_day_qc, "n_day_all": n_day_all, "denom": denom}


# ----------------------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------------------
def write_tex(rows: list, key: dict, path: Path, source_note: str,
              fp_seconds: float = 150.0) -> None:
    """The ledger as the table the paper would carry: the three costs, the ratio, the
    break-even, and the crossover. Everything else stays in the CSV."""
    def val(name, unit=None):
        for r in rows:
            if r["quantity"] == name and (unit is None or r["unit"] == unit):
                return r["value"]
        return float("nan")

    lines = [
        "% GENERATED by experiments/scoring/amortization.py -- do not edit by hand.",
        f"% training cost source: {source_note}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\begin{tabular}{lrl}",
        "\\toprule",
        "Quantity & Value & Unit \\\\",
        "\\midrule",
        f"Full-physics retrieval, $t_F$ & {key['t_f']:.0f} & core-s per sounding \\\\",
        f"Amortized predictive, $t_N$ & {key['t_n'] * 1000:.2f} & "
        "GPU-ms per sounding \\\\",
        f"Throughput ratio per device & {key['ratio']:,.0f}$\\times$ & --- \\\\",
        "\\midrule",
        f"Search and training, one model & {key['t0_model'] / 3600:.1f} & GPU-h \\\\",
        f"Search and training, one task & {key['t0_task'] / 3600:.1f} & GPU-h \\\\",
        f"Search and training, six models & {key['t0_prog'] / 3600:.1f} & GPU-h \\\\",
        f"Regenerating the training ensemble & {key['t_sim'] / 3600:.0f} & core-h \\\\",
        "\\midrule",
        f"Break-even, one model & {val('N*, break-even soundings (one model)'):,.0f} & "
        "soundings \\\\",
        f"Break-even, six models & {val('N*, break-even soundings (whole program)'):,.0f} "
        "& soundings \\\\",
        f"Break-even, ensemble regeneration & "
        f"{val('N*, break-even soundings (ensemble regeneration)'):,.0f} & soundings \\\\",
        f"\\quad as hours of quality-passing data & "
        f"{val('N* as hours of the QC stream (ensemble regeneration)'):.1f} & h \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Offline and online cost of the amortized retrieval. $t_F$ is "
        f"{fp_caption_clause(fp_seconds)}; "
        "$t_N$ is measured here on one GPU. A core-second and a GPU-second are different "
        "units, so "
        "the ratio is between devices and not a controlled benchmark. The break-even count "
        "is $N^\\star = T_0/(t_F - t_N)$, and the ensemble row is the one comparison in "
        "which both sides are core-seconds.}",
        "\\label{tab:amortization}",
        "\\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(key: dict, args, path: Path) -> bool:
    """Cumulative retrievals against wall clock, with the crossover marked.

    Optional on purpose: this script's job is the arithmetic, and it must run on a login
    node with nothing installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable; skipping the figure")
        return False

    t_f, t_n, t0 = key["t_f"], key["t_n"], key["t0_prog"]
    delta = args.head_start_hours * 3600.0
    horizon = max(4.0 * t0, delta + 4.0 * t0, 6 * 3600.0)
    ts = [horizon * i / 600.0 for i in range(601)]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for k in args.cores:
        ax.plot([t / 3600.0 for t in ts],
                [max((t + delta) * k / t_f, 1e-1) for t in ts],
                lw=2, label=f"full physics, {k:,} core{'s' if k > 1 else ''}"
                            f" (started {args.head_start_hours:g} h earlier)")
    ax.plot([t / 3600.0 for t in ts],
            [max((t - t0) * args.gpus / t_n, 1e-1) if t > t0 else 1e-1 for t in ts],
            lw=2.5, color="k",
            label=f"amortized, {args.gpus} GPU (training first)")
    ax.axvline(t0 / 3600.0, ls="--", color="0.4", lw=1)
    ax.annotate("training ends", (t0 / 3600.0, 1e2),
                xytext=(6, 0), textcoords="offset points", rotation=90,
                va="bottom", fontsize=9, color="0.3")
    ax.axhline(key["n_day_qc"], ls=":", color="0.5", lw=1)
    ax.annotate("one day of quality-passing soundings",
                (horizon / 3600.0, key["n_day_qc"]), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=9, color="0.3")
    ax.set_yscale("log")
    ax.set_xlabel("wall clock since the amortized project started (h)")
    ax.set_ylabel("soundings retrieved (cumulative)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return True


def markdown(rows: list, key: dict, cost: dict, source_note: str) -> str:
    """The block to paste into."""
    def val(name, unit=None):
        for r in rows:
            if r["quantity"] == name and (unit is None or r["unit"] == unit):
                return r["value"]
        return float("nan")

    out = ["| quantity | value | unit |", "|---|---:|---|"]
    for r in rows:
        v = r["value"]
        shown = (f"{v:,.4g}" if isinstance(v, float) and abs(v) < 1e-2
                 else f"{v:,.2f}" if isinstance(v, float) else f"{v:,}")
        out.append(f"| {r['quantity']} | {shown} | {r['unit']} |")
    head = [
        f"**Training cost source:** {source_note}",
        "",
        f"**Per study** ({len(cost['per_study'])} studies, "
        f"{cost['totals']['n_trials']} trials):",
        "",
        "| study | trials | trial GPU-h | median trial (min) | retrain GPU-h | "
        "total GPU-h |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for study, v in sorted(cost["per_study"].items()):
        head.append(f"| `{study}` | {v['n_trials']} | "
                    f"{v['trial_seconds_total'] / 3600:.2f} | "
                    f"{v['trial_seconds_median'] / 60:.1f} | "
                    f"{v['final_seconds'] / 3600:.2f} | "
                    f"{v['total_seconds'] / 3600:.2f} |")
    if cost["totals"]["gpus_seen"]:
        head += ["", "GPUs seen: " + ", ".join(cost["totals"]["gpus_seen"])]
    head += ["", "**The ledger:**", ""]
    return "\n".join(head + out)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # where the training cost comes from
    ap.add_argument("--trial-archives", type=Path, default=None,
                    help="directory of fetched job archives; run_meta_*.json is read "
                         "straight out of them (best source)")
    ap.add_argument("--runs-csv", type=Path, default=None,
                    help="an `amortization_runs.csv` this script wrote on the access "
                         "point; highest priority, so the ledger can be rebuilt where the "
                         "paper is without re-reading /staging")
    ap.add_argument("--condor-logs", type=Path, default=Path("chtc/logs"))
    ap.add_argument("--history-json", type=Path, default=None,
                    help="output of `condor_history -json`")
    ap.add_argument("--trials-per-study", type=int, default=25)
    ap.add_argument("--assume-trial-seconds", type=float, default=600.0,
                    help="fallback per-trial wall clock (default 600 s = the 20-epoch "
                         "trials of the hyperparameter sweep)")
    ap.add_argument("--assume-final-seconds", type=float, default=1200.0,
                    help="fallback wall clock for the 40-epoch retrain")
    # where the inference cost comes from
    ap.add_argument("--benchmark-csv", type=Path,
                    default=Path("experiments/scoring/tables/benchmark.csv"))
    ap.add_argument("--task", default="profiles")
    ap.add_argument("--method", default="flow",
                    choices=("flow", "laplace", "point", "point_estimate"))
    ap.add_argument("--scope", default="model", choices=("model", "end_to_end"))
    ap.add_argument("--batch-size", type=int, default=None,
                    help="pick this batch size instead of the largest available")
    ap.add_argument("--assume-ms-per-sounding", type=float, default=4.09,
                    help="fallback t_N (default: the flow row of Table 8 at batch 256)")
    # the operational side
    ap.add_argument("--fp-seconds", type=float, default=150.0,
                    help="full-physics cost per sounding per core. Default is the figure "
                         "the manuscript quotes, from the group that produced the "
                         "ensemble; --fp-seconds 300 gives Taylor et al. (2023)'s about "
                         "5 min per sounding on a single processor, which is larger and so "
                         "more favourable to us")
    ap.add_argument("--fp-iterations", type=float, default=4.0,
                    help="median iteration count of the operational retrieval on this "
                         "ensemble (Section 4.1); used for the consistency check between "
                         "the two published cost figures")
    ap.add_argument("--forward-seconds", type=float, default=33.45,
                    help="cost of one forward-model (RT) evaluation. Default is the "
                         "measured ReFRACtor figure from Lamminpaa et al. (2025) Table 2, "
                         "AMT 18, 673-694, on a 2023 MacBook Pro")
    ap.add_argument("--forward-jacobian-seconds", type=float, default=55.26,
                    help="one RT evaluation plus its Jacobian, i.e. what one iteration of "
                         "the inversion costs; same table")
    ap.add_argument("--soundings-per-day", type=float, default=1e6)
    ap.add_argument("--qc-fraction", type=float, default=0.10,
                    help="fraction passing all quality tests (Eldering et al. 2017: 7-12 %%)")
    ap.add_argument("--prescreen-fraction", type=float, default=0.225,
                    help="fraction of soundings the full-physics retrieval is actually run "
                         "on, i.e. those passing cloud screening (Taylor et al. 2016: "
                         "20-25 %%). This, not the quality-passing fraction, is the "
                         "operational workload")
    # the scenarios
    ap.add_argument("--cores", type=int, nargs="+", default=[1, 1000, 10000])
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--head-start-hours", type=float, default=24.0)
    ap.add_argument("--retrains-per-year", type=int, nargs="+", default=[1, 4, 12])
    ap.add_argument("--gpu-core-equivalence", type=int, nargs="+", default=[1, 100, 1000])
    # the ensemble
    ap.add_argument("--split-csv", type=Path, default=Path("group_split_v1_cov.csv"))
    ap.add_argument("--n-ensemble", type=int, default=None)
    # output
    ap.add_argument("--out", type=Path, default=Path("experiments/scoring/tables"))
    ap.add_argument("--fig", type=Path, default=None,
                    help="write the crossover figure here, e.g. latex/figures")
    args = ap.parse_args()
    if args.method == "point_estimate":
        args.method = "point"

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    runs, source_note, duplicates = collect_runs(args)
    cost = summarize_runs(runs)
    t_n, t_n_desc = inference_cost(args)
    n_sim, n_refs, sim_desc = ensemble_size(args)
    rows, key = build_ledger(args, cost, t_n, t_n_desc, n_sim, n_refs, sim_desc)

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "amortization.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["quantity", "value", "unit", "basis"])
        w.writeheader()
        w.writerows(rows)

    runs_path = args.out / "amortization_runs.csv"
    # `--runs-csv <out>/amortization_runs.csv` is the natural way to rebuild the ledger in
    # place, and it destroys its own input: the de-duplicated rows are written over the raw
    # measurement, so the duplicates cannot be inspected afterwards and a second run sees
    # nothing to de-duplicate. Keep the original. It happened once, on the only copy of a
    # 231-row measurement, which is why this is here rather than in a comment.
    if args.runs_csv and args.runs_csv.resolve() == runs_path.resolve():
        raw_path = args.out / "amortization_runs_raw.csv"
        if not raw_path.exists():
            raw_path.write_bytes(args.runs_csv.read_bytes())
            logger.info("input preserved as %s before it is overwritten", raw_path.name)
    with open(runs_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["job", "kind", "study", "seconds", "gpu",
                                           "exit_status", "finished", "source"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(runs)

    # The de-duplication is a judgement the reader has to be able to overturn, so the
    # discarded archives are written out rather than only warned about.
    dup_path = args.out / "amortization_runs_duplicates.csv"
    if duplicates:
        with open(dup_path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["job", "ignored_archive", "charged_archive",
                        "ignored_seconds", "charged_seconds"])
            w.writerows(duplicates)
    elif dup_path.exists():
        dup_path.unlink()

    prov = {
        "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "training_cost_source": source_note,
        "training_cost_is_measured": not source_note.startswith("ASSUMED"),
        "inference_cost_source": t_n_desc,
        "inference_cost_is_measured": not t_n_desc.startswith("ASSUMED"),
        "ensemble_source": sim_desc,
        "n_simulated_soundings": n_sim,
        "n_reference_soundings": n_refs,
        "n_duplicate_job_names": len(duplicates),
        "duplicates": [{"job": j, "ignored": o, "charged": n,
                        "ignored_seconds": so, "charged_seconds": sn}
                       for j, o, n, so, sn in duplicates],
        "operational_cost_seconds": args.fp_seconds,
        "operational_cost_citation": fp_provenance(args.fp_seconds),
        "forward_model_cost_citation": (
            "Lamminpaa, O. et al. (2025), Forward model emulator for atmospheric "
            "radiative transfer using Gaussian processes and cross validation, Atmos. "
            "Meas. Tech. 18, 673-694, doi:10.5194/amt-18-673-2025, Table 2: ReFRACtor "
            "RT 33.45 s, RT + Jacobian 55.26 s, on a 2023 MacBook Pro. A different "
            "implementation and different hardware from the build that generated our "
            "ensemble, so it is an order-of-magnitude figure for one forward evaluation, "
            "not a measurement of our own pipeline."),
        "collection_rate_citation": (
            "Crisp, D. et al. (2017), The on-orbit performance of the Orbiting Carbon "
            "Observatory-2 (OCO-2) instrument and its radiometrically calibrated products, "
            "Atmos. Meas. Tech. 10, 59-81, doi:10.5194/amt-10-59-2017: 'yielding almost 1 "
            "million soundings each day'; and Taylor, T. E. et al. (2016), OCO-2 cloud "
            "screening algorithms, Atmos. Meas. Tech. 9, 973-989, "
            "doi:10.5194/amt-9-973-2016: 'approximately 1 million soundings per day', with "
            "'processing of ~20-25 % of all OCO-2 soundings' after cloud screening."),
        "quality_fraction_citation": (
            "Eldering, A. et al. (2017), The Orbiting Carbon Observatory-2: first 18 "
            "months of science data products, Atmos. Meas. Tech. 10, 549-563, "
            "doi:10.5194/amt-10-549-2017: 'On monthly timescales, 7 to 12 % of these "
            "measurements are sufficiently cloud and aerosol free to yield estimates of "
            "the column-averaged atmospheric CO2 dry air mole fraction, XCO2, that pass "
            "all quality tests.'"),
        "caveat": ("t_F is a published single-processor figure and t_N is measured on one "
                   "GPU. These are different units; the --gpu-core-equivalence rows exist "
                   "so the conclusion can be checked after charging the GPU as 100 or "
                   "1000 cores."),
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "per_study": cost["per_study"],
        "per_task": cost["per_task"],
        "totals": cost["totals"],
    }
    (args.out / "amortization.json").write_text(json.dumps(prov, indent=2),
                                                encoding="utf-8")
    write_tex(rows, key, args.out / "amortization.tex", source_note, args.fp_seconds)

    if args.fig is not None:
        if write_figure(key, args, args.fig / "amortization_crossover.png"):
            logger.info("-> %s", args.fig / "amortization_crossover.png")

    print()
    print(markdown(rows, key, cost, source_note))
    print()
    for p in (csv_path, runs_path, args.out / "amortization.json",
              args.out / "amortization.tex"):
        print(f"-> {p}")
    if source_note.startswith("ASSUMED"):
        print("\n!! The training cost is ASSUMED, not measured. Re-run with "
              "--condor-logs / --trial-archives pointing at the real run before any of "
              "these numbers goes into the paper.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

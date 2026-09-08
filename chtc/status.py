#!/usr/bin/env python3
"""What is running, what finished, what failed and why -- in one screen.

    python3 chtc/status.py              # one line per job, failures first
    python3 chtc/status.py -v           # plus the last few lines of each failure's .err
    python3 chtc/status.py --archive    # move the current logs aside and start clean

`logs/` accumulates. HTCondor truncates `<job>.out` and `<job>.err` on each run but
*appends* to `<job>.log`, so after a few attempts the directory is forty files in which the
only way to find the one that failed is to cat them in turn. This reads the HTCondor event
log -- which is structured, unlike the job's own stdout -- takes the outcome of the most
recent cluster for each job, and pairs it with the first real error line from that job's
`.err`.

Reading the event log rather than the job output is the point: a job that was held during
file transfer, or evicted, never wrote a line of its own, and those are exactly the
failures that are hardest to find by eye.
"""

from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"

# "005 (6037497.000.000) 2026-08-14 16:41:14 Job terminated."
EVENT = re.compile(r"^(\d{3}) \((\d+)\.(\d+)\.(\d+)\) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (.*)$")
RETVAL = re.compile(r"\((\d+)\) Normal termination \(return value (\d+)\)")

# Event codes we care about. Anything else is progress noise.
STATES = {
    "000": "submitted",
    "001": "running",
    "004": "evicted",
    "005": "terminated",
    "009": "aborted",
    "012": "HELD",
    "013": "released",
}

# Lines in a .err that are never the actual problem.
NOISE = re.compile(
    r"UserWarning|warnings\.warn|Lazy modules are a new feature|"
    r"^\s*$|^\s+warnings\.warn|FutureWarning|DeprecationWarning"
)


def parse_log(path: Path) -> dict:
    """Outcome of the most recent cluster in one HTCondor event log."""
    clusters: dict[str, list] = {}
    order: list[str] = []
    lines = path.read_text(errors="replace").splitlines()

    for i, line in enumerate(lines):
        m = EVENT.match(line)
        if not m:
            continue
        code, cluster, _proc, _sub, when, text = m.groups()
        if cluster not in clusters:
            clusters[cluster] = []
            order.append(cluster)
        rec = {"code": code, "when": when, "text": text, "exit": None, "detail": ""}
        if code == "005":
            for follow in lines[i + 1: i + 4]:
                r = RETVAL.search(follow)
                if r:
                    rec["exit"] = int(r.group(2))
                    break
        if code in ("012", "009", "004"):
            # The hold/abort reason is on the following lines and is the whole story.
            rec["detail"] = " ".join(l.strip() for l in lines[i + 1: i + 3]).strip()
        clusters[cluster].append(rec)

    if not order:
        return {"state": "no events", "cluster": "-", "when": "-", "exit": None, "detail": ""}

    cluster = order[-1]
    events = clusters[cluster]

    # Only some events change a job's state. 006 ("Image size of job updated") and 040
    # ("transferring files") are progress notes that a long-running job emits repeatedly,
    # and taking the literal last event made a healthy running job display as "006".
    # Walk back to the most recent event that actually says where the job is.
    last = next((e for e in reversed(events) if e["code"] in STATES), events[-1])
    state = STATES.get(last["code"], last["code"])

    if last["code"] == "005":
        state = "done" if last["exit"] == 0 else "FAILED"

    start = next((e["when"] for e in events if e["code"] == "001"), None)
    runtime = ""
    if start:
        try:
            t0 = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
            # For a job still on a machine the last event IS the start, so measuring to it
            # reports 0m00s for something that may have been running for hours. Measure to
            # now instead. HTCondor writes these in the submit machine's local time, which
            # is what datetime.now() gives us.
            end = datetime.now() if state in ("running", "submitted") \
                else datetime.strptime(last["when"], "%Y-%m-%d %H:%M:%S")
            secs = max(int((end - t0).total_seconds()), 0)
            runtime = (f"{secs // 3600}h{(secs % 3600) // 60:02d}m" if secs >= 3600
                       else f"{secs // 60}m{secs % 60:02d}s")
        except ValueError:
            pass

    return {"state": state, "cluster": cluster, "when": last["when"],
            "exit": last["exit"], "detail": last["detail"], "runtime": runtime}


ERRORISH = re.compile(
    r"Traceback \(most recent call last\)|^\w*(Error|Exception)\b|Error:|ERROR|"
    r"assert|Killed|Segmentation fault|CUDA out of memory|No such file"
)


def first_error(job: str, n: int = 1) -> list:
    """The lines of a job's .err most likely to be the actual failure.

    Not the first lines. Anything using `logging` writes to stderr by default, so for
    predict.py the head of the file is `INFO predict: test=4489 train=36003 ...` -- a
    perfectly healthy startup message reported as the cause of death. Python also puts the
    useful half of a traceback at the *end*.

    So: find the last error-shaped line and show from there. Failing that, show the tail.
    """
    p = LOGS / f"{job}.err"
    if not p.exists():
        return []
    lines = [l.rstrip() for l in p.read_text(errors="replace").splitlines()
             if l.strip() and not NOISE.search(l)]
    if not lines:
        return []

    start = None
    for i, line in enumerate(lines):
        if ERRORISH.search(line):
            start = i
    if start is None:
        return [l.strip() for l in lines[-n:]]
    # A traceback's exception line is at its end, so bias towards the tail of the block.
    block = lines[start:]
    return [l.strip() for l in (block[-n:] if len(block) > n else block)]


def archive() -> int:
    if not LOGS.exists():
        print(f"{LOGS} does not exist.")
        return 1
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = LOGS / "archive" / stamp
    dest.mkdir(parents=True)
    moved = 0
    for f in sorted(LOGS.glob("*")):
        if f.is_dir():
            continue
        shutil.move(str(f), str(dest / f.name))
        moved += 1
    print(f"moved {moved} files to {dest.relative_to(HERE)}")
    print("logs/ is clean; the next submit starts from nothing.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show more of each failure's .err")
    ap.add_argument("--archive", action="store_true",
                    help="move current logs to logs/archive/<utc>/ and stop")
    args = ap.parse_args()

    if args.archive:
        return archive()
    if not LOGS.exists():
        print(f"no {LOGS}. Run `mkdir -p chtc/logs` and submit something.")
        return 1

    rows = []
    for log in sorted(LOGS.glob("*.log")):
        job = log.stem
        info = parse_log(log)
        info["job"] = job
        rows.append(info)

    if not rows:
        print("no .log files yet.")
        return 0

    # Failures first, then running, then done -- the order you want to read them in.
    rank = {"FAILED": 0, "HELD": 1, "aborted": 2, "evicted": 3, "running": 4,
            "submitted": 5, "done": 6}
    rows.sort(key=lambda r: (rank.get(r["state"], 3), r["job"]))

    w = max(len(r["job"]) for r in rows)
    print(f"{'JOB':{w}s}  {'STATE':9s} {'EXIT':>4s}  {'RUNTIME':>7s}  {'LAST EVENT':19s}  CLUSTER")
    print("-" * (w + 56))
    n_bad = 0
    for r in rows:
        ex = "" if r["exit"] is None else str(r["exit"])
        print(f"{r['job']:{w}s}  {r['state']:9s} {ex:>4s}  {r.get('runtime',''):>7s}  "
              f"{r['when']:19s}  {r['cluster']}")
        bad = r["state"] in ("FAILED", "HELD", "aborted")
        if bad:
            n_bad += 1
            if r["detail"]:
                print(f"{'':{w}s}    ! {r['detail'][:150]}")
            for line in first_error(r["job"], 6 if args.verbose else 1):
                print(f"{'':{w}s}    > {line[:150]}")

    print()
    if n_bad:
        print(f"{n_bad} job(s) need attention. For the full error:  cat chtc/logs/<job>.err")
        print("Remember: a failed job already uploaded its archive, so before resubmitting")
        print("          ./fetch_results.sh --clean <job>")
    else:
        print("Nothing failed.")
    print("Once a batch is finished and collected:  python3 chtc/status.py --archive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

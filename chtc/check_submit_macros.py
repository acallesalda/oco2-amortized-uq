#!/usr/bin/env python3
"""Predict condor_submit's "was unused by condor_submit. Is it a typo?" warnings.

    python3 chtc/check_submit_macros.py

condor_submit warns about every macro a submit file defines and never references, and an
`include`d file counts as part of the submit file. Sharing site.conf across eight jobs
therefore trades one edit point for a column of warnings on every job that does not use
all of it -- and a submit that always warns is a submit whose warnings nobody reads, which
is how a real typo gets through.

This reproduces the check offline so the layout can be kept honest without a round trip to
the access point. It is not condor_submit and does not know every subtlety of the submit
language; it strips comments, expands macro definitions transitively, and reports what is
left. Treat a clean run as "no obvious noise", not as proof.

Exit status is 1 if any file has an unexpected unused macro, so it can go in a pre-commit
hook if that ever seems worth it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Warnings we have decided to live with, and why. Anything not listed here is a finding.
# Deliberately empty. It briefly held two entries -- reprocess.sub and audit.sub, which
# size their own disk and so could not use $(DISK_GB) -- on the reasoning that a warning
# was cheaper than over-requesting 40 GB for a job needing 8. That was the wrong call:
# every submit then printed a warning, which is precisely the state this script exists to
# prevent, and the choice was between inflating a request and tolerating noise only
# because DISK_GB was in the wrong file. Moving it to job_resources.conf, which those two
# jobs do not include, removed the dilemma.
#
# If you find yourself wanting to add an entry here, that is usually the same signal: a
# macro is living somewhere more general than its users. Move it before excepting it.
EXPECTED: set = set()

MACRO_RE = re.compile(r"\$\((\w+)\)")
DEF_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.*)$")
INCLUDE_RE = re.compile(r"^\s*include\s*:\s*(\S+)")

# Submit commands, not macros: condor_submit consumes these itself and never warns.
COMMANDS = {
    "universe", "executable", "arguments", "log", "output", "error", "queue",
    "requirements", "environment", "getenv", "input", "notification", "priority",
    "rank", "initialdir", "transfer_input_files", "transfer_output_files",
    "transfer_output_remaps", "should_transfer_files", "when_to_transfer_output",
    "request_cpus", "request_memory", "request_disk", "request_gpus",
    "container_image", "container_target_dir", "max_retries", "batch_name",
    "gpus_minimum_capability", "gpus_maximum_capability", "gpus_minimum_memory",
    # The expression form of the three above, and the only way to ask for a particular
    # card by name -- which benchmark.sub does, because there the card is the measurement.
    # Without it here, `require_gpus = (DeviceName == "$(device)")` reads as a macro
    # definition, so the macro it references looks unused and both get flagged.
    "require_gpus",
    "concurrency_limits", "periodic_release", "periodic_remove", "on_exit_hold",
    "stream_output", "stream_error", "want_graceful_removal",
    # Late materialization. If your condor_submit is old enough not to know these it will
    # warn that they are unused -- harmless, they only throttle how fast a large `queue`
    # materializes.
    "max_idle", "max_materialize",
    # Job control and notification.
    "on_exit_remove", "remove_kill_sig", "kill_sig", "hold_kill_sig",
    "job_lease_duration", "notify_user", "nice_user", "email_attributes",
    "copy_to_spool", "submit_event_notes", "want_remote_io", "leave_in_queue",
}


def strip_comments(text: str) -> str:
    # condor_submit treats a leading # as a comment. It does not strip trailing comments
    # on a definition line, so neither do we.
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))


def parse(path: Path, seen=None):
    """-> (defined {name: value}, referenced {name})"""
    seen = seen if seen is not None else set()
    if path in seen:
        return {}, set()
    seen.add(path)

    defined, referenced = {}, set()
    for line in strip_comments(path.read_text(encoding="utf-8")).splitlines():
        inc = INCLUDE_RE.match(line)
        if inc:
            d, r = parse(path.parent / inc.group(1), seen)
            defined.update(d)
            referenced |= r
            continue
        m = DEF_RE.match(line)
        if m and m.group(1).lower() not in COMMANDS and not m.group(1).startswith("+"):
            defined[m.group(1)] = m.group(2)
        else:
            referenced |= set(MACRO_RE.findall(line))
    return defined, referenced


def resolve(defined: dict, referenced: set) -> set:
    """Referencing $(SIF) also uses everything SIF's own definition mentions."""
    out, queue = set(referenced), list(referenced)
    while queue:
        name = queue.pop()
        for dep in MACRO_RE.findall(defined.get(name, "")):
            if dep not in out:
                out.add(dep)
                queue.append(dep)
    return out


def check_site_conf() -> list:
    """Catch an unconfigured site.conf here rather than in the job log.

    An unedited NetID is not a submit-time error: condor_submit expands `$(NETID)` to
    CHANGEME quite happily, the job is accepted, sits in the queue, matches a slot, and
    then fails file transfer with

        Permission denied ... URL file = osdf:///chtc/staging/c/CHANGEME/oco2/...

    which reads like a credentials problem and is really a one-line edit. Sixty seconds
    and a held job to learn that, three separate times, is what this function is for.
    """
    problems = []
    path = HERE / "site.conf"
    if not path.exists():
        return [f"{path} does not exist. Copy site.conf.example to it and set your NetID."]

    values = {}
    for line in strip_comments(path.read_text(encoding="utf-8")).splitlines():
        m = DEF_RE.match(line)
        if m:
            values[m.group(1)] = m.group(2).strip()

    netid = values.get("NETID", "")
    letter = values.get("NETID_LETTER", "")
    if not netid or netid == "CHANGEME":
        problems.append(
            "site.conf still says NETID = CHANGEME. Every OSDF path derives from it, so "
            "jobs will be accepted and then fail file transfer with 'Permission denied'."
        )
    elif letter and netid[0].lower() != letter.lower():
        problems.append(
            f"site.conf has NETID = {netid} but NETID_LETTER = {letter}. /staging is "
            f"sharded by first letter, so this should be '{netid[0].lower()}' -- "
            "otherwise every path points at a directory you cannot read."
        )
    return problems


VARS_RE = re.compile(r"^\s*VARS\s+(\S+)\s+(.*)$", re.IGNORECASE)
JOB_RE = re.compile(r"^\s*JOB\s+(\S+)\s+(\S+)", re.IGNORECASE)
VARS_KV_RE = re.compile(r"([A-Za-z_]\w*)\s*=\s*\"")


def check_dag_vars_are_not_shadowed() -> list:
    """A macro a submit file DEFINES beats the VARS of the same name a .dag passes it.

    This cost three jobs and four hours on 2026-09-07. `eval_ext.sub` declared

        tag       = ext
        partition = test

    as defaults, and `ext.dag` passed `tag="resim" partition="resim_test"` through VARS for
    the three tail-scene nodes, expecting the VARS to win. They do not. The jobs scored the
    `test` partition -- not the one asked for -- and each tried to upload
    `eval_<task>_ext.tar.gz` over the archive its own twin had already written. OSDF refused
    and held them:

        Pelican Client Error: remote object already exists, upload aborted
        ( URL file = osdf:///chtc/staging/.../results/eval_xco2_ext.tar.gz )

    Nothing was overwritten, so nothing was lost but the time. What makes it worth a
    preflight check rather than a comment is the failure mode when the names DON'T collide:
    the job runs, uploads a plausible archive, and scores the wrong partition silently.

    The same .dag files prove the rule: `task`, `det_study`, `flow_study`, `month` and
    `study` are passed by VARS, are NOT defined in any submit file, and have always worked.
    So the check is one-sided and cheap -- for every JOB node, take the names its VARS set
    and complain if the submit file it names defines any of them.
    """
    problems = []
    for dag in sorted(HERE.glob("*.dag")) + sorted(HERE.glob("*/*.dag")):
        text = strip_comments(dag.read_text(encoding="utf-8"))
        node_sub, node_vars = {}, {}
        for line in text.splitlines():
            m = JOB_RE.match(line)
            if m:
                node_sub[m.group(1)] = m.group(2)
                continue
            m = VARS_RE.match(line)
            if m:
                node_vars.setdefault(m.group(1), set()).update(
                    VARS_KV_RE.findall(m.group(2)))
        for node, names in sorted(node_vars.items()):
            sub = node_sub.get(node)
            if not sub:
                continue
            sub_path = (dag.parent / sub)
            if not sub_path.exists():
                problems.append(f"{dag.name}: node {node} names {sub}, which does not exist")
                continue
            defined, _ = parse(sub_path)
            clash = sorted(n for n in names if n in defined)
            if clash:
                problems.append(
                    f"{dag.name}: node {node} passes VARS {clash} to {sub}, which DEFINES "
                    f"{clash}. The submit file's value wins and the VARS are ignored -- the "
                    f"job runs with the wrong value and, if the filenames it builds happen "
                    f"not to collide, says nothing. Hardcode the constant in a submit file "
                    f"of its own (as eval_resim.sub and eval_month_full.sub do), or remove "
                    f"the definition and let every node supply it.")
    return problems


SPLIT_ENV_RE = re.compile(r"OCO2_SPLIT=([A-Za-z0-9_]+)")
SPLIT_ARCHIVE_RE = re.compile(r"\boco2_((?:ext_cap|ext|full_feb|replicate_split|size_\d+)_v1)\.tar\.gz")


def check_split_env_matches_archive() -> list:
    """A job that WRITES `oco2_<split>.tar.gz` must set `OCO2_SPLIT=<split>`.

    This is the 2026-09-08 hold. `prep_ext_cap.sub` was generated from `prep_ext.sub` by
    substituting archive names, but `prep_ext.sub` had no `environment` line at all -- the
    script it runs used to hardcode the split name. So there was no
    `OCO2_SPLIT=...` string to substitute, the capped job ran with the variable unset,
    fell back to the uncapped split, and built the CAPPED table under the UNCAPPED
    archive's name.

    It was held rather than accepted, and only because OSDF refuses to overwrite an object
    that the first run had already put at that path. Had the name been free the job would
    have completed, written a plausible archive, and every number downstream would have
    been attributed to the wrong arm of the experiment. That is the failure this checks
    for, and the reason it checks the *output* name rather than the input list: the output
    name is what a later reader uses to decide what an archive contains.

    Only files that both declare an OCO2_SPLIT and name a split archive are checked, so a
    job that reads one split's archive while writing something else entirely -- the month
    preps, which write `oco2_month_<month>_<suffix>.tar.gz` -- is not caught up in it.
    """
    problems = []
    for sub in sorted(HERE.glob("*.sub")) + sorted(HERE.glob("*/*.sub")):
        text = strip_comments(sub.read_text(encoding="utf-8"))
        out_lines = [ln for ln in text.splitlines()
                     if ln.strip().lower().startswith(("transfer_output_files",
                                                       "transfer_output_remaps"))]
        written = {m.group(1) for ln in out_lines for m in SPLIT_ARCHIVE_RE.finditer(ln)}
        if not written:
            continue
        declared = {m.group(1) for ln in text.splitlines()
                    if ln.strip().lower().startswith("environment")
                    for m in SPLIT_ENV_RE.finditer(ln)}
        if not declared:
            problems.append(
                f"{sub.name}: writes {sorted(written)} but sets no OCO2_SPLIT in its "
                f"`environment`. The script then falls back to its own default and can "
                f"build one split's table under another's archive name -- which is what "
                f"held prep_ext_cap on 2026-09-08. Add "
                f'environment = "OCO2_SPLIT={sorted(written)[0]}".')
            continue
        wrong = written - declared
        if wrong:
            problems.append(
                f"{sub.name}: writes {sorted(wrong)} but declares "
                f"OCO2_SPLIT={sorted(declared)}. The archive name and the table inside it "
                f"would disagree, and nothing downstream can tell.")
    return problems


QUEUE_FROM = re.compile(r"^\s*queue\b.*\bfrom\s+(\S+)", re.IGNORECASE)


def check_queue_files() -> list:
    """`queue ... from <file>` makes one job per line -- including comments.

    It does not skip `#` lines or blank lines. A three-line header on a 16-trial grid
    silently became 19 jobs, three of them parsing as base="# det_last4: 16 trials" with
    output filenames containing spaces and a '#'. Nothing warns; you find out from the job
    count in condor_submit's output, if you happen to be counting.
    """
    problems = []
    for sub in submit_files():
        for line in strip_comments(sub.read_text(encoding="utf-8")).splitlines():
            m = QUEUE_FROM.match(line)
            if not m:
                continue
            # The path may be built from a macro, e.g. sweeps/$(grid).txt -- glob it.
            pattern = re.sub(r"\$\(\w+\)", "*", m.group(1))
            matches = sorted((HERE / pattern).parent.glob(Path(pattern).name))
            if not matches:
                problems.append(f"{sub.name}: `queue ... from {m.group(1)}` matches no file")
                continue
            for f in matches:
                lines = f.read_text(encoding="utf-8").splitlines()
                bad = [i for i, l in enumerate(lines, 1)
                       if not l.strip() or l.lstrip().startswith("#")]
                if bad:
                    problems.append(
                        f"{sub.name}: {f.relative_to(HERE)} has {len(bad)} comment/blank "
                        f"line(s) at {bad[:5]} -- condor_submit will queue each as a job "
                        f"({len(lines)} jobs, not {len(lines) - len(bad)})")
    return problems


# condor_submit_dag writes <name>.dag.condor.sub beside the DAG. It is generated, we do
# not maintain it, and its contents are HTCondor's business -- linting it produces findings
# nobody can act on.
GENERATED = ("*.dag.condor.sub",)


def is_ours(path: Path) -> bool:
    return not any(path.match(pat) for pat in GENERATED)


def submit_files() -> list:
    """Every submit file we maintain, including the generated BO ones.

    chtc/bo/*.sub are written by make_bo_dag.py and are as capable of an unused macro as a
    hand-written one -- more so, since a template mistake is repeated fifteen times. They
    include ../site.conf, which `parse` resolves relative to the submit file, so nothing
    else has to change.
    """
    # bo*/ rather than bo/: make_bo_dag.py writes one directory per split, and a
    # generated submit file nobody checks is exactly the one that warns.
    return sorted(f for f in list(HERE.glob("*.sub")) + list(HERE.glob("bo*/*.sub"))
                  if is_ours(f))


def main() -> int:
    site_problems = ([("site.conf", p) for p in check_site_conf()]
                     + [("queue files", p) for p in check_queue_files()]
                     + [("dag VARS", p) for p in check_dag_vars_are_not_shadowed()]
                     + [("split naming", p) for p in check_split_env_matches_archive()])
    for where, p in site_problems:
        print(f"  {where:24s} ERROR: {p}")
    if site_problems:
        print()

    findings = []
    for sub in submit_files():
        defined, referenced = parse(sub)
        unused = sorted(set(defined) - resolve(defined, referenced))
        surprises = [u for u in unused if (sub.name, u) not in EXPECTED]
        expected = [u for u in unused if (sub.name, u) in EXPECTED]

        status = "clean"
        if expected:
            status = f"expected: {', '.join(expected)}"
        if surprises:
            status = f"WOULD WARN: {', '.join(surprises)}"
            findings.append((sub.name, surprises))
        label = str(sub.relative_to(HERE))
        print(f"  {label:34s} {status}")

    print()
    if findings:
        print("These macros are defined (directly or via an include) and never used.")
        print("Either reference them, move them to a config only the users include, or")
        print("add them to EXPECTED here with a reason.")
    if site_problems:
        print("Preflight failed. Fix these before submitting: none of them make")
        print("condor_submit complain, and all of them cost you a queue wait.")
    if findings or site_problems:
        return 1
    print("No unexpected unused macros. site.conf looks configured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

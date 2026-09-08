#!/usr/bin/env python3
"""Pack the archives that go into the Zenodo data record, and say what is missing.

    python3 scripts/make_zenodo_bundles.py --list           # what each bundle needs, and
                                                            # whether this machine has it
    python3 scripts/make_zenodo_bundles.py splits artifacts
    python3 scripts/make_zenodo_bundles.py all --out dist/zenodo

Writes one `.tar.gz` per bundle into `--out`, plus a `SHA256SUMS.txt` covering them and a
`CONTENTS.md` listing what went into each. Upload all of it, including those two files:
`scripts/get_data.sh` reads the record's own `SHA256SUMS.txt` to verify what it downloads,
so a record without it cannot be verified by the tool that fetches it.

Nothing is invented. A bundle whose inputs are not on this machine is reported as missing,
with the path it wanted, and no partial archive is written -- a half-populated weights
bundle uploaded to a DOI is not fixable afterwards.

The bundle names match the ones `scripts/get_data.sh` looks for. Changing one means
changing both, which is why they are listed in one place in each file.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# name -> (members, description)
#
# `members` are (source path relative to ROOT, path inside the archive). A source that is
# a directory is added whole. The archive-internal paths are chosen so that
# `get_data.sh`'s unpack rules land them where the code defaults to; do not "tidy" them.
BUNDLES: dict[str, tuple[list[tuple[str, str]], str]] = {
    "splits": (
        [
            # The primary table lives at the repository root in a working checkout,
            # because that is where the pipeline writes it and where rebuild.sh reads it;
            # inside the archive it goes with the others.
            ("group_split_v1_cov.csv",
             "scratch/simulated/files/splits/group_split_v1_cov.csv"),
            ("scratch/simulated/files/splits", "scratch/simulated/files/splits"),
            # One entry per split rather than the whole stats/ directory, so a missing
            # one is named. `stats/` also holds the ORIGINAL replicate-level profile
            # statistics, which are already published as oco2_profile_stats.tar.gz and
            # must not be duplicated into a second archive under a different name.
            ("experiments/pipelines/stats/group_split_v1",
             "experiments/pipelines/stats/group_split_v1"),
            ("experiments/pipelines/stats/replicate_split_v1",
             "experiments/pipelines/stats/replicate_split_v1"),
            ("experiments/pipelines/stats/full_feb_v1",
             "experiments/pipelines/stats/full_feb_v1"),
        ],
        "every split table with its own standardization constants and covariate encodings",
    ),
    "weights": (
        [
            ("experiments/weights/group_split_v1", "experiments/weights/group_split_v1"),
            ("experiments/weights/replicate_split_v1", "experiments/weights/replicate_split_v1"),
            ("experiments/weights/full_feb_v1", "experiments/weights/full_feb_v1"),
        ],
        "the trained checkpoints of every model reported in the paper",
    ),
    "artifacts": (
        [
            ("experiments/scoring/xco2_bo.npz", "xco2_bo.npz"),
            ("experiments/scoring/xco2_bo.meta.json", "xco2_bo.meta.json"),
            ("experiments/scoring/profiles_bo.npz", "profiles_bo.npz"),
            ("experiments/scoring/profiles_bo.meta.json", "profiles_bo.meta.json"),
            ("experiments/scoring/last4_bo.npz", "last4_bo.npz"),
            ("experiments/scoring/last4_bo.meta.json", "last4_bo.meta.json"),
            ("experiments/scoring/xco2.npz", "xco2.npz"),
            ("experiments/scoring/xco2.meta.json", "xco2.meta.json"),
            ("experiments/scoring/profiles.npz", "profiles.npz"),
            ("experiments/scoring/profiles.meta.json", "profiles.meta.json"),
            ("experiments/scoring/last4.npz", "last4.npz"),
            ("experiments/scoring/last4.meta.json", "last4.meta.json"),
            ("experiments/scoring/xco2_gauss.npz", "xco2_gauss.npz"),
            ("experiments/scoring/xco2_gauss.meta.json", "xco2_gauss.meta.json"),
            ("experiments/scoring/profiles_gauss.npz", "profiles_gauss.npz"),
            ("experiments/scoring/profiles_gauss.meta.json", "profiles_gauss.meta.json"),
            ("experiments/scoring/last4_gauss.npz", "last4_gauss.npz"),
            ("experiments/scoring/last4_gauss.meta.json", "last4_gauss.meta.json"),
            ("experiments/scoring/results/group_split_v1", "results/group_split_v1"),
            ("experiments/scoring/results/replicate_split_v1", "results/replicate_split_v1"),
            ("experiments/scoring/results/month_202003_full", "results/month_202003_full"),
            ("experiments/scoring/results/month_202007_full", "results/month_202007_full"),
            ("experiments/scoring/results/size_90_v1", "results/size_90_v1"),
            ("experiments/scoring/results/size_180_v1", "results/size_180_v1"),
            ("experiments/scoring/results/size_270_v1", "results/size_270_v1"),
        ],
        "per-sounding predictive artifacts: every table and figure in the paper, with no GPU",
    ),
}

FILENAME = {
    "splits": "oco2_splits.tar.gz",
    "weights": "oco2_model_weights_v2.tar.gz",
    "artifacts": "oco2_scoring_artifacts.tar.gz",
}

# Directories that must never travel inside a bundle, whatever a member glob picks up.
EXCLUDE_NAMES = {"__pycache__", ".ipynb_checkpoints", ".DS_Store"}
# The February extension is not in the paper; its artifacts must not ship with the record
# that the paper cites. Drop these two lines when it is.
EXCLUDE_PREFIXES = ("results/ext_v1", "results/ext_cap_v1",
                    "results/month_202003_ext", "results/month_202007_ext")


def human(n: int) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def walk(src: Path):
    if src.is_file():
        yield src
        return
    for f in sorted(src.rglob("*")):
        if f.is_file() and not any(p in EXCLUDE_NAMES for p in f.parts):
            yield f


def bundle_inputs(name: str):
    members, _desc = BUNDLES[name]
    present, missing, total = [], [], 0
    for rel, arc in members:
        src = ROOT / rel
        if not src.exists():
            missing.append(rel)
            continue
        for f in walk(src):
            inner = arc if src.is_file() else f"{arc}/{f.relative_to(src).as_posix()}"
            if inner.startswith(EXCLUDE_PREFIXES):
                continue
            present.append((f, inner))
            total += f.stat().st_size
    return present, missing, total


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundles", nargs="*", help=f"any of {', '.join(BUNDLES)}, or 'all'")
    ap.add_argument("--out", default=str(ROOT / "dist" / "zenodo"))
    ap.add_argument("--list", action="store_true",
                    help="report what each bundle would contain and stop")
    args = ap.parse_args(argv)

    names = list(BUNDLES) if (args.list or args.bundles == ["all"]) else args.bundles
    if not names:
        ap.error(f"name at least one bundle ({', '.join(BUNDLES)}) or 'all', or pass --list")
    for n in names:
        if n not in BUNDLES:
            ap.error(f"unknown bundle {n!r}; known: {', '.join(BUNDLES)}")

    if args.list:
        for name in names:
            present, missing, total = bundle_inputs(name)
            print(f"--- {name}  ->  {FILENAME[name]}")
            print(f"    {BUNDLES[name][1]}")
            print(f"    {len(present)} files, {human(total)} uncompressed")
            for m in missing:
                print(f"    MISSING  {m}")
            if missing:
                print("    -> fetch these onto this machine before packing "
                      "(chtc/fetch_results.sh, or scripts/get_data.sh)")
            print()
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name in names:
        present, missing, total = bundle_inputs(name)
        if missing:
            print(f"SKIPPING {name}: {len(missing)} input(s) absent")
            for m in missing:
                print(f"    {m}")
            print("    A partial archive uploaded to a DOI cannot be fixed afterwards, so")
            print("    nothing is written. Fetch the inputs and re-run.")
            continue
        if not present:
            print(f"SKIPPING {name}: no files matched")
            continue
        dest = out / FILENAME[name]
        print(f"--- {name}: {len(present)} files, {human(total)} -> {dest.name}")
        with tarfile.open(dest, "w:gz") as tar:
            for src, inner in present:
                tar.add(src, arcname=inner)
        print(f"    {human(dest.stat().st_size)} compressed")
        written.append(dest)

    if not written:
        print("\nnothing written")
        return 1

    sums = out / "SHA256SUMS.txt"
    existing = sums.read_text(encoding="utf-8") if sums.is_file() else ""
    keep = [l for l in existing.splitlines()
            if l.strip() and l.split()[-1] not in {d.name for d in written}]
    rows = keep + [f"{sha256(d)}  {d.name}" for d in written]
    sums.write_text("\n".join(sorted(rows, key=lambda r: r.split()[-1])) + "\n",
                    encoding="utf-8")
    print(f"\n{sums} updated ({len(rows)} entries)")

    contents = out / "CONTENTS.md"
    lines = ["# What is in each archive", ""]
    for name in BUNDLES:
        present, missing, total = bundle_inputs(name)
        lines += [f"## `{FILENAME[name]}`", "", BUNDLES[name][1], "",
                  f"{len(present)} files, {human(total)} uncompressed.", ""]
        if missing:
            lines += ["Not packed on this machine: " +
                      ", ".join(f"`{m}`" for m in missing), ""]
    contents.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{contents} written")
    print("\nUpload every file in this directory, SHA256SUMS.txt included.")
    print("release/ZENODO.md has the rest of the checklist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

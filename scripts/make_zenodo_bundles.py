#!/usr/bin/env python3
"""Pack the archives that go into the Zenodo data record, and say what is missing.

    python3 scripts/make_zenodo_bundles.py --list           # what each bundle needs, and
                                                            # whether this machine has it
    python3 scripts/make_zenodo_bundles.py splits artifacts
    python3 scripts/make_zenodo_bundles.py all --out dist/zenodo
    python3 scripts/make_zenodo_bundles.py --finalize       # on the machine that UPLOADS,
                                                            # once every archive is in --out

Writes one `.tar.gz` per bundle into `--out`, and a `SHA256SUMS.txt` covering them. Upload
all of it, including that file: `scripts/get_data.sh` reads the record's own
`SHA256SUMS.txt` to verify what it downloads, so a record without it cannot be verified by
the tool that fetches it.

`--finalize` is the step that makes the record's two metadata files, and it is separate
from packing because the archives are built in two places -- the artifacts on a laptop, the
splits and the weights on the cluster where the checkpoints are -- and only the machine
that uploads ever sees all of them at once. It hashes every archive in `--out`, CARRIES
FORWARD the checksums of the archives a new version imports unchanged, and writes
`SHA256SUMS.txt` and `CONTENTS.md` describing the whole record.

The carry-forward is not a nicety. A new Zenodo version imports the previous version's
files without re-uploading them, but the new `SHA256SUMS.txt` replaces the old one wholesale
-- so a checksum file listing only the newly packed archives silently un-publishes every
imported one as far as `get_data.sh` is concerned: it resolves an archive's real filename
through that file, and a base name with no entry there reads as "not in the record".

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

# The concept record: the parent of every version, which is what to ask for the newest
# version's files. Hardcoding a version id here would mean carrying forward the checksums
# of whichever version happened to be current when this line was written.
ZENODO_CONCEPT = 20617717

# Every archive the record holds, in the order CONTENTS.md lists them, with where each one
# comes from. The record is assembled from three places and a reader checking one of them
# needs to know which:
#
#   bundle   packed by this script from a checkout
#   staged   uploaded as the very tarball the GPU jobs read, not repacked -- so the bytes
#            scored in the paper and the bytes in the record are the same bytes
#   v1       already in the record; a new version imports it unchanged
#
# The descriptions are deliberately the same sentences as `scripts/get_data.sh`'s ITEMS
# table, which is the other half of this contract: that script looks archives up by these
# filenames, so a name changed in one place has to change in both.
RECORD: list[tuple[str, str, str]] = [
    ("oco2_processed_train.tar.gz", "v1",
     "February processed training spectra + aux_data.csv"),
    ("oco2_processed_test.tar.gz", "v1",
     "February processed test spectra, aux_data.csv, ACOS profile covariances"),
    ("oco2_profile_stats.tar.gz", "v1",
     "profile_mu.npy / profile_sigma.npy for the original replicate-level split"),
    ("oco2_unprocessed_og_data.tar.gz", "v1",
     "the raw .h5 soundings, needed only to redo the preprocessing"),
    ("oco2_model_weights.tar.gz", "v1",
     "the ORIGINAL submission's checkpoints, kept for provenance"),
    ("MANIFEST.tsv", "v1",
     "the per-file inventory of the raw delivery"),
    ("oco2_splits.tar.gz", "bundle", BUNDLES["splits"][1]),
    ("oco2_model_weights_v2.tar.gz", "bundle", BUNDLES["weights"][1]),
    ("oco2_scoring_artifacts.tar.gz", "bundle", BUNDLES["artifacts"][1]),
    ("oco2_month_202003_full.tar.gz", "staged",
     "March 2020: processed spectra, ACOS profile covariances, covariates and the month "
     "split table"),
    ("oco2_month_202007_full.tar.gz", "staged",
     "July 2020: the same, for the second held-out month"),
]
SOURCE_NOTE = {
    "bundle": "packed from a checkout by `scripts/make_zenodo_bundles.py`",
    "staged": "uploaded as the tarball the scoring jobs read, unrepacked",
    "v1": "imported unchanged from version 1 of this record",
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


def read_sums(text: str) -> dict[str, str]:
    """`SHA256SUMS.txt` -> {filename: digest}. Tolerates the `*name` binary marker."""
    out = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            out[parts[-1].lstrip("*")] = parts[0]
    return out


def previous_sums(concept: int) -> dict[str, str]:
    """The published `SHA256SUMS.txt` of the newest existing version of the record.

    Resolved through the concept record so that this does not have to know which version
    is current. Raises on any failure -- the caller must not quietly write a checksum file
    that is missing the archives a new version imports.
    """
    import json
    import urllib.request

    def get(url: str) -> bytes:
        with urllib.request.urlopen(url, timeout=60) as r:   # noqa: S310 (fixed host)
            return r.read()

    latest = json.loads(get(f"https://zenodo.org/api/records/{concept}/versions/latest"))
    rec = latest.get("id")
    text = get(f"https://zenodo.org/api/records/{rec}/files/SHA256SUMS.txt/content")
    sums = read_sums(text.decode("utf-8"))
    if not sums:
        raise RuntimeError(f"record {rec} has a SHA256SUMS.txt, but it parsed to nothing")
    print(f"    carried forward {len(sums)} checksum(s) from record {rec}")
    return sums


def finalize(out: Path, carry: Path | None, no_carry: bool, concept: int) -> int:
    """Write the record's `SHA256SUMS.txt` and `CONTENTS.md` from what is in `out`."""
    if not out.is_dir():
        print(f"{out} does not exist; nothing to finalize")
        return 1

    here = {f.name: f for f in sorted(out.iterdir())
            if f.is_file() and f.name not in {"SHA256SUMS.txt", "CONTENTS.md"}}
    if not here:
        print(f"{out} holds no archives; nothing to finalize")
        return 1

    print(f"--- hashing {len(here)} file(s) in {out}")
    digests = {}
    for name, f in here.items():
        digests[name] = sha256(f)
        print(f"    {name:38s} {human(f.stat().st_size):>10s}  {digests[name][:16]}...")

    # The imported archives' checksums, which this machine cannot compute because it does
    # not have the files -- and must not omit, because the new checksum file REPLACES the
    # old one in the record rather than adding to it.
    carried: dict[str, str] = {}
    if carry is not None:
        carried = read_sums(carry.read_text(encoding="utf-8"))
        print(f"--- carried forward {len(carried)} checksum(s) from {carry}")
    elif not no_carry:
        print(f"--- fetching the previous version's checksums (concept {concept})")
        try:
            carried = previous_sums(concept)
        except Exception as exc:                                  # noqa: BLE001
            print(f"    FAILED: {exc}")
            print("\nNothing was written. The record's SHA256SUMS.txt has to cover the")
            print("archives a new version imports unchanged as well as the new ones, or")
            print("`scripts/get_data.sh` will report every imported archive as absent.")
            print("Pass --carry-sums <file> with the previous version's SHA256SUMS.txt")
            print("(downloadable from the record page), or --no-carry if this record")
            print("genuinely has no earlier version.")
            return 1

    # A locally hashed file always wins: it is the copy being uploaded.
    rows = dict(carried)
    rows.update(digests)

    sums = out / "SHA256SUMS.txt"
    sums.write_text("".join(f"{rows[n]}  {n}\n" for n in sorted(rows)), encoding="utf-8")
    print(f"\n{sums} written ({len(rows)} entries: "
          f"{len(digests)} hashed here, {len(rows) - len(digests)} carried forward)")

    known = {n for n, _src, _d in RECORD}
    lines = [
        "# What is in this record",
        "",
        "Everything here unpacks into a checkout of the code repository at the paths its",
        "own scripts default to. `scripts/get_data.sh` does that and verifies every",
        "download against the `SHA256SUMS.txt` published beside these files:",
        "",
        "```bash",
        "./scripts/get_data.sh --list        # what this record holds, and what you have",
        "./scripts/get_data.sh artifacts     # every table and figure, no GPU, no training",
        "```",
        "",
        "The code that reads all of it, and the documentation for each archive, is at",
        "<https://github.com/acallesalda/oco2-amortized-uq>.",
        "",
    ]
    for name, src, desc in RECORD:
        f = out / name
        lines += [f"## `{name}`", "", desc + ".", ""]
        if name in digests:
            lines += [f"{human(f.stat().st_size)}. SHA-256 `{digests[name]}`.", ""]
        elif name in rows:
            lines += [f"{SOURCE_NOTE[src].capitalize()}. "
                      f"SHA-256 `{rows[name]}`.", ""]
        else:
            lines += ["**Not in this record.** Listed here because the code expects it; "
                      "see the description above for what depends on it.", ""]
    extra = sorted(set(rows) - known)
    if extra:
        lines += ["## Also in this record", ""]
        lines += [f"* `{n}`" for n in extra] + [""]
    contents = out / "CONTENTS.md"
    contents.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{contents} written")

    absent = [n for n, _s, _d in RECORD if n not in rows]
    if absent:
        print("\nThe record will NOT hold these, and CONTENTS.md says so:")
        for n in absent:
            print(f"    {n}")
        print("  Every one of them is something `scripts/get_data.sh` offers to fetch, so")
        print("  a reader who asks for it is told it is absent rather than getting a 404.")
    print("\nUpload every file in this directory. release/ZENODO.md has the checklist.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundles", nargs="*", help=f"any of {', '.join(BUNDLES)}, or 'all'")
    ap.add_argument("--out", default=str(ROOT / "dist" / "zenodo"))
    ap.add_argument("--list", action="store_true",
                    help="report what each bundle would contain and stop")
    ap.add_argument("--finalize", action="store_true",
                    help="write the record's SHA256SUMS.txt and CONTENTS.md from whatever "
                         "archives are in --out; run this on the machine that uploads")
    ap.add_argument("--carry-sums", type=Path, metavar="FILE",
                    help="the previous version's SHA256SUMS.txt, if it cannot be fetched")
    ap.add_argument("--no-carry", action="store_true",
                    help="this record has no earlier version whose files it imports")
    ap.add_argument("--concept", type=int, default=ZENODO_CONCEPT,
                    help=f"Zenodo concept record id (default {ZENODO_CONCEPT})")
    args = ap.parse_args(argv)

    if args.finalize:
        if args.bundles:
            ap.error("--finalize takes no bundle names: it describes what is already packed")
        return finalize(Path(args.out), args.carry_sums, args.no_carry, args.concept)

    names = list(BUNDLES) if (args.list or args.bundles == ["all"]) else args.bundles
    if not names:
        ap.error(f"name at least one bundle ({', '.join(BUNDLES)}) or 'all', "
                 "or pass --list, or --finalize")
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

    # CONTENTS.md is NOT written here. It describes the record, and this machine packed
    # only part of it; the whole-record view exists on the machine that uploads, which is
    # where --finalize runs. Writing a partial one here is how the file that goes into a
    # DOI ends up saying "0 files, 0 B" about an archive that was packed elsewhere.
    print("\nUpload every file in this directory, SHA256SUMS.txt included -- but first,")
    print("on the machine that uploads, once every archive is in one directory:")
    print("\n    python3 scripts/make_zenodo_bundles.py --finalize\n")
    print("which merges the checksums and writes CONTENTS.md for the whole record.")
    print("release/ZENODO.md has the rest of the checklist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

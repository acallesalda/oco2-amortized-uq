"""artifacts.py

The on-disk contract between the two halves of evaluation.

Scoring used to be four notebooks that each did the whole job: fit a Laplace
approximation, sample the flow, compute the metrics, draw the figures. On a shared
cluster that is the wrong shape. The expensive half needs a GPU, the training data and a
batch slot; the cheap half is a few seconds of numpy and gets re-run twenty times while
the table caption is being argued about.

So the GPU half (`experiments/scoring/predict.py`) writes exactly one file per task --
every predictive quantity for every test sounding, and nothing else -- and the CPU half
(`score.py`, `figures.py`) reads it. Tables and figures can then be regenerated on a
laptop, offline, without a checkpoint or a queue in sight. The file is also the archival
record: it is what a reader would need to check any number in Tables 1-3 without
retraining anything.

Everything is stored in the *standardized* coordinates the models work in, together with
the constants needed to leave them (`y_mu`, `y_sigma`, in mole fraction). Converting on
the way out rather than on the way in means the stored numbers are directly comparable to
what the training scripts print, and the ppm conversion lives in one place
(`evals.metrics`) instead of being repeated per notebook -- which is how the manuscript
ended up reporting NLLs in three mutually incomparable units.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

SCHEMA_VERSION = 1

# IDs are written by oco2_utils.add_uniq_id as clst%04d_ref%04d_rep%04d.
SCENE_ID_RE = re.compile(r"clst(\d+)_ref(\d+)_rep(\d+)")


def scene_labels(art) -> Optional[np.ndarray]:
    """`"<cluster>/<reference>"` per sounding, parsed from the artifact's own IDs.

    The reference sounding -- the *scene* -- is the unit the group split is disjoint in,
    and it is the unit almost every question about generalization is asked in: how well
    was this scene retrieved, was it covered, is the bootstrap resampling independent
    things. It is not stored as a column, but it is recoverable from the ID, which is
    built from the filename and is therefore attached to the right sounding under any
    pairing.

    `None` when the artifact has no IDs or they are not in that format -- the caller then
    falls back to per-sounding treatment rather than crashing.

    The cluster is part of the key on purpose: reference numbers restart within each
    cluster, so `ref` alone would merge nine different scenes into one.
    """
    if not art.has("ids"):
        return None
    m = [SCENE_ID_RE.match(str(i)) for i in np.asarray(art["ids"]).ravel()]
    if any(x is None for x in m):
        return None
    # Zero padding dropped: these become axis labels and dictionary keys.
    return np.array([f"{int(x.group(1))}/{int(x.group(2))}" for x in m])

# Arrays every artifact must carry. Anything else is optional and may be absent, so a
# consumer checks with `has()` rather than assuming.
_REQUIRED = ("ids", "y_std")


def _git_commit(repo_root: Optional[Path] = None) -> str:
    """Best-effort commit hash, so a table can be traced to the code that made it."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    # On an execute node the checkout is a tarball, not a repository; chtc/GIT_PROVENANCE.txt
    # is written by make_code_tarball.sh precisely for this case.
    prov = root / "chtc" / "GIT_PROVENANCE.txt"
    if prov.exists():
        for line in prov.read_text().splitlines():
            if line.startswith("commit:"):
                return line.split(":", 1)[1].strip()
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


@dataclass
class ArtifactMeta:
    """Everything needed to interpret the arrays, and to reproduce them."""

    task: str                       # "xco2" | "profiles" | "last4"
    split_name: str                 # "group_split_v1" | "replicate" | ...
    n: int
    dim: int
    y_mu: list                      # [dim], mole fraction
    y_sigma: list                   # [dim], mole fraction
    levels: list = field(default_factory=list)      # 1-based pressure levels, [] for xco2
    ppm_scale: float = 1e6
    n_nf_samples: int = 0
    schema_version: int = SCHEMA_VERSION
    created_utc: str = ""
    git_commit: str = ""
    device: str = ""
    checkpoints: Dict[str, Any] = field(default_factory=dict)
    laplace: Dict[str, Any] = field(default_factory=dict)
    nf: Dict[str, Any] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_utc:
            self.created_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if not self.git_commit:
            self.git_commit = _git_commit()


class Artifact:
    """A meta block plus a dict of numpy arrays, saved as one .npz.

    Read-only in practice: `predict.py` builds one and saves it, everything else loads
    and queries. Missing optional arrays return None from `get`, which is deliberate --
    the ACOS covariance is absent for a nontrivial fraction of soundings and the scalar
    task has no covariance file at all.
    """

    def __init__(self, meta: ArtifactMeta, arrays: Dict[str, np.ndarray]):
        missing = [k for k in _REQUIRED if k not in arrays]
        if missing:
            raise KeyError(f"artifact is missing required arrays: {missing}")
        n = int(meta.n)
        for k, v in arrays.items():
            if v is None:
                continue
            if len(v) != n:
                raise ValueError(
                    f"array '{k}' has leading dimension {len(v)}, expected n={n}. "
                    "Every stored array is per-sounding and must align with `ids`."
                )
        self.meta = meta
        self.arrays = {k: v for k, v in arrays.items() if v is not None}

    # -- access -------------------------------------------------------------------
    def has(self, name: str) -> bool:
        return name in self.arrays

    def get(self, name: str, default=None):
        return self.arrays.get(name, default)

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self.arrays:
            raise KeyError(
                f"'{name}' is not in this artifact. Present: {sorted(self.arrays)}"
            )
        return self.arrays[name]

    @property
    def y_mu(self) -> np.ndarray:
        return np.asarray(self.meta.y_mu, dtype=np.float64)

    @property
    def y_sigma(self) -> np.ndarray:
        return np.asarray(self.meta.y_sigma, dtype=np.float64)

    # -- io -----------------------------------------------------------------------
    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.arrays)
        payload["__meta__"] = np.frombuffer(
            json.dumps(asdict(self.meta)).encode("utf-8"), dtype=np.uint8
        )
        # Uncompressed: these are mostly random floats, where compression buys a few
        # percent for minutes of CPU, and the file is written from a batch job whose
        # wall time we are paying for.
        np.savez(path, **payload)
        # A sidecar copy of the metadata, so `cat` answers "what is this file" without
        # loading 200 MB of samples.
        path.with_suffix(".meta.json").write_text(
            json.dumps(asdict(self.meta), indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: Path | str, lazy: bool = False) -> "Artifact":
        """Load an artifact. `lazy=True` keeps the NpzFile open and reads on demand,
        which matters for the flow samples -- they are most of the file and the table
        code does not touch them."""
        path = Path(path)
        npz = np.load(path, allow_pickle=False, mmap_mode=None)
        meta = ArtifactMeta(**json.loads(bytes(npz["__meta__"]).decode("utf-8")))
        if lazy:
            arrays = _LazyArrays(npz, meta.n)
            obj = cls.__new__(cls)
            obj.meta = meta
            obj.arrays = arrays
            return obj
        arrays = {k: npz[k] for k in npz.files if k != "__meta__"}
        return cls(meta, arrays)

    def summary(self) -> str:
        lines = [
            f"task={self.meta.task}  split={self.meta.split_name}  "
            f"n={self.meta.n}  dim={self.meta.dim}",
            f"created={self.meta.created_utc}  commit={self.meta.git_commit[:12]}",
            f"arrays: {', '.join(sorted(self.arrays))}",
        ]
        return "\n".join(lines)


class _LazyArrays(dict):
    """Dict view over an open NpzFile; decompresses an array only when asked for it."""

    def __init__(self, npz, n: int):
        super().__init__()
        self._npz = npz
        self._n = n

    def __contains__(self, key) -> bool:
        return key in self._npz.files and key != "__meta__"

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(key)
        return self._npz[key]

    def get(self, key, default=None):
        return self[key] if key in self else default

    def keys(self):
        return [k for k in self._npz.files if k != "__meta__"]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

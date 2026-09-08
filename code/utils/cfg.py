"""Dotted command-line overrides for the YAML-configured trainers.

The three flow trainers read one YAML file rather than a Hydra config group, so they have
no `key=value` override mechanism of their own. `train_nf_profiles_4d.py` grew one --
`--set nflow.num_layers=11 optim.lr=1e-3` -- and it is what lets a static submit file run
an adaptive search: the config is fixed and only the overrides change from trial to trial.
This is that function, moved here so all three trainers share one copy rather than three
that can drift.

Semantics are `train_nf_profiles_4d.py`'s: a missing intermediate key is created as a
dict, and values are coerced by shape (`true`/`false` to bool, then int, then float, then
left as a string). One addition -- `null` becomes None, so a warm start can be switched
off from the command line, which is what the from-scratch flow search needs.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable, Optional


def coerce(v: str):
    """`"11"` -> 11, `"1e-3"` -> 0.001, `"true"` -> True, `"scene_prior"` -> itself."""
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() in ("null", "none"):
        return None
    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except ValueError:
        return v


def dotted_overrides(cfg: dict, overrides: Optional[Iterable[str]]) -> dict:
    """Apply `--set a.b.c=val` overrides to a loaded config, returning a copy."""
    out = deepcopy(cfg)
    for s in overrides or []:
        if "=" not in s:
            raise ValueError(f"--set expects key=value, got {s!r}")
        k, v = s.split("=", 1)
        *ks, last = k.split(".")
        node = out
        for kk in ks:
            if kk not in node or not isinstance(node[kk], dict):
                node[kk] = {}
            node = node[kk]
        node[last] = coerce(v)
    return out

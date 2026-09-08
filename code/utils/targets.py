"""targets.py -- which levels of the 20-level CO2 profile a model predicts.

One function, in one place, because this has now been wrong twice in two different files
and in two different directions:

  1. `det_last1` was configured with `target_indices: [19]` and trained all twenty levels
     anyway. Hydra hands a YAML list to Python as an `omegaconf.ListConfig`, which is a
     `MutableSequence` but is neither a `list` nor a `tuple`, so an `isinstance(t, (list,
     tuple))` test was dead for every Hydra-loaded config and the value fell through to a
     silent "all". Twenty-five minutes of GPU, and a checkpoint indistinguishable from
     `det_profiles` written into the surface task's weights directory.

  2. The fix for (1) -- widen the test to `collections.abc.Sequence` -- was ported to
     `train_nf_profiles.py` without the string handling that went with it. **`str` is a
     `Sequence`**, so `target_indices: all` reached `[int(i) for i in "all"]` and every
     `nf_profiles` run died with `invalid literal for int() with base 10: 'a'`. That is
     why `nf_profiles_final_full` and `nf_profiles_final_bal` are checkpoint-free 30 MB
     archives while `nf_last4` -- whose config says `last4`, which returned before
     reaching the sequence branch -- was fine.

The ordering below is the whole point: **strings are resolved first and exhaustively**, and
anything unrecognised raises rather than defaulting to all twenty levels. A silent default
is what let both bugs run for days.

The caller does its own lookup, because the two trainer families store this in different
places -- `cfg.target_indices` for the Hydra trainers, `cfg["model"]["target_indices"]` for
the flow trainers, which read a plain dict -- and only the resolution is shared.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, List

__all__ = ["resolve_target_indices"]


def resolve_target_indices(value: Any, D_full: int = 20) -> List[int]:
    """0-based indices of the profile levels a model predicts.

    `value` is the raw config value, already looked up by the caller.

      "all"          -> every level
      "lastN"        -> the N levels nearest the surface (last4 is the near-surface task,
                        last1 the surface-only one)
      [i, j, ...]    -> those 0-based indices, any Sequence including omegaconf.ListConfig
      anything else  -> raises

    >>> resolve_target_indices("all", 20)[:3]
    [0, 1, 2]
    >>> resolve_target_indices("last4", 20)
    [17, 18, 19]
    >>> resolve_target_indices([19], 20)
    [19]
    """
    # Strings first, and completely. `str` satisfies `isinstance(t, Sequence)`, so any
    # string that reaches the sequence branch below is iterated character by character.
    if isinstance(value, str):
        t = value.strip().lower()
        if t == "all":
            return list(range(D_full))
        m = re.fullmatch(r"last(\d+)", t)
        if m:
            k = int(m.group(1))
            if not 1 <= k <= D_full:
                raise ValueError(
                    f"target_indices={value!r} asks for {k} of {D_full} levels")
            return list(range(D_full - k, D_full))
        raise ValueError(
            f"Unknown target_indices string: {value!r}. Use 'all', 'lastN', or a list "
            f"of 0-based indices.")

    # Sequence, not (list, tuple): a Hydra YAML list arrives as an omegaconf.ListConfig,
    # which is a MutableSequence and neither a list nor a tuple. bytes is excluded for the
    # same reason str is handled above.
    if isinstance(value, (bytes, bytearray)):
        raise TypeError(f"target_indices must not be bytes: {value!r}")
    if isinstance(value, Sequence):
        idx = [int(i) for i in value]
        if not idx:
            raise ValueError("target_indices is an empty sequence")
        bad = [i for i in idx if not 0 <= i < D_full]
        if bad:
            raise ValueError(
                f"target_indices {bad} are outside 0..{D_full - 1}")
        return idx

    # Not a silent default: falling back to all twenty levels is exactly how both bugs
    # above went unnoticed, and an unreadable target spec is a configuration error.
    raise TypeError(
        f"target_indices has type {type(value).__name__}, which is neither a string nor "
        f"a sequence: {value!r}. Use 'all', 'lastN', or a list of 0-based indices.")

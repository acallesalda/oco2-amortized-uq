"""test_target_indices.py
Tests for `utils.targets.resolve_target_indices`, the one place that decides which levels
of the 20-level CO2 profile a model predicts.

This function has been wrong twice, in two files, in opposite directions, and each time it
failed in a way that cost a queue day:

  * `det_last1` was configured with `target_indices: [19]` and trained all twenty levels,
    silently, because Hydra hands a YAML list to Python as an `omegaconf.ListConfig` and
    the branch tested `isinstance(t, (list, tuple))`.
  * `nf_profiles` died on `int('a')` for a month because the fix for the above was ported
    without its string handling, and `str` is a `collections.abc.Sequence`, so the default
    `"all"` was iterated character by character.

The regression tests for both are `test_a_bare_string_is_not_iterated_as_a_sequence` and
`test_a_listconfig_like_sequence_is_accepted`. The rest pin the contract that makes those
two impossible to reintroduce: strings are resolved first and exhaustively, and anything
unrecognised raises instead of defaulting to all twenty levels.
"""
import sys
from collections.abc import MutableSequence
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from utils.targets import resolve_target_indices  # noqa: E402


class FakeListConfig(MutableSequence):
    """Stands in for omegaconf.ListConfig: a MutableSequence that is not list or tuple.

    This is what a Hydra YAML list actually arrives as, and testing against a plain list
    is what let the det_last1 bug through.
    """

    def __init__(self, data):
        self._data = list(data)

    def __getitem__(self, i):
        return self._data[i]

    def __setitem__(self, i, v):
        self._data[i] = v

    def __delitem__(self, i):
        del self._data[i]

    def __len__(self):
        return len(self._data)

    def insert(self, i, v):
        self._data.insert(i, v)


# --------------------------------------------------------------------- the two bugs
def test_a_bare_string_is_not_iterated_as_a_sequence():
    """Regression: `str` is a Sequence, so "all" must be handled before that branch."""
    assert resolve_target_indices("all", 20) == list(range(20))


def test_a_listconfig_like_sequence_is_accepted():
    """Regression: the list branch must not test for `list`/`tuple` specifically."""
    assert not isinstance(FakeListConfig([19]), (list, tuple))
    assert resolve_target_indices(FakeListConfig([19]), 20) == [19]


# --------------------------------------------------------------------- the contract
@pytest.mark.parametrize("value,expected", [
    ("all", list(range(20))),
    ("ALL", list(range(20))),
    (" all ", list(range(20))),
    ("last1", [19]),
    ("last4", [16, 17, 18, 19]),
    ("LAST4", [16, 17, 18, 19]),
    ("last20", list(range(20))),
])
def test_strings(value, expected):
    assert resolve_target_indices(value, 20) == expected


@pytest.mark.parametrize("value,expected", [
    ([19], [19]),
    ((17, 18, 19), [17, 18, 19]),
    ([0], [0]),
    (["3", "4"], [3, 4]),
])
def test_sequences(value, expected):
    assert resolve_target_indices(value, 20) == expected


def test_every_target_indices_value_used_in_the_repo_resolves():
    """The four spellings that appear in code/shared_conf/experiment/*.yaml today."""
    for value, n in (("all", 20), ("last4", 4), ([19], 1), (FakeListConfig([19]), 1)):
        assert len(resolve_target_indices(value, 20)) == n


# --------------------------------------------------------------------- it must not guess
@pytest.mark.parametrize("value", ["", "nonsense", "last", "last0", "last21", "first4"])
def test_unknown_strings_raise_rather_than_defaulting_to_all(value):
    with pytest.raises(ValueError):
        resolve_target_indices(value, 20)


@pytest.mark.parametrize("value", [None, 3, 3.5, {"a": 1}, object()])
def test_non_sequences_raise(value):
    with pytest.raises(TypeError):
        resolve_target_indices(value, 20)


def test_bytes_are_rejected_rather_than_iterated():
    with pytest.raises(TypeError):
        resolve_target_indices(b"all", 20)


@pytest.mark.parametrize("value", [[], [20], [-1], [0, 20]])
def test_out_of_range_and_empty_sequences_raise(value):
    with pytest.raises(ValueError):
        resolve_target_indices(value, 20)


def test_d_full_is_honoured():
    assert resolve_target_indices("all", 4) == [0, 1, 2, 3]
    assert resolve_target_indices("last2", 4) == [2, 3]
    with pytest.raises(ValueError):
        resolve_target_indices("last5", 4)
    with pytest.raises(ValueError):
        resolve_target_indices([4], 4)

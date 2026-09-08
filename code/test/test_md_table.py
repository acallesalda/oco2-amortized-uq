"""Tests for `evals.tables.md_table`.

The point of this function is that it cannot fail, and the reason it exists is that the
one-liner it replaces did: `DataFrame.to_markdown` raises `ImportError` without `tabulate`,
which the container does not have, and it killed all three `attribution.dag` jobs on
4 Sep 2026 at the very end of the run. So the test that matters is the third one -- the
fallback is exercised with `to_markdown` forced to raise, because otherwise the branch is
only ever reached on a machine nobody runs the suite on.

Pandas only, no torch: this file has to run in the environment where the failure happened.
"""

import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(_ROOT / "code"))

from evals.tables import md_table                                # noqa: E402


@pytest.fixture
def df():
    return pd.DataFrame({"config": ["intact", "occlude b2"],
                         "det_rmse_ppm": [5.003123, 6.160321],
                         "n": [4489, 4489]})


def test_the_markdown_path_is_a_pipe_table_when_tabulate_is_available(df):
    pytest.importorskip("tabulate")
    out = md_table(df)
    assert "|" in out
    assert "occlude b2" in out
    assert "5.0031" in out and "6.1603" in out
    assert "5.003123" not in out                    # floatfmt was applied
    assert out.endswith("\n")


def test_the_fallback_does_not_raise_and_keeps_the_numbers(df, monkeypatch):
    """The regression test: `to_markdown` raising must not cost the caller anything."""
    def boom(self, *a, **k):
        raise ImportError("Missing optional dependency 'tabulate'.")

    monkeypatch.setattr(pd.DataFrame, "to_markdown", boom, raising=True)

    out = md_table(df)                              # must not raise
    assert out.startswith("```\n") and out.rstrip().endswith("```")
    assert "occlude b2" in out
    assert "5.0031" in out and "6.1603" in out
    assert "5.003123" not in out                    # the float format survived the fallback
    assert "\n" in out


def test_the_fallback_honours_a_different_float_format(df, monkeypatch):
    monkeypatch.setattr(pd.DataFrame, "to_markdown",
                        lambda self, *a, **k: (_ for _ in ()).throw(ImportError("no tabulate")),
                        raising=True)
    out = md_table(df, floatfmt=".1f")
    assert "5.0" in out
    assert "5.0031" not in out


def test_an_empty_frame_is_a_placeholder_not_a_table(df):
    assert md_table(df.iloc[:0]).strip() == "_(nothing to report)_"
    assert md_table(pd.DataFrame()).strip() == "_(nothing to report)_"


def test_none_is_a_placeholder_too():
    """`main` builds the correlation frame from a list that can legitimately be empty."""
    assert md_table(None).strip() == "_(nothing to report)_"


def test_a_custom_placeholder_is_used(df):
    assert md_table(df.iloc[:0], empty="_(no windows)_").strip() == "_(no windows)_"

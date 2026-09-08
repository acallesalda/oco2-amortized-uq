"""test_table_guard.py
Tests for the scale check on a predictive distribution, and the warning it puts in the
LaTeX that goes into the manuscript.

The defect this guards against shipped for the whole project and was invisible: the scalar
Laplace predictive carried no observation-noise term, so it was 27.6x too narrow in
variance, and every column beside it -- an NLL, a coverage percentage -- looked exactly like
a result. Nothing in the pipeline objected, because nothing compared the predictive to the
errors it was placed against.

The two properties that matter are asymmetric, so both are pinned here:

  * a predictive fitted here that is grossly mis-scaled must stop the table;
  * ACOS being grossly mis-scaled must NOT, because that is the paper's central
    measurement and a warning there would bury it.

Pure numpy and pandas -- no artifact, no torch.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from evals.metrics import _var_ratio                                  # noqa: E402


def _load_module(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sc = _load_module("experiments/scoring/score.py", "score_mod")


# ------------------------------------------------------------------ the ratio --
def test_ratio_is_one_for_a_correctly_scaled_predictive():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 2.0, size=(20000, 1))
    mu = np.zeros_like(y)
    var = np.full_like(y, 4.0)
    assert _var_ratio(y, mu, var) == pytest.approx(1.0, rel=0.05)


def test_ratio_reports_the_shortfall_of_a_too_narrow_predictive():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 2.0, size=(20000, 1))
    mu = np.zeros_like(y)
    # The real case: an epistemic-only predictive, 5x too small in standard deviation.
    var = np.full_like(y, (2.0 / 5.0) ** 2)
    assert _var_ratio(y, mu, var) == pytest.approx(25.0, rel=0.05)


def test_ratio_is_finite_and_nan_rather_than_infinite_on_a_degenerate_predictive():
    y = np.zeros((10, 1))
    assert np.isnan(_var_ratio(y, y, np.zeros((10, 1))))


def test_ratio_handles_multiple_dimensions():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 3.0, size=(5000, 4))
    assert _var_ratio(y, np.zeros_like(y), np.full_like(y, 9.0)) == pytest.approx(1.0, rel=0.05)


# ------------------------------------------------------------------- the guard --
def _frame(rows):
    base = dict(rmse_ppm=1.0, rmse_ppm_se=0.1, mae_ppm=1.0, mae_ppm_se=0.1,
                nll_ppm=1.0, nll_se=0.1, nll_std=1.0,
                cov95_joint=0.95, cov95_marginal=0.95,
                cov68_joint=0.68, cov68_marginal=0.68)
    return pd.DataFrame([{**base, **r} for r in rows])


def test_a_mis_scaled_row_of_ours_stops_the_table():
    tex = sc.to_latex(_frame([
        dict(split="s", method="Laplace", var_ratio=27.9),
    ]), {})
    assert "NOT REPORTABLE" in tex
    assert "Laplace: residual variance is 27.9x" in tex


def test_a_well_scaled_row_of_ours_does_not():
    tex = sc.to_latex(_frame([
        dict(split="s", method="Laplace", var_ratio=0.97),
        dict(split="s", method="Normalizing flow", var_ratio=1.23),
    ]), {})
    assert "NOT REPORTABLE" not in tex


def test_acos_being_mis_scaled_is_reported_as_a_finding_not_a_defect():
    """The one row where a large ratio is the result rather than a bug."""
    tex = sc.to_latex(_frame([
        dict(split="s", method="ACOS L2FP (v10)", var_ratio=32.5),
        dict(split="s", method="Normalizing flow", var_ratio=1.2),
    ]), {})
    assert "NOT REPORTABLE" not in tex
    assert "this is a finding, not a defect" in tex
    assert "32.5x" in tex


def test_both_can_appear_and_are_not_confused():
    tex = sc.to_latex(_frame([
        dict(split="s", method="ACOS L2FP (v10)", var_ratio=32.5),
        dict(split="s", method="Laplace", var_ratio=27.9),
        dict(split="s", method="Normalizing flow", var_ratio=1.2),
    ]), {})
    assert "NOT REPORTABLE" in tex
    warn = tex.split("NOT REPORTABLE")[1].split("Re-score")[0]
    assert "Laplace" in warn
    assert "ACOS" not in warn, "ACOS must not be inside the not-reportable block"


def test_a_missing_ratio_column_does_not_crash_older_tables():
    """Artifacts scored before this column existed still have to render."""
    df = _frame([dict(split="s", method="Laplace")])
    tex = sc.to_latex(df.drop(columns=[]), {})
    assert "Laplace &" in tex
    assert "NOT REPORTABLE" not in tex


def test_the_numbers_still_render_beside_the_warning():
    """The warning is a comment: it must not replace the row or break the LaTeX."""
    tex = sc.to_latex(_frame([dict(split="s", method="Laplace", var_ratio=27.9)]), {})
    body = [l for l in tex.splitlines() if not l.startswith("%")]
    assert any(l.strip().endswith(r"\\") for l in body)

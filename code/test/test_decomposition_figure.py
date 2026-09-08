"""test_decomposition_figure.py
The width panel of `decomposition.png` must not draw a predictive it cannot report.

After the 31 August re-score the three evaluations are no longer scored from one generation
of artifact: February's XCO2 Laplace carries the observation-noise term and the two months'
do not. Drawing all three put a 1.71 ppm bar beside two 0.43 ppm ones, which reads as "the
reported width collapses out of distribution" and is nothing but a stale artifact -- Note
14's failure mode, produced by fixing it halfway.

`width_void` marks those rows in `decomposition_summary.csv`; the figure skips them. Only
the width panel is affected: the placement and within/bound panels are point estimates and
were never at risk.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "experiments" / "scoring"))


def _load_module(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


de = _load_module("experiments/scoring/decompose_error.py", "decompose_error_mod")

EXPS = ["group split (Feb)", "March", "July"]


def _summary(with_flag=True, void_months=True):
    rows = []
    for task, floor in (("near surface", 11.8), ("XCO2", 1.82)):
        for exp in EXPS:
            for m, sd in (("flow", 1.72), ("Laplace", 1.71), ("ACOS", 0.50)):
                # the defect makes the months' Laplace width a third of February's
                void = void_months and task == "XCO2" and m == "Laplace" \
                    and exp != "group split (Feb)"
                r = dict(task=task, experiment=exp, level="pooled", method=m,
                         n=4489, rmse=2.0, placement=1.4, within=1.5, floor=floor,
                         over_floor=1.1, within_over_floor=0.9,
                         pred_sd=0.43 if void else sd, z=1.2, cov95=93.0)
                if with_flag:
                    r["width_void"] = void
                rows.append(r)
    return pd.DataFrame(rows)


def test_the_flagged_rows_are_the_two_month_laplace_widths(tmp_path):
    """The figure draws, and what it withholds is exactly the two defective cells."""
    S = _summary()
    path = de.fig_decomposition(S, tmp_path, EXPS, stem="d")
    assert path is not None and path.exists()
    void = S[S.width_void]
    assert len(void) == 2                      # March and July, XCO2, Laplace
    assert set(void.experiment) == {"March", "July"}
    assert (void.task == "XCO2").all() and (void.method == "Laplace").all()


def test_the_point_estimate_panels_keep_every_row():
    """`width_void` must not reach the placement or within/bound panels: those columns are
    point estimates and the defect does not touch them."""
    S = _summary()
    for col in ("placement", "within_over_floor", "rmse", "floor"):
        assert S[col].notna().all()
        assert S.loc[S.width_void, col].notna().all()


def test_a_summary_without_the_column_still_draws(tmp_path):
    """`--fig-only` on a CSV written before the column existed draws the old figure rather
    than crashing."""
    S = _summary(with_flag=False)
    assert "width_void" not in S.columns
    path = de.fig_decomposition(S, tmp_path, EXPS, stem="old")
    assert path is not None and path.exists()


def test_epistemic_only_is_an_exact_test_on_the_artifact(tmp_path):
    """Same detector as make_temporal_table's, pinned here too: the two scripts must agree
    about which artifacts are defective or the figure and the table will disagree."""
    epi = np.full((6, 1, 1), 0.01)
    same = tmp_path / "same.npz"
    diff = tmp_path / "diff.npz"
    np.savez(same, lap_cov_epi_std=epi, lap_cov_std=epi.copy())
    np.savez(diff, lap_cov_epi_std=epi, lap_cov_std=epi + 0.5)
    assert de.epistemic_only(same) is True
    assert de.epistemic_only(diff) is False


def test_the_two_detectors_are_the_same_function(tmp_path):
    mt = _load_module("experiments/scoring/make_temporal_table.py", "mtt_for_fig_test")
    epi = np.full((4, 1, 1), 0.02)
    p = tmp_path / "a.npz"
    np.savez(p, lap_cov_epi_std=epi, lap_cov_std=epi.copy())
    assert de.epistemic_only(p) == mt.epistemic_only(p) is True

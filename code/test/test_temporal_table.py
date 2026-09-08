"""test_temporal_table.py
Tests for the temporal-generalization table generator.

Three properties, and each of them is a mistake this project has already made once:

  * the Note-14 cells must not print. The XCO2 Laplace predictive carried no
    observation-noise term in every artifact produced before the `laplace_mode="mc"` fix,
    and its NLL and coverage look exactly like results. The detector is an exact test on
    the artifact -- `lap_cov_std` bit-identical to `lap_cov_epi_std` -- rather than a
    variance-ratio heuristic, so it is pinned as one.
  * the near-surface Laplace cells must still print. The defect is the scalar task's; a
    guard that blanked the whole Laplace row would delete four correct numbers per
    evaluation and would be read as the method having no result at all.
  * `n` and the screened count must reach the block header. One July sounding carries a
    non-physical retrieved state (`clst0002_ref0011_rep0010`), and the whole reason
    a control cell went missing once is that nothing said so out loud.

Pure numpy/pandas plus one temporary .npz -- no artifact, no torch, no GPU.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))


def _load_module(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mt = _load_module("experiments/scoring/make_temporal_table.py", "make_temporal_table_mod")

EXPS = [mt.FEB, "March 2020", "July 2020"]


def _rows(lap_epistemic_only=True, july_screened=1):
    """The nine rows of the table, in the shape `score.rows_for` produces them."""
    out = []
    for exp, n in ((mt.FEB, 4489), ("March 2020", 44985), ("July 2020", 44327)):
        for task, dim in (("xco2", 1), ("last4", 4)):
            for m in mt.METHOD_ORDER:
                drop = july_screened if exp == "July 2020" else 0
                out.append(dict(
                    experiment=exp, task=task, method=m, n=n, n_point=n - drop,
                    n_nonphysical=drop, rmse_ppm=2.5 if task == "xco2" else 13.0,
                    rmse_ppm_se=0.2, nll_ppm=3.0, nll_std=0.5, cov95_marginal=0.91,
                    cov95_joint=0.88, dim=dim, artifact=f"{exp}_{task}.npz",
                    rmse_over_feb=1.0 if exp == mt.FEB else 1.21,
                    lap_epistemic_only=(lap_epistemic_only and task == "xco2"),
                ))
    return pd.DataFrame(out)


# ------------------------------------------------- the not-reportable guard --
def test_void_xco2_laplace_cells_are_blanked_and_announced():
    tex = mt.to_latex(_rows(), EXPS, ["February / Laplace / XCO2: no noise term"])
    assert "NOT REPORTABLE" in tex
    lap = [l for l in tex.splitlines() if l.startswith(r"\quad Laplace")]
    assert len(lap) == 3
    for line in lap:
        cells = [c.strip() for c in line.split("&")]
        # method | RMSE, vs Feb, NLL, cov (XCO2) | the same four (near surface)
        assert cells[1].startswith("2.50")          # the point estimate is unaffected
        assert cells[2] in ("1.00", "1.21")         # so is the ratio built from it
        assert cells[3] == "--" and cells[4] == "--"


def test_near_surface_laplace_cells_survive_the_guard():
    """The defect is the scalar task's. Blanking the profile task too would delete four
    correct numbers per evaluation."""
    tex = mt.to_latex(_rows(), EXPS, ["x"])
    for line in [l for l in tex.splitlines() if l.startswith(r"\quad Laplace")]:
        cells = [c.strip() for c in line.split("&")]
        assert cells[7] == "3.00"                    # near-surface NLL
        assert cells[8].startswith("88.0 / 91.0")    # joint / marginal


def test_a_rescored_artifact_prints_every_cell_and_no_warning():
    tex = mt.to_latex(_rows(lap_epistemic_only=False), EXPS, [])
    assert "NOT REPORTABLE" not in tex
    for line in [l for l in tex.splitlines() if l.startswith(r"\quad Laplace")]:
        assert "--" not in line


# --------------------------------------------------------- the artifact detector --
def _write(tmp_path, identical: bool):
    n = 8
    epi = np.full((n, 1, 1), 0.01)
    cov = epi.copy() if identical else epi + 0.5
    p = tmp_path / f"xco2_{identical}.npz"
    np.savez(p, lap_cov_epi_std=epi, lap_cov_std=cov)
    return p


def test_epistemic_only_is_an_exact_test_on_the_artifact(tmp_path):
    assert mt.epistemic_only(_write(tmp_path, True)) is True
    assert mt.epistemic_only(_write(tmp_path, False)) is False


def test_epistemic_only_is_false_when_the_keys_are_absent(tmp_path):
    p = tmp_path / "empty.npz"
    np.savez(p, y_std=np.zeros(3))
    assert mt.epistemic_only(p) is False


# ------------------------------------------------------------------- the header --
def test_block_header_carries_n_and_the_screened_sounding():
    tex = mt.to_latex(_rows(), EXPS, [])
    july = [l for l in tex.splitlines() if "July 2020" in l and "multicolumn" in l]
    assert len(july) == 1
    assert "44{,}327" in july[0]
    assert "one sounding screened" in july[0]
    feb = [l for l in tex.splitlines() if mt.FEB in l and "multicolumn" in l][0]
    assert "4{,}489" in feb and "screened" not in feb


def test_the_table_has_one_row_per_method_per_evaluation():
    tex = mt.to_latex(_rows(), EXPS, [])
    assert len([l for l in tex.splitlines() if l.startswith(r"\quad ")]) == 9
    assert tex.count(r"\multicolumn{9}{l}") == 3
    assert tex.rstrip().endswith(r"\end{table*}")


# ------------------------------------------------------- the companion float ----
def _decomp():
    rows = []
    for exp, n, floor in (("group split (Feb)", 4489, 11.77), ("March", 44985, 13.80),
                          ("July", 44326, 14.18)):
        for m, wof in (("ACOS", 0.847), ("Laplace", 0.961), ("flow", 1.0013)):
            rows.append(dict(task="near surface", experiment=exp, level="pooled", method=m,
                             n=n, rmse=12.0, placement=4.0, within=11.0, floor=floor,
                             within_over_floor=wof))
    return pd.DataFrame(rows)


def test_decomposition_float_keeps_three_decimals_on_the_ratio():
    """1.001 is the finding; 1.00 is not."""
    tex = mt.decomposition_latex(_decomp())
    assert "1.001" in tex
    assert r"\label{tab:decomp}" in tex
    # the console's experiment names are translated into the table's
    assert "group split" not in tex and mt.FEB in tex


def test_decomposition_float_is_empty_when_the_task_is_absent():
    assert mt.decomposition_latex(_decomp(), task="XCO2") == ""


# ------------------------------------------------------- the re-score hint ------
def test_months_and_splits_are_separate_invocations():
    """`rescore_submit.sh` rejects --split and --month in one call ("different
    experiments. Run them separately."), so a hint that prints both on one line is a
    command that errors out. It did, once."""
    cmds = mt.rescore_commands([
        "experiments/scoring/xco2_bo.npz",
        "experiments/scoring/results/month_202003_full/xco2.npz",
        "experiments/scoring/results/month_202007_full/xco2.npz",
    ])
    assert len(cmds) == 2
    months = [c for c in cmds if "--month" in c]
    splits = [c for c in cmds if "--split" in c]
    assert len(months) == 1 and len(splits) == 1
    assert "--split" not in months[0] and "--month" not in splits[0]
    # both months in the one invocation that takes them, in a stable order
    assert months[0].endswith("--month 202003 --month 202007")
    assert splits[0].endswith("--split group")


def test_the_hint_names_the_right_partition_for_each_artifact():
    assert mt.rescore_commands(["a/xco2_borep.npz"]) == [
        "./chtc/rescore_submit.sh xco2 --split replicate"]
    assert mt.rescore_commands(["experiments/scoring/xco2.npz"]) == [
        "./chtc/rescore_submit.sh xco2 --split published"]


def test_no_void_artifacts_means_no_command():
    assert mt.rescore_commands([]) == []

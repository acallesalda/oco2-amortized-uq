"""test_input_table.py
Tests for the definitive input table.

The table's whole value is that it cannot disagree with the loader. A hand-written table
that omits an input, or keeps a row for one the model no longer receives, is precisely the
defect this table exists to prevent -- and it is invisible: the manuscript compiles, the numbers
are unchanged, and only a reader with the code can tell. So the properties that make the
table definitive are asserted here rather than eyeballed once.

Pure frame operations -- no simulation data, no checkpoint, no split table.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from data.loading.covariates import LEGACY_AEROSOL_VOCAB, get_spec  # noqa: E402


def _load_module(relpath: str, name: str):
    """Import a script that lives outside the importable package tree."""
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mit = _load_module("experiments/pipelines/make_input_table.py", "make_input_table")


# ---------------------------------------------------------------- fixtures --
def fake_split_table(n_scenes=12, n_rep=5, seed=0):
    """A table carrying every column any reported preset names, plus split/clst/ref."""
    rng = np.random.default_rng(seed)
    cols = sorted({c for name in (mit.PUBLISHED, mit.REVISED, mit.ABLATION)
                   for c in get_spec(name).columns})
    rows = []
    for scene in range(1, n_scenes + 1):
        split = {1: "test", 2: "val"}.get(scene, "train")
        for rep in range(n_rep):
            row = {"clst": 1 + scene % 3, "ref": scene, "split": split}
            for c in cols:
                if c.startswith("Footprint"):
                    row[c] = 1 + (rep % 8)
                elif c.startswith("AerType"):
                    row[c] = ["DU", "SO", "SS", "OC"][rep % 4]
                else:
                    row[c] = float(rng.normal(1.0, 0.25))
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def table(tmp_path):
    p = tmp_path / "split_cov.csv"
    fake_split_table().to_csv(p, index=False)
    return p


# ------------------------------------------------------- coverage of the presets --
def test_every_covariate_of_every_reported_preset_has_a_row():
    rows = mit.aux_rows(None, None)
    listed = {r["column"] for r in rows}
    for name in (mit.PUBLISHED, mit.REVISED, mit.ABLATION):
        missing = set(get_spec(name).columns) - listed
        assert not missing, f"preset '{name}' names {missing}, which the table omits"


def test_a_preset_gaining_a_covariate_fails_loudly(monkeypatch):
    """The failure mode that matters: a new input, and a table that quietly omits it."""
    spec = get_spec(mit.REVISED)
    grown = type(spec)(name=spec.name,
                       continuous=spec.continuous + ("a_brand_new_covariate",),
                       categorical=spec.categorical, legacy=spec.legacy, tier=spec.tier)
    monkeypatch.setattr(mit, "get_spec",
                        lambda n: grown if n == mit.REVISED else get_spec(n))
    with pytest.raises(KeyError, match="a_brand_new_covariate"):
        mit.aux_rows(None, None)


def test_no_row_is_marked_for_a_preset_that_does_not_name_it():
    rows = mit.aux_rows(None, None)
    for r in rows:
        assert r["published"] == (r["column"] in get_spec(mit.PUBLISHED).columns)
        assert r["revised"] == (r["column"] in get_spec(mit.REVISED).columns)
        assert r["ablation"] == (r["column"] in get_spec(mit.ABLATION).columns)


def test_counts_are_the_ones_the_manuscript_quotes():
    rows = mit.aux_rows(None, None)
    assert sum(r["published"] for r in rows) == 4
    assert sum(r["revised"] for r in rows) == 12
    assert sum(r["ablation"] for r in rows) == 14


def test_continuous_and_categorical_agree_with_the_spec():
    rows = mit.aux_rows(None, None)
    cat = set(get_spec(mit.REVISED).categorical) | set(get_spec(mit.PUBLISHED).categorical)
    for r in rows:
        if r["column"] in cat:
            assert r["kind"] == "categorical", r["column"]


def test_the_published_rows_report_the_hardcoded_vocabulary():
    """`legacy` encodes through LEGACY_AEROSOL_VOCAB, not through the training rows.

    Printing an observed level count there would say the released encoding tolerates an
    unseen aerosol type. It raises a KeyError.
    """
    rows = {r["column"]: r for r in mit.aux_rows(None, None)}
    for col in ("AerType1", "AerType2"):
        assert str(len(LEGACY_AEROSOL_VOCAB)) in rows[col]["unit"]
        assert "unseen value is an error" in rows[col]["range"]


# ------------------------------------------------------------------ statistics --
def test_statistics_come_from_the_training_rows_only(table):
    df = pd.read_csv(table)
    stats, meta = mit.compute_stats(table)
    assert meta["n_train"] == int((df["split"] == "train").sum())
    assert meta["n_train"] < meta["n_total"]
    train = df[df["split"] == "train"]
    st = stats["SlrZen_fix"]
    assert st["mean"] == pytest.approx(float(train["SlrZen_fix"].mean()))
    assert st["min"] == pytest.approx(float(train["SlrZen_fix"].min()))


def test_a_level_only_in_the_test_partition_is_not_in_the_vocabulary(table):
    """The reserved slot exists for exactly this, and the table must say so."""
    df = pd.read_csv(table)
    df.loc[df["split"] == "test", "AerType2_fix"] = "BC"
    df.to_csv(table, index=False)
    stats, _ = mit.compute_stats(table)
    st = stats["AerType2_fix"]
    assert "BC" not in st["levels"]
    assert st["cardinality"] == len(st["levels"]) + 1


# ----------------------------------------------------------------- the outputs --
def test_writes_all_four_files_and_they_are_not_empty(table, tmp_path):
    out = tmp_path / "tables"
    assert mit.main(["--split-table", str(table), "--out", str(out)]) == 0
    for name in ("inputs_spectral.tex", "inputs_aux.tex", "inputs.md", "inputs.json"):
        assert (out / name).stat().st_size > 0, name


def test_the_tex_has_one_row_per_variable(table, tmp_path):
    out = tmp_path / "tables"
    mit.main(["--split-table", str(table), "--out", str(out)])
    body = (out / "inputs_aux.tex").read_text(encoding="utf-8")
    n_rows = sum(1 for line in body.splitlines()
                 if line.endswith(r"\\") and "Variable &" not in line)
    assert n_rows == len(mit.aux_rows(None, None))


def test_the_tex_escapes_nothing_inside_math(table, tmp_path):
    """`_tex_escape` used to escape underscores, which prints `\\_` inside $...$."""
    out = tmp_path / "tables"
    mit.main(["--split-table", str(table), "--out", str(out)])
    for name in ("inputs_aux.tex", "inputs_spectral.tex"):
        body = (out / name).read_text(encoding="utf-8")
        assert r"\_" not in body, name


def test_it_runs_without_a_split_table(tmp_path):
    """The schema half must not need the data: it is what a reader can check."""
    out = tmp_path / "tables"
    assert mit.main(["--split-table", str(tmp_path / "absent.csv"),
                     "--out", str(out)]) == 0
    assert (out / "inputs_aux.tex").stat().st_size > 0


def test_spectral_channel_counts_sum_to_the_kept_total():
    assert sum(n for _, _, n, _, _ in mit.BAND_ROWS) == mit.N_CHANNELS_KEPT
    assert mit.N_CHANNELS_KEPT < mit.N_CHANNELS_RAW
    for _, _, n, _, _ in mit.BAND_ROWS:
        assert n <= mit.PAD_WIDTH, "a band longer than the pad width would be truncated"

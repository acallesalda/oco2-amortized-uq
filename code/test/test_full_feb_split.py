"""test_full_feb_split.py
Tests for the ALL-February split builder.

The table this script writes decides what six retrained models are fitted on, which scenes
select their checkpoints, and which scenes calibrate the aleatoric diagonal of the Laplace
predictive at scoring time. Every one of those properties is invisible in the numbers that
come out: a table whose validation scenes are not the ones the hyperparameter search ranked
on trains perfectly well, selects a checkpoint, calibrates a Laplace, and produces a
plausible March RMSE with nothing anywhere to say the experiment was not the one described.
So the properties are asserted here rather than inferred from a log line later.

Pure frame operations -- no simulation data, no checkpoint.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))


def _load_module(relpath: str, name: str):
    """Import a script that lives outside the importable package tree."""
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mff = _load_module("experiments/pipelines/make_full_feb_split.py", "make_full_feb_split")


# ---------------------------------------------------------------- fixtures --
def make_group_split(refs_per_cluster=(6, 4, 3), n_rep=10, seed=0):
    """A group_split_v1-shaped table: whole reference soundings in one partition each."""
    rng = np.random.default_rng(seed)
    rows = []
    for clst, n_refs in enumerate(refs_per_cluster, start=1):
        # One test scene and one val scene per cluster; the rest train. The real builder
        # apportions by cluster size, but all that matters downstream is that the
        # assignment is by whole reference sounding.
        for ref in range(1, n_refs + 1):
            split = {1: "test", 2: "val"}.get(ref, "train")
            for rep in range(1, n_rep + 1):
                rows.append({
                    "ID": f"clst{clst:04d}_ref{ref:04d}_rep{rep:04d}",
                    "source_dir": "train" if rng.random() < 0.8 else "test",
                    "clst": clst,
                    "ref": ref,
                    "rep": rep,
                    "true_xco2": 4.0e-4 + 1e-5 * rng.standard_normal(),
                    "split": split,
                })
    return pd.DataFrame(rows)


def scenes(df, split):
    s = df[df["split"] == split]
    return set(map(tuple, s[["clst", "ref"]].drop_duplicates().to_numpy()))


# ------------------------------------------------------------ the builder --
def test_no_row_is_gained_or_lost():
    src = make_group_split()
    out = mff.absorb_test(src)
    assert len(out) == len(src)
    assert set(out["ID"]) == set(src["ID"])


def test_no_test_partition_survives():
    out = mff.absorb_test(make_group_split())
    assert not (out["split"] == "test").any()


def test_the_test_scenes_move_into_training():
    src = make_group_split()
    out = mff.absorb_test(src)
    assert scenes(out, "train") == scenes(src, "train") | scenes(src, "test")


def test_validation_is_untouched():
    """The property both the searched hyperparameters and the Laplace depend on.

    The configurations were ranked on these scenes, and predict.py calibrates the
    aleatoric diagonal of the Laplace predictive on them at scoring time. If either read a
    different set, nothing would fail -- the numbers would just mean something else.
    """
    src = make_group_split()
    out = mff.absorb_test(src)
    assert scenes(out, "val") == scenes(src, "val")
    assert (out["split"] == "val").sum() == (src["split"] == "val").sum()


def test_train_and_val_stay_scene_disjoint():
    """What makes validation a held-out set rather than a subsample of training."""
    out = mff.absorb_test(make_group_split())
    assert not (scenes(out, "train") & scenes(out, "val"))


def test_nothing_but_the_split_column_changes():
    src = make_group_split()
    out = mff.absorb_test(src)
    other = [c for c in src.columns if c != "split"]
    pd.testing.assert_frame_equal(out[other], src[other])


def test_the_source_table_is_not_mutated():
    src = make_group_split()
    before = src["split"].value_counts().to_dict()
    mff.absorb_test(src)
    assert src["split"].value_counts().to_dict() == before


# ---------------------------------------------------------------- validate --
def test_validate_accepts_the_builder_output():
    src = make_group_split()
    mff.validate(mff.absorb_test(src), src)


def test_validate_rejects_a_surviving_test_partition():
    src = make_group_split()
    with pytest.raises(AssertionError, match="test partition"):
        mff.validate(src, src)


def test_validate_rejects_an_empty_validation_partition():
    src = make_group_split()
    out = mff.absorb_test(src)
    out.loc[out["split"] == "val", "split"] = "train"
    with pytest.raises(AssertionError, match="partition 'val' is empty"):
        mff.validate(out, src)


def test_validate_rejects_a_scene_in_both_train_and_val():
    src = make_group_split()
    out = mff.absorb_test(src)
    # Move one replicate of a training scene into val: the scene is then in both.
    first_train = out.index[out["split"] == "train"][0]
    out.loc[first_train, "split"] = "val"
    with pytest.raises(AssertionError, match="both train and val"):
        mff.validate(out, src)


def test_validate_rejects_moved_validation_scenes():
    """A val partition of the right SIZE but the wrong scenes must still fail."""
    src = make_group_split()
    out = mff.absorb_test(src)
    swap_out = scenes(out, "val").pop()
    victim = (out["clst"] == swap_out[0]) & (out["ref"] == swap_out[1])
    donor_scene = sorted(scenes(out, "train"))[0]
    donor = (out["clst"] == donor_scene[0]) & (out["ref"] == donor_scene[1])
    out.loc[victim, "split"] = "train"
    out.loc[donor, "split"] = "val"
    with pytest.raises(AssertionError, match="not group_split_v1's"):
        mff.validate(out, src)


def test_validate_rejects_dropped_rows():
    src = make_group_split()
    out = mff.absorb_test(src).iloc[:-1]
    with pytest.raises(AssertionError, match="Nothing here drops rows"):
        mff.validate(out, src)


def test_validate_rejects_duplicate_ids():
    src = make_group_split()
    out = mff.absorb_test(src)
    out = pd.concat([out, out.iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError, match="duplicate sounding IDs"):
        mff.validate(out, src)


# ------------------------------------------------------------------ counts --
def test_n_groups_per_split_counts_scenes_not_rows():
    out = mff.absorb_test(make_group_split(refs_per_cluster=(6, 4, 3), n_rep=10))
    # 13 scenes, one val per cluster -> 3 val, 10 train.
    assert mff.n_groups_per_split(out) == {"train": 10, "val": 3}

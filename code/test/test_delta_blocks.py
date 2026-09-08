"""test_delta_blocks.py
Tests for the discrepancy-block error decomposition.

The number this produces -- "67 % of the operational retrieval's error variance is which
draw of the discrepancy the sounding received, and 0 % of ours is" -- is the whole of the
headline number, and it comes out of a nested ANOVA on an unbalanced design. That estimator is
easy to get subtly wrong in a way no output inspection catches: the variance of group means
is biased upward by exactly the within-group noise it is supposed to separate out, and the
bias is largest when groups are small, which is the regime here (ten replicates per block).
So the estimator is checked against synthetic data with known components.

Pure numpy -- no artifact, no simulation data.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))


def _load_module(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


db = _load_module("experiments/scoring/delta_blocks.py", "delta_blocks")


# ---------------------------------------------------------------- fixtures --
def synthetic(v_scene, v_block, v_within, n_scenes=45, n_blocks=10, n_rep=10, seed=0):
    """The real design: 45 scenes, 10 discrepancy blocks each, 10 replicates each."""
    rng = np.random.default_rng(seed)
    y, scene, blk = [], [], []
    for s in range(n_scenes):
        a = rng.normal(0, np.sqrt(v_scene))
        for b in range(n_blocks):
            c = rng.normal(0, np.sqrt(v_block))
            for _ in range(n_rep):
                y.append(a + c + rng.normal(0, np.sqrt(v_within)))
                scene.append(f"{s}")
                blk.append(f"{s}#{b}")
    return np.array(y), np.array(scene), np.array(blk)


# ------------------------------------------------------- the estimator --
@pytest.mark.parametrize("v_scene,v_block,v_within", [
    (4.0, 2.0, 1.0),
    (0.0, 1.0, 1.0),      # the ACOS shape: no scene component, a large block one
    (1.0, 0.0, 1.0),      # the learned-model shape: the discrepancy is gone
    (9.0, 0.25, 0.5),
])
def test_components_recover_known_variances(v_scene, v_block, v_within):
    y, scene, blk = synthetic(v_scene, v_block, v_within, seed=1)
    out = db.nested_components(y, scene, blk)
    # 45 scenes is a small sample for the top level, so it gets the loosest tolerance;
    # the block and within levels have 450 and 4500 units behind them.
    assert out["within"] == pytest.approx(v_within, rel=0.10)
    assert out["block"] == pytest.approx(v_block, abs=0.10 + 0.15 * v_block)
    assert out["scene"] == pytest.approx(v_scene, abs=0.35 + 0.35 * v_scene)


def test_the_naive_estimator_would_have_been_biased():
    """Why this is an ANOVA and not `np.var` of the block means.

    With ten replicates per block, the variance of the block means overstates the block
    component by about `v_within / 10` -- which is the entire signal when the block
    component is near zero, exactly the case for the learned models.
    """
    y, scene, blk = synthetic(0.0, 0.0, 1.0, seed=2)
    naive = float(np.var([y[blk == b].mean() for b in np.unique(blk)]))
    out = db.nested_components(y, scene, blk)
    assert naive > 0.05, "the naive estimator should be visibly positive here"
    assert abs(out["block"]) < 0.02, "the ANOVA estimator should be near zero"


def test_a_pure_discrepancy_signal_is_attributed_to_the_block_level():
    y, scene, blk = synthetic(0.0, 4.0, 0.01, seed=3)
    out = db.nested_components(y, scene, blk)
    share = out["block"] / (out["block"] + out["within"])
    assert share > 0.99


def test_unbalanced_blocks_do_not_bias_the_estimate():
    """Soundings are dropped by the fill-value cleaning, so blocks are not all size 10."""
    y, scene, blk = synthetic(2.0, 1.0, 1.0, seed=4)
    rng = np.random.default_rng(5)
    keep = rng.random(len(y)) > 0.15
    out = db.nested_components(y[keep], scene[keep], blk[keep])
    assert out["within"] == pytest.approx(1.0, rel=0.10)
    assert out["block"] == pytest.approx(1.0, abs=0.25)


def test_degenerate_designs_return_nan_rather_than_dividing_by_zero():
    y = np.arange(10.0)
    one_scene = np.array(["a"] * 10)
    one_block = np.array(["a#0"] * 10)
    out = db.nested_components(y, one_scene, one_block)
    assert np.isnan(out["within"]) and np.isnan(out["block"])


# ------------------------------------------------------------ the labels --
class _FakeArt:
    """The two artifact fields `block_labels` reads, and nothing else."""

    def __init__(self, ids, rep):
        self._d = {"ids": np.array(ids), "rep": np.array(rep)}

    def has(self, k):
        return k in self._d

    def __getitem__(self, k):
        return self._d[k]


def test_block_index_is_ten_replicates_wide():
    ids = [f"clst0001_ref0046_rep{r:04d}" for r in range(1, 101)]
    scene, blk = db.block_labels(_FakeArt(ids, list(range(1, 101))), block=10)
    assert len(set(scene)) == 1
    assert len(set(blk)) == 10
    assert blk[0] == blk[9] and blk[9] != blk[10], "the boundary is between rep 10 and 11"


def test_blocks_never_straddle_scenes():
    ids = ([f"clst0001_ref0001_rep{r:04d}" for r in range(1, 101)]
           + [f"clst0001_ref0002_rep{r:04d}" for r in range(1, 101)])
    rep = list(range(1, 101)) * 2
    scene, blk = db.block_labels(_FakeArt(ids, rep), block=10)
    for b in np.unique(blk):
        assert len(set(scene[blk == b])) == 1
    assert len(np.unique(blk)) == 20


def test_missing_replicate_index_is_reported_not_guessed():
    ids = [f"clst0001_ref0001_rep{r:04d}" for r in range(1, 11)]
    art = _FakeArt(ids, list(range(1, 11)))
    del art._d["rep"]
    scene, blk = db.block_labels(art)
    assert scene is None and blk is None


# ------------------------------------------------------------- combining --
def test_multivariate_components_sum_over_dimensions():
    y1, scene, blk = synthetic(1.0, 1.0, 1.0, seed=6)
    y2 = synthetic(1.0, 1.0, 1.0, seed=7)[0]
    d1 = db.decompose(y1[:, None], scene, blk)
    both = db.decompose(np.stack([y1, y2], axis=1), scene, blk)
    assert both["within"] > d1["within"]
    assert both["total"] == pytest.approx(both["scene"] + both["block"] + both["within"])
    assert both["dims"] == 2

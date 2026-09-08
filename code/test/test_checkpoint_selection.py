"""Which checkpoint `require()` picks when several candidate paths exist.

`train_profiles.py` writes `net_profiles.pt` twice -- unconditionally into
`weights_dir/global_best/`, and into `weights_dir/` itself only when the run beats the
val_rmse recorded under `log/best_runs/`. Both can therefore be present, and they are not
guaranteed to be the same model. Taking the first path that merely *exists* is how
`eval_last1` died in `load_state_dict` with a 20-output head against a 1-output model,
seconds after the datasets were built, with a torch shape error as the only diagnostic.

LAYOUT already records how many outputs each job has, so these tests pin down that it is
used: prefer a candidate that fits, and when none fits, say which widths were found rather
than leaving it to a stack trace.
"""

import logging

import pytest
import torch

from evals import checkpoints as ck


def _head(out_dim: int) -> dict:
    """A state dict shaped like FullRegressorProfiles with `out_dim` outputs."""
    return {
        "backbone.0.weight": torch.zeros(320, 64),
        "predictor.4.weight": torch.zeros(out_dim, 320),
        "predictor.4.bias": torch.zeros(out_dim),
    }


@pytest.fixture
def last1_tree(tmp_path):
    d = tmp_path / "full_regressor_profiles_last1"
    (d / "global_best").mkdir(parents=True)
    return tmp_path, d


def test_prefers_the_candidate_whose_width_matches(last1_tree, caplog):
    """The eval_last1 failure: the first candidate exists but is the wrong model."""
    root, d = last1_tree
    torch.save(_head(20), d / "net_profiles.pt")
    torch.save(_head(1), d / "global_best" / "net_profiles.pt")

    with caplog.at_level(logging.WARNING):
        picked = ck.require(root, "det_last1")

    assert picked == d / "global_best" / "net_profiles.pt"
    assert "out_dim=20" in caplog.text


def test_first_candidate_wins_when_it_fits(last1_tree, caplog):
    """The ordinary case must not be perturbed, and must not warn."""
    root, d = last1_tree
    torch.save(_head(1), d / "net_profiles.pt")
    torch.save(_head(1), d / "global_best" / "net_profiles.pt")

    with caplog.at_level(logging.WARNING):
        picked = ck.require(root, "det_last1")

    assert picked == d / "net_profiles.pt"
    assert caplog.text == ""


def test_no_candidate_fits_warns_with_the_widths(last1_tree, caplog):
    """Both wrong means the training job is at fault; the log has to say so."""
    root, d = last1_tree
    torch.save(_head(20), d / "net_profiles.pt")
    torch.save(_head(20), d / "global_best" / "net_profiles.pt")

    with caplog.at_level(logging.WARNING):
        picked = ck.require(root, "det_last1")

    assert picked == d / "net_profiles.pt"
    assert "no candidate checkpoint has out_dim=1" in caplog.text
    assert "net_profiles.pt=20" in caplog.text


def test_single_candidate_is_returned_unread(last1_tree):
    """One path present: unchanged behaviour, and no checkpoint is loaded to decide."""
    root, d = last1_tree
    torch.save(_head(20), d / "net_profiles.pt")
    assert ck.require(root, "det_last1") == d / "net_profiles.pt"


def test_missing_everything_still_raises(last1_tree):
    root, _ = last1_tree
    with pytest.raises(FileNotFoundError, match="no checkpoint for 'det_last1'"):
        ck.require(root, "det_last1")


def test_unreadable_candidate_does_not_stop_the_search(last1_tree):
    """A payload _ckpt_out_dim cannot parse is skipped, not raised on."""
    root, d = last1_tree
    (d / "net_profiles.pt").write_bytes(b"not a torch file")
    torch.save(_head(1), d / "global_best" / "net_profiles.pt")
    assert ck.require(root, "det_last1") == d / "global_best" / "net_profiles.pt"

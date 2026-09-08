"""Every generated config family must differ from its `*_cov_grouped` twin only in the split.

There are two families, generated the same way and for the same kind of reason:

  `*_cov_replicate`  the rerun. It exists to put two columns side by side whose only
                     difference is the partition: reference-disjoint against
                     within-reference, same covariates, same architecture, same search
                     protocol. A reader can then attribute the difference to the split.

  `*_cov_fullfeb`    the temporal hold-out. It exists so that "trained on all of
                     February" is a claim about the training rows and nothing else -- the
                     same searched configuration, the same trainers, more scenes. If an
                     epoch count drifted here, the March and July numbers would no longer
                     be the group-split model with more data; they would be a different
                     model, and we would have no way to say which change did what.

  `*_cov_ext`        the February extension. 447 training reference soundings
                     instead of 361, with the same 45 validation and 45 test scenes on the
                     same realisations. The claim it has to support is "the same model, more
                     scenes and more replicates", so the longer epoch budget and the
                     annealing knobs are passed on the command line by chtc/ext.dag and are
                     deliberately NOT in the config -- if they were, this test would fail
                     and the run would no longer be a controlled comparison.

  `*_cov_extcap`     the same extension with the replicates capped at 100 per scene, so the
                     training rows are scene-proportional. Same
                     rule: the per-study budgets and the flow's reduced coupling stack are
                     in chtc/ext_cap.dag's VARS, never in the config.

Both claims are properties of a set of YAML files, and both are exactly the kind of
property that decays silently. So: load each pair, flatten both, and assert that every key
either matches or is one of the paths that is *supposed* to move. A new key on one side is
a failure too, because a config that gained an option its twin lacks is a difference in the
experiment.

Pure PyYAML: no torch, no Hydra composition (the `defaults:` block is compared as data,
which is enough -- the two data configs it names are checked as a pair of their own).
"""

import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "code"))

CONF = _ROOT / "code" / "shared_conf" / "experiment"
DATA = _ROOT / "code" / "shared_conf" / "data"

STUDIES = [
    "full_regressor",
    "full_regressor_profiles",
    "full_regressor_profiles_last4",
    "nf_full_regressor_nflows",
    "nf_profiles",
    "nf_profiles4",
]

# family suffix -> (the values allowed to differ, and the substitution that must turn the
# grouped one into this family's). Anything else differing is a finding.
FAMILIES = {
    "cov_replicate": [
        ("group_split_v1", "replicate_split_v1"),
        ("local_simulated_grouped_cov", "local_simulated_replicate_cov"),
    ],
    "cov_fullfeb": [
        ("group_split_v1", "full_feb_v1"),
        ("local_simulated_grouped_cov", "local_simulated_full_feb_cov"),
    ],
    "cov_ext": [
        ("group_split_v1", "ext_v1"),
        ("local_simulated_grouped_cov", "local_simulated_ext_cov"),
    ],
    "cov_extcap": [
        ("group_split_v1", "ext_cap_v1"),
        ("local_simulated_grouped_cov", "local_simulated_ext_cap_cov"),
    ],
}

# family suffix -> the data config it is the twin of local_simulated_grouped_cov in.
DATA_TWINS = {
    "cov_replicate": "local_simulated_replicate_cov",
    "cov_fullfeb": "local_simulated_full_feb_cov",
    "cov_ext": "local_simulated_ext_cov",
    "cov_extcap": "local_simulated_ext_cap_cov",
}

PAIRS = [(study, family) for family in FAMILIES for study in STUDIES]


def require(*paths):
    """Skip, rather than fail, when a config family is not part of this checkout.

    Not every distribution of this repository carries every twin -- a release built for
    one paper ships the families that paper reports and leaves the rest out. A family
    whose configs are absent has nothing to assert about; a family whose configs are
    present must still satisfy every assertion below. Failing instead would turn "this
    experiment is not in this tree" into "these configs disagree", which is a different
    and much more alarming sentence.
    """
    missing = [p for p in paths if not p.is_file()]
    if missing:
        pytest.skip("not in this checkout: "
                    + ", ".join(m.name for m in missing))


def flatten(node, prefix=""):
    """`{dotted.key: scalar}` for a nested mapping, lists rendered as their repr."""
    out = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = node
    return out


def load(path: Path) -> dict:
    return flatten(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def rewritten(value, subs):
    """The grouped value with every allowed substitution applied."""
    if not isinstance(value, str):
        return value
    for a, b in subs:
        value = value.replace(a, b)
    return value


@pytest.mark.parametrize("study,family", PAIRS)
def test_pair_differs_only_in_the_split(study, family):
    subs = FAMILIES[family]
    require(CONF / f"{study}_cov_grouped.yaml", CONF / f"{study}_{family}.yaml")
    grouped = load(CONF / f"{study}_cov_grouped.yaml")
    twin = load(CONF / f"{study}_{family}.yaml")

    assert set(grouped) == set(twin), (
        f"{study}/{family}: the two configs declare different keys: "
        f"only grouped {sorted(set(grouped) - set(twin))}, "
        f"only {family} {sorted(set(twin) - set(grouped))}"
    )
    for key, want in grouped.items():
        assert twin[key] == rewritten(want, subs), (
            f"{study}/{family}: '{key}' is {twin[key]!r} in the {family} config but "
            f"{want!r} in the grouped one. Either the substitution is missing or a "
            f"hyperparameter has drifted; the two are only a controlled comparison "
            f"while this passes."
        )


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_data_configs_differ_only_in_the_split(family):
    subs = FAMILIES[family]
    require(DATA / "local_simulated_grouped_cov.yaml", DATA / f"{DATA_TWINS[family]}.yaml")
    grouped = load(DATA / "local_simulated_grouped_cov.yaml")
    twin = load(DATA / f"{DATA_TWINS[family]}.yaml")
    assert set(grouped) == set(twin)
    for key, want in grouped.items():
        assert twin[key] == rewritten(want, subs), key


@pytest.mark.parametrize("study,family", PAIRS)
def test_twin_configs_point_at_their_own_split(study, family):
    """Belt and braces: no path may still name the group split.

    The pairwise test above would catch a missed substitution as a mismatch, but only if
    the grouped twin has that key. A path added to the twin alone -- by an edit, not by the
    generator -- would slip through as a new key on both sides if someone "fixed" the first
    failure by copying it across.
    """
    require(CONF / f"{study}_{family}.yaml")
    text = (CONF / f"{study}_{family}.yaml").read_text(encoding="utf-8")
    body = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    assert "group_split_v1" not in body, (
        f"{study}/{family}: a non-comment line still points at the group split"
    )


def test_the_splits_write_to_different_weights_directories():
    """The failure this prevents is a silent one: several models at one path.

    All three families train `full_regressor_scene_prior`. If the split did not change the
    weights root they would land on the same directory, and whichever archive was unpacked
    last would become "the" model -- with nothing in any log to say so.
    """
    paths = {
        "group_split_v1": load(CONF / "full_regressor_cov_grouped.yaml")["data.weight_path"],
        "replicate_split_v1": load(CONF / "full_regressor_cov_replicate.yaml")["data.weight_path"],
        "full_feb_v1": load(CONF / "full_regressor_cov_fullfeb.yaml")["data.weight_path"],
    }
    assert len(set(paths.values())) == len(paths), paths
    for split, path in paths.items():
        assert split in path, f"{path} does not name {split}"

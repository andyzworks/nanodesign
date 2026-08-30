from copy import deepcopy

import pytest

from nanodesign.v0.config import ConfigError, load_config, validate_v0_config


def test_default_config_is_spec_valid_but_explicitly_not_science_ready():
    report = validate_v0_config(load_config("configs/v0.yaml"))
    assert report.valid
    assert not report.ready
    assert "data.antibody_cdr.cdr_design" in report.blockers
    assert "data.rna_aptamer.usable_complex_inventory.path" in report.blockers
    assert "evaluation.rna_aptamer.rna_structure_predictor" in report.blockers
    with pytest.raises(ConfigError, match="unresolved v0 decisions"):
        report.require_ready()


def test_config_rejects_task_or_source_drift():
    config = load_config("configs/v0.yaml")
    changed = deepcopy(config)
    changed["tasks"] = ["protein_binder", "antibody_cdr", "rna"]
    with pytest.raises(ConfigError, match="tasks must be exactly"):
        validate_v0_config(changed)
    changed = deepcopy(config)
    changed["data"]["rna_aptamer"]["auxiliary_sources"] = [
        "pdb_rna_target_complex"
    ]
    with pytest.raises(ConfigError, match="RNAsolo2"):
        validate_v0_config(changed)


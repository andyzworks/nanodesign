from copy import deepcopy

import pytest

from nanodesign.v0.config import (
    ConfigError,
    load_config,
    validate_resolved_assets,
    validate_v0_config,
)


def test_default_config_is_fully_resolved_and_asset_pinned():
    config = load_config("configs/v0.yaml")
    report = validate_v0_config(config)
    assert report.valid and report.ready and not report.blockers
    inventory = validate_resolved_assets(config, config_directory="configs")
    assert inventory["usable_complexes"] == 2016


def test_config_rejects_task_source_or_official_model_drift():
    config = load_config("configs/v0.yaml")
    changed = deepcopy(config)
    changed["tasks"] = ["protein_binder", "antibody_cdr", "rna"]
    with pytest.raises(ConfigError, match="tasks must be exactly"):
        validate_v0_config(changed)
    changed = deepcopy(config)
    changed["data"]["rna_aptamer"]["auxiliary_sources"] = ["pdb_rna_target_complex"]
    with pytest.raises(ConfigError, match="RNAsolo2"):
        validate_v0_config(changed)
    changed = deepcopy(config)
    changed["model"]["foundry_commit"] = "0" * 40
    with pytest.raises(ConfigError, match="foundry_commit"):
        validate_v0_config(changed)


def test_config_rejects_sequence_leaking_atom_slots():
    config = load_config("configs/v0.yaml")
    config["model"]["atom_slot_schema"] = "native-residue-sidechains"
    with pytest.raises(ConfigError, match="hide design"):
        validate_v0_config(config)

import json
from copy import deepcopy

import pytest

from nanodesign.v0.config import (
    ConfigError,
    load_config,
    validate_resolved_assets,
    validate_v0_config,
)
from nanodesign.v0.constants import DataSource
from nanodesign.v0.data.inventory import (
    RnaComplexInventoryRecord,
    audit_rna_complex_inventory,
)


def test_default_config_is_spec_valid_but_explicitly_not_science_ready():
    report = validate_v0_config(load_config("configs/v0.yaml"))
    assert report.valid
    assert not report.ready
    assert "data.protein_binder.source" in report.blockers
    assert "data.antibody_cdr.cdr_design" in report.blockers
    assert "data.rna_aptamer.usable_complex_inventory.path" in report.blockers
    assert "model.atom_slot_schema" in report.blockers
    assert "model.capacity_benchmark" in report.blockers
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
    changed["data"]["rna_aptamer"]["auxiliary_sources"] = ["pdb_rna_target_complex"]
    with pytest.raises(ConfigError, match="RNAsolo2"):
        validate_v0_config(changed)


def test_resolved_config_verifies_frozen_rna_inventory(tmp_path):
    records = [
        RnaComplexInventoryRecord(
            source=source,
            source_version="frozen-rna-version",
            candidate_complexes=10,
            usable_complexes=7,
            exclusive_rejection_counts={"quality_filter": 3},
            selection_protocol="frozen protocol",
        )
        for source in (
            DataSource.RIBOCENTRE_APTAMER,
            DataSource.PDB_RNA_TARGET_COMPLEX,
        )
    ]
    inventory_path = tmp_path / "rna-inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": "nanodesign.v0",
                "records": [record.to_dict() for record in records],
            }
        ),
        encoding="utf-8",
    )
    inventory_report = audit_rna_complex_inventory(records)

    config = load_config("configs/v0.yaml")
    config["data"]["protein_binder"].update(
        {
            "source": "ppiref50k",
            "version": "frozen-version",
            "split_method": "cluster-disjoint-v1",
            "redundancy_filter": "frozen-filter",
            "chain_assignment": "frozen-rule",
        }
    )
    config["data"]["antibody_cdr"].update(
        {
            "version": "frozen-version",
            "cdr_design": "h3_only",
            "split_method": "cluster-disjoint-v1",
            "quality_filter": "frozen-filter",
        }
    )
    rna = config["data"]["rna_aptamer"]
    rna["versions"] = {
        "ribocentre_aptamer": "frozen-rna-version",
        "pdb_rna_target_complex": "frozen-rna-version",
        "rnasolo2": "frozen-prior-version",
    }
    rna["usable_complex_inventory"] = {
        "path": inventory_path.name,
        "sha256": inventory_report["sha256"],
    }
    rna["split_method"] = "cluster-disjoint-v1"
    rna["quality_filter"] = "frozen-filter"
    config["model"]["capacity_benchmark"] = "frozen-gpu-profile"
    config["model"]["atom_slot_schema"] = "sequence-independent-v1"
    for task_config in config["evaluation"].values():
        for key in task_config:
            task_config[key] = "frozen-protocol"

    assert validate_v0_config(config).ready
    report = validate_resolved_assets(config, config_directory=tmp_path)
    assert report["usable_complexes"] == 14

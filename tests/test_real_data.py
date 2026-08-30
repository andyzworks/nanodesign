import json
from pathlib import Path

import pytest

from nanodesign.v0.data.real import load_catalog_example

EXPECTED = {
    "protein_binder": {"train": 40883, "validation": 5110, "test": 5110},
    "antibody_h3": {"train": 3878, "validation": 438, "test": 984},
    "rna_aptamer_binding": {"train": 2117, "validation": 83, "test": 88},
    "rna_structure_prior_auxiliary": {"train": 419, "validation": 234, "test": 229},
}


def test_frozen_real_split_counts_when_data_snapshot_is_present():
    root = Path(__file__).resolve().parents[1]
    split_root = root / "data/processed/v0/splits"
    if not split_root.is_dir():
        pytest.skip("real data snapshot is not part of a source-only checkout")
    for task, counts in EXPECTED.items():
        for split, expected in counts.items():
            path = split_root / task / f"{split}.jsonl"
            assert sum(1 for line in path.open(encoding="utf-8") if line.strip()) == expected


def test_each_binding_task_loads_real_coordinates_when_snapshot_is_present():
    root = Path(__file__).resolve().parents[1]
    catalogs = {
        "protein_binder": root / "data/processed/v0/catalogs/ppiref50k.jsonl",
        "antibody_h3": root / "data/processed/v0/catalogs/sabdab2.jsonl",
        "rna_aptamer": root / "data/processed/v0/catalogs/pdb_rna_target.jsonl",
    }
    if not all(path.is_file() for path in catalogs.values()):
        pytest.skip("real data snapshot is not part of a source-only checkout")
    for task, path in catalogs.items():
        rows = (json.loads(line) for line in path.open(encoding="utf-8") if line.strip())
        row = min(rows, key=lambda value: sum(c["resolved_residues"] for c in value["chains"]))
        example = load_catalog_example(root, row)
        assert example.atom_positions.shape[0] > 0, task
        assert example.design_mask.sum() > 0, task

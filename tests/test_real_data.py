import hashlib
import json
from pathlib import Path

import pytest

from nanodesign.v0.data.real import load_catalog_example, load_foundry_training_example
from nanodesign.v0.distributed import catalog_model_atom_count

EXPECTED = {
    "protein_binder": {"train": 40883, "validation": 5110, "test": 5110},
    "antibody_h3": {"train": 3878, "validation": 438, "test": 984},
    "rna_binding": {"train": 2117, "validation": 83, "test": 88},
    "rna_structure_prior_auxiliary": {"train": 419, "validation": 234, "test": 229},
}


def test_frozen_real_split_counts_when_data_snapshot_is_present():
    root = Path(__file__).resolve().parents[1]
    split_root = root / "data/processed/v0/splits"
    if not split_root.is_dir():
        pytest.skip("real data snapshot is not part of a source-only checkout")
    frozen = json.loads((root / "docs/data_v0_stats.json").read_text(encoding="utf-8"))
    for task, counts in EXPECTED.items():
        for split, expected in counts.items():
            path = split_root / task / f"{split}.jsonl"
            assert sum(1 for line in path.open(encoding="utf-8") if line.strip()) == expected
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest()
                == frozen["manifest_files"][f"{task}/{split}"]["sha256"]
            )


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


def test_catalog_size_identity_matches_real_model_ready_atoms_when_snapshot_is_present():
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "data/processed/v0/splits/protein_binder/train.jsonl",
        root / "data/processed/v0/splits/antibody_h3/train.jsonl",
        root / "data/processed/v0/splits/rna_binding/train.jsonl",
    ]
    if not all(path.is_file() for path in paths):
        pytest.skip("real data snapshot is not part of a source-only checkout")
    for path in paths:
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        row = min(rows, key=lambda value: sum(c["resolved_residues"] for c in value["chains"]))
        batch = load_foundry_training_example(root, row, noise_level=0.5, max_context_tokens=384)
        assert batch["f"]["atom_to_token_map"].numel() == catalog_model_atom_count(
            row, max_context_tokens=384
        )


def test_rna_sources_have_frozen_non_overclaiming_semantics_when_snapshot_is_present():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "ribocentre.jsonl": "true_aptamer",
        "pdb_rna_target.jsonl": "general_rna_protein_interaction",
        "rnasolo2.jsonl": "rna_structural_prior",
    }
    catalog_root = root / "data/processed/v0/catalogs"
    if not catalog_root.is_dir():
        pytest.skip("real data snapshot is not part of a source-only checkout")
    for filename, semantics in expected.items():
        with (catalog_root / filename).open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        assert all(row["data_semantics"] == semantics for row in rows)
        rna_roles = {
            chain["role"] for row in rows for chain in row["chains"] if chain["role"] != "target"
        }
        expected_roles = {
            "true_aptamer": {"rna_aptamer"},
            "general_rna_protein_interaction": {"rna_design_region"},
            "rna_structural_prior": {"rna_structure_prior"},
        }
        assert rna_roles == expected_roles[semantics]


def test_antibody_context_crop_preserves_the_complete_fixed_framework_when_present():
    root = Path(__file__).resolve().parents[1]
    split = root / "data/processed/v0/splits/antibody_h3/validation.jsonl"
    if not split.is_file():
        pytest.skip("real data snapshot is not part of a source-only checkout")
    row = json.loads(next(line for line in split.open(encoding="utf-8") if line.strip()))
    heavy = next(chain for chain in row["chains"] if chain["role"] == "antibody_framework+cdr_h3")
    light = [chain for chain in row["chains"] if chain["role"] == "antibody_framework"]
    design_count = len(heavy["design_residue_keys"])
    framework_count = (
        heavy["resolved_residues"]
        - design_count
        + sum(chain["resolved_residues"] for chain in light)
    )
    assert framework_count > 16  # Exercise the protected-over-budget branch.

    batch = load_foundry_training_example(root, row, max_context_tokens=16)
    assert int(batch["ground_truth_sequence_mask"].sum()) == design_count
    assert int(batch["ground_truth_sequence"].shape[0]) == framework_count + design_count


def test_microheterogeneous_rna_residue_uses_catalog_sequence_and_accepted_altloc():
    root = Path(__file__).resolve().parents[1]
    split = root / "data/processed/v0/splits/rna_binding/train.jsonl"
    if not split.is_file():
        pytest.skip("real data snapshot is not part of a source-only checkout")
    row = next(
        json.loads(line)
        for line in split.open(encoding="utf-8")
        if '"sample_id":"pdb_rna_target:9kfx:0"' in line
    )
    rna = next(chain for chain in row["chains"] if chain["role"] == "rna_design_region")

    example = load_catalog_example(root, row)
    batch = load_foundry_training_example(root, row, noise_level=0.5, max_context_tokens=384)

    assert int(example.design_mask.sum()) == rna["resolved_residues"]
    assert int(batch["ground_truth_sequence_mask"].sum()) == rna["resolved_residues"]
    assert bool(batch["ground_truth_atom_mask"].any())

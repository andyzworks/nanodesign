from copy import deepcopy

import gemmi
import pytest
import torch

from nanodesign.v0.config import load_config
from nanodesign.v0.model import NanoDesignTinyConfig
from nanodesign.v0.training import (
    TrainingConfig,
    _assert_model_matches_resolved_config,
    write_generation_structure,
)


def test_training_config_rejects_invalid_optimizer_values():
    with pytest.raises(ValueError, match="invalid"):
        TrainingConfig(learning_rate=0)


def test_runtime_model_config_must_match_resolved_train_and_inference_config():
    class ModelStub:
        config = NanoDesignTinyConfig()

    resolved = load_config("configs/v0.yaml")
    _assert_model_matches_resolved_config(ModelStub(), resolved)
    changed = deepcopy(resolved)
    changed["model"]["sampling_steps"] = 2
    with pytest.raises(ValueError, match="train/inference config"):
        _assert_model_matches_resolved_config(ModelStub(), changed)


def test_generation_export_enforces_polymer_alphabets_and_writes_predicted_atoms(tmp_path):
    logits = torch.zeros(2, 32)
    logits[0, 28] = 100  # DNA is invalid for protein and must not be selected.
    logits[0, 1] = 10  # ARG
    logits[1, 0] = 100  # amino acid is invalid for RNA and must not be selected.
    logits[1, 23] = 10  # G
    output = {
        "X_L": torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]]),
        "sequence_logits_I": logits,
    }
    batch = {
        "sample_id": "export",
        "f": {"is_protein": torch.tensor([True, False]), "is_rna": torch.tensor([False, True])},
        "ground_truth_sequence": torch.nn.functional.one_hot(
            torch.tensor([0, 21]), num_classes=32
        ).float(),
        "ground_truth_sequence_mask": torch.tensor([True, True]),
        "output_metadata": {
            "atom_names": ["N", "CA", "C4'", "C1'"],
            "atom_to_token": [0, 0, 1, 1],
            "atom_output_mask": [True, True, True, True],
            "token_chain_names": ["H", "R"],
            "token_residue_keys": [(105, ""), (1, "")],
        },
    }
    path = tmp_path / "generated.pdb"
    sequences = write_generation_structure(output, batch, path)

    model = gemmi.read_structure(str(path))[0]
    assert sequences == {"H": "R", "R": "G"}
    assert model["H"][0].name == "ARG"
    assert model["R"][0].name == "G"
    assert [atom.name for atom in model["H"][0]] == ["N", "CA"]

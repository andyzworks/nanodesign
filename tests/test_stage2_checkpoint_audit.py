import pytest
import torch

from scripts.audit_stage2_checkpoint import (
    _design_coordinate_metrics,
    _detach_fixed_context_geometry,
    _prediction_metrics,
    _sequence_collapse_metrics,
    _shuffle_fixed_context_sequence,
)


def test_design_coordinate_metrics_use_only_resolved_design_atoms():
    ground_truth = torch.zeros(1, 4, 3)
    predicted = ground_truth.clone()
    predicted[0, 1, 0] = 3.0
    predicted[0, 2, 0] = 100.0
    batch = {
        "ground_truth_positions": ground_truth,
        "ground_truth_atom_mask": torch.tensor([True, True, True, False]),
        "ground_truth_sequence_mask": torch.tensor([False, True, False]),
        "f": {"atom_to_token_map": torch.tensor([0, 1, 2, 1])},
    }
    rmsd, coordinates = _design_coordinate_metrics({"X_L": predicted}, batch)
    assert rmsd == pytest.approx(3.0)
    assert torch.equal(coordinates, predicted[0, [1]])


def test_context_shuffle_changes_only_fixed_sequence_position_association():
    restype = torch.eye(4)
    batch = {
        "ground_truth_sequence_mask": torch.tensor([False, True, False, False]),
        "f": {"restype": restype},
    }
    shuffled = _shuffle_fixed_context_sequence(batch)
    fixed = torch.tensor([True, False, True, True])
    assert torch.equal(shuffled["f"]["restype"][~fixed], restype[~fixed])
    assert torch.equal(
        shuffled["f"]["restype"][fixed], torch.roll(restype[fixed], shifts=1, dims=0)
    )
    assert torch.equal(batch["f"]["restype"], restype)


def test_context_detachment_moves_only_fixed_coordinates():
    noisy = torch.zeros(1, 4, 3)
    motif = torch.zeros(4, 3)
    fixed = torch.tensor([True, False, True, False])
    batch = {
        "X_noisy_L": noisy,
        "f": {
            "motif_pos": motif,
            "is_motif_atom_with_fixed_coord": fixed,
        },
    }
    detached = _detach_fixed_context_geometry(batch)
    expected = torch.tensor([100.0, 0.0, 0.0])
    assert torch.equal(detached["X_noisy_L"][0, fixed], expected.expand(2, 3))
    assert torch.equal(detached["f"]["motif_pos"][fixed], expected.expand(2, 3))
    assert torch.equal(detached["X_noisy_L"][0, ~fixed], noisy[0, ~fixed])
    assert torch.equal(detached["f"]["motif_pos"][~fixed], motif[~fixed])
    assert torch.equal(batch["X_noisy_L"], noisy)
    assert torch.equal(batch["f"]["motif_pos"], motif)


def test_prediction_and_collapse_metrics_have_expected_extremes():
    targets = torch.tensor([0, 1, 2])
    output = {
        "sequence_logits_I": 10 * torch.eye(3).unsqueeze(0),
        "sequence_indices_I": targets.unsqueeze(0),
    }
    batch = {
        "ground_truth_sequence_mask": torch.ones(3, dtype=torch.bool),
        "ground_truth_sequence": torch.eye(3),
    }
    recovery, cross_entropy, predictions = _prediction_metrics(output, batch)
    assert recovery == 1.0
    assert cross_entropy < 1e-3
    assert torch.equal(predictions, targets)

    collapsed = _sequence_collapse_metrics(torch.tensor([2, 2, 2, 2]))
    diverse = _sequence_collapse_metrics(torch.tensor([0, 1, 2, 3]))
    assert collapsed["dominant_token_fraction"] == 1.0
    assert collapsed["normalized_token_entropy"] == 0.0
    assert diverse["dominant_token_fraction"] == pytest.approx(0.25)
    assert diverse["normalized_token_entropy"] == pytest.approx(1.0)

import pytest
import torch

from scripts.audit_stage2_checkpoint import (
    _prediction_metrics,
    _sequence_collapse_metrics,
    _shuffle_fixed_context_sequence,
)


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

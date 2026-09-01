import random
from copy import deepcopy

import gemmi
import numpy as np
import pytest
import torch

from nanodesign.v0.config import load_config
from nanodesign.v0.model import NanoDesignTinyConfig
from nanodesign.v0.training import (
    ExponentialMovingAverage,
    TrainingConfig,
    _assert_model_matches_resolved_config,
    _design_normalized_sequence_mask,
    _official_generation_coordinates,
    build_optimizer,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    write_generation_structure,
)


def test_training_config_rejects_invalid_optimizer_values():
    with pytest.raises(ValueError, match="invalid"):
        TrainingConfig(learning_rate=0)
    with pytest.raises(ValueError, match="invalid"):
        TrainingConfig(gradient_clip=0)


def test_sequence_mask_is_normalized_over_design_tokens_only():
    weights = _design_normalized_sequence_mask(torch.tensor([False, True, False, True]))
    assert torch.equal(weights, torch.tensor([0.0, 2.0, 0.0, 2.0]))
    token_losses = torch.tensor([100.0, 1.0, 100.0, 3.0])
    assert torch.isclose((weights * token_losses).mean(), torch.tensor(2.0))
    with pytest.raises(ValueError, match="at least one"):
        _design_normalized_sequence_mask(torch.zeros(3, dtype=torch.bool))


def test_regular_generation_centers_fixed_context_and_removes_native_design_coordinates():
    batch = {
        "coord_atom_lvl_to_be_noised": torch.tensor(
            [[[10.0, 2.0, 0.0], [14.0, 2.0, 0.0], [99.0, 99.0, 99.0]]]
        ),
        "f": {
            "is_motif_atom_with_fixed_coord": torch.tensor([True, True, False]),
        },
    }
    coordinates = _official_generation_coordinates(batch)
    assert torch.equal(
        coordinates,
        torch.tensor([[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
    )
    # The cached/native batch is immutable so validation and future noising remain exact.
    assert torch.equal(
        batch["coord_atom_lvl_to_be_noised"][0, 2], torch.tensor([99.0, 99.0, 99.0])
    )


def test_runtime_model_config_must_match_resolved_train_and_inference_config():
    class ModelStub:
        config = NanoDesignTinyConfig()

    resolved = load_config("configs/v0.yaml")
    _assert_model_matches_resolved_config(ModelStub(), resolved)
    changed = deepcopy(resolved)
    changed["model"]["sampling_steps"] = 2
    with pytest.raises(ValueError, match="train/inference config"):
        _assert_model_matches_resolved_config(ModelStub(), changed)


class _CheckpointModel(torch.nn.Module):
    config = NanoDesignTinyConfig()

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def validate_parameter_budget(self):
        return None


def test_ema_updates_and_can_supply_inference_weights():
    model = _CheckpointModel()
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    ema = ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(2.0)
    ema.update(model)
    for name, value in model.state_dict().items():
        assert torch.allclose(ema.shadow[name], initial[name] * 0.5 + value * 0.5)
    online = {name: value.clone() for name, value in model.state_dict().items()}
    with ema.average_parameters(model):
        for name, value in model.state_dict().items():
            assert torch.equal(value, ema.shadow[name])
    _assert_nested_equal(online, model.state_dict())


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _stochastic_training_step(model, optimizer, step, history):
    optimizer.zero_grad(set_to_none=True)
    multiplier = random.random() + float(np.random.random()) + float(torch.rand(()))
    inputs = torch.randn(4, 3)
    loss = (model.linear(inputs) * multiplier).square().mean()
    loss.backward()
    optimizer.step()
    history.append({"step": step, "loss": float(loss)})


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_checkpoint_resume_matches_uninterrupted_stochastic_training(tmp_path):
    resolved = load_config("configs/v0.yaml")
    manifest_sha = "a" * 64
    total_steps, interrupted_at = 5, 2

    _seed_all(19)
    continuous_model = _CheckpointModel()
    continuous_optimizer = build_optimizer(continuous_model)
    continuous_history = []
    for step in range(1, total_steps + 1):
        _stochastic_training_step(continuous_model, continuous_optimizer, step, continuous_history)
    continuous_next_rng = (random.random(), float(np.random.random()), torch.rand(()))

    _seed_all(19)
    interrupted_model = _CheckpointModel()
    interrupted_optimizer = build_optimizer(interrupted_model)
    interrupted_history = []
    for step in range(1, interrupted_at + 1):
        _stochastic_training_step(
            interrupted_model, interrupted_optimizer, step, interrupted_history
        )
    checkpoint_path = tmp_path / "periodic.pt"
    save_checkpoint(
        checkpoint_path,
        model=interrupted_model,
        optimizer=interrupted_optimizer,
        step=interrupted_at,
        samples_seen=interrupted_at,
        task_cursors={"protein_binder": 1, "antibody_h3": 1, "rna": 0},
        task_steps={"protein_binder": 1, "antibody_h3": 1, "rna": 0},
        history=interrupted_history,
        training_run_config={"seed": 19},
        validation_before={"protein_binder": {"loss": 1.0}},
        manifest_sha256=manifest_sha,
        resolved_config=resolved,
    )

    _seed_all(999)  # Construction and unrelated work must be undone by restore_rng.
    resumed_model = _CheckpointModel()
    resumed_optimizer = build_optimizer(resumed_model)
    loaded = load_checkpoint(
        checkpoint_path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        expected_manifest_sha256=manifest_sha,
        restore_rng=True,
    )
    resumed_history = list(loaded["history"])
    for step in range(interrupted_at + 1, total_steps + 1):
        _stochastic_training_step(resumed_model, resumed_optimizer, step, resumed_history)
    resumed_next_rng = (random.random(), float(np.random.random()), torch.rand(()))

    _assert_nested_equal(continuous_model.state_dict(), resumed_model.state_dict())
    _assert_nested_equal(continuous_optimizer.state_dict(), resumed_optimizer.state_dict())
    assert continuous_history == resumed_history
    _assert_nested_equal(continuous_next_rng, resumed_next_rng)
    assert loaded["samples_seen"] == interrupted_at
    assert loaded["task_cursors"] == {"protein_binder": 1, "antibody_h3": 1, "rna": 0}


def test_checkpoint_restores_rank_specific_rng_and_has_unwrapped_model_keys(tmp_path):
    resolved = load_config("configs/v0.yaml")
    _seed_all(3)
    model = _CheckpointModel()
    optimizer = build_optimizer(model)
    rank_states = []
    expected_next = []
    for seed in (101, 102):
        _seed_all(seed)
        state = capture_rng_state()
        rank_states.append(state)
        expected_next.append((random.random(), float(np.random.random()), torch.rand(())))
        restore_rng_state(state)
    path = tmp_path / "distributed.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=1,
        samples_seen=2,
        rng_states_by_rank=rank_states,
        manifest_sha256="b" * 64,
        resolved_config=resolved,
    )
    restored = _CheckpointModel()
    load_checkpoint(
        path,
        model=restored,
        expected_manifest_sha256="b" * 64,
        restore_rng=True,
        rng_rank=1,
    )
    observed = (random.random(), float(np.random.random()), torch.rand(()))
    _assert_nested_equal(expected_next[1], observed)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert all(not key.startswith("module.") for key in payload["model"])


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

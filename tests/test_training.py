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
    _sequence_supervision_mask,
    build_learning_rate_scheduler,
    build_optimizer,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    validate_resume_training_run_config,
    write_generation_structure,
)


def test_training_config_rejects_invalid_optimizer_values():
    with pytest.raises(ValueError, match="invalid"):
        TrainingConfig(learning_rate=0)
    with pytest.raises(ValueError, match="invalid"):
        TrainingConfig(gradient_clip=0)
    with pytest.raises(ValueError, match="optimizer"):
        TrainingConfig(optimizer="sgd")
    with pytest.raises(ValueError, match="sequence supervision"):
        TrainingConfig(sequence_supervision="context")
    with pytest.raises(ValueError, match="at least one"):
        TrainingConfig(coordinate_loss_weight=0, sequence_loss_weight=0)


def test_build_optimizer_preserves_adamw_default_and_supports_official_adam_control():
    model = _CheckpointModel()
    assert isinstance(build_optimizer(model), torch.optim.AdamW)
    assert isinstance(build_optimizer(model, TrainingConfig(optimizer="adam")), torch.optim.Adam)


def test_resume_config_accepts_only_a_strict_completed_milestone_extension():
    checkpoint = {
        "seed": 17,
        "learning_rate": 5e-4,
        "milestone_samples": [300, 900, 3000, 9000],
    }
    extended = {**checkpoint, "milestone_samples": [300, 900, 3000, 9000, 18000]}
    validate_resume_training_run_config(checkpoint, checkpoint, samples_seen=7250)
    validate_resume_training_run_config(checkpoint, extended, samples_seen=9000)
    with pytest.raises(TypeError, match="must be a mapping"):
        validate_resume_training_run_config(None, extended, samples_seen=9000)
    with pytest.raises(ValueError, match="configuration mismatch"):
        validate_resume_training_run_config(checkpoint, extended, samples_seen=8500)
    with pytest.raises(ValueError, match="configuration mismatch"):
        validate_resume_training_run_config(
            checkpoint,
            {**extended, "learning_rate": 1e-3},
            samples_seen=9000,
        )
    with pytest.raises(ValueError, match="configuration mismatch"):
        validate_resume_training_run_config(
            checkpoint,
            {**checkpoint, "milestone_samples": [300, 3000, 9000, 18000]},
            samples_seen=9000,
        )


def test_pinned_af3_scheduler_uses_official_warmup_and_constant_path_is_unchanged():
    model = _CheckpointModel()
    optimizer = build_optimizer(model, TrainingConfig(learning_rate=1.8e-3))
    assert (
        build_learning_rate_scheduler(optimizer, schedule="constant", base_learning_rate=1.8e-3)
        is None
    )
    scheduler = build_learning_rate_scheduler(optimizer, schedule="af3", base_learning_rate=1.8e-3)
    assert scheduler is not None
    assert optimizer.param_groups[0]["lr"] == 0.0
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.8e-6)
    with pytest.raises(ValueError, match="constant or af3"):
        build_learning_rate_scheduler(optimizer, schedule="cosine", base_learning_rate=1.8e-3)


def test_sequence_mask_is_normalized_over_design_tokens_only():
    weights = _design_normalized_sequence_mask(torch.tensor([False, True, False, True]))
    assert torch.equal(weights, torch.tensor([0.0, 2.0, 0.0, 2.0]))
    token_losses = torch.tensor([100.0, 1.0, 100.0, 3.0])
    assert torch.isclose((weights * token_losses).mean(), torch.tensor(2.0))
    with pytest.raises(ValueError, match="at least one"):
        _design_normalized_sequence_mask(torch.zeros(3, dtype=torch.bool))


def test_sequence_supervision_can_match_official_all_valid_rule():
    sequence = torch.tensor([0, 20, 21, 25, 30, 31])
    design = torch.tensor([False, True, True, False, False, False])
    assert torch.equal(
        _sequence_supervision_mask(sequence, design, mode="all_valid"),
        torch.tensor([True, False, True, False, False, False]),
    )
    assert torch.equal(
        _sequence_supervision_mask(sequence, design, mode="design"),
        torch.tensor([0.0, 3.0, 3.0, 0.0, 0.0, 0.0]),
    )


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
    assert torch.equal(batch["coord_atom_lvl_to_be_noised"][0, 2], torch.tensor([99.0, 99.0, 99.0]))


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


def test_checkpoint_restores_learning_rate_scheduler_state(tmp_path):
    resolved = load_config("configs/v0.yaml")
    model = _CheckpointModel()
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    optimizer.step()
    scheduler.step()
    path = tmp_path / "scheduled.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        step=1,
        manifest_sha256="c" * 64,
        resolved_config=resolved,
    )

    restored_model = _CheckpointModel()
    restored_optimizer = build_optimizer(restored_model)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1, gamma=0.5)
    load_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        lr_scheduler=restored_scheduler,
        expected_manifest_sha256="c" * 64,
    )
    _assert_nested_equal(scheduler.state_dict(), restored_scheduler.state_dict())
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]

    with pytest.raises(ValueError, match="runtime does not"):
        load_checkpoint(
            path,
            model=_CheckpointModel(),
            optimizer=build_optimizer(_CheckpointModel()),
            expected_manifest_sha256="c" * 64,
        )


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

"""Minimal training/checkpoint infrastructure shared by every v0 task."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from nanodesign.v0.config import validate_v0_config
from nanodesign.v0.model import NanoDesignTiny
from nanodesign.v0.spec import SPEC_VERSION


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("invalid optimizer or gradient-clipping configuration")


def train_step(
    model: NanoDesignTiny,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, object],
    config: TrainingConfig | None = None,
) -> dict[str, float]:
    """Run one step with the public RFD3NA diffusion and sequence losses."""

    config = config or TrainingConfig()
    try:
        from rfd3na.metrics.losses import DiffusionLoss, SequenceLoss
    except ImportError as error:
        raise ImportError("training requires the project 'model' extra") from error
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    features = batch["f"]
    if not isinstance(features, Mapping):
        raise TypeError("batch.f must be an RFD3NA feature mapping")
    gt_positions = batch["ground_truth_positions"]
    gt_atom_mask = batch["ground_truth_atom_mask"]
    gt_sequence = batch["ground_truth_sequence"]
    gt_sequence_mask = batch["ground_truth_sequence_mask"]
    if not all(
        isinstance(value, torch.Tensor)
        for value in (gt_positions, gt_atom_mask, gt_sequence, gt_sequence_mask)
    ):
        raise TypeError("batch is missing tensor ground truth")
    loss_input = {
        "X_gt_L_in_input_frame": gt_positions,
        "crd_mask_L": gt_atom_mask,
        "is_original_unindexed_token": torch.zeros(
            gt_sequence.shape[0], dtype=torch.bool, device=gt_sequence.device
        ),
        "seq_token_lvl": gt_sequence.argmax(dim=-1),
        "sequence_valid_mask": gt_sequence_mask.float(),
    }
    coordinate_loss_module = DiffusionLoss(
        weight=4.0,
        sigma_data=16.0,
        lddt_weight=0.25,
        alpha_virtual_atom=1.0,
        alpha_polar_residues=1.0,
        alpha_ligand=10.0,
        lp_weight=0.0,
    )
    sequence_loss_module = SequenceLoss(weight=0.1, max_t=1.0)
    coordinate_loss, coordinate_metrics = coordinate_loss_module(batch, output, loss_input)
    sequence_loss, sequence_metrics = sequence_loss_module(batch, output, loss_input)
    loss = coordinate_loss + sequence_loss
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    optimizer.step()
    metrics = {
        "loss": loss,
        "coordinate_loss": coordinate_loss,
        "sequence_loss": sequence_loss,
        **coordinate_metrics,
        **sequence_metrics,
    }
    return {
        key: float(value.detach().item())
        for key, value in metrics.items()
        if isinstance(value, torch.Tensor) and value.numel() == 1
    } | {"gradient_norm": float(gradient_norm)}


@torch.no_grad()
def generate(model: NanoDesignTiny, batch: Mapping[str, object]) -> dict[str, torch.Tensor]:
    """Generate sequence and atom23 coordinates with RFD3NA's official EDM sampler."""

    coordinates = batch.get("coord_atom_lvl_to_be_noised")
    if not isinstance(coordinates, torch.Tensor):
        raise TypeError("batch is missing coord_atom_lvl_to_be_noised")
    model.eval()
    return model(batch, coord_atom_lvl_to_be_noised=coordinates)


def build_optimizer(
    model: NanoDesignTiny, config: TrainingConfig | None = None
) -> torch.optim.Optimizer:
    config = config or TrainingConfig()
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{name} must contain 64 hexadecimal characters")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must contain 64 hexadecimal characters") from error


def save_checkpoint(
    path: str | Path,
    *,
    model: NanoDesignTiny,
    optimizer: torch.optim.Optimizer,
    step: int,
    manifest_sha256: str,
    resolved_config: Mapping[str, object],
) -> None:
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    _require_sha256(manifest_sha256, "manifest_sha256")
    validate_v0_config(resolved_config).require_ready()
    model.validate_parameter_budget()
    config_json = json.dumps(resolved_config, sort_keys=True, separators=(",", ":"))
    state = {
        "schema_version": SPEC_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "manifest_sha256": manifest_sha256,
        "resolved_config": dict(resolved_config),
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "model_config": asdict(model.config),
        "parameter_count": model.parameter_count,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, destination)


def load_checkpoint(
    path: str | Path,
    *,
    model: NanoDesignTiny,
    optimizer: torch.optim.Optimizer | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != SPEC_VERSION:
        raise ValueError("checkpoint does not implement NanoDesign v0")
    if checkpoint.get("model_config") != asdict(model.config):
        raise ValueError("checkpoint model configuration mismatch")
    if int(checkpoint.get("parameter_count", -1)) != model.parameter_count:
        raise ValueError("checkpoint parameter count mismatch")
    resolved_config = checkpoint.get("resolved_config")
    if not isinstance(resolved_config, Mapping):
        raise TypeError("checkpoint is missing its resolved configuration")
    config_json = json.dumps(resolved_config, sort_keys=True, separators=(",", ":"))
    config_sha256 = hashlib.sha256(config_json.encode()).hexdigest()
    if checkpoint.get("config_sha256") != config_sha256:
        raise ValueError("checkpoint resolved configuration fingerprint mismatch")
    validate_v0_config(resolved_config).require_ready()
    if expected_manifest_sha256 is not None:
        _require_sha256(expected_manifest_sha256, "expected_manifest_sha256")
        if checkpoint.get("manifest_sha256") != expected_manifest_sha256:
            raise ValueError("checkpoint dataset manifest mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint

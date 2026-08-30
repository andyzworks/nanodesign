"""Minimal training/checkpoint infrastructure shared by every v0 task."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch

from nanodesign.v0.config import validate_v0_config
from nanodesign.v0.diffusion import UnifiedDiffusion
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
    diffusion: UnifiedDiffusion,
    optimizer: torch.optim.Optimizer,
    clean_batch: Mapping[str, torch.Tensor],
    config: TrainingConfig | None = None,
) -> dict[str, float]:
    config = config or TrainingConfig()
    model.train()
    batch = diffusion.corrupt(clean_batch)
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    losses = diffusion.loss(output, batch)
    losses["loss"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    optimizer.step()
    return {
        key: float(value.detach().item())
        for key, value in losses.items()
        if value.ndim == 0
    } | {"gradient_norm": float(gradient_norm)}


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
        raise ValueError("checkpoint is missing its resolved configuration")
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


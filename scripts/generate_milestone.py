#!/usr/bin/env python3
"""Generate the three frozen test designs from one NanoDesign milestone checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nanodesign.v0.config import load_config, validate_v0_config
from nanodesign.v0.data.real import load_foundry_training_example, load_split_catalog
from nanodesign.v0.model import NanoDesignTiny, NanoDesignTinyConfig
from nanodesign.v0.training import generate, load_checkpoint, write_generation_structure

SPLITS = {
    "protein_binder": "data/processed/v0/splits/protein_binder/test.jsonl",
    "antibody_h3": "data/processed/v0/splits/antibody_h3/test.jsonl",
    "rna": "data/processed/v0/splits/rna_binding/test.jsonl",
}


def _model_config(resolved: dict[str, Any]) -> NanoDesignTinyConfig:
    return NanoDesignTinyConfig.from_mapping(
        {key: resolved["model"][key] for key in NanoDesignTinyConfig.__dataclass_fields__}
    )


def _fixed_context_tokens(row: dict[str, Any]) -> int:
    design_tokens = 0
    for chain in row["chains"]:
        if chain["role"] in {"binder", "rna_aptamer", "rna_design_region"}:
            design_tokens += int(chain["resolved_residues"])
        elif chain["role"] == "antibody_framework+cdr_h3":
            design_tokens += len(chain["design_residue_keys"])
    return sum(int(chain["resolved_residues"]) for chain in row["chains"]) - design_tokens


def _generation_row(
    rows: list[dict[str, Any]], *, max_context_tokens: int
) -> tuple[dict[str, Any], bool]:
    """Match train_v0's frozen final-generation sample selection exactly."""

    complete = next((row for row in rows if _fixed_context_tokens(row) <= max_context_tokens), None)
    return (complete, True) if complete is not None else (rows[0], False)


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _batch(
    root: Path,
    row: dict[str, Any],
    *,
    device: torch.device,
    max_context_tokens: int,
) -> dict[str, Any]:
    return _to_device(
        load_foundry_training_example(
            root,
            row,
            noise_level=0.5,
            diffusion_batch_size=1,
            max_context_tokens=max_context_tokens,
        ),
        device,
    )


def _seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _config_sha256(resolved: dict[str, Any]) -> str:
    encoded = json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run(args: argparse.Namespace, *, root: Path | None = None) -> dict[str, Any]:
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    resolved = load_config(config_path)
    validate_v0_config(resolved).require_ready()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA generation was requested but no CUDA device is available")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    _seed(args.seed, device)
    model = NanoDesignTiny(_model_config(resolved)).to(device)
    stats_path = root / "docs/data_v0_stats.json"
    manifest_sha = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    checkpoint = load_checkpoint(
        args.checkpoint,
        model=model,
        expected_manifest_sha256=manifest_sha,
    )
    samples_seen = int(checkpoint.get("samples_seen", -1))
    if samples_seen != args.samples_seen:
        raise ValueError(
            f"checkpoint samples_seen is {samples_seen}, expected milestone {args.samples_seen}"
        )
    config_sha = _config_sha256(resolved)
    if checkpoint.get("config_sha256") != config_sha:
        raise ValueError("checkpoint does not match the requested frozen configuration")

    max_context_tokens = int(resolved["model"]["max_context_tokens"])
    output_dir = Path(args.output_root) / f"samples-{samples_seen:08d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks: dict[str, dict[str, Any]] = {}
    model.eval()
    for task_index, (task, relative_split) in enumerate(SPLITS.items()):
        rows = load_split_catalog(root / relative_split)
        row, context_complete = _generation_row(rows, max_context_tokens=max_context_tokens)
        task_seed = args.seed + task_index
        _seed(task_seed, device)
        batch = _batch(
            root,
            row,
            device=device,
            max_context_tokens=max_context_tokens,
        )
        with torch.no_grad():
            output = generate(model, batch)
        structure_path = output_dir / f"{task}.pdb"
        sequences = write_generation_structure(output, batch, structure_path)
        tasks[task] = {
            "sample_id": row["sample_id"],
            "seed": task_seed,
            "structure_path": str(structure_path.resolve()),
            "sequences": sequences,
            "fixed_context_complete": context_complete,
            "coordinates_shape": list(output["X_L"].shape),
            "sequence_logits_shape": list(output["sequence_logits_I"].shape),
            "finite": bool(
                torch.isfinite(output["X_L"]).all()
                and torch.isfinite(output["sequence_logits_I"]).all()
            ),
            "execution_mode": getattr(model, "last_execution_mode", "standard"),
        }
    metadata = {
        "samples_seen": samples_seen,
        "optimizer_step": int(checkpoint["step"]),
        "seed": args.seed,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest_sha256": manifest_sha,
        "config_sha256": config_sha,
        "model_parameter_count": model.parameter_count,
        "generation_split": "test",
        "selection": "first complete fixed-context sample; first-row fallback",
        "tasks": tasks,
        # Preserve the existing training-report key so the antibody and RNA evaluator
        # runners can consume this metadata directly via --training-report.
        "generation": tasks,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples-seen", type=int, required=True)
    parser.add_argument("--config", default="configs/v0.yaml")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.samples_seen <= 0:
        raise ValueError("samples-seen milestone must be positive")
    run(args)


if __name__ == "__main__":
    main()

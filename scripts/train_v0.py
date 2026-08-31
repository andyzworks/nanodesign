#!/usr/bin/env python3
"""Train the frozen NanoDesign v0 RFD3NA-Tiny model on all three real tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nanodesign.v0.config import load_config, validate_v0_config
from nanodesign.v0.data.real import load_foundry_training_example, load_split_catalog
from nanodesign.v0.model import NanoDesignTiny, NanoDesignTinyConfig
from nanodesign.v0.training import (
    TrainingConfig,
    build_optimizer,
    evaluate_loss,
    generate,
    load_checkpoint,
    save_checkpoint,
    train_step,
    write_generation_structure,
)

SPLITS = {
    "protein_binder": "data/processed/v0/splits/protein_binder/{split}.jsonl",
    "antibody_h3": "data/processed/v0/splits/antibody_h3/{split}.jsonl",
    "rna": "data/processed/v0/splits/rna_binding/{split}.jsonl",
}


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _model_config(resolved: dict[str, Any]) -> NanoDesignTinyConfig:
    return NanoDesignTinyConfig.from_mapping(
        {key: resolved["model"][key] for key in NanoDesignTinyConfig.__dataclass_fields__}
    )


def _load_rows(root: Path, split: str) -> dict[str, list[dict[str, Any]]]:
    return {
        task: load_split_catalog(root / pattern.format(split=split))
        for task, pattern in SPLITS.items()
    }


def _first_fully_represented_row(
    rows: list[dict[str, Any]], *, max_context_tokens: int
) -> dict[str, Any]:
    """Select the first held-out sample whose complete fixed context fits the config."""

    for row in rows:
        design_tokens = 0
        for chain in row["chains"]:
            if chain["role"] in {"binder", "rna_aptamer", "rna_design_region"}:
                design_tokens += int(chain["resolved_residues"])
            elif chain["role"] == "antibody_framework+cdr_h3":
                design_tokens += len(chain["design_residue_keys"])
        fixed_tokens = sum(int(chain["resolved_residues"]) for chain in row["chains"])
        fixed_tokens -= design_tokens
        if fixed_tokens <= max_context_tokens:
            return row
    raise ValueError("no held-out sample has a fully representable fixed context")


def _batch(
    root: Path,
    row: dict[str, Any],
    *,
    device: torch.device,
    max_context_tokens: int,
    noise_level: float | None = None,
    diffusion_batch_size: int = 1,
) -> dict[str, Any]:
    return _to_device(
        load_foundry_training_example(
            root,
            row,
            noise_level=noise_level,
            diffusion_batch_size=diffusion_batch_size,
            max_context_tokens=max_context_tokens,
        ),
        device,
    )


def _validation(
    model: NanoDesignTiny,
    root: Path,
    rows: dict[str, list[dict[str, Any]]],
    *,
    device: torch.device,
    max_context_tokens: int,
    samples_per_task: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    report = {}
    for task_index, (task, task_rows) in enumerate(rows.items()):
        selected = random.Random(seed + 1000 + task_index).sample(
            task_rows, min(samples_per_task, len(task_rows))
        )
        values = defaultdict(list)
        for sample_index, row in enumerate(selected):
            validation_seed = seed + 10_000 * task_index + sample_index
            torch.manual_seed(validation_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(validation_seed)
            metrics = evaluate_loss(
                model,
                _batch(
                    root,
                    row,
                    device=device,
                    max_context_tokens=max_context_tokens,
                    noise_level=0.5,
                ),
            )
            for name, value in metrics.items():
                values[name].append(value)
        report[task] = {name: float(np.mean(items)) for name, items in values.items()}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v0.yaml")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--validation-samples-per-task", type=int, default=4)
    parser.add_argument(
        "--diffusion-batch-size",
        type=int,
        default=4,
        help="EDM realizations per complex (official RFD3NA uses 32; v0 tiny uses 4)",
    )
    parser.add_argument("--max-context-tokens", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if min(args.steps, args.validation_samples_per_task, args.diffusion_batch_size) < 1:
        raise ValueError("steps, validation samples, and diffusion batch size must be positive")

    root = Path(__file__).resolve().parents[1]
    resolved = load_config(root / args.config)
    validate_v0_config(resolved).require_ready()
    max_context_tokens = args.max_context_tokens or int(resolved["model"]["max_context_tokens"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but no CUDA device is available")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = NanoDesignTiny(_model_config(resolved)).to(device)
    training_config = TrainingConfig()
    optimizer = build_optimizer(model, training_config)
    train_rows = _load_rows(root, "train")
    validation_rows = _load_rows(root, "validation")
    test_rows = _load_rows(root, "test")
    shuffled = {}
    for task_index, (task, rows) in enumerate(train_rows.items()):
        shuffled[task] = list(rows)
        random.Random(args.seed + task_index).shuffle(shuffled[task])

    validation_before = _validation(
        model,
        root,
        validation_rows,
        device=device,
        max_context_tokens=max_context_tokens,
        samples_per_task=args.validation_samples_per_task,
        seed=args.seed,
    )
    task_names = list(SPLITS)
    cursors = defaultdict(int)
    history = []
    task_steps = defaultdict(int)
    for step in range(args.steps):
        task = task_names[step % len(task_names)]
        rows = shuffled[task]
        row = rows[cursors[task] % len(rows)]
        cursors[task] += 1
        batch = _batch(
            root,
            row,
            device=device,
            max_context_tokens=max_context_tokens,
            diffusion_batch_size=args.diffusion_batch_size,
        )
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                metrics = train_step(model, optimizer, batch, training_config)
        else:
            metrics = train_step(model, optimizer, batch, training_config)
        history.append({"step": step + 1, "task": task, "sample_id": row["sample_id"], **metrics})
        task_steps[task] += 1
        print(
            f"step={step + 1} task={task} loss={metrics['loss']:.6f} "
            f"coord={metrics['coordinate_loss']:.6f} seq={metrics['sequence_loss']:.6f}",
            flush=True,
        )

    validation_after = _validation(
        model,
        root,
        validation_rows,
        device=device,
        max_context_tokens=max_context_tokens,
        samples_per_task=args.validation_samples_per_task,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generation = {}
    for task, rows in test_rows.items():
        row = _first_fully_represented_row(rows, max_context_tokens=max_context_tokens)
        generation_batch = _batch(
            root,
            row,
            device=device,
            max_context_tokens=max_context_tokens,
            noise_level=0.5,
        )
        output = generate(
            model,
            generation_batch,
        )
        structure_path = output_dir / "generations" / f"{task}.pdb"
        sequences = write_generation_structure(output, generation_batch, structure_path)
        generation[task] = {
            "sample_id": row["sample_id"],
            "coordinates_shape": list(output["X_L"].shape),
            "sequence_logits_shape": list(output["sequence_logits_I"].shape),
            "structure_path": str(structure_path.resolve()),
            "sequences": sequences,
            "finite": bool(
                torch.isfinite(output["X_L"]).all()
                and torch.isfinite(output["sequence_logits_I"]).all()
            ),
        }

    stats_path = root / "docs/data_v0_stats.json"
    manifest_sha = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    checkpoint_path = output_dir / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=args.steps,
        manifest_sha256=manifest_sha,
        resolved_config=resolved,
    )
    restored = NanoDesignTiny(_model_config(resolved)).to(device)
    restored_optimizer = build_optimizer(restored, training_config)
    loaded = load_checkpoint(
        checkpoint_path,
        model=restored,
        optimizer=restored_optimizer,
        expected_manifest_sha256=manifest_sha,
    )
    report = {
        "steps": args.steps,
        "samples_seen": args.steps,
        "batch_size_complexes": 1,
        "diffusion_realizations_per_complex": args.diffusion_batch_size,
        "seed": args.seed,
        "device": str(device),
        "model_parameter_count": model.parameter_count,
        "max_context_tokens": max_context_tokens,
        "task_steps": dict(task_steps),
        "validation_samples_per_task": args.validation_samples_per_task,
        "generation_split": "test",
        "generation_selection": "first sample with complete fixed context within token budget",
        "validation_before": validation_before,
        "validation_after": validation_after,
        "generation": generation,
        "checkpoint_step": loaded["step"],
        "history": history,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

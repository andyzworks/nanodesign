#!/usr/bin/env python3
"""Run real examples from all three tasks through official RFD3NA train/generate paths."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

import torch

from nanodesign.v0.config import load_config
from nanodesign.v0.data.real import load_foundry_training_example
from nanodesign.v0.model import NanoDesignTiny, NanoDesignTinyConfig
from nanodesign.v0.training import (
    build_optimizer,
    generate,
    load_checkpoint,
    save_checkpoint,
    train_step,
)

CATALOGS = {
    "protein_binder": "data/processed/v0/splits/protein_binder/train.jsonl",
    "antibody_h3": "data/processed/v0/splits/antibody_h3/train.jsonl",
    "rna": "data/processed/v0/splits/rna_binding/train.jsonl",
}


def _smallest_row(path: Path) -> dict:
    rows = (json.loads(line) for line in path.open(encoding="utf-8") if line.strip())
    return min(rows, key=lambda row: sum(chain["resolved_residues"] for chain in row["chains"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v0.yaml")
    parser.add_argument("--sampling-steps", type=int, default=2)
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolved_config = load_config(root / args.config)
    smoke_config = copy.deepcopy(resolved_config)
    smoke_config["model"]["sampling_steps"] = args.sampling_steps
    model_values = {
        key: smoke_config["model"][key] for key in NanoDesignTinyConfig.__dataclass_fields__
    }
    model = NanoDesignTiny(NanoDesignTinyConfig.from_mapping(model_values))
    optimizer = build_optimizer(model)
    report = {
        "parameter_count": model.parameter_count,
        "foundry_model": f"{model.net.__class__.__module__}.{model.net.__class__.__name__}",
        "sampling_steps": model.config.sampling_steps,
        "tasks": {},
    }
    last_batch = None
    for task, relative_catalog in CATALOGS.items():
        row = _smallest_row(root / relative_catalog)
        training_batch = load_foundry_training_example(
            root,
            row,
            noise_level=0.5,
            diffusion_batch_size=1,
            max_context_tokens=4,
        )
        metrics = train_step(model, optimizer, training_batch)
        required_metrics = ("loss", "coordinate_loss", "sequence_loss", "gradient_norm")
        if not all(torch.isfinite(torch.tensor(metrics[name])) for name in required_metrics):
            raise RuntimeError(f"{task}: non-finite training metric")
        generation_batch = load_foundry_training_example(
            root,
            row,
            noise_level=0.5,
            diffusion_batch_size=1,
            max_context_tokens=4,
        )
        output = generate(model, generation_batch)
        if (
            not torch.isfinite(output["X_L"]).all()
            or not torch.isfinite(output["sequence_logits_I"]).all()
        ):
            raise RuntimeError(f"{task}: non-finite generation output")
        last_batch = generation_batch
        report["tasks"][task] = {
            "sample_id": row["sample_id"],
            "tokens": int(generation_batch["f"]["restype"].shape[0]),
            "atoms": int(generation_batch["coord_atom_lvl_to_be_noised"].shape[1]),
            "loss": metrics["loss"],
            "coordinate_loss": metrics["coordinate_loss"],
            "sequence_loss": metrics["sequence_loss"],
            "gradient_norm": metrics["gradient_norm"],
            "generated_coordinates": list(output["X_L"].shape),
            "generated_sequence_logits": list(output["sequence_logits_I"].shape),
        }
    manifest_sha = json.loads((root / "docs/data_v0_stats.json").read_text())["manifest_files"][
        "protein_binder/train"
    ]["sha256"]
    with tempfile.TemporaryDirectory(prefix="nanodesign-checkpoint-") as directory:
        checkpoint_path = Path(directory) / "smoke.pt"
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            step=len(CATALOGS),
            manifest_sha256=manifest_sha,
            resolved_config=smoke_config,
        )
        restored = NanoDesignTiny(NanoDesignTinyConfig.from_mapping(model_values))
        restored_optimizer = build_optimizer(restored)
        checkpoint = load_checkpoint(
            checkpoint_path,
            model=restored,
            optimizer=restored_optimizer,
            expected_manifest_sha256=manifest_sha,
        )
        if last_batch is None:
            raise RuntimeError("no task batch was evaluated")
        restored_output = generate(restored, last_batch)
        report["checkpoint"] = {
            "step": checkpoint["step"],
            "save_load": "passed",
            "restored_generation_coordinates": list(restored_output["X_L"].shape),
        }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        destination = root / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

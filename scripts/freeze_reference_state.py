#!/usr/bin/env python3
"""Fingerprint the 0K/9K/18K NanoDesign reference checkpoints and recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    training_report = json.loads(
        (run_dir / "training_report.json").read_text(encoding="utf-8")
    )

    checkpoints = {}
    fingerprints = set()
    for label, samples in (("untrained", 0), ("9k", 9000), ("18k", 18000)):
        path = run_dir / "milestones" / f"samples-{samples:08d}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        fingerprints.add(
            (
                checkpoint["config_sha256"],
                checkpoint["manifest_sha256"],
                checkpoint["parameter_count"],
            )
        )
        checkpoints[label] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "samples_seen": int(checkpoint["samples_seen"]),
            "optimizer_steps": int(checkpoint["step"]),
            "global_samples_per_task": {
                task: int(steps) for task, steps in checkpoint["task_steps"].items()
            },
            "ema_available": isinstance(checkpoint.get("ema"), dict),
        }
        checkpoints[label]["global_samples_per_task"] = {
            task: value for task, value in checkpoints[label]["global_samples_per_task"].items()
        }
    if len(fingerprints) != 1:
        raise ValueError("reference checkpoints do not share model/data/config fingerprints")
    config_sha, manifest_sha, parameter_count = fingerprints.pop()
    final_checkpoint = torch.load(
        run_dir / "milestones/samples-00018000.pt", map_location="cpu", weights_only=False
    )
    payload = {
        "schema": "nanodesign.reference_state.v1",
        "selection": {
            "run": str(run_dir.resolve()),
            "reason": (
                "frozen current seed-17 lr5e-4 diffusion-batch-16 gradient-clip-10 "
                "reference used by the existing 9K/18K reports"
            ),
        },
        "model": {
            "architecture": "pinned RosettaCommons RFD3NA RFD3",
            "parameter_count": int(parameter_count),
            "model_config": final_checkpoint["model_config"],
            "config_sha256": config_sha,
        },
        "data_manifest_sha256": manifest_sha,
        "training_recipe": final_checkpoint["training_run_config"],
        "reported_training_mechanics": training_report["training_mechanics"],
        "checkpoints": checkpoints,
        "legacy_validation_panel": {
            "samples_per_task": training_report["validation_samples_per_task"],
            "warning": "superseded for evaluator audit by frozen learnability.v1 panel",
            "validation_before": training_report["validation_before"],
            "validation_after": training_report["validation_after"],
        },
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

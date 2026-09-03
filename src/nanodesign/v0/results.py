"""Compact machine-readable experiment rows derived from complete training reports."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def training_result_rows(
    report: Mapping[str, Any], *, experiment: str
) -> list[dict[str, Any]]:
    """Return one comparable row per trained task without discarding the full report."""

    history = report.get("history")
    validation = report.get("validation_after")
    per_task_samples = report.get("global_samples_per_task")
    if not isinstance(history, list) or not isinstance(validation, Mapping):
        raise TypeError("training report lacks history or validation_after")
    if not isinstance(per_task_samples, Mapping):
        raise TypeError("training report lacks global_samples_per_task")
    mechanics = report.get("training_mechanics", {})
    generation = report.get("generation", {})
    rows = []
    for task, samples in per_task_samples.items():
        records = [record for record in history if record.get("task") == task]
        if not records:
            raise ValueError(f"training report has no history for task {task!r}")
        task_validation = validation.get(task)
        if not isinstance(task_validation, Mapping):
            raise TypeError(f"training report has no validation metrics for task {task!r}")
        rows.append(
            {
                "experiment": experiment,
                "model_size": int(report["model_parameter_count"]),
                "task": task,
                "budget_samples_seen": int(report["samples_seen"]),
                "task_samples_seen": int(samples),
                "seed": int(report["seed"]),
                "optimizer": mechanics.get("optimizer"),
                "learning_rate": mechanics.get("learning_rate"),
                "train_loss": float(np.mean([record["loss"] for record in records])),
                "val_loss": float(task_validation["loss"]),
                "sequence_loss": float(task_validation["sequence_loss"]),
                "structure_loss": float(task_validation["coordinate_loss"]),
                "sequence_recovery": float(task_validation["seq_recovery"]),
                "generation_metrics": (
                    generation.get(task, {}) if isinstance(generation, Mapping) else {}
                ),
                "runtime_seconds": float(report["optimization_wall_seconds"]),
                "gpu_hours": float(report["optimization_gpu_hours"]),
            }
        )
    return rows


def write_training_result_rows(
    report: Mapping[str, Any], *, experiment: str, destination: str | Path
) -> Path:
    """Atomically persist the compact rows beside every completed experiment."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "nanodesign.experiment_rows.v1",
        "rows": training_result_rows(report, experiment=experiment),
    }
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination

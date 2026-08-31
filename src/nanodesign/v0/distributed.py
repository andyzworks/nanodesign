"""Deterministic sample assignment helpers for NanoDesign v0 DDP training."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    """Process identity supplied by ``torchrun``."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError("world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be inside world_size")
        if self.local_rank < 0:
            raise ValueError("local_rank must be non-negative")

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def task_for_step(task_names: Sequence[str], optimizer_step: int) -> str:
    """Return the shared task for one optimizer step on every rank."""

    if not task_names:
        raise ValueError("task_names cannot be empty")
    if optimizer_step < 0:
        raise ValueError("optimizer_step must be non-negative")
    return task_names[optimizer_step % len(task_names)]


def row_for_rank(
    rows_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
    task_names: Sequence[str],
    *,
    optimizer_step: int,
    rank: int,
    world_size: int,
) -> Mapping[str, Any]:
    """Assign distinct consecutive samples to ranks without a replicated sampler.

    All ranks train on the same task at a given optimizer step, so DDP executes the
    same graph.  Across occurrences of that task, the global cursor advances by the
    world size.  The modulo is an explicit epoch wrap, not an accidental duplicate.
    """

    context = DistributedContext(rank=rank, local_rank=rank, world_size=world_size)
    task = task_for_step(task_names, optimizer_step)
    rows = rows_by_task.get(task)
    if not rows:
        raise ValueError(f"task {task!r} has no samples")
    task_occurrence = optimizer_step // len(task_names)
    global_index = task_occurrence * context.world_size + context.rank
    return rows[global_index % len(rows)]


def validation_indices(sample_count: int, *, rank: int, world_size: int) -> range:
    """Shard validation exactly once across ranks."""

    DistributedContext(rank=rank, local_rank=rank, world_size=world_size)
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    return range(rank, sample_count, world_size)


def samples_seen(completed_optimizer_steps: int, world_size: int) -> int:
    """Count complexes globally rather than once per rank."""

    if completed_optimizer_steps < 0:
        raise ValueError("completed_optimizer_steps must be non-negative")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    return completed_optimizer_steps * world_size


def context_from_environment() -> DistributedContext:
    """Read the process identity exported by torchrun, defaulting to one process."""

    return DistributedContext(
        rank=int(os.environ.get("RANK", "0")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
    )


def reduce_scalar_metrics(
    metrics: Mapping[str, float], *, device: torch.device, world_size: int
) -> dict[str, float]:
    """Average scalar training metrics across ranks in a stable key order."""

    if world_size == 1:
        return {key: float(value) for key, value in metrics.items()}
    if not dist.is_initialized():
        raise RuntimeError("distributed metrics require an initialized process group")
    names = sorted(metrics)
    values = torch.tensor([metrics[name] for name in names], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= world_size
    return {name: float(value) for name, value in zip(names, values.cpu().tolist(), strict=True)}


def all_gather_objects(value: Any, world_size: int) -> list[Any]:
    """Collect one small provenance object per rank."""

    if world_size == 1:
        return [value]
    if not dist.is_initialized():
        raise RuntimeError("distributed gathering requires an initialized process group")
    values: list[Any] = [None] * world_size
    dist.all_gather_object(values, value)
    return values

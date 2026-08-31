"""Deterministic sample assignment helpers for NanoDesign v0 DDP training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


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

"""Deterministic sample assignment helpers for NanoDesign v0 DDP training."""

from __future__ import annotations

import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

SIZE_PACKING_VERSION = "catalog-polymer-slots-v1"
PROTEIN_ATOM_SLOTS = 14
RNA_ATOM_SLOTS = 23


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


def catalog_model_token_count(row: Mapping[str, Any], *, max_context_tokens: int | None) -> int:
    """Return the exact post-crop token count using catalog-stored residue identities.

    The Foundry loader creates one atom23 token per resolved catalog residue. Its
    context crop is also fully determined by chain roles, CDR-H3 keys, and the context
    limit.  Consequently this value is reproducible without opening a structure, and
    the feature-cache identity already binds both the complete catalog row and the
    context limit.
    """

    if max_context_tokens is not None and max_context_tokens < 0:
        raise ValueError("max_context_tokens must be non-negative or None")
    total = 0
    design = 0
    protected_context = 0
    for chain in row.get("chains", []):
        residues = int(chain["resolved_residues"])
        if residues < 0:
            raise ValueError("resolved_residues must be non-negative")
        role = str(chain["role"])
        total += residues
        if role in {"binder", "rna_aptamer", "rna_design_region"}:
            design += residues
        elif role == "antibody_framework+cdr_h3":
            cdr_h3 = len(chain.get("design_residue_keys", []))
            if cdr_h3 > residues:
                raise ValueError("CDR-H3 keys exceed resolved chain residues")
            design += cdr_h3
            protected_context += residues - cdr_h3
        elif role == "antibody_framework":
            protected_context += residues
    context = total - design
    if max_context_tokens is None or not design or context <= max_context_tokens:
        return total
    optional_context = context - protected_context
    optional_budget = max(0, max_context_tokens - protected_context)
    return design + protected_context + min(optional_context, optional_budget)


def catalog_model_atom_count(row: Mapping[str, Any], *, max_context_tokens: int | None) -> int:
    """Return the exact model atom tensor length implied by a catalog row.

    Foundry association schemes use 14 slots for protein tokens and 23 for RNA
    tokens. NanoDesign's optional cropped context is protein target in all three
    frozen tasks; reject a future mixed-polymer optional context rather than silently
    turn this scheduling identity into an approximation.
    """

    if max_context_tokens is not None and max_context_tokens < 0:
        raise ValueError("max_context_tokens must be non-negative or None")
    counts = {
        "design_protein": 0,
        "design_rna": 0,
        "protected_protein": 0,
        "protected_rna": 0,
        "optional_protein": 0,
        "optional_rna": 0,
    }
    rna_roles = {"rna_aptamer", "rna_design_region", "rna_structure_prior"}
    for chain in row.get("chains", []):
        residues = int(chain["resolved_residues"])
        if residues < 0:
            raise ValueError("resolved_residues must be non-negative")
        role = str(chain["role"])
        polymer = "rna" if role in rna_roles else "protein"
        if role in {"binder", "rna_aptamer", "rna_design_region"}:
            counts[f"design_{polymer}"] += residues
        elif role == "antibody_framework+cdr_h3":
            design = len(chain.get("design_residue_keys", []))
            if design > residues:
                raise ValueError("CDR-H3 keys exceed resolved chain residues")
            counts["design_protein"] += design
            counts["protected_protein"] += residues - design
        elif role == "antibody_framework":
            counts["protected_protein"] += residues
        else:
            counts[f"optional_{polymer}"] += residues

    design = counts["design_protein"] + counts["design_rna"]
    protected = counts["protected_protein"] + counts["protected_rna"]
    optional = counts["optional_protein"] + counts["optional_rna"]
    if max_context_tokens is None or not design or protected + optional <= max_context_tokens:
        selected_optional_protein = counts["optional_protein"]
        selected_optional_rna = counts["optional_rna"]
    else:
        optional_budget = max(0, max_context_tokens - protected)
        if optional_budget < optional and counts["optional_protein"] and counts["optional_rna"]:
            raise ValueError(
                "cannot infer a partial mixed-polymer context crop from catalog counts"
            )
        selected_optional_protein = min(counts["optional_protein"], optional_budget)
        selected_optional_rna = min(counts["optional_rna"], optional_budget)
    protein_tokens = (
        counts["design_protein"] + counts["protected_protein"] + selected_optional_protein
    )
    rna_tokens = counts["design_rna"] + counts["protected_rna"] + selected_optional_rna
    return protein_tokens * PROTEIN_ATOM_SLOTS + rna_tokens * RNA_ATOM_SLOTS


def size_aware_rank_packing(
    rows: Sequence[Mapping[str, Any]],
    *,
    world_size: int,
    seed: int,
    max_context_tokens: int | None,
) -> list[Mapping[str, Any]]:
    """Deterministically group similarly sized samples into global DDP steps.

    Every input row occurs exactly once. Full ``world_size`` groups are shuffled as
    units, preserving stochastic step order while preventing large within-step size
    differences. A final incomplete group stays last so no group boundaries shift.
    """

    if world_size < 1:
        raise ValueError("world_size must be positive")
    packed = list(rows)
    sample_ids = [str(row["sample_id"]) for row in packed]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("size-aware packing requires unique sample IDs")
    rng = random.Random(seed)
    rng.shuffle(packed)
    packed.sort(
        key=lambda row: catalog_model_atom_count(row, max_context_tokens=max_context_tokens)
    )
    full_count = len(packed) // world_size
    groups = [packed[index * world_size : (index + 1) * world_size] for index in range(full_count)]
    remainder = packed[full_count * world_size :]
    rng.shuffle(groups)
    for group in groups:
        rng.shuffle(group)
    return [row for group in groups for row in group] + remainder


def synchronize_training_execution_mode(
    local_atom_count: int,
    *,
    device: torch.device,
    world_size: int,
    standard_max_atoms: int,
    force_chunked: bool = False,
) -> str:
    """Choose one reduction-graph-compatible execution mode on every DDP rank."""

    if local_atom_count < 0 or standard_max_atoms < 1:
        raise ValueError("atom counts and threshold must be positive")
    maximum = torch.tensor(local_atom_count, dtype=torch.int64, device=device)
    if world_size > 1:
        if not dist.is_initialized():
            raise RuntimeError("distributed execution-mode routing requires a process group")
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return "chunked" if force_chunked or int(maximum.item()) > standard_max_atoms else "standard"


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

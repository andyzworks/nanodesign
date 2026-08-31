#!/usr/bin/env python3
"""Train the frozen NanoDesign v0 RFD3NA-Tiny model on all three real tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from nanodesign.v0.config import load_config, validate_v0_config
from nanodesign.v0.data.cache import FeatureCacheError, FeatureCacheSpec
from nanodesign.v0.data.loader import (
    CachedFeatureDataset,
    build_async_feature_loader,
    recursive_to_device,
    stage_catalog_cache,
)
from nanodesign.v0.data.real import load_foundry_training_example, load_split_catalog
from nanodesign.v0.distributed import (
    SIZE_PACKING_VERSION,
    DistributedContext,
    all_gather_objects,
    context_from_environment,
    reduce_scalar_metrics,
    row_for_rank,
    size_aware_rank_packing,
    synchronize_training_execution_mode,
    task_for_step,
    validation_indices,
)
from nanodesign.v0.model import STANDARD_MODE_MAX_ATOMS, NanoDesignTiny, NanoDesignTinyConfig
from nanodesign.v0.training import (
    TrainingConfig,
    build_optimizer,
    capture_rng_state,
    evaluate_loss,
    generate,
    load_checkpoint,
    restore_rng_state,
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


def _sample_milestones(value: str) -> list[int]:
    try:
        milestones = [int(item) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("milestones must be comma-separated integers") from error
    if not milestones or any(item <= 0 for item in milestones):
        raise argparse.ArgumentTypeError("milestones must be positive")
    if milestones != sorted(set(milestones)):
        raise argparse.ArgumentTypeError("milestones must be strictly increasing")
    return milestones


def _load_rows(root: Path, split: str) -> dict[str, list[dict[str, Any]]]:
    return {
        task: load_split_catalog(root / pattern.format(split=split))
        for task, pattern in SPLITS.items()
    }


def _fixed_context_tokens(row: dict[str, Any]) -> int:
    design_tokens = 0
    for chain in row["chains"]:
        if chain["role"] in {"binder", "rna_aptamer", "rna_design_region"}:
            design_tokens += int(chain["resolved_residues"])
        elif chain["role"] == "antibody_framework+cdr_h3":
            design_tokens += len(chain["design_residue_keys"])
    return sum(int(chain["resolved_residues"]) for chain in row["chains"]) - design_tokens


def _generation_row(rows: list[dict[str, Any]], *, max_context_tokens: int) -> dict[str, Any]:
    """Prefer a complete held-out context, with a deterministic smoke fallback."""

    return next((row for row in rows if _fixed_context_tokens(row) <= max_context_tokens), rows[0])


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
    distributed: DistributedContext,
) -> dict[str, dict[str, float]]:
    report = {}
    for task_index, (task, task_rows) in enumerate(rows.items()):
        selected = random.Random(seed + 1000 + task_index).sample(
            task_rows, min(samples_per_task, len(task_rows))
        )
        totals = defaultdict(float)
        local_count = 0
        for sample_index in validation_indices(
            len(selected), rank=distributed.rank, world_size=distributed.world_size
        ):
            row = selected[sample_index]
            validation_seed = seed + 10_000 * task_index + sample_index
            torch.manual_seed(validation_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(validation_seed)
            precision = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with precision:
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
                totals[name] += value
            local_count += 1
        names = ("loss", "coordinate_loss", "sequence_loss")
        packed = torch.tensor(
            [*(totals[name] for name in names), local_count],
            dtype=torch.float64,
            device=device,
        )
        if distributed.world_size > 1:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        count = int(packed[-1].item())
        if count != len(selected):
            raise RuntimeError("distributed validation did not cover every selected sample once")
        report[task] = {
            name: float(packed[index].item() / count) for index, name in enumerate(names)
        }
    return report


def _seed_process(seed: int, rank: int) -> None:
    process_seed = seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)


def _generation_outputs(
    model: NanoDesignTiny,
    root: Path,
    rows_by_task: dict[str, list[dict[str, Any]]],
    *,
    device: torch.device,
    max_context_tokens: int,
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    generation = {}
    for task, rows in rows_by_task.items():
        row = _generation_row(rows, max_context_tokens=max_context_tokens)
        generation_batch = _batch(
            root,
            row,
            device=device,
            max_context_tokens=max_context_tokens,
            noise_level=0.5,
        )
        precision = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with precision:
            output = generate(model, generation_batch)
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
            "fixed_context_complete": _fixed_context_tokens(row) <= max_context_tokens,
        }
    return generation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v0.yaml")
    parser.add_argument("--steps", type=int, help="target optimizer steps (smoke/backward mode)")
    parser.add_argument(
        "--milestone-samples",
        type=_sample_milestones,
        help="global-sample milestones, e.g. 3000,9000,18000,36000",
    )
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
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="save a resumable numbered checkpoint every N steps; 0 disables periodic saves",
    )
    parser.add_argument("--resume", type=Path, help="resume the same sample-budget run")
    parser.add_argument(
        "--feature-cache-root",
        type=Path,
        help="optional finalized SQLite feature-cache root; raw preprocessing remains fallback",
    )
    parser.add_argument(
        "--feature-cache-stage-root",
        type=Path,
        help="optional node-local directory receiving whole task/split cache databases",
    )
    parser.add_argument("--data-workers", type=int, default=4)
    parser.add_argument("--data-prefetch-factor", type=int, default=4)
    parser.add_argument("--data-pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--feature-cache-fallback", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if min(args.validation_samples_per_task, args.diffusion_batch_size) < 1:
        raise ValueError("validation samples and diffusion batch size must be positive")
    if args.steps is None and args.milestone_samples is None:
        raise ValueError("provide --steps or --milestone-samples")
    if args.steps is not None and args.steps < 1:
        raise ValueError("steps must be positive")
    if args.checkpoint_every < 0:
        raise ValueError("checkpoint interval must be non-negative")
    if args.data_workers < 0 or args.data_prefetch_factor < 1:
        raise ValueError("data workers must be non-negative and prefetch factor positive")
    if args.feature_cache_stage_root is not None and args.feature_cache_root is None:
        raise ValueError("feature-cache staging requires --feature-cache-root")

    root = Path(__file__).resolve().parents[1]
    resolved = load_config(root / args.config)
    validate_v0_config(resolved).require_ready()
    max_context_tokens = args.max_context_tokens or int(resolved["model"]["max_context_tokens"])
    distributed = context_from_environment()
    milestones = args.milestone_samples or []
    if any(value % distributed.world_size for value in milestones):
        raise ValueError("every sample milestone must be divisible by world size")
    if any((value // distributed.world_size) % len(SPLITS) for value in milestones):
        raise ValueError("sample milestones must end on an exact 1:1:1 task cycle")
    milestone_steps = [value // distributed.world_size for value in milestones]
    total_steps = milestone_steps[-1] if milestone_steps else int(args.steps)
    if args.steps is not None and milestone_steps and args.steps != total_steps:
        raise ValueError("--steps must equal the final sample milestone divided by world size")
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda":
        device = torch.device("cuda", distributed.local_rank)
    else:
        device = requested_device
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but no CUDA device is available")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if distributed.world_size > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")

    # Every rank constructs identical initial parameters; DDP also broadcasts rank 0.
    _seed_process(args.seed, 0)

    base_model = NanoDesignTiny(_model_config(resolved)).to(device)
    force_chunked_execution = base_model.execution_mode == "chunked"
    model = (
        DistributedDataParallel(
            base_model,
            device_ids=[distributed.local_rank] if device.type == "cuda" else None,
            output_device=distributed.local_rank if device.type == "cuda" else None,
            # Every frozen RFD3NA-Tiny parameter participates in both official
            # execution paths.  Foundry also activation-checkpoints its blocks;
            # enabling DDP's unused-parameter graph traversal can deadlock on the
            # second reentrant backward even though no parameter is unused.  Declaring
            # the unchanged RFD3 parameter graph static is PyTorch's supported DDP
            # path for reentrant activation checkpointing.
            find_unused_parameters=False,
            static_graph=True,
        )
        if distributed.world_size > 1
        else base_model
    )
    training_config = TrainingConfig()
    optimizer = build_optimizer(base_model, training_config)
    output_dir = Path(args.output_dir)
    if distributed.is_primary:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed.world_size > 1:
        dist.barrier()
    stats_path = root / "docs/data_v0_stats.json"
    manifest_sha = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    train_rows = _load_rows(root, "train")
    validation_rows = _load_rows(root, "validation")
    test_rows = _load_rows(root, "test")
    shuffled = {}
    for task_index, (task, rows) in enumerate(train_rows.items()):
        shuffled[task] = size_aware_rank_packing(
            rows,
            world_size=distributed.world_size,
            seed=args.seed + task_index,
            max_context_tokens=max_context_tokens,
        )
    task_names = list(SPLITS)
    cursors = defaultdict(int)
    history = []
    task_steps = defaultdict(int)
    execution_mode_counts = {task: {"standard": 0, "chunked": 0} for task in task_names}
    samples_seen = 0
    start_step = 0
    milestone_records = []
    elapsed_before_resume = 0.0
    run_started = time.monotonic()
    training_run_config = {
        "seed": args.seed,
        "diffusion_batch_size": args.diffusion_batch_size,
        "max_context_tokens": max_context_tokens,
        "validation_samples_per_task": args.validation_samples_per_task,
        "task_names": task_names,
        "world_size": distributed.world_size,
        "global_batch_size_complexes": distributed.world_size,
        "milestone_samples": milestones,
        "feature_cache_enabled": args.feature_cache_root is not None,
        "size_packing_version": SIZE_PACKING_VERSION,
    }
    if args.resume is not None:
        loaded = load_checkpoint(
            args.resume,
            model=base_model,
            optimizer=optimizer,
            expected_manifest_sha256=manifest_sha,
            restore_rng=True,
            rng_rank=distributed.rank,
        )
        if loaded.get("training_run_config") != training_run_config:
            raise ValueError("resume checkpoint training-run configuration mismatch")
        start_step = int(loaded["step"])
        if start_step > total_steps:
            raise ValueError("resume checkpoint step exceeds requested total steps")
        samples_seen = int(loaded["samples_seen"])
        loaded_cursors = {str(key): int(value) for key, value in loaded["task_cursors"].items()}
        loaded_task_steps = {str(key): int(value) for key, value in loaded["task_steps"].items()}
        if (set(loaded_cursors) | set(loaded_task_steps)) - set(task_names):
            raise ValueError("resume checkpoint contains an unknown task cursor")
        cursors.update(loaded_cursors)
        task_steps.update(loaded_task_steps)
        for task in task_names:
            cursors[task] += 0
            task_steps[task] += 0
        history = list(loaded["history"])
        for record in history:
            for execution_mode in record.get("execution_modes", []):
                if execution_mode not in {"standard", "chunked"}:
                    raise ValueError("resume checkpoint has an unknown execution mode")
                execution_mode_counts[record["task"]][execution_mode] += 1
        if not (
            samples_seen == start_step * distributed.world_size
            and start_step == len(history) == sum(task_steps.values())
            and samples_seen == sum(cursors.values())
            and all(
                cursors[task] == task_steps[task] * distributed.world_size for task in task_names
            )
        ):
            raise ValueError("resume checkpoint step/sample/task state is inconsistent")
        validation_before = loaded.get("validation_before")
        if not isinstance(validation_before, dict) or not validation_before:
            raise ValueError("resume checkpoint lacks initial validation state")
        milestone_records = list(loaded.get("milestone_records", []))
        elapsed_before_resume = float(loaded.get("elapsed_wall_seconds", 0.0))
    else:
        validation_before = _validation(
            base_model,
            root,
            validation_rows,
            device=device,
            max_context_tokens=max_context_tokens,
            samples_per_task=args.validation_samples_per_task,
            seed=args.seed,
            distributed=distributed,
        )
        _seed_process(args.seed, distributed.rank)

    scheduled_rows = [
        row_for_rank(
            shuffled,
            task_names,
            optimizer_step=step,
            rank=distributed.rank,
            world_size=distributed.world_size,
        )
        for step in range(start_step, total_steps)
    ]
    async_loader = None
    async_iterator = None
    selected_cache_root = args.feature_cache_root
    if args.feature_cache_root is not None:
        if args.feature_cache_stage_root is not None:
            try:
                stage_catalog_cache(
                    args.feature_cache_root, args.feature_cache_stage_root, scheduled_rows
                )
                selected_cache_root = args.feature_cache_stage_root
            except FeatureCacheError:
                if not args.feature_cache_fallback:
                    raise
                selected_cache_root = args.feature_cache_root
        cache_spec = FeatureCacheSpec(
            manifest_sha256=manifest_sha,
            max_context_tokens=max_context_tokens,
            diffusion_batch_size=args.diffusion_batch_size,
            noise_level=None,
            random_seed=None,
        )
        sampling_seeds = [
            args.seed + 1_000_000 + step * distributed.world_size + distributed.rank
            for step in range(start_step, total_steps)
        ]
        cached_dataset = CachedFeatureDataset(
            root,
            scheduled_rows,
            cache_spec,
            cache_root=selected_cache_root,
            allow_fallback=args.feature_cache_fallback,
            lru_size=8,
            sampling_seeds=sampling_seeds,
        )
        async_loader = build_async_feature_loader(
            cached_dataset,
            num_workers=args.data_workers,
            prefetch_factor=args.data_prefetch_factor,
            persistent_workers=args.data_workers > 0,
            pin_memory=args.data_pin_memory and device.type == "cuda",
            multiprocessing_context="spawn",
        )
        # DataLoader creates a base seed in the parent process. Worker sampling itself
        # is explicitly seeded per optimizer step, so preserve the model RNG exactly.
        parent_rng = capture_rng_state()
        async_iterator = iter(async_loader)
        restore_rng_state(parent_rng)

    def elapsed_wall_seconds() -> float:
        elapsed = torch.tensor(
            elapsed_before_resume + time.monotonic() - run_started,
            dtype=torch.float64,
            device=device,
        )
        if distributed.world_size > 1:
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        return float(elapsed.item())

    def save_training_checkpoint(path: Path, completed_step: int) -> None:
        ranked_rng = all_gather_objects(capture_rng_state(), distributed.world_size)
        elapsed = elapsed_wall_seconds()
        if distributed.is_primary:
            save_checkpoint(
                path,
                model=base_model,
                optimizer=optimizer,
                step=completed_step,
                samples_seen=samples_seen,
                task_cursors=cursors,
                task_steps=task_steps,
                history=history,
                training_run_config=training_run_config,
                validation_before=validation_before,
                rng_states_by_rank=ranked_rng,
                milestone_records=milestone_records,
                elapsed_wall_seconds=elapsed,
                manifest_sha256=manifest_sha,
                resolved_config=resolved,
            )
        if distributed.world_size > 1:
            dist.barrier()

    def append_milestone_record(completed_step: int) -> None:
        local_rng = capture_rng_state()
        milestone_validation = _validation(
            base_model,
            root,
            validation_rows,
            device=device,
            max_context_tokens=max_context_tokens,
            samples_per_task=args.validation_samples_per_task,
            seed=args.seed,
            distributed=distributed,
        )
        restore_rng_state(local_rng)
        elapsed = elapsed_wall_seconds()
        milestone_records.append(
            {
                "global_samples_seen": samples_seen,
                "optimizer_step": completed_step,
                "global_samples_per_task": {
                    task_name: task_steps[task_name] * distributed.world_size
                    for task_name in task_names
                },
                "wall_time_seconds": elapsed,
                "gpu_hours": (
                    elapsed * distributed.world_size / 3600 if device.type == "cuda" else 0.0
                ),
                "train_loss_per_task": {
                    task_name: float(
                        np.mean(
                            [record["loss"] for record in history if record["task"] == task_name]
                        )
                    )
                    for task_name in task_names
                },
                "validation_loss_per_task": {
                    task_name: milestone_validation[task_name]["loss"] for task_name in task_names
                },
            }
        )

    recorded_milestones = {record["global_samples_seen"] for record in milestone_records}
    if samples_seen in milestones and samples_seen not in recorded_milestones:
        append_milestone_record(start_step)
        save_training_checkpoint(
            output_dir / "milestones" / f"samples-{samples_seen:08d}.pt", start_step
        )

    for step in range(start_step, total_steps):
        task = task_for_step(task_names, step)
        row = row_for_rank(
            shuffled,
            task_names,
            optimizer_step=step,
            rank=distributed.rank,
            world_size=distributed.world_size,
        )
        if async_iterator is not None:
            batch = next(async_iterator)
            if batch["sample_id"] != row["sample_id"]:
                raise RuntimeError("asynchronous loader changed the scheduled sample order")
            batch = recursive_to_device(batch, device, non_blocking=True)
        else:
            batch = _batch(
                root,
                row,
                device=device,
                max_context_tokens=max_context_tokens,
                diffusion_batch_size=args.diffusion_batch_size,
            )
        atom_map = batch.get("f", {}).get("atom_to_token_map")
        if not isinstance(atom_map, torch.Tensor):
            raise TypeError("batch.f.atom_to_token_map must be a tensor")
        synchronized_mode = synchronize_training_execution_mode(
            atom_map.numel(),
            device=device,
            world_size=distributed.world_size,
            standard_max_atoms=STANDARD_MODE_MAX_ATOMS,
            force_chunked=force_chunked_execution,
        )
        base_model.execution_mode = synchronized_mode
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                metrics = train_step(model, optimizer, batch, training_config)
        else:
            metrics = train_step(model, optimizer, batch, training_config)
        metrics = reduce_scalar_metrics(metrics, device=device, world_size=distributed.world_size)
        sample_ids = all_gather_objects(row["sample_id"], distributed.world_size)
        local_execution_mode = getattr(base_model, "last_execution_mode", "standard")
        if local_execution_mode not in {"standard", "chunked"}:
            raise RuntimeError(f"model reported unknown execution mode {local_execution_mode!r}")
        execution_modes = all_gather_objects(local_execution_mode, distributed.world_size)
        history.append(
            {
                "step": step + 1,
                "task": task,
                "sample_ids": sample_ids,
                "execution_modes": execution_modes,
                **metrics,
            }
        )
        for execution_mode in execution_modes:
            execution_mode_counts[task][execution_mode] += 1
        task_steps[task] += 1
        cursors[task] += distributed.world_size
        samples_seen += distributed.world_size
        if distributed.is_primary:
            print(
                f"samples={samples_seen} step={step + 1} task={task} "
                f"loss={metrics['loss']:.6f} "
                f"coord={metrics['coordinate_loss']:.6f} seq={metrics['sequence_loss']:.6f}",
                flush=True,
            )
        completed_step = step + 1
        if samples_seen in milestones:
            milestone_path = output_dir / "milestones" / f"samples-{samples_seen:08d}.pt"
            # First make the training state durable. Validation uses fixed seeds, then
            # the exact per-rank training RNG state is restored before continuing.
            save_training_checkpoint(milestone_path, completed_step)
            append_milestone_record(completed_step)
            save_training_checkpoint(milestone_path, completed_step)
        elif args.checkpoint_every and completed_step % args.checkpoint_every == 0:
            save_training_checkpoint(
                output_dir / "checkpoints" / f"step-{completed_step:08d}.pt", completed_step
            )

    checkpoint_path = output_dir / "checkpoint.pt"
    save_training_checkpoint(checkpoint_path, total_steps)

    validation_after = _validation(
        base_model,
        root,
        validation_rows,
        device=device,
        max_context_tokens=max_context_tokens,
        samples_per_task=args.validation_samples_per_task,
        seed=args.seed,
        distributed=distributed,
    )
    generation = (
        _generation_outputs(
            base_model,
            root,
            test_rows,
            device=device,
            max_context_tokens=max_context_tokens,
            output_dir=output_dir,
        )
        if distributed.is_primary
        else {}
    )
    if not distributed.is_primary:
        dist.barrier()
        dist.destroy_process_group()
        return
    report = {
        "samples_seen": samples_seen,
        "global_samples_per_task": {
            task: task_steps[task] * distributed.world_size for task in task_names
        },
        "optimizer_steps": total_steps,
        "steps": total_steps,
        "starting_step": start_step,
        "resumed_from": str(args.resume.resolve()) if args.resume is not None else None,
        "checkpoint_every": args.checkpoint_every,
        "batch_size_complexes": distributed.world_size,
        "batch_size_complexes_per_rank": 1,
        "global_batch_size_complexes": distributed.world_size,
        "world_size": distributed.world_size,
        "diffusion_realizations_per_complex": args.diffusion_batch_size,
        "seed": args.seed,
        "device": str(device),
        "model_parameter_count": base_model.parameter_count,
        "max_context_tokens": max_context_tokens,
        "task_steps": dict(task_steps),
        "execution_mode_counts_per_task": execution_mode_counts,
        "size_aware_rank_packing": {
            "version": SIZE_PACKING_VERSION,
            "size_identity": "catalog row + max_context_tokens -> Foundry protein14/RNA23 atoms",
            "global_group_size": distributed.world_size,
            "synchronized_standard_max_atoms": STANDARD_MODE_MAX_ATOMS,
        },
        "validation_samples_per_task": args.validation_samples_per_task,
        "generation_split": "test",
        "generation_selection": (
            "first sample with complete fixed context; first-sample fallback for low-budget smoke"
        ),
        "validation_before": validation_before,
        "validation_after": validation_after,
        "milestones": milestone_records,
        "generation": generation,
        "checkpoint_step": total_steps,
        "history": history,
        "data_loader": {
            "feature_cache_enabled": args.feature_cache_root is not None,
            "selected_cache_root": (
                str(Path(selected_cache_root).resolve())
                if selected_cache_root is not None
                else None
            ),
            "num_workers": args.data_workers if args.feature_cache_root is not None else 0,
            "prefetch_factor": (
                args.data_prefetch_factor if args.feature_cache_root is not None else None
            ),
            "persistent_workers": bool(args.feature_cache_root and args.data_workers > 0),
            "pin_memory": bool(
                args.feature_cache_root and args.data_pin_memory and device.type == "cuda"
            ),
            "fallback_enabled": args.feature_cache_fallback,
        },
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if distributed.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

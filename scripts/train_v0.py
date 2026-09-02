#!/usr/bin/env python3
"""Train the frozen NanoDesign v0 RFD3NA-Tiny model on selected real tasks."""

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
from nanodesign.v0.data.cache import FeatureCacheError, FeatureCacheSpec, SQLiteFeatureCache
from nanodesign.v0.data.loader import (
    CachedFeatureDataset,
    build_async_feature_loader,
    recursive_to_device,
    stage_catalog_cache,
)
from nanodesign.v0.data.real import load_foundry_training_example, load_split_catalog
from nanodesign.v0.distributed import (
    SIZE_PACKING_GROUP_SIZE,
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
    ExponentialMovingAverage,
    TrainingConfig,
    build_learning_rate_scheduler,
    build_optimizer,
    capture_rng_state,
    evaluate_loss,
    generate,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    train_step,
    validate_resume_training_run_config,
    write_generation_structure,
)

SPLITS = {
    "protein_binder": "data/processed/v0/splits/protein_binder/{split}.jsonl",
    "antibody_h3": "data/processed/v0/splits/antibody_h3/{split}.jsonl",
    "rna": "data/processed/v0/splits/rna_binding/{split}.jsonl",
}
TASK_INDEX = {task: index for index, task in enumerate(SPLITS)}


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


def _task_names(value: str) -> list[str]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if not requested:
        raise argparse.ArgumentTypeError("tasks must not be empty")
    unknown = sorted(set(requested) - set(SPLITS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown task(s): {', '.join(unknown)}; choose from {', '.join(SPLITS)}"
        )
    if len(requested) != len(set(requested)):
        raise argparse.ArgumentTypeError("tasks must not contain duplicates")
    # Canonical ordering preserves the frozen unified task cycle, sample shuffles,
    # validation subsets, and generation seeds when a diagnostic selects one task.
    return [task for task in SPLITS if task in requested]


def _load_rows(
    root: Path, split: str, task_names: list[str] | None = None
) -> dict[str, list[dict[str, Any]]]:
    selected = list(SPLITS) if task_names is None else task_names
    return {task: load_split_catalog(root / SPLITS[task].format(split=split)) for task in selected}


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
    augment_coordinates: bool = False,
) -> dict[str, Any]:
    return _to_device(
        load_foundry_training_example(
            root,
            row,
            noise_level=noise_level,
            diffusion_batch_size=diffusion_batch_size,
            max_context_tokens=max_context_tokens,
            augment_coordinates=augment_coordinates,
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
    feature_cache_root: Path | None = None,
    feature_cache_fallback: bool = True,
    manifest_sha256: str | None = None,
    force_chunked: bool = False,
) -> dict[str, dict[str, float]]:
    if feature_cache_root is not None and manifest_sha256 is None:
        raise ValueError("cached validation requires the frozen manifest SHA256")
    cache = (
        SQLiteFeatureCache(feature_cache_root, readonly=True, lru_size=samples_per_task)
        if feature_cache_root is not None
        else None
    )
    report = {}
    try:
        for task, task_rows in rows.items():
            task_index = TASK_INDEX[task]
            # Keep the frozen sample selection and rank sharding independent of the
            # storage path.  A cache hit replaces only deterministic CIF parsing and
            # feature construction; the same fixed validation seed freshly samples
            # the noise_level=0.5 diffusion input on every invocation.
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
                batch = None
                if cache is not None:
                    try:
                        batch = recursive_to_device(
                            cache.get(
                                row,
                                FeatureCacheSpec(
                                    manifest_sha256=str(manifest_sha256),
                                    max_context_tokens=max_context_tokens,
                                    diffusion_batch_size=1,
                                    noise_level=0.5,
                                ),
                            ),
                            device,
                            non_blocking=device.type == "cuda",
                        )
                    except FeatureCacheError:
                        if not feature_cache_fallback:
                            raise
                if batch is None:
                    batch = _batch(
                        root,
                        row,
                        device=device,
                        max_context_tokens=max_context_tokens,
                        noise_level=0.5,
                    )
                atom_map = batch.get("f", {}).get("atom_to_token_map")
                if not isinstance(atom_map, torch.Tensor):
                    raise TypeError("batch.f.atom_to_token_map must be a tensor")
                # Training sets an explicit mode on every step so all DDP ranks use
                # one backward graph.  Do not inherit the final training batch's
                # mode here: validation has no backward/gradient synchronization and
                # each rank may evaluate differently sized samples.  Route every
                # local sample independently using the same profiled H100 threshold.
                model.execution_mode = (
                    "chunked"
                    if force_chunked or atom_map.numel() > STANDARD_MODE_MAX_ATOMS
                    else "standard"
                )
                precision = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with precision:
                    metrics = evaluate_loss(model, batch)
                for name, value in metrics.items():
                    totals[name] += value
                local_count += 1
            names = ("loss", "coordinate_loss", "sequence_loss", "seq_recovery")
            packed = torch.tensor(
                [*(totals[name] for name in names), local_count],
                dtype=torch.float64,
                device=device,
            )
            if distributed.world_size > 1:
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            count = int(packed[-1].item())
            if count != len(selected):
                raise RuntimeError(
                    "distributed validation did not cover every selected sample once"
                )
            report[task] = {
                name: float(packed[index].item() / count) for index, name in enumerate(names)
            }
    finally:
        if cache is not None:
            cache.close()
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
    parser.add_argument(
        "--tasks",
        type=_task_names,
        default=list(SPLITS),
        help=(
            "comma-separated frozen task names; defaults to the unchanged "
            "protein_binder,antibody_h3,rna unified cycle"
        ),
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
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "af3"),
        default="constant",
        help="constant control or the pinned public RFD3NA AF3 warmup/decay schedule",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help=(
            "global gradient-norm ceiling; 1.0 preserves the existing NanoDesign "
            "control and 10.0 matches the pinned public RFD3NA trainer"
        ),
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="RFD3NA EMA decay; set to 0 only for a controlled ablation",
    )
    parser.add_argument(
        "--coordinate-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the pinned RFD3NA centering/rigid/COM training augmentation",
    )
    parser.add_argument(
        "--feature-cache-fallback", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--final-generation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the unchanged generation path for the selected task(s) after training",
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
    if args.ema_decay != 0.0 and not 0.0 < args.ema_decay < 1.0:
        raise ValueError("EMA decay must be zero (disabled) or between zero and one")
    if args.feature_cache_stage_root is not None and args.feature_cache_root is None:
        raise ValueError("feature-cache staging requires --feature-cache-root")

    root = Path(__file__).resolve().parents[1]
    resolved = load_config(root / args.config)
    validate_v0_config(resolved).require_ready()
    max_context_tokens = args.max_context_tokens or int(resolved["model"]["max_context_tokens"])
    distributed = context_from_environment()
    task_names = list(args.tasks)
    milestones = args.milestone_samples or []
    if any(value % distributed.world_size for value in milestones):
        raise ValueError("every sample milestone must be divisible by world size")
    if any((value // distributed.world_size) % len(task_names) for value in milestones):
        raise ValueError("sample milestones must end on an exact selected-task cycle")
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
            # RFD3NA's process_pll path is mask/task dependent, so the set of
            # parameters receiving gradients can change between Binder, H3, and RNA
            # steps. Foundry activation checkpointing uses use_reentrant=False, which
            # supports PyTorch's dynamic unused-parameter traversal. Do not declare a
            # static graph here: doing so fails on the first repeated task cycle.
            find_unused_parameters=True,
            static_graph=False,
        )
        if distributed.world_size > 1
        else base_model
    )
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        adam_beta2=args.adam_beta2,
    )
    optimizer = build_optimizer(base_model, training_config)
    lr_scheduler = build_learning_rate_scheduler(
        optimizer,
        schedule=args.lr_schedule,
        base_learning_rate=training_config.learning_rate,
    )
    ema = (
        ExponentialMovingAverage(base_model, decay=args.ema_decay) if args.ema_decay > 0.0 else None
    )
    output_dir = Path(args.output_dir)
    if distributed.is_primary:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed.world_size > 1:
        dist.barrier()
    stats_path = root / "docs/data_v0_stats.json"
    manifest_sha = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    train_rows = _load_rows(root, "train", task_names)
    validation_rows = _load_rows(root, "validation", task_names)
    test_rows = _load_rows(root, "test", task_names)
    shuffled = {}
    for task, rows in train_rows.items():
        task_index = TASK_INDEX[task]
        shuffled[task] = size_aware_rank_packing(
            rows,
            # A fixed flattened order makes 1/2/4-GPU scaling consume exactly the
            # same samples while retaining size-homogeneous 4-rank global batches.
            world_size=SIZE_PACKING_GROUP_SIZE,
            seed=args.seed + task_index,
            max_context_tokens=max_context_tokens,
        )
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
        "learning_rate": training_config.learning_rate,
        "lr_schedule": args.lr_schedule,
        "lr_warmup_steps": 1000 if args.lr_schedule == "af3" else 0,
        "lr_decay_factor": 0.95 if args.lr_schedule == "af3" else 1.0,
        "lr_decay_steps": 50000 if args.lr_schedule == "af3" else 0,
        "weight_decay": training_config.weight_decay,
        "gradient_clip": training_config.gradient_clip,
        "adam_betas": [training_config.adam_beta1, training_config.adam_beta2],
        "ema_decay": args.ema_decay,
        "coordinate_augmentation": args.coordinate_augmentation,
        "sequence_mask_normalization": "per_design_token",
    }
    if args.resume is not None:
        loaded = load_checkpoint(
            args.resume,
            model=base_model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            expected_manifest_sha256=manifest_sha,
            restore_rng=True,
            rng_rank=distributed.rank,
            ema=ema,
        )
        validate_resume_training_run_config(
            loaded.get("training_run_config"),
            training_run_config,
            samples_seen=int(loaded["samples_seen"]),
        )
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
            feature_cache_root=args.feature_cache_root,
            feature_cache_fallback=args.feature_cache_fallback,
            manifest_sha256=manifest_sha,
            force_chunked=force_chunked_execution,
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
            stage_succeeded = True
            if distributed.is_primary:
                try:
                    # One node-local database copy serves all local ranks.  Copying the
                    # same multi-GB SQLite files concurrently from every rank only
                    # amplifies GPFS traffic and can race the atomic sidecars.
                    stage_catalog_cache(
                        args.feature_cache_root, args.feature_cache_stage_root, scheduled_rows
                    )
                except FeatureCacheError:
                    stage_succeeded = False
            if distributed.world_size > 1:
                stage_status = torch.tensor(int(stage_succeeded), dtype=torch.int8, device=device)
                dist.broadcast(stage_status, src=0)
                stage_succeeded = bool(stage_status.item())
                dist.barrier()
            if stage_succeeded:
                selected_cache_root = args.feature_cache_stage_root
            elif not args.feature_cache_fallback:
                raise FeatureCacheError("rank 0 could not stage the finalized feature cache")
            else:
                selected_cache_root = args.feature_cache_root
        cache_spec = FeatureCacheSpec(
            manifest_sha256=manifest_sha,
            max_context_tokens=max_context_tokens,
            diffusion_batch_size=args.diffusion_batch_size,
            noise_level=None,
            random_seed=None,
            augment_coordinates=args.coordinate_augmentation,
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
                lr_scheduler=lr_scheduler,
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
                ema=ema,
            )
        if distributed.world_size > 1:
            dist.barrier()

    def append_milestone_record(completed_step: int) -> None:
        local_rng = capture_rng_state()
        validation_weights = (
            ema.average_parameters(base_model) if ema is not None else nullcontext()
        )
        with validation_weights:
            milestone_validation = _validation(
                base_model,
                root,
                validation_rows,
                device=device,
                max_context_tokens=max_context_tokens,
                samples_per_task=args.validation_samples_per_task,
                seed=args.seed,
                distributed=distributed,
                feature_cache_root=args.feature_cache_root,
                feature_cache_fallback=args.feature_cache_fallback,
                manifest_sha256=manifest_sha,
                force_chunked=force_chunked_execution,
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
                "optimization_wall_seconds": float(
                    sum(float(record.get("step_wall_seconds", 0.0)) for record in history)
                ),
                "optimization_gpu_hours": float(
                    sum(float(record.get("step_wall_seconds", 0.0)) for record in history)
                    * distributed.world_size
                    / 3600
                    if device.type == "cuda"
                    else 0.0
                ),
                "train_loss_per_task": {
                    task_name: float(
                        np.mean(
                            [record["loss"] for record in history if record["task"] == task_name]
                        )
                    )
                    for task_name in task_names
                },
                "train_metrics_per_task": {
                    task_name: {
                        metric_name: float(
                            np.mean(
                                [
                                    record[metric_name]
                                    for record in history
                                    if record["task"] == task_name
                                ]
                            )
                        )
                        # Some random EDM batches have no t < 1 realization, so the
                        # official SequenceLoss does not emit recovery on every train
                        # step.  Stable recovery is recorded on fixed-t validation
                        # batches above; keep train aggregates to always-defined loss.
                        for metric_name in ("loss", "coordinate_loss", "sequence_loss")
                    }
                    for task_name in task_names
                },
                "validation_loss_per_task": {
                    task_name: milestone_validation[task_name]["loss"] for task_name in task_names
                },
                "validation_metrics_per_task": milestone_validation,
            }
        )

    recorded_milestones = {record["global_samples_seen"] for record in milestone_records}
    if start_step == 0 and milestones:
        # Preserve the exact random-initialization reference used by this sweep so
        # milestone metrics can demonstrate learning rather than only lower loss.
        save_training_checkpoint(output_dir / "milestones" / "samples-00000000.pt", 0)
    if samples_seen in milestones and samples_seen not in recorded_milestones:
        append_milestone_record(start_step)
        save_training_checkpoint(
            output_dir / "milestones" / f"samples-{samples_seen:08d}.pt", start_step
        )

    for step in range(start_step, total_steps):
        step_started = time.monotonic()
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
                augment_coordinates=args.coordinate_augmentation,
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
                metrics = train_step(
                    model,
                    optimizer,
                    batch,
                    training_config,
                    ema,
                    lr_scheduler,
                )
        else:
            metrics = train_step(
                model,
                optimizer,
                batch,
                training_config,
                ema,
                lr_scheduler,
            )
        metrics = reduce_scalar_metrics(metrics, device=device, world_size=distributed.world_size)
        sample_ids = all_gather_objects(row["sample_id"], distributed.world_size)
        local_execution_mode = getattr(base_model, "last_execution_mode", "standard")
        if local_execution_mode not in {"standard", "chunked"}:
            raise RuntimeError(f"model reported unknown execution mode {local_execution_mode!r}")
        execution_modes = all_gather_objects(local_execution_mode, distributed.world_size)
        step_wall = torch.tensor(
            time.monotonic() - step_started, dtype=torch.float64, device=device
        )
        if distributed.world_size > 1:
            dist.all_reduce(step_wall, op=dist.ReduceOp.MAX)
        history.append(
            {
                "step": step + 1,
                "task": task,
                "sample_ids": sample_ids,
                "execution_modes": execution_modes,
                "step_wall_seconds": float(step_wall.item()),
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

    final_milestone = milestone_records[-1] if milestone_records else {}
    if (
        final_milestone.get("global_samples_seen") == samples_seen
        and "validation_metrics_per_task" in final_milestone
    ):
        validation_after = final_milestone["validation_metrics_per_task"]
    else:
        validation_weights = (
            ema.average_parameters(base_model) if ema is not None else nullcontext()
        )
        with validation_weights:
            validation_after = _validation(
                base_model,
                root,
                validation_rows,
                device=device,
                max_context_tokens=max_context_tokens,
                samples_per_task=args.validation_samples_per_task,
                seed=args.seed,
                distributed=distributed,
                feature_cache_root=args.feature_cache_root,
                feature_cache_fallback=args.feature_cache_fallback,
                manifest_sha256=manifest_sha,
                force_chunked=force_chunked_execution,
            )
    generation = {}
    if distributed.is_primary and args.final_generation:
        generation_weights = (
            ema.average_parameters(base_model) if ema is not None else nullcontext()
        )
        with generation_weights:
            generation = _generation_outputs(
                base_model,
                root,
                test_rows,
                device=device,
                max_context_tokens=max_context_tokens,
                output_dir=output_dir,
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
        "training_mechanics": {
            "sequence_mask_normalization": "per_design_token",
            "coordinate_loss_mask": "all_resolved_atoms_foundry_occupancy",
            "coordinate_augmentation": args.coordinate_augmentation,
            "generation_origin": "foundry_fixed_motif_com_with_zeroed_design_coordinates",
            "optimizer": "AdamW",
            "learning_rate": training_config.learning_rate,
            "lr_schedule": args.lr_schedule,
            "lr_warmup_steps": 1000 if args.lr_schedule == "af3" else 0,
            "lr_decay_factor": 0.95 if args.lr_schedule == "af3" else 1.0,
            "lr_decay_steps": 50000 if args.lr_schedule == "af3" else 0,
            "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "weight_decay": training_config.weight_decay,
            "gradient_clip": training_config.gradient_clip,
            "adam_betas": [training_config.adam_beta1, training_config.adam_beta2],
            "ema_decay": args.ema_decay,
            "validation_and_generation_weights": "ema" if ema is not None else "online",
        },
        "max_context_tokens": max_context_tokens,
        "task_steps": dict(task_steps),
        "execution_mode_counts_per_task": execution_mode_counts,
        "size_aware_rank_packing": {
            "version": SIZE_PACKING_VERSION,
            "size_identity": "catalog row + max_context_tokens -> Foundry protein14/RNA23 atoms",
            "global_group_size": SIZE_PACKING_GROUP_SIZE,
            "synchronized_standard_max_atoms": STANDARD_MODE_MAX_ATOMS,
        },
        "validation_samples_per_task": args.validation_samples_per_task,
        "generation_split": "test",
        "final_generation_enabled": args.final_generation,
        "generation_selection": (
            "first sample with complete fixed context; first-sample fallback for low-budget smoke"
        ),
        "validation_before": validation_before,
        "validation_after": validation_after,
        "milestones": milestone_records,
        "generation": generation,
        "checkpoint_step": total_steps,
        "history": history,
        "optimization_wall_seconds": float(
            sum(float(record.get("step_wall_seconds", 0.0)) for record in history)
        ),
        "optimization_gpu_hours": float(
            sum(float(record.get("step_wall_seconds", 0.0)) for record in history)
            * distributed.world_size
            / 3600
            if device.type == "cuda"
            else 0.0
        ),
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

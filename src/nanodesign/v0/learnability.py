"""Frozen deterministic denoising evaluation for NanoDesign learnability.

This protocol is deliberately separate from true autoregressive/diffusion generation
and external biological evaluators.  It measures whether training improves the exact
RFD3NA denoising objective on a stable validation panel.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch

from nanodesign.v0.data.cache import FeatureCacheSpec, SQLiteFeatureCache, sha256_file
from nanodesign.v0.data.loader import recursive_to_device
from nanodesign.v0.data.real import load_split_catalog
from nanodesign.v0.model import STANDARD_MODE_MAX_ATOMS, NanoDesignTiny
from nanodesign.v0.training import evaluate_loss

TASK_INDEX = {"protein_binder": 0, "antibody_h3": 1, "rna": 2}


def _ids_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "".join(f"{row['sample_id']}\n" for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def load_frozen_panel(
    root: str | Path, protocol_path: str | Path
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Load and fingerprint the exact validation rows selected by the protocol."""

    root = Path(root).resolve()
    protocol_path = Path(protocol_path)
    if not protocol_path.is_absolute():
        protocol_path = root / protocol_path
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol") not in {
        "nanodesign.learnability.v1",
        "nanodesign.learnability.v2",
    }:
        raise ValueError("unsupported learnability protocol")
    seed = int(protocol["seed"])
    selection_seed_offset = int(protocol.get("selection_seed_offset", 1000))
    tasks = protocol.get("tasks")
    if not isinstance(tasks, Mapping) or list(tasks) != list(TASK_INDEX):
        raise ValueError("learnability tasks must use the frozen canonical order")

    selected_by_task: dict[str, list[dict[str, Any]]] = {}
    for task, task_index in TASK_INDEX.items():
        task_spec = tasks[task]
        catalog_path = root / str(task_spec["catalog"])
        actual_catalog_sha = sha256_file(catalog_path)
        if actual_catalog_sha != task_spec["catalog_sha256"]:
            raise ValueError(f"{task}: validation catalog SHA256 changed")
        rows = load_split_catalog(catalog_path)
        panel_size = int(task_spec["panel_size"])
        if panel_size < 1 or panel_size > len(rows):
            raise ValueError(f"{task}: invalid panel size {panel_size}")
        selected = random.Random(seed + selection_seed_offset + task_index).sample(
            rows, panel_size
        )
        if _ids_sha256(selected) != task_spec["selected_sample_ids_sha256"]:
            raise ValueError(f"{task}: frozen panel sample IDs changed")
        selected_by_task[task] = selected
    return protocol, selected_by_task


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


@contextmanager
def _deterministic_linear_pool(model: NanoDesignTiny):
    """Replace Foundry's non-deterministic CUDA index-reduce during evaluation.

    Atom-to-token indices are contiguous and sorted in every NanoDesign atom23
    example. A mean over each consecutive segment is mathematically identical to
    ``index_reduce(..., 'mean')`` but avoids CUDA atomics. This changes no parameter,
    feature, loss or architecture and is never enabled by training/generation.
    """

    pool = model.net.diffusion_module.process_a

    def deterministic_forward(self, coordinates, tok_idx):
        token_indices = tok_idx.detach().cpu().long()
        unique, counts = torch.unique_consecutive(token_indices, return_counts=True)
        expected = torch.arange(len(unique), dtype=unique.dtype)
        if not torch.equal(unique, expected):
            raise ValueError("deterministic pooling requires contiguous sorted token indices")
        embedded = self.linear(coordinates)
        device_counts = counts.to(device=embedded.device)
        ends = torch.cumsum(device_counts, dim=0) - 1
        cumulative = torch.cumsum(embedded, dim=1)
        sums = cumulative.index_select(1, ends).clone()
        if len(ends) > 1:
            sums[:, 1:, :] -= cumulative.index_select(1, ends[:-1])
        return sums / device_counts[None, :, None]

    pool.forward = MethodType(deterministic_forward, pool)
    try:
        yield
    finally:
        del pool.forward


@torch.no_grad()
def evaluate_frozen_panel(
    model: NanoDesignTiny,
    *,
    root: str | Path,
    protocol_path: str | Path,
    feature_cache_root: str | Path,
    manifest_sha256: str,
    device: torch.device,
    max_context_tokens: int,
) -> dict[str, Any]:
    """Evaluate every frozen panel row once with exact per-sample diffusion seeds."""

    root = Path(root).resolve()
    protocol, rows_by_task = load_frozen_panel(root, protocol_path)
    diffusion_t = float(protocol["diffusion_t"])
    diffusion_batch_size = int(protocol["diffusion_realizations_per_complex"])
    if diffusion_batch_size != 1:
        raise ValueError("learnability.v1 freezes one diffusion realization per complex")

    started = time.monotonic()
    task_reports: dict[str, Any] = {}
    # Validation is a reward signal, so repeatability takes precedence over the
    # small throughput cost of deterministic CUDA kernels.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with (
        _deterministic_linear_pool(model),
        SQLiteFeatureCache(feature_cache_root, readonly=True, lru_size=4) as cache,
    ):
        for task, rows in rows_by_task.items():
            metrics_by_name: dict[str, list[float]] = defaultdict(list)
            execution_modes: Counter[str] = Counter()
            task_started = time.monotonic()
            task_index = TASK_INDEX[task]
            for sample_index, row in enumerate(rows):
                sample_seed = int(protocol["seed"]) + 10_000 * task_index + sample_index
                batch = cache.get(
                    row,
                    FeatureCacheSpec(
                        manifest_sha256=manifest_sha256,
                        max_context_tokens=max_context_tokens,
                        diffusion_batch_size=diffusion_batch_size,
                        noise_level=diffusion_t,
                        random_seed=sample_seed,
                        augment_coordinates=bool(protocol["coordinate_augmentation"]),
                    ),
                )
                batch = recursive_to_device(
                    batch, device, non_blocking=device.type == "cuda"
                )
                # RFD3NA's local attention index builder uses random padding when a
                # neighbourhood is undersubscribed.  The diffusion input seed alone
                # is therefore insufficient: reset CPU and CUDA RNG immediately
                # before every model forward as part of the frozen protocol.
                torch.manual_seed(sample_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(sample_seed)
                atom_map = batch["f"]["atom_to_token_map"]
                model.execution_mode = (
                    "standard"
                    if int(atom_map.numel()) <= STANDARD_MODE_MAX_ATOMS
                    else "chunked"
                )
                precision = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with precision:
                    metrics = evaluate_loss(model, batch)
                for name, value in metrics.items():
                    metrics_by_name[name].append(float(value))
                execution_modes[model.last_execution_mode or model.execution_mode] += 1
            task_reports[task] = {
                "sample_count": len(rows),
                "sample_ids_sha256": _ids_sha256(rows),
                "metrics": {
                    name: _summary(values) for name, values in sorted(metrics_by_name.items())
                },
                "execution_mode_counts": dict(sorted(execution_modes.items())),
                "wall_time_seconds": time.monotonic() - task_started,
            }

    return {
        "protocol": protocol["protocol"],
        "protocol_sha256": sha256_file(
            Path(protocol_path) if Path(protocol_path).is_absolute() else root / protocol_path
        ),
        "manifest_sha256": manifest_sha256,
        "device": str(device),
        "precision": "bfloat16" if device.type == "cuda" else "float32",
        "tasks": task_reports,
        "wall_time_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }

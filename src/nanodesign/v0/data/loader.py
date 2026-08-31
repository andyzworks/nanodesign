"""Asynchronous DataLoader helpers for the integrity-checked RFD3 feature cache."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from nanodesign.v0.data.cache import (
    FeatureCacheError,
    FeatureCacheSpec,
    SQLiteFeatureCache,
    preprocess_feature_batch,
    stage_cache_database,
)


class CachedFeatureDataset(Dataset[dict[str, Any]]):
    """Ordered rows backed by one lazy, per-worker read-only SQLite connection set.

    The caller owns task scheduling, shuffling, and DDP rank partitioning by passing the
    exact row sequence for this worker/rank. This class only loads that sequence.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        rows: Sequence[dict[str, Any]],
        spec: FeatureCacheSpec,
        *,
        cache_root: str | Path,
        allow_fallback: bool = True,
        lru_size: int = 8,
        sampling_seeds: Sequence[int] | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.rows = list(rows)
        self.spec = spec
        self.cache_root = Path(cache_root)
        self.allow_fallback = allow_fallback
        self.lru_size = lru_size
        self.sampling_seeds = list(sampling_seeds) if sampling_seeds is not None else None
        if self.sampling_seeds is not None and len(self.sampling_seeds) != len(self.rows):
            raise ValueError("sampling_seeds must match the ordered row sequence")
        self._cache: SQLiteFeatureCache | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def _worker_cache(self) -> SQLiteFeatureCache:
        if self._cache is None:
            self._cache = SQLiteFeatureCache(self.cache_root, readonly=True, lru_size=self.lru_size)
        return self._cache

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        spec = (
            replace(self.spec, random_seed=self.sampling_seeds[index])
            if self.sampling_seeds is not None
            else self.spec
        )
        try:
            return self._worker_cache().get(row, spec)
        except FeatureCacheError:
            if not self.allow_fallback:
                raise
            return preprocess_feature_batch(self.dataset_root, row, spec)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # SQLite connections are process-local and must never cross a spawn/fork boundary.
        state["_cache"] = None
        return state

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    def __del__(self) -> None:
        self.close()


def _single_item(items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(items) != 1:
        raise ValueError("NanoDesign v0 loader expects one variable-size complex per batch")
    return items[0]


def build_async_feature_loader(
    dataset: CachedFeatureDataset,
    *,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    multiprocessing_context: str | None = "spawn",
) -> DataLoader:
    """Build a one-complex asynchronous loader without changing caller row order."""

    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive")
    options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": 1,
        "shuffle": False,
        "collate_fn": _single_item,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers:
        options.update(
            {
                "prefetch_factor": prefetch_factor,
                "persistent_workers": persistent_workers,
                "multiprocessing_context": multiprocessing_context,
            }
        )
    return DataLoader(**options)


def recursive_to_device(
    value: Any,
    device: torch.device | str,
    *,
    non_blocking: bool = True,
) -> Any:
    """Recursively transfer tensor leaves while preserving provenance containers."""

    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, dict):
        return {
            key: recursive_to_device(item, device, non_blocking=non_blocking)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [recursive_to_device(item, device, non_blocking=non_blocking) for item in value]
    if isinstance(value, tuple):
        return tuple(recursive_to_device(item, device, non_blocking=non_blocking) for item in value)
    return value


def stage_catalog_cache(
    shared_cache_root: str | Path,
    stage_root: str | Path,
    rows: Sequence[dict[str, Any]],
) -> list[Path]:
    """Stage each distinct task/split database once before worker processes start."""

    databases = sorted({(str(row["task"]), str(row["split"])) for row in rows})
    return [
        stage_cache_database(shared_cache_root, stage_root, task, split)
        for task, split in databases
    ]

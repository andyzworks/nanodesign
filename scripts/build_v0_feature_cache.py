#!/usr/bin/env python3
"""Build or preflight the integrity-checked NanoDesign v0 SQLite feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from nanodesign.v0.data.cache import (
    FeatureCacheSpec,
    SQLiteFeatureCache,
    cache_database_path,
    finalize_cache_database,
    model_ready_batches_equal,
    preprocess_feature_batch,
    sha256_file,
    stage_cache_database,
    verify_finalized_database,
)


def _rows(paths: list[Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            result.extend(json.loads(line) for line in handle if line.strip())
    if len({row["sample_id"] for row in result}) != len(result):
        raise ValueError("cache catalogs contain duplicate sample IDs")
    return result


def _sample_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def _read_worker(arguments: tuple[Path, dict[str, Any], FeatureCacheSpec, int]) -> float:
    cache_root, row, spec, reads = arguments
    with SQLiteFeatureCache(cache_root, readonly=True, lru_size=1) as cache:
        cache.get(row, spec)
        started = time.perf_counter()
        for _ in range(reads):
            cache.get(row, spec)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--catalog", type=Path, action="append", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-context-tokens", type=int, default=384)
    parser.add_argument("--diffusion-batch-size", type=int, default=1)
    parser.add_argument("--noise-level", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--benchmark-samples", type=int, default=3)
    parser.add_argument("--read-workers", type=int, default=4)
    parser.add_argument("--reads-per-worker", type=int, default=10)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--stage-root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.benchmark_samples < 0:
        raise ValueError("benchmark-samples must be non-negative")
    if args.read_workers < 1:
        raise ValueError("read-workers must be positive")
    if args.reads_per_worker < 1:
        raise ValueError("reads-per-worker must be positive")

    root = args.dataset_root.resolve()
    catalog_paths = [path if path.is_absolute() else root / path for path in args.catalog]
    rows = _rows(catalog_paths)
    if args.limit is not None:
        rows = rows[: args.limit]
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    manifest_sha = sha256_file(manifest)
    specs = {
        row["sample_id"]: FeatureCacheSpec(
            manifest_sha256=manifest_sha,
            max_context_tokens=args.max_context_tokens,
            diffusion_batch_size=args.diffusion_batch_size,
            noise_level=args.noise_level,
            random_seed=_sample_seed(args.seed, row["sample_id"]),
        )
        for row in rows
    }
    timings = defaultdict(float)
    built_batches: dict[str, dict[str, Any]] = {}
    if not args.preflight_only:
        with SQLiteFeatureCache(args.cache_root, readonly=False, lru_size=0) as cache:
            for index, row in enumerate(rows, 1):
                started = time.perf_counter()
                batch = preprocess_feature_batch(root, row, specs[row["sample_id"]])
                timings["preprocessing_seconds"] += time.perf_counter() - started
                started = time.perf_counter()
                cache.put(row, specs[row["sample_id"]], batch)
                timings["write_seconds"] += time.perf_counter() - started
                if index <= args.benchmark_samples:
                    built_batches[row["sample_id"]] = batch
                if index % 100 == 0 or index == len(rows):
                    print(f"cached {index}/{len(rows)}", flush=True)

    databases = sorted({(row["task"], row["split"]) for row in rows})
    if not args.preflight_only:
        for task, split in databases:
            finalize_cache_database(args.cache_root, task, split)

    exact_matches = 0
    benchmark_rows = rows[: args.benchmark_samples]
    for task, split in databases:
        verify_finalized_database(cache_database_path(args.cache_root, task, split))
    with SQLiteFeatureCache(args.cache_root, readonly=True, lru_size=0) as cache:
        for task, split in databases:
            cache.quick_check(task, split)
        for row in rows:
            started = time.perf_counter()
            cached = cache.get(row, specs[row["sample_id"]])
            timings["cache_read_seconds"] += time.perf_counter() - started
            original = built_batches.get(row["sample_id"])
            if original is not None:
                if not model_ready_batches_equal(original, cached):
                    raise RuntimeError(f"cached tensors differ for {row['sample_id']}")
                exact_matches += 1

    uncached_benchmark_seconds = 0.0
    cached_benchmark_seconds = 0.0
    if benchmark_rows:
        with SQLiteFeatureCache(args.cache_root, readonly=True, lru_size=0) as cache:
            for row in benchmark_rows:
                started = time.perf_counter()
                uncached = preprocess_feature_batch(root, row, specs[row["sample_id"]])
                uncached_benchmark_seconds += time.perf_counter() - started
                started = time.perf_counter()
                cached = cache.get(row, specs[row["sample_id"]])
                cached_benchmark_seconds += time.perf_counter() - started
                if not model_ready_batches_equal(uncached, cached):
                    raise RuntimeError(f"preprocessing/cache mismatch for {row['sample_id']}")
                exact_matches += int(row["sample_id"] not in built_batches)

    staged = []
    if args.stage_root is not None:
        for task, split in databases:
            staged.append(
                str(stage_cache_database(args.cache_root, args.stage_root, task, split).resolve())
            )
    multiworker_root = args.stage_root or args.cache_root
    multiworker_arguments = (
        [
            (
                multiworker_root,
                benchmark_rows[index % len(benchmark_rows)],
                specs[benchmark_rows[index % len(benchmark_rows)]["sample_id"]],
                args.reads_per_worker,
            )
            for index in range(args.read_workers)
        ]
        if benchmark_rows
        else []
    )
    multiworker_started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.read_workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        multiworker_read_seconds = list(executor.map(_read_worker, multiworker_arguments))
    multiworker_wall_seconds = time.perf_counter() - multiworker_started
    report = {
        "cache_format": "sqlite-task-split-v1",
        "manifest": str(manifest.resolve()),
        "manifest_sha256": manifest_sha,
        "rows": len(rows),
        "databases": [
            str((args.cache_root / task / f"{split}.sqlite3").resolve())
            for task, split in databases
        ],
        "preflight_only": args.preflight_only,
        "exact_tensor_matches": exact_matches,
        "benchmark_samples": len(benchmark_rows),
        "uncached_benchmark_seconds": uncached_benchmark_seconds,
        "cached_benchmark_seconds": cached_benchmark_seconds,
        "benchmark_speedup": (
            uncached_benchmark_seconds / cached_benchmark_seconds
            if cached_benchmark_seconds > 0
            else None
        ),
        "multiworker": {
            "workers": args.read_workers,
            "reads_per_worker": args.reads_per_worker,
            "reads": len(multiworker_read_seconds) * args.reads_per_worker,
            "wall_seconds": multiworker_wall_seconds,
            "sum_read_seconds": sum(multiworker_read_seconds),
            "reads_per_second": (
                len(multiworker_read_seconds)
                * args.reads_per_worker
                / max(multiworker_read_seconds)
                if multiworker_read_seconds
                else None
            ),
            "cache_root": str(Path(multiworker_root).resolve()),
        },
        "timings": dict(timings),
        "staged_databases": staged,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

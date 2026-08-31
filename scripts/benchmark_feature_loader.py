#!/usr/bin/env python3
"""Benchmark async SQLite feature loading and non-blocking H2D transfer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from nanodesign.v0.data.cache import FeatureCacheSpec, sha256_file
from nanodesign.v0.data.loader import (
    CachedFeatureDataset,
    build_async_feature_loader,
    recursive_to_device,
    stage_catalog_cache,
)


def _rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--catalog", type=Path, action="append", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--catalog-limit", type=int)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--max-context-tokens", type=int, default=384)
    parser.add_argument("--diffusion-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--simulated-compute-ms", type=float, default=10.0)
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.samples, args.warmup + 1, args.diffusion_batch_size) < 1:
        raise ValueError("samples, warmup, and diffusion batch size must be positive")
    if args.simulated_compute_ms < 0:
        raise ValueError("simulated-compute-ms must be non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but unavailable")

    root = args.dataset_root.resolve()
    paths = [path if path.is_absolute() else root / path for path in args.catalog]
    available_rows = _rows(paths)
    if args.catalog_limit is not None:
        if args.catalog_limit < 1:
            raise ValueError("catalog-limit must be positive")
        available_rows = available_rows[: args.catalog_limit]
    if not available_rows:
        raise ValueError("benchmark catalogs contain no rows")
    scheduled_rows = [available_rows[index % len(available_rows)] for index in range(args.samples)]
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    spec = FeatureCacheSpec(
        manifest_sha256=sha256_file(manifest),
        max_context_tokens=args.max_context_tokens,
        diffusion_batch_size=args.diffusion_batch_size,
        noise_level=None,
        random_seed=None,
    )
    staged = stage_catalog_cache(args.cache_root, args.stage_root, scheduled_rows)
    configurations = [(0, None)] + [
        (workers, prefetch) for workers in (2, 4, 8) for prefetch in (2, 4)
    ]
    results = []
    for workers, prefetch in configurations:
        dataset = CachedFeatureDataset(
            root,
            scheduled_rows,
            spec,
            cache_root=args.stage_root,
            allow_fallback=args.allow_fallback,
            lru_size=8,
        )
        loader = build_async_feature_loader(
            dataset,
            num_workers=workers,
            prefetch_factor=prefetch or 2,
            persistent_workers=workers > 0,
            pin_memory=device.type == "cuda",
            multiprocessing_context="spawn",
        )
        iterator = iter(loader)
        for _ in range(min(args.warmup, args.samples - 1)):
            batch = recursive_to_device(next(iterator), device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            del batch
        measured = args.samples - min(args.warmup, args.samples - 1)
        next_wait = 0.0
        h2d_wait = 0.0
        started_total = time.perf_counter()
        for _ in range(measured):
            started = time.perf_counter()
            batch = next(iterator)
            next_wait += time.perf_counter() - started
            started = time.perf_counter()
            batch = recursive_to_device(batch, device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            h2d_wait += time.perf_counter() - started
            if args.simulated_compute_ms:
                time.sleep(args.simulated_compute_ms / 1000.0)
            del batch
        wall = time.perf_counter() - started_total
        gpu_wait = next_wait + h2d_wait
        results.append(
            {
                "num_workers": workers,
                "prefetch_factor": prefetch,
                "persistent_workers": workers > 0,
                "pin_memory": device.type == "cuda",
                "samples": measured,
                "wall_seconds": wall,
                "samples_per_second": measured / wall,
                "next_wait_seconds": next_wait,
                "h2d_wait_seconds": h2d_wait,
                "gpu_data_wait_seconds": gpu_wait,
                "gpu_data_wait_fraction": gpu_wait / wall,
            }
        )
        del iterator, loader, dataset
    best = max(results, key=lambda result: result["samples_per_second"])
    report = {
        "device": str(device),
        "diffusion_batch_size": args.diffusion_batch_size,
        "max_context_tokens": args.max_context_tokens,
        "simulated_compute_ms": args.simulated_compute_ms,
        "staged_databases": [str(path.resolve()) for path in staged],
        "results": results,
        "recommended": {
            "num_workers": best["num_workers"],
            "prefetch_factor": best["prefetch_factor"],
            "pin_memory": best["pin_memory"],
            "persistent_workers": best["persistent_workers"],
            "samples_per_second": best["samples_per_second"],
            "gpu_data_wait_fraction": best["gpu_data_wait_fraction"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

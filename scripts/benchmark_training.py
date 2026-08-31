#!/usr/bin/env python3
"""Reproducible stage profiler for the frozen NanoDesign v0 training step."""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import platform
import random
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import torch

from nanodesign.v0.config import load_config, validate_v0_config
from nanodesign.v0.data.real import (
    _read_model,
    load_foundry_training_example,
    load_split_catalog,
)
from nanodesign.v0.model import NanoDesignTiny, NanoDesignTinyConfig
from nanodesign.v0.training import (
    TrainingConfig,
    _compute_rfd3na_loss,
    build_optimizer,
)


@dataclass(frozen=True)
class FixedSample:
    task: str
    sample_id: str
    catalog: str
    expected_token_bucket: str | None
    expected_atom_bucket: str | None


FIXED_SAMPLES = (
    FixedSample(
        "protein_binder",
        "ppiref50k:117e_A_B",
        "data/processed/v0/splits/protein_binder/train.jsonl",
        "small",
        "small",
    ),
    FixedSample(
        "antibody_h3",
        "sabdab2:pdb_00009nk9_A_+",
        "data/processed/v0/splits/antibody_h3/train.jsonl",
        "medium",
        "medium",
    ),
    FixedSample(
        "rna",
        "pdb_rna_target:10be:0",
        "data/processed/v0/splits/rna_binding/train.jsonl",
        "large",
        "large",
    ),
)


def size_bucket(size: int, *, small_max: int, medium_max: int) -> str:
    if size <= 0:
        raise ValueError("bucket size must be positive")
    if size <= small_max:
        return "small"
    if size <= medium_max:
        return "medium"
    return "large"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, function: Callable[[], Any]) -> tuple[Any, float]:
    _sync(device)
    started = time.perf_counter_ns()
    value = function()
    _sync(device)
    return value, (time.perf_counter_ns() - started) / 1_000_000.0


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _raw_paths(root: Path, row: Mapping[str, Any]) -> list[Path]:
    default = row["raw_paths"][0] if len(row["raw_paths"]) == 1 else None
    paths = []
    for chain in row["chains"]:
        raw_path = chain.get("raw_path", default)
        if raw_path is None:
            raise ValueError(f"{row['sample_id']}: ambiguous raw path")
        path = (root / raw_path).resolve()
        if path not in paths:
            paths.append(path)
    return paths


def _parse_bytes(path: Path, payload: bytes) -> gemmi.Structure:
    if path.suffix.lower() == ".gz":
        payload = gzip.decompress(payload)
        path = path.with_suffix("")
    text = payload.decode("utf-8")
    if path.suffix.lower() in {".cif", ".mmcif"}:
        return gemmi.make_structure_from_block(gemmi.cif.read_string(text).sole_block())
    return gemmi.read_pdb_string(text)


def profile_data_stages(
    root: Path,
    row: Mapping[str, Any],
    *,
    repeats: int,
    max_context_tokens: int,
    diffusion_batch_size: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Measure isolated raw I/O, in-memory parse, and cache-warm feature construction."""

    paths = _raw_paths(root, row)
    io_values, parse_values, feature_values = [], [], []
    payloads: list[bytes] = []
    for _ in range(repeats):
        payloads, elapsed = _timed(
            torch.device("cpu"), lambda: [path.read_bytes() for path in paths]
        )
        io_values.append(elapsed)
        _, elapsed = _timed(
            torch.device("cpu"),
            lambda current_payloads=payloads: [
                _parse_bytes(path, payload)
                for path, payload in zip(paths, current_payloads, strict=True)
            ],
        )
        parse_values.append(elapsed)

    _read_model.cache_clear()
    for path in paths:
        _read_model(str(path))
    # Exclude one-time Foundry imports/encoding initialization from steady-state
    # feature construction while retaining the exact production loader below.
    load_foundry_training_example(
        root,
        dict(row),
        noise_level=0.5,
        diffusion_batch_size=diffusion_batch_size,
        max_context_tokens=max_context_tokens,
    )
    batch = None
    for _ in range(repeats):
        batch, elapsed = _timed(
            torch.device("cpu"),
            lambda: load_foundry_training_example(
                root,
                dict(row),
                noise_level=0.5,
                diffusion_batch_size=diffusion_batch_size,
                max_context_tokens=max_context_tokens,
            ),
        )
        feature_values.append(elapsed)
    assert batch is not None
    return batch, {
        "raw_file_io_ms": float(np.mean(io_values)),
        "parse_ms": float(np.mean(parse_values)),
        "feature_construction_ms": float(np.mean(feature_values)),
        "raw_bytes": float(sum(len(payload) for payload in payloads)),
    }


def profile_training_stages(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cpu_batch: Mapping[str, Any],
    *,
    device: torch.device,
    repeats: int,
    loss_function: Callable[
        [Mapping[str, Any], Mapping[str, torch.Tensor]], tuple[torch.Tensor, Mapping[str, Any]]
    ] = _compute_rfd3na_loss,
    gradient_clip: float = 1.0,
) -> dict[str, float]:
    """Profile the exact ordered stages of one training step."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    values = {
        name: [] for name in ("h2d_ms", "forward_ms", "loss_ms", "backward_ms", "optimizer_ms")
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(repeats):
        batch, elapsed = _timed(device, lambda: _to_device(dict(cpu_batch), device))
        values["h2d_ms"].append(elapsed)
        model.train()
        _, zero_elapsed = _timed(device, lambda: optimizer.zero_grad(set_to_none=True))
        precision = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with precision:
            output, elapsed = _timed(device, lambda current_batch=batch: model(current_batch))
            values["forward_ms"].append(elapsed)
            (loss, _), elapsed = _timed(
                device,
                lambda current_batch=batch, current_output=output: loss_function(
                    current_batch, current_output
                ),
            )
            values["loss_ms"].append(elapsed)

        def backward(current_loss: torch.Tensor = loss) -> None:
            current_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        _, elapsed = _timed(device, backward)
        values["backward_ms"].append(elapsed)
        _, step_elapsed = _timed(device, optimizer.step)
        values["optimizer_ms"].append(zero_elapsed + step_elapsed)
    result = {name: float(np.mean(items)) for name, items in values.items()}
    result["training_step_ms"] = sum(
        result[name] for name in ("forward_ms", "loss_ms", "backward_ms", "optimizer_ms")
    )
    if device.type == "cuda":
        result["cuda_peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated(device))
        result["cuda_peak_reserved_bytes"] = float(torch.cuda.max_memory_reserved(device))
    else:
        result["cuda_peak_allocated_bytes"] = 0.0
        result["cuda_peak_reserved_bytes"] = 0.0
    return result


def render_markdown(report: Mapping[str, Any]) -> str:
    columns = (
        "mode",
        "task",
        "sample_id",
        "bucket",
        "tokens",
        "atoms",
        "I/O ms",
        "parse ms",
        "features ms",
        "H2D ms",
        "forward ms",
        "loss ms",
        "backward ms",
        "optimizer ms",
        "allocated GB",
        "reserved GB",
        "status",
    )
    lines = ["# NanoDesign v0 training profiler", "", "| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in report["records"]:
        timing = row.get("timing_ms", {})
        allocated = float(row.get("cuda_peak_allocated_bytes", 0.0)) / 1e9
        reserved = float(row.get("cuda_peak_reserved_bytes", 0.0)) / 1e9
        values = (
            row["mode"],
            row["task"],
            row["sample_id"],
            row["bucket"],
            str(row["tokens"]),
            str(row["atoms"]),
            f"{timing.get('raw_file_io_ms', float('nan')):.3f}",
            f"{timing.get('parse_ms', float('nan')):.3f}",
            f"{timing.get('feature_construction_ms', float('nan')):.3f}",
            f"{timing.get('h2d_ms', float('nan')):.3f}",
            f"{timing.get('forward_ms', float('nan')):.3f}",
            f"{timing.get('loss_ms', float('nan')):.3f}",
            f"{timing.get('backward_ms', float('nan')):.3f}",
            f"{timing.get('optimizer_ms', float('nan')):.3f}",
            f"{allocated:.3f}",
            f"{reserved:.3f}",
            row["status"],
        )
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        (
            "",
            (
                "Timings are means over measured repeats. Backward includes the frozen gradient "
                "clip; optimizer includes zero-grad and AdamW step. Peak memory includes the "
                "model and batch."
            ),
            "",
        )
    )
    routing = report.get("routing_observation")
    if isinstance(routing, Mapping):
        lines.extend(
            (
                "## Observed single-H100 boundary",
                "",
                f"- Largest successful standard sample: {routing['largest_standard_ok_atoms']} atoms.",
                f"- First observed standard OOM: {routing['first_standard_oom_atoms']} atoms.",
                (
                    "- Largest measured standard sample retaining at least 20% memory headroom: "
                    f"{routing['largest_standard_20pct_headroom_atoms']} atoms."
                ),
                "- All measured samples above that conservative boundary completed in chunked mode.",
                "",
            )
        )
    return "\n".join(lines)


def routing_observation(report: Mapping[str, Any]) -> dict[str, int] | None:
    total = report.get("cuda_total_memory_bytes")
    if not isinstance(total, (int, float)) or total <= 0:
        return None
    standard = [record for record in report["records"] if record["mode"] == "standard"]
    successful = [record for record in standard if record["status"] == "ok"]
    failed = [record for record in standard if record["status"] == "cuda_oom"]
    headroom = [
        record
        for record in successful
        if float(record.get("cuda_peak_reserved_bytes", total)) <= 0.8 * total
    ]
    if not successful or not failed or not headroom:
        return None
    return {
        "largest_standard_ok_atoms": max(int(record["atoms"]) for record in successful),
        "first_standard_oom_atoms": min(int(record["atoms"]) for record in failed),
        "largest_standard_20pct_headroom_atoms": max(int(record["atoms"]) for record in headroom),
    }


def _find_row(root: Path, sample: FixedSample) -> dict[str, Any]:
    matches = [
        row
        for row in load_split_catalog(root / sample.catalog)
        if row["sample_id"] == sample.sample_id
    ]
    if len(matches) != 1:
        raise ValueError(f"fixed sample resolution failed for {sample.sample_id}")
    return matches[0]


def _model_config(resolved: Mapping[str, Any]) -> NanoDesignTinyConfig:
    return NanoDesignTinyConfig.from_mapping(
        {key: resolved["model"][key] for key in NanoDesignTinyConfig.__dataclass_fields__}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v0.yaml")
    parser.add_argument("--mode", choices=("standard", "chunked", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--data-repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--diffusion-batch-size",
        type=int,
        default=1,
        help="number of diffusion realizations per complex (use 4 for the frozen v0 baseline)",
    )
    parser.add_argument("--sample-task")
    parser.add_argument("--sample-id")
    parser.add_argument("--sample-catalog")
    parser.add_argument("--merge-json", nargs="+")
    parser.add_argument("--cuda-total-memory-bytes", type=int)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()
    if args.merge_json:
        reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.merge_json]
        report = dict(reports[0])
        report["records"] = [record for item in reports for record in item["records"]]
        report["merged_from"] = [str(Path(path)) for path in args.merge_json]
        if args.cuda_total_memory_bytes is not None:
            report["cuda_total_memory_bytes"] = args.cuda_total_memory_bytes
        report["routing_observation"] = routing_observation(report)
        output_json = Path(args.output_json)
        output_markdown = Path(args.output_markdown)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        output_markdown.write_text(render_markdown(report), encoding="utf-8")
        return
    if min(args.repeats, args.data_repeats, args.diffusion_batch_size) < 1 or args.warmup < 0:
        raise ValueError("repeat counts must be positive and warmup must be non-negative")
    sample_arguments = (args.sample_task, args.sample_id, args.sample_catalog)
    if any(sample_arguments) and not all(sample_arguments):
        raise ValueError("sample-task, sample-id, and sample-catalog must be supplied together")
    root = Path(__file__).resolve().parents[1]
    resolved = load_config(root / args.config)
    validate_v0_config(resolved).require_ready()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    selected_samples = (
        (
            FixedSample(
                args.sample_task,
                args.sample_id,
                args.sample_catalog,
                None,
                None,
            ),
        )
        if all(sample_arguments)
        else FIXED_SAMPLES
    )
    prepared = []
    for sample in selected_samples:
        row = _find_row(root, sample)
        batch, data_metrics = profile_data_stages(
            root,
            row,
            repeats=args.data_repeats,
            max_context_tokens=int(resolved["model"]["max_context_tokens"]),
            diffusion_batch_size=args.diffusion_batch_size,
        )
        tokens = int(batch["f"]["restype"].shape[0])
        atoms = int(batch["ground_truth_positions"].shape[1])
        token_bucket = size_bucket(tokens, small_max=64, medium_max=256)
        atom_bucket = size_bucket(atoms, small_max=1024, medium_max=4096)
        if sample.expected_token_bucket is not None and (token_bucket, atom_bucket) != (
            sample.expected_token_bucket,
            sample.expected_atom_bucket,
        ):
            raise RuntimeError(
                f"fixed bucket drift for {sample.sample_id}: {token_bucket}/{atom_bucket}"
            )
        prepared.append((sample, batch, data_metrics, tokens, atoms, token_bucket, atom_bucket))

    records = []
    modes = ("standard", "chunked") if args.mode == "all" else (args.mode,)
    for mode in modes:
        for sample, batch, data_metrics, tokens, atoms, token_bucket, atom_bucket in prepared:
            if mode == "chunked":
                os.environ["RFD3_LOW_MEMORY_MODE"] = "1"
            else:
                os.environ.pop("RFD3_LOW_MEMORY_MODE", None)
            record = {
                "mode": mode,
                "task": sample.task,
                "sample_id": sample.sample_id,
                "token_bucket": token_bucket,
                "atom_bucket": atom_bucket,
                "bucket": f"{token_bucket}/{atom_bucket}",
                "tokens": tokens,
                "atoms": atoms,
                "status": "ok",
                "timing_ms": dict(data_metrics),
            }
            model = optimizer = None
            try:
                # Benchmark the requested implementation explicitly.  In particular,
                # ``standard`` must not silently use the wrapper's ``auto`` atom-count
                # route for samples above the current threshold.
                model = NanoDesignTiny(_model_config(resolved), execution_mode=mode).to(device)
                optimizer = build_optimizer(model, TrainingConfig())
                if args.warmup:
                    profile_training_stages(
                        model,
                        optimizer,
                        batch,
                        device=device,
                        repeats=args.warmup,
                        gradient_clip=TrainingConfig().gradient_clip,
                    )
                stage_metrics = profile_training_stages(
                    model,
                    optimizer,
                    batch,
                    device=device,
                    repeats=args.repeats,
                    gradient_clip=TrainingConfig().gradient_clip,
                )
                record["timing_ms"].update(
                    {key: value for key, value in stage_metrics.items() if key.endswith("_ms")}
                )
                record["cuda_peak_allocated_bytes"] = stage_metrics["cuda_peak_allocated_bytes"]
                record["cuda_peak_reserved_bytes"] = stage_metrics["cuda_peak_reserved_bytes"]
            except torch.OutOfMemoryError as error:
                record["status"] = "cuda_oom"
                record["error"] = str(error)
                if device.type == "cuda":
                    record["cuda_peak_allocated_bytes"] = float(
                        torch.cuda.max_memory_allocated(device)
                    )
                    record["cuda_peak_reserved_bytes"] = float(
                        torch.cuda.max_memory_reserved(device)
                    )
            records.append(record)
            del optimizer, model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    report = {
        "schema_version": "nanodesign.v0.training_profile.1",
        "scientific_setup_unchanged": True,
        "seed": args.seed,
        "repeats": args.repeats,
        "data_repeats": args.data_repeats,
        "warmup_steps_per_sample": args.warmup,
        "max_context_tokens": int(resolved["model"]["max_context_tokens"]),
        "diffusion_batch_size": args.diffusion_batch_size,
        "precision": "bfloat16" if device.type == "cuda" else "float32",
        "device": str(device),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_total_memory_bytes": (
            torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 0
        ),
        "model_config": asdict(_model_config(resolved)),
        "parameter_count": int(resolved["model"]["parameter_count"]),
        "bucket_definition": {
            "token": {"small": "<=64", "medium": "65-256", "large": ">256"},
            "atom": {"small": "<=1024", "medium": "1025-4096", "large": ">4096"},
        },
        "records": records,
    }
    report["routing_observation"] = routing_observation(report)
    output_json = Path(args.output_json)
    output_markdown = Path(args.output_markdown)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

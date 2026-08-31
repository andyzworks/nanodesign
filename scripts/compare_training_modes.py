#!/usr/bin/env python3
"""Compare standard and chunked RFD3NA training numerics on fixed batches."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from nanodesign.v0.config import load_config, validate_v0_config
from nanodesign.v0.data.real import load_foundry_training_example, load_split_catalog
from nanodesign.v0.model import NanoDesignTiny, NanoDesignTinyConfig
from nanodesign.v0.training import _compute_rfd3na_loss

SAMPLES = (
    (
        "small",
        "protein_binder",
        "ppiref50k:117e_A_B",
        "data/processed/v0/splits/protein_binder/train.jsonl",
    ),
    (
        "medium",
        "antibody_h3",
        "sabdab2:pdb_00009nk9_A_+",
        "data/processed/v0/splits/antibody_h3/train.jsonl",
    ),
)


def tensor_difference(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"tensor shapes differ: {reference.shape} != {candidate.shape}")
    reference = reference.detach().float()
    candidate = candidate.detach().float()
    difference = candidate - reference
    reference_norm = torch.linalg.vector_norm(reference)
    return {
        "max_absolute": float(difference.abs().max().item()),
        "mean_absolute": float(difference.abs().mean().item()),
        "rms_absolute": float(torch.sqrt(torch.mean(difference.square())).item()),
        "relative_l2": float(
            (torch.linalg.vector_norm(difference) / reference_norm.clamp_min(1e-12)).item()
        ),
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


def _model(mode: str, config: NanoDesignTinyConfig, seed: int, device: torch.device):
    if mode == "chunked":
        os.environ["RFD3_LOW_MEMORY_MODE"] = "1"
    else:
        os.environ.pop("RFD3_LOW_MEMORY_MODE", None)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return NanoDesignTiny(config).to(device).train()


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Standard vs chunked numerical comparison",
        "",
        "Both modes use identical initialized parameters, the same fixed batch/noise, and H100 bf16.",
        "",
        "| bucket | sample | value | max abs | mean abs | relative L2 |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for record in report["records"]:
        for name, values in record["differences"].items():
            lines.append(
                f"| {record['bucket']} | {record['sample_id']} | {name} | "
                f"{values['max_absolute']:.8g} | {values['mean_absolute']:.8g} | "
                f"{values['relative_l2']:.8g} |"
            )
    lines.extend(
        (
            "",
            f"State-dict key symmetric difference: {report['state_dict_key_difference']}. ",
            (
                "Maximum same-seed parameter difference before loading: "
                f"{report['same_seed_parameter_max_absolute']:.8g}."
            ),
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v0.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolved = load_config(root / args.config)
    validate_v0_config(resolved).require_ready()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("mode equivalence is defined for the formal CUDA bf16 path")
    config = _model_config(resolved)

    standard = _model("standard", config, args.seed, device)
    chunked = _model("chunked", config, args.seed, device)
    standard_state, chunked_state = standard.state_dict(), chunked.state_dict()
    key_difference = sorted(set(standard_state) ^ set(chunked_state))
    common_keys = sorted(set(standard_state) & set(chunked_state))
    parameter_max = max(
        (
            float((standard_state[key] - chunked_state[key]).abs().max().item())
            if standard_state[key].is_floating_point()
            else float(not torch.equal(standard_state[key], chunked_state[key]))
        )
        for key in common_keys
    )
    chunked.load_state_dict(standard_state, strict=True)

    records = []
    for bucket, task, sample_id, catalog in SAMPLES:
        row = next(
            item for item in load_split_catalog(root / catalog) if item["sample_id"] == sample_id
        )
        torch.manual_seed(args.seed)
        batch = load_foundry_training_example(
            root,
            row,
            noise_level=0.5,
            diffusion_batch_size=1,
            max_context_tokens=int(resolved["model"]["max_context_tokens"]),
        )
        batch = _to_device(batch, device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            os.environ.pop("RFD3_LOW_MEMORY_MODE", None)
            standard_output = standard(batch)
            standard_loss, standard_metrics = _compute_rfd3na_loss(batch, standard_output)
            os.environ["RFD3_LOW_MEMORY_MODE"] = "1"
            chunked_output = chunked(batch)
            chunked_loss, chunked_metrics = _compute_rfd3na_loss(batch, chunked_output)
        output_keys = sorted(
            key
            for key in set(standard_output) & set(chunked_output)
            if isinstance(standard_output[key], torch.Tensor)
            and isinstance(chunked_output[key], torch.Tensor)
            and standard_output[key].shape == chunked_output[key].shape
        )
        differences = {
            f"output.{key}": tensor_difference(standard_output[key], chunked_output[key])
            for key in output_keys
        }
        differences.update(
            {
                "loss.total": tensor_difference(standard_loss[None], chunked_loss[None]),
                "loss.coordinate": tensor_difference(
                    standard_metrics["coordinate_loss"][None],
                    chunked_metrics["coordinate_loss"][None],
                ),
                "loss.sequence": tensor_difference(
                    standard_metrics["sequence_loss"][None],
                    chunked_metrics["sequence_loss"][None],
                ),
            }
        )
        records.append(
            {
                "bucket": bucket,
                "task": task,
                "sample_id": sample_id,
                "tokens": int(batch["f"]["restype"].shape[0]),
                "atoms": int(batch["ground_truth_positions"].shape[1]),
                "standard_output_keys": sorted(standard_output),
                "chunked_output_keys": sorted(chunked_output),
                "differences": differences,
            }
        )
    report = {
        "schema_version": "nanodesign.v0.training_mode_equivalence.1",
        "seed": args.seed,
        "precision": "bfloat16",
        "device": torch.cuda.get_device_name(device),
        "model_config": asdict(config),
        "state_dict_key_difference": key_difference,
        "same_seed_parameter_max_absolute": parameter_max,
        "records": records,
    }
    output_json, output_markdown = Path(args.output_json), Path(args.output_markdown)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_markdown.write_text(_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()

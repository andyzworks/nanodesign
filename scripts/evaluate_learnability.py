#!/usr/bin/env python3
"""Run the frozen deterministic NanoDesign learnability evaluation panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from nanodesign.v0.config import load_config, validate_v0_config
from nanodesign.v0.learnability import evaluate_frozen_panel
from nanodesign.v0.model import NanoDesignTiny, NanoDesignTinyConfig
from nanodesign.v0.training import load_checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/v0.yaml"))
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/evaluation/learnability_v1.json")
    )
    parser.add_argument("--feature-cache-root", type=Path, default=Path("data/cache/v0"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--weight-source", choices=("ema", "online"), default="ema"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    resolved = load_config(config_path)
    validate_v0_config(resolved).require_ready()
    model_config = NanoDesignTinyConfig.from_mapping(
        {key: resolved["model"][key] for key in NanoDesignTinyConfig.__dataclass_fields__}
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but no CUDA device is available")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("high")
    model = NanoDesignTiny(model_config).to(device)

    checkpoint_path = (
        args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    )
    manifest_path = root / "docs/data_v0_stats.json"
    manifest_sha = _sha256(manifest_path)
    checkpoint = load_checkpoint(
        checkpoint_path,
        model=model,
        expected_manifest_sha256=manifest_sha,
        prefer_ema=args.weight_source == "ema",
    )
    actual_source = str(checkpoint["loaded_weight_source"])
    if args.weight_source == "ema" and actual_source != "ema":
        raise ValueError("checkpoint has no EMA weights required by the frozen protocol")

    result = evaluate_frozen_panel(
        model,
        root=root,
        protocol_path=args.protocol,
        feature_cache_root=(
            args.feature_cache_root
            if args.feature_cache_root.is_absolute()
            else root / args.feature_cache_root
        ),
        manifest_sha256=manifest_sha,
        device=device,
        max_context_tokens=int(resolved["model"]["max_context_tokens"]),
    )
    result["checkpoint"] = {
        "path": str(checkpoint_path.resolve()),
        "sha256": _sha256(checkpoint_path),
        "samples_seen": int(checkpoint["samples_seen"]),
        "optimizer_steps": int(checkpoint["step"]),
        "weight_source": actual_source,
        "parameter_count": int(checkpoint["parameter_count"]),
        "config_sha256": str(checkpoint["config_sha256"]),
        "training_run_config": checkpoint.get("training_run_config", {}),
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

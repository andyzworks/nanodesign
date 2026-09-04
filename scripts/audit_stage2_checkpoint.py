#!/usr/bin/env python3
"""Audit one Stage-2 checkpoint for context use and generation collapse."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from nanodesign.v0.config import load_config, validate_v0_config
from nanodesign.v0.data.cache import FeatureCacheSpec, SQLiteFeatureCache
from nanodesign.v0.data.loader import recursive_to_device
from nanodesign.v0.learnability import TASK_INDEX, _deterministic_linear_pool, load_frozen_panel
from nanodesign.v0.model import STANDARD_MODE_MAX_ATOMS, NanoDesignTiny, NanoDesignTinyConfig
from nanodesign.v0.training import generate, load_checkpoint


def _seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _model_config(resolved: dict[str, Any]) -> NanoDesignTinyConfig:
    return NanoDesignTinyConfig.from_mapping(
        {key: resolved["model"][key] for key in NanoDesignTinyConfig.__dataclass_fields__}
    )


def _precision(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def _prediction_metrics(
    output: dict[str, torch.Tensor], batch: dict[str, Any]
) -> tuple[float, float, torch.Tensor]:
    design = batch["ground_truth_sequence_mask"].bool()
    targets = batch["ground_truth_sequence"].argmax(dim=-1)[design]
    logits = output["sequence_logits_I"][:, design]
    predictions = output["sequence_indices_I"][:, design]
    expanded_targets = targets.unsqueeze(0).expand(predictions.shape[0], -1)
    recovery = float((predictions == expanded_targets).float().mean().item())
    cross_entropy = float(
        F.cross_entropy(logits.reshape(-1, logits.shape[-1]), expanded_targets.reshape(-1)).item()
    )
    return recovery, cross_entropy, predictions[0].detach().cpu()


def _design_coordinate_metrics(
    output: dict[str, torch.Tensor], batch: dict[str, Any]
) -> tuple[float, torch.Tensor]:
    """Return direct-frame design-atom RMSD and the first denoised realization."""

    coordinates = output["X_L"]
    ground_truth = batch["ground_truth_positions"]
    if coordinates.ndim != 3 or ground_truth.shape != coordinates.shape:
        raise ValueError("predicted and ground-truth coordinates must have shape [D, A, 3]")
    atom_to_token = batch["f"]["atom_to_token_map"].long()
    design_token = batch["ground_truth_sequence_mask"].bool()
    design_atom = design_token[atom_to_token] & batch["ground_truth_atom_mask"].bool()
    if not bool(design_atom.any()):
        raise ValueError("coordinate audit requires at least one resolved design atom")
    delta = coordinates[:, design_atom] - ground_truth[:, design_atom]
    rmsd = float(delta.square().sum(dim=-1).mean().sqrt().item())
    return rmsd, coordinates[0, design_atom].detach().float().cpu()


def _shuffle_fixed_context_sequence(batch: dict[str, Any]) -> dict[str, Any]:
    shuffled = dict(batch)
    shuffled["f"] = dict(batch["f"])
    restype = batch["f"]["restype"].clone()
    fixed = ~batch["ground_truth_sequence_mask"].bool()
    if int(fixed.sum()) < 2:
        raise ValueError("context shuffle requires at least two fixed tokens")
    # A cyclic permutation preserves the exact context composition and changes only
    # its association with residue position/geometry. Design GAP tokens are untouched.
    restype[fixed] = torch.roll(restype[fixed], shifts=1, dims=0)
    shuffled["f"]["restype"] = restype
    return shuffled


def _detach_fixed_context_geometry(
    batch: dict[str, Any], *, displacement: float = 100.0
) -> dict[str, Any]:
    """Move only the fixed motif away from the design region.

    The sequence-only control above asks whether residue identities in the fixed
    context matter.  That is insufficient for structure-conditioned RNA design,
    where target geometry can be informative even when target residue identities
    are not.  This second control preserves the internal target geometry and all
    design-region inputs while breaking the target/design spatial relationship.
    """

    detached = dict(batch)
    detached["f"] = dict(batch["f"])
    fixed_atom = batch["f"]["is_motif_atom_with_fixed_coord"].bool()
    if int(fixed_atom.sum()) < 1:
        raise ValueError("context detachment requires at least one fixed atom")
    shift = torch.zeros(3, dtype=batch["X_noisy_L"].dtype, device=batch["X_noisy_L"].device)
    shift[0] = displacement

    noisy = batch["X_noisy_L"].clone()
    noisy[:, fixed_atom] += shift
    detached["X_noisy_L"] = noisy

    motif_pos = batch["f"]["motif_pos"].clone()
    motif_pos[fixed_atom] += shift
    detached["f"]["motif_pos"] = motif_pos
    return detached


def _sequence_collapse_metrics(indices: torch.Tensor) -> dict[str, Any]:
    values = [int(value) for value in indices.tolist()]
    counts = Counter(values)
    probabilities = [count / len(values) for count in counts.values()]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    normalizer = math.log(min(28, len(values))) if len(values) > 1 else 1.0
    return {
        "length": len(values),
        "token_indices": values,
        "unique_token_count": len(counts),
        "dominant_token_fraction": max(counts.values()) / len(values),
        "normalized_token_entropy": entropy / normalizer if normalizer > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--task", choices=("protein_binder", "antibody_h3", "rna"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/v0.yaml"))
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/evaluation/overfit32_v2.json")
    )
    parser.add_argument("--feature-cache-root", type=Path, default=Path("data/cache/v0"))
    parser.add_argument("--weight-source", choices=("ema", "online"), default="ema")
    parser.add_argument("--generation-examples", type=int, default=8)
    parser.add_argument(
        "--diffusion-t",
        type=float,
        default=None,
        help=(
            "override the frozen protocol timestep for an explicitly labelled "
            "Stage-2 diagnostic; the default keeps the frozen protocol unchanged"
        ),
    )
    parser.add_argument(
        "--coordinate-augmentation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override the protocol coordinate-frame setting for a diagnostic control",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.generation_examples < 0:
        raise ValueError("generation examples must be non-negative")
    if args.diffusion_t is not None and args.diffusion_t <= 0:
        raise ValueError("diffusion timestep must be positive")

    # Required by torch.use_deterministic_algorithms for CUDA >= 10.2. Set it
    # before the first CUDA model operation so the audit is self-contained.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    root = Path(__file__).resolve().parents[1]
    resolved = load_config(root / args.config)
    validate_v0_config(resolved).require_ready()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA audit requested but no CUDA device is available")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("high")

    protocol, rows_by_task = load_frozen_panel(root, root / args.protocol)
    rows = rows_by_task[args.task]
    manifest_sha = hashlib.sha256((root / "docs/data_v0_stats.json").read_bytes()).hexdigest()
    model = NanoDesignTiny(_model_config(resolved)).to(device)
    checkpoint = load_checkpoint(
        root / args.checkpoint,
        model=model,
        expected_manifest_sha256=manifest_sha,
        prefer_ema=args.weight_source == "ema",
    )
    actual_weight_source = str(checkpoint["loaded_weight_source"])
    if args.weight_source == "ema" and actual_weight_source != "ema":
        raise ValueError("checkpoint does not contain EMA weights")

    correct_recovery: list[float] = []
    shuffled_recovery: list[float] = []
    correct_ce: list[float] = []
    shuffled_ce: list[float] = []
    prediction_change: list[float] = []
    detached_recovery: list[float] = []
    detached_ce: list[float] = []
    detached_prediction_change: list[float] = []
    correct_design_rmsd: list[float] = []
    shuffled_design_rmsd: list[float] = []
    shuffled_coordinate_change: list[float] = []
    detached_design_rmsd: list[float] = []
    detached_coordinate_change: list[float] = []
    generations: list[dict[str, Any]] = []
    protocol_diffusion_t = float(protocol["diffusion_t"])
    diffusion_t = protocol_diffusion_t if args.diffusion_t is None else args.diffusion_t
    coordinate_augmentation = (
        bool(protocol["coordinate_augmentation"])
        if args.coordinate_augmentation is None
        else args.coordinate_augmentation
    )
    max_context_tokens = int(resolved["model"]["max_context_tokens"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    with (
        _deterministic_linear_pool(model),
        SQLiteFeatureCache(root / args.feature_cache_root, readonly=True, lru_size=4) as cache,
        torch.no_grad(),
    ):
        for sample_index, row in enumerate(rows):
            sample_seed = (
                int(protocol["seed"]) + 10_000 * TASK_INDEX[args.task] + sample_index
            )
            batch = cache.get(
                row,
                FeatureCacheSpec(
                    manifest_sha256=manifest_sha,
                    max_context_tokens=max_context_tokens,
                    diffusion_batch_size=1,
                    noise_level=diffusion_t,
                    random_seed=sample_seed,
                    augment_coordinates=coordinate_augmentation,
                ),
            )
            batch = recursive_to_device(batch, device, non_blocking=device.type == "cuda")
            atom_count = int(batch["f"]["atom_to_token_map"].numel())
            model.execution_mode = (
                "standard" if atom_count <= STANDARD_MODE_MAX_ATOMS else "chunked"
            )
            model.train()
            _seed(sample_seed, device)
            with _precision(device):
                correct_output = model(batch)
            recovery, cross_entropy, correct_indices = _prediction_metrics(
                correct_output, batch
            )
            coordinate_rmsd, correct_coordinates = _design_coordinate_metrics(
                correct_output, batch
            )
            shuffled_batch = _shuffle_fixed_context_sequence(batch)
            _seed(sample_seed, device)
            with _precision(device):
                shuffled_output = model(shuffled_batch)
            shuffled_rec, shuffled_cross_entropy, shuffled_indices = _prediction_metrics(
                shuffled_output, batch
            )
            shuffled_coordinate_rmsd, shuffled_coordinates = _design_coordinate_metrics(
                shuffled_output, batch
            )
            correct_recovery.append(recovery)
            shuffled_recovery.append(shuffled_rec)
            correct_ce.append(cross_entropy)
            shuffled_ce.append(shuffled_cross_entropy)
            prediction_change.append(
                float((correct_indices != shuffled_indices).float().mean().item())
            )
            correct_design_rmsd.append(coordinate_rmsd)
            shuffled_design_rmsd.append(shuffled_coordinate_rmsd)
            shuffled_coordinate_change.append(
                float(
                    (correct_coordinates - shuffled_coordinates)
                    .square()
                    .sum(dim=-1)
                    .mean()
                    .sqrt()
                    .item()
                )
            )
            detached_batch = _detach_fixed_context_geometry(batch)
            _seed(sample_seed, device)
            with _precision(device):
                detached_output = model(detached_batch)
            detached_rec, detached_cross_entropy, detached_indices = _prediction_metrics(
                detached_output, batch
            )
            detached_coordinate_rmsd, detached_coordinates = _design_coordinate_metrics(
                detached_output, batch
            )
            detached_recovery.append(detached_rec)
            detached_ce.append(detached_cross_entropy)
            detached_prediction_change.append(
                float((correct_indices != detached_indices).float().mean().item())
            )
            detached_design_rmsd.append(detached_coordinate_rmsd)
            detached_coordinate_change.append(
                float(
                    (correct_coordinates - detached_coordinates)
                    .square()
                    .sum(dim=-1)
                    .mean()
                    .sqrt()
                    .item()
                )
            )

            if sample_index < min(args.generation_examples, len(rows)):
                generation_seed = int(protocol["seed"]) + 100_000 + sample_index
                _seed(generation_seed, device)
                with _precision(device):
                    generated = generate(model, batch)
                design = batch["ground_truth_sequence_mask"].bool()
                indices = generated["sequence_indices_I"][0, design].detach().cpu()
                generations.append(
                    {
                        "sample_id": row["sample_id"],
                        "seed": generation_seed,
                        "finite_coordinates": bool(torch.isfinite(generated["X_L"]).all()),
                        **_sequence_collapse_metrics(indices),
                    }
                )

    dominant_fractions = [item["dominant_token_fraction"] for item in generations]
    generation_sequences = [tuple(item["token_indices"]) for item in generations]
    result = {
        "schema": "nanodesign.stage2_checkpoint_audit.v2",
        "task": args.task,
        "checkpoint": str((root / args.checkpoint).resolve()),
        "samples_seen": int(checkpoint["samples_seen"]),
        "weight_source": actual_weight_source,
        "protocol": protocol["protocol"],
        "protocol_diffusion_t": protocol_diffusion_t,
        "diffusion_t": diffusion_t,
        "diffusion_t_overridden": args.diffusion_t is not None,
        "coordinate_augmentation": coordinate_augmentation,
        "panel_sample_count": len(rows),
        "context_control": {
            "definition": "cyclic permutation of fixed-context restype over fixed positions",
            "correct_recovery_mean": float(np.mean(correct_recovery)),
            "shuffled_recovery_mean": float(np.mean(shuffled_recovery)),
            "correct_sequence_ce_mean": float(np.mean(correct_ce)),
            "shuffled_sequence_ce_mean": float(np.mean(shuffled_ce)),
            "prediction_change_fraction_mean": float(np.mean(prediction_change)),
            "correct_design_coordinate_rmsd_mean": float(np.mean(correct_design_rmsd)),
            "shuffled_design_coordinate_rmsd_mean": float(np.mean(shuffled_design_rmsd)),
            "design_coordinate_prediction_change_rmsd_mean": float(
                np.mean(shuffled_coordinate_change)
            ),
        },
        "detached_context_control": {
            "definition": (
                "translate the intact fixed context by +100 Angstrom on x while "
                "leaving the design region unchanged"
            ),
            "detached_recovery_mean": float(np.mean(detached_recovery)),
            "detached_sequence_ce_mean": float(np.mean(detached_ce)),
            "prediction_change_fraction_mean": float(np.mean(detached_prediction_change)),
            "detached_design_coordinate_rmsd_mean": float(np.mean(detached_design_rmsd)),
            "design_coordinate_prediction_change_rmsd_mean": float(
                np.mean(detached_coordinate_change)
            ),
        },
        "generation": {
            "sample_count": len(generations),
            "distinct_sequence_count": len(set(generation_sequences)),
            "dominant_token_fraction_mean": (
                float(np.mean(dominant_fractions)) if dominant_fractions else None
            ),
            "dominant_token_fraction_max": (
                float(np.max(dominant_fractions)) if dominant_fractions else None
            ),
            "samples": generations,
        },
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

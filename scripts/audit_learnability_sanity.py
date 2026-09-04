#!/usr/bin/env python3
"""Check Perfect/Perturbed/Broken ordering of the frozen learnability evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from nanodesign.v0.data.cache import FeatureCacheSpec, SQLiteFeatureCache
from nanodesign.v0.learnability import TASK_INDEX, load_frozen_panel
from nanodesign.v0.training import _compute_rfd3na_loss


def _output(batch: dict, quality: str, *, seed: int) -> dict[str, torch.Tensor]:
    ground_truth = batch["ground_truth_sequence"].argmax(dim=-1)
    design_token_mask = batch["ground_truth_sequence_mask"].bool()
    atom_to_token = batch["f"]["atom_to_token_map"].long()
    design_atom_mask = design_token_mask[atom_to_token]
    positions = batch["ground_truth_positions"].clone()
    predictions = ground_truth.clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    design_indices = torch.where(design_token_mask)[0]
    if quality == "perturbed":
        positions[:, design_atom_mask] += 0.25 * torch.randn(
            positions[:, design_atom_mask].shape, generator=generator
        )
        changed = design_indices[::4]
        predictions[changed] = (ground_truth[changed] + 1) % 32
    elif quality == "broken":
        positions[:, design_atom_mask] += torch.tensor([20.0, -15.0, 10.0])
        positions[:, design_atom_mask] += 3.0 * torch.randn(
            positions[:, design_atom_mask].shape, generator=generator
        )
        predictions[design_indices] = (ground_truth[design_indices] + 7) % 32
    elif quality != "perfect":
        raise ValueError(f"unknown synthetic prediction quality {quality!r}")

    logits = torch.full(
        (positions.shape[0], ground_truth.numel(), 32),
        -10.0,
        dtype=positions.dtype,
    )
    logits.scatter_(2, predictions[None, :, None].expand(positions.shape[0], -1, -1), 10.0)
    return {
        "X_L": positions,
        "sequence_logits_I": logits,
        "sequence_indices_I": predictions[None].expand(positions.shape[0], -1),
    }


def _score(batch: dict, output: dict[str, torch.Tensor]) -> dict[str, float]:
    loss, metrics = _compute_rfd3na_loss(batch, output)
    selected = {
        "loss": loss,
        "coordinate_loss": metrics["coordinate_loss"],
        "sequence_loss": metrics["sequence_loss"],
        "sequence_ce": metrics["token_lvl_sequence_loss"],
        "sequence_recovery": metrics["seq_recovery"],
        "coordinate_mse": metrics["mse_loss_mean"],
        "lddt_loss": metrics["mean_lddt"],
    }
    return {name: float(value.detach().item()) for name, value in selected.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/evaluation/learnability_v2.json")
    )
    parser.add_argument("--feature-cache-root", type=Path, default=Path("data/cache/v0"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol, rows_by_task = load_frozen_panel(root, args.protocol)
    manifest_sha = hashlib.sha256((root / "docs/data_v0_stats.json").read_bytes()).hexdigest()
    cache_root = (
        args.feature_cache_root
        if args.feature_cache_root.is_absolute()
        else root / args.feature_cache_root
    )
    results = {}
    with SQLiteFeatureCache(cache_root, readonly=True, lru_size=1) as cache:
        for task, task_index in TASK_INDEX.items():
            row = rows_by_task[task][0]
            batch = cache.get(
                row,
                FeatureCacheSpec(
                    manifest_sha256=manifest_sha,
                    max_context_tokens=384,
                    diffusion_batch_size=1,
                    noise_level=float(protocol["diffusion_t"]),
                    random_seed=int(protocol["seed"]) + 10_000 * task_index,
                    augment_coordinates=False,
                ),
            )
            task_results = {
                quality: _score(
                    batch,
                    _output(batch, quality, seed=int(protocol["seed"]) + task_index),
                )
                for quality in ("perfect", "perturbed", "broken")
            }
            lower_metrics = ("loss", "coordinate_loss", "sequence_ce", "lddt_loss")
            higher_metrics = ("sequence_recovery",)
            checks = {
                metric: (
                    task_results["perfect"][metric]
                    < task_results["perturbed"][metric]
                    < task_results["broken"][metric]
                )
                for metric in lower_metrics
            } | {
                metric: (
                    task_results["perfect"][metric]
                    > task_results["perturbed"][metric]
                    > task_results["broken"][metric]
                )
                for metric in higher_metrics
            }
            results[task] = {
                "sample_id": row["sample_id"],
                "predictions": task_results,
                "strict_ordering_checks": checks,
                "weighted_sequence_loss_saturates_by_design": (
                    task_results["perfect"]["sequence_loss"]
                    < task_results["perturbed"]["sequence_loss"]
                    <= task_results["broken"]["sequence_loss"]
                ),
                "passed": all(checks.values()),
            }

    payload = {
        "protocol": protocol["protocol"],
        "definition": "synthetic model outputs scored by the exact training loss implementation",
        "tasks": results,
        "passed": all(result["passed"] for result in results.values()),
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit("learnability evaluator sanity ordering failed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit frozen Stage-2 inputs without changing the model or evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from nanodesign.v0.data.cache import FeatureCacheSpec, SQLiteFeatureCache
from nanodesign.v0.learnability import load_frozen_panel
from nanodesign.v0.training import _sequence_supervision_mask

PROTEIN_TARGETS = set(range(20))
RNA_TARGETS = set(range(21, 25))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/evaluation/overfit32_v1.json")
    )
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/v0"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    protocol, rows_by_task = load_frozen_panel(root, args.protocol)
    manifest_sha = _sha256(root / "docs/data_v0_stats.json")
    checks: dict[str, dict[str, object]] = {}
    all_passed = True
    with SQLiteFeatureCache(root / args.cache_root, readonly=True, lru_size=4) as cache:
        for task, rows in rows_by_task.items():
            totals = {
                "tokens": 0,
                "design_tokens": 0,
                "atoms": 0,
                "resolved_atoms": 0,
            }
            task_checks = {
                "one_hot_targets": True,
                "design_input_is_gap": True,
                "fixed_input_equals_target": True,
                "design_target_alphabet": True,
                "token_and_atom_masks_agree": True,
                "fixed_coordinates_are_not_noised": True,
                "design_coordinates_are_noised": True,
                "motif_token_classes_agree": True,
                "design_normalization_mean_is_one": True,
                "all_valid_supervision_matches_native_targets": True,
            }
            design_lengths: list[int] = []
            atom_counts: list[int] = []
            for index, row in enumerate(rows):
                batch = cache.get(
                    row,
                    FeatureCacheSpec(
                        manifest_sha256=manifest_sha,
                        max_context_tokens=384,
                        diffusion_batch_size=16,
                        random_seed=17 + 1_000_000 + index,
                        augment_coordinates=True,
                    ),
                )
                features = batch["f"]
                design = batch["ground_truth_sequence_mask"].bool()
                target = batch["ground_truth_sequence"]
                target_index = target.argmax(dim=-1)
                input_index = features["restype"].argmax(dim=-1)
                atom_to_token = features["atom_to_token_map"].long()
                atom_design = design[atom_to_token]
                expected_fixed = ~atom_design
                fixed_coord = features["is_motif_atom_with_fixed_coord"].bool()
                fixed_seq = features["is_motif_atom_with_fixed_seq"].bool()
                delta = batch["X_noisy_L"] - batch["ground_truth_positions"]
                motif_class = features["ref_motif_token_type"].argmax(dim=-1)
                design_weights = _sequence_supervision_mask(
                    target_index, design, mode="design"
                )
                all_valid = _sequence_supervision_mask(
                    target_index, design, mode="all_valid"
                )

                allowed = PROTEIN_TARGETS if task != "rna" else RNA_TARGETS
                task_checks["one_hot_targets"] &= bool(
                    torch.equal(target.sum(dim=-1), torch.ones(len(target)))
                )
                task_checks["design_input_is_gap"] &= bool((input_index[design] == 31).all())
                task_checks["fixed_input_equals_target"] &= bool(
                    torch.equal(input_index[~design], target_index[~design])
                )
                task_checks["design_target_alphabet"] &= set(
                    target_index[design].tolist()
                ).issubset(allowed)
                task_checks["token_and_atom_masks_agree"] &= bool(
                    torch.equal(fixed_coord, expected_fixed)
                    and torch.equal(fixed_seq, expected_fixed)
                )
                task_checks["fixed_coordinates_are_not_noised"] &= bool(
                    torch.equal(delta[:, expected_fixed], torch.zeros_like(delta[:, expected_fixed]))
                )
                task_checks["design_coordinates_are_noised"] &= bool(
                    torch.count_nonzero(delta[:, atom_design]) > 0
                )
                task_checks["motif_token_classes_agree"] &= bool(
                    torch.equal(motif_class, (~design).long())
                )
                task_checks["design_normalization_mean_is_one"] &= bool(
                    torch.isclose(design_weights.mean(), torch.tensor(1.0))
                )
                expected_all_valid = ~torch.isin(
                    target_index, torch.tensor((20, 25, 30, 31))
                )
                task_checks["all_valid_supervision_matches_native_targets"] &= bool(
                    torch.equal(all_valid, expected_all_valid)
                )

                totals["tokens"] += len(target)
                totals["design_tokens"] += int(design.sum())
                totals["atoms"] += len(atom_to_token)
                totals["resolved_atoms"] += int(batch["ground_truth_atom_mask"].sum())
                design_lengths.append(int(design.sum()))
                atom_counts.append(len(atom_to_token))

            task_passed = all(task_checks.values())
            all_passed &= task_passed
            checks[task] = {
                "sample_count": len(rows),
                "checks": task_checks,
                "passed": task_passed,
                "totals": totals,
                "design_length_range": [min(design_lengths), max(design_lengths)],
                "atom_count_range": [min(atom_counts), max(atom_counts)],
            }

    payload = {
        "schema": "nanodesign.stage2_batch_audit.v1",
        "protocol": protocol["protocol"],
        "protocol_sha256": _sha256(root / args.protocol),
        "manifest_sha256": manifest_sha,
        "tasks": checks,
        "passed": all_passed,
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

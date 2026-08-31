#!/usr/bin/env python3
"""Collect completed NanoDesign v0 training/evaluator outputs into the required table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--binder", type=Path, required=True)
    parser.add_argument("--antibody", type=Path, required=True)
    parser.add_argument("--rna", type=Path, required=True)
    parser.add_argument("--data-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    training = _read(args.training_report)
    binder = _read(args.binder)
    antibody = _read(args.antibody)
    rna = _read(args.rna)
    stats = _read(args.data_stats)
    antibody_metrics = antibody["metrics"]
    payload = {
        "status": "initial_real_test_sample_baseline",
        "evaluation_sample_count_per_task": 1,
        "warning": "N=1 is an initial real run, not a full test-set benchmark.",
        "training": {
            key: training[key]
            for key in (
                "steps",
                "samples_seen",
                "batch_size_complexes",
                "diffusion_realizations_per_complex",
                "seed",
                "model_parameter_count",
                "max_context_tokens",
                "task_steps",
            )
        },
        "data_split_counts": stats["split_counts"],
        "results": [
            {
                "task": "Protein Binder",
                "metric": "In-silico Success Rate",
                "value": binder["in_silico_success_rate"],
            },
            {"task": "Antibody H3", "metric": "H3 AAR", "value": antibody_metrics["h3_aar"]},
            {
                "task": "Antibody H3",
                "metric": "H3 RMSD",
                "value": antibody_metrics["h3_rmsd"],
            },
            {"task": "Antibody H3", "metric": "DockQ", "value": antibody_metrics["dockq"]},
            {"task": "RNA", "metric": "scTM", "value": rna["scTM"]},
            {"task": "RNA", "metric": "scRMSD", "value": rna["scRMSD"]},
            {"task": "RNA", "metric": "DockQ", "value": rna["DockQ"]},
        ],
        "rna_metric_scope": "computational structure/interface evaluation; not binding affinity",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

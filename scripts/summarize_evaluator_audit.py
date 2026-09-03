#!/usr/bin/env python3
"""Summarize frozen learnability evaluations and verify repeat determinism."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _means(report: dict) -> dict[str, dict[str, float]]:
    return {
        task: {
            metric: float(statistics["mean"])
            for metric, statistics in task_report["metrics"].items()
        }
        for task, task_report in report["tasks"].items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = {
        name: json.loads((args.input_dir / filename).read_text(encoding="utf-8"))
        for name, filename in {
            "0k": "0k.json",
            "9k": "9k.json",
            "18k-repeat-1": "18k-repeat-1.json",
            "18k-repeat-2": "18k-repeat-2.json",
            "18k-repeat-3": "18k-repeat-3.json",
        }.items()
    }
    means = {name: _means(report) for name, report in reports.items()}
    reference = means["18k-repeat-1"]
    maximum_difference = 0.0
    differences = {}
    for repeat in ("18k-repeat-2", "18k-repeat-3"):
        repeat_maximum = 0.0
        for task, metrics in reference.items():
            for metric, reference_value in metrics.items():
                difference = abs(reference_value - means[repeat][task][metric])
                repeat_maximum = max(repeat_maximum, difference)
                maximum_difference = max(maximum_difference, difference)
        differences[repeat] = repeat_maximum

    payload = {
        "protocol": reports["0k"]["protocol"],
        "protocol_sha256": reports["0k"]["protocol_sha256"],
        "checkpoint_metrics": {
            "0k": means["0k"],
            "9k": means["9k"],
            "18k": reference,
        },
        "determinism": {
            "repeat_count": 3,
            "maximum_absolute_mean_metric_difference": maximum_difference,
            "maximum_absolute_mean_metric_difference_by_repeat": differences,
            "tolerance": 1e-7,
            "passed": maximum_difference <= 1e-7,
        },
        "evaluation_wall_time_seconds": {
            name: float(report["wall_time_seconds"]) for name, report in reports.items()
        },
        "peak_gpu_memory_bytes": {
            name: int(report["peak_gpu_memory_bytes"]) for name, report in reports.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

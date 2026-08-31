#!/usr/bin/env python3
"""Run the frozen NanoDesign v0 Antibody H3 evaluation and write JSON."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from nanodesign.v0.evaluators import EvaluationError, evaluate_antibody_h3


def _catalog_row(catalogs: Sequence[Path], sample_id: str) -> dict[str, Any]:
    matches = []
    for catalog in catalogs:
        with catalog.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("sample_id") == sample_id:
                        matches.append(row)
    if len(matches) != 1:
        raise EvaluationError(
            f"expected exactly one catalog row for {sample_id!r}, found {len(matches)}"
        )
    return matches[0]


def _catalog_chain_roles(row: dict[str, Any]) -> tuple[str, str | None, list[str]]:
    if row.get("task") != "antibody_cdr":
        raise EvaluationError("catalog row is not an antibody_cdr sample")
    chains = row.get("chains")
    if not isinstance(chains, list):
        raise EvaluationError("catalog row has no chain records")
    heavy = [c.get("chain_id") for c in chains if c.get("role") == "antibody_framework+cdr_h3"]
    light = [c.get("chain_id") for c in chains if c.get("role") == "antibody_framework"]
    antigen = [c.get("chain_id") for c in chains if c.get("role") == "antigen"]
    if len(heavy) != 1 or len(light) > 1 or not antigen:
        raise EvaluationError(
            "catalog antibody roles require exactly one heavy chain, at most one light chain, "
            "and at least one antigen chain"
        )
    role_ids = [heavy[0], *light, *antigen]
    if not all(isinstance(chain_id, str) and chain_id for chain_id in role_ids):
        raise EvaluationError("catalog antibody chain IDs must be non-empty strings")
    if len(role_ids) != len(set(role_ids)):
        raise EvaluationError("catalog antibody chain roles overlap")
    return heavy[0], light[0] if light else None, antigen


def _from_training_report(report_path: Path) -> tuple[Path, str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    generation = report.get("generation", {}).get("antibody_h3")
    if not isinstance(generation, dict):
        raise EvaluationError("training report has no antibody_h3 generation")
    prediction, sample_id = generation.get("structure_path"), generation.get("sample_id")
    if not isinstance(prediction, str) or not isinstance(sample_id, str):
        raise EvaluationError("training report antibody generation lacks structure_path/sample_id")
    return Path(prediction), sample_id


def _resolve_reference(root: Path, row: dict[str, Any]) -> Path:
    raw_paths = row.get("raw_paths")
    if not isinstance(raw_paths, list) or len(raw_paths) != 1:
        raise EvaluationError("antibody catalog row must identify exactly one reference structure")
    reference = root / raw_paths[0]
    if not reference.is_file():
        raise EvaluationError(f"reference structure does not exist: {reference}")
    return reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--sample-id")
    parser.add_argument(
        "--catalog",
        type=Path,
        help="optional exact split catalog; by default search frozen train/validation/test",
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dockq-executable", default="DockQ")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.training_report is not None:
        if args.prediction is not None or args.sample_id is not None:
            parser.error("--training-report cannot be combined with --prediction or --sample-id")
        prediction, sample_id = _from_training_report(args.training_report)
    else:
        if args.prediction is None or args.sample_id is None:
            parser.error("provide --training-report or both --prediction and --sample-id")
        prediction, sample_id = args.prediction, args.sample_id

    root = args.repo_root.resolve()
    if args.catalog is not None:
        catalogs = [args.catalog if args.catalog.is_absolute() else root / args.catalog]
    else:
        split_root = root / "data/processed/v0/splits/antibody_h3"
        catalogs = [split_root / f"{split}.jsonl" for split in ("train", "validation", "test")]
    missing_catalogs = [str(path) for path in catalogs if not path.is_file()]
    if missing_catalogs:
        raise EvaluationError(f"antibody split catalogs do not exist: {missing_catalogs}")
    row = _catalog_row(catalogs, sample_id)
    heavy_chain, light_chain, antigen_chains = _catalog_chain_roles(row)
    reference = args.reference or _resolve_reference(root, row)
    if not prediction.is_file():
        raise EvaluationError(f"prediction structure does not exist: {prediction}")
    if not reference.is_file():
        raise EvaluationError(f"reference structure does not exist: {reference}")

    result = evaluate_antibody_h3(
        prediction,
        reference,
        heavy_chain=heavy_chain,
        light_chain=light_chain,
        antigen_chains=antigen_chains,
        dockq_executable=args.dockq_executable,
    )
    payload = {
        "task": "antibody_h3",
        "sample_id": sample_id,
        "prediction": str(prediction.resolve()),
        "reference": str(reference.resolve()),
        "chains": {
            "heavy": heavy_chain,
            "light": light_chain,
            "antigen": antigen_chains,
        },
        "metrics": {
            "h3_aar": result.h3_aar,
            "h3_rmsd": result.h3_rmsd,
            "dockq": result.dockq,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

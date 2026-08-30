#!/usr/bin/env python3
"""Run real examples from all three tasks through official RFD3NA train/generate paths."""

from __future__ import annotations

import json
from pathlib import Path

from nanodesign.v0.data.real import load_foundry_training_example
from nanodesign.v0.model import NanoDesignTiny, NanoDesignTinyConfig
from nanodesign.v0.training import build_optimizer, generate, train_step

CATALOGS = {
    "protein_binder": "data/processed/v0/catalogs/ppiref50k.jsonl",
    "antibody_h3": "data/processed/v0/catalogs/sabdab2.jsonl",
    "rna_aptamer": "data/processed/v0/catalogs/pdb_rna_target.jsonl",
}


def _smallest_row(path: Path) -> dict:
    rows = (json.loads(line) for line in path.open(encoding="utf-8") if line.strip())
    return min(rows, key=lambda row: sum(chain["resolved_residues"] for chain in row["chains"]))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model = NanoDesignTiny(NanoDesignTinyConfig(sampling_steps=2))
    optimizer = build_optimizer(model)
    report = {"parameter_count": model.parameter_count, "tasks": {}}
    for task, relative_catalog in CATALOGS.items():
        row = _smallest_row(root / relative_catalog)
        training_batch = load_foundry_training_example(
            root,
            row,
            noise_level=0.5,
            max_context_tokens=4,
        )
        metrics = train_step(model, optimizer, training_batch)
        generation_batch = load_foundry_training_example(
            root,
            row,
            noise_level=0.5,
            max_context_tokens=4,
        )
        output = generate(model, generation_batch)
        report["tasks"][task] = {
            "sample_id": row["sample_id"],
            "tokens": int(generation_batch["f"]["restype"].shape[0]),
            "atoms": int(generation_batch["coord_atom_lvl_to_be_noised"].shape[1]),
            "loss": metrics["loss"],
            "generated_coordinates": list(output["X_L"].shape),
            "generated_sequence_logits": list(output["sequence_logits_I"].shape),
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

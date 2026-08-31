#!/usr/bin/env python3
"""Run the frozen NanoDesign v0 RNA evaluation pipeline for one generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nanodesign.v0.evaluators import evaluate_rna


def _rna_sequence(value: str) -> str:
    sequence = "".join(value.split()).upper()
    if not sequence:
        raise argparse.ArgumentTypeError("RNA sequence must not be empty")
    invalid = sorted(set(sequence) - set("ACGUN"))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"RNA sequence contains unsupported symbols: {''.join(invalid)}"
        )
    return sequence


def _read_single_fasta(path: Path) -> str:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    records: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line:
            continue
        if line.startswith(">"):
            if current:
                records.append("".join(current))
                current = []
            continue
        current.append(line)
    if current:
        records.append("".join(current))
    if len(records) != 1:
        raise ValueError(f"generated FASTA must contain exactly one record, found {len(records)}")
    return _rna_sequence(records[0])


def _from_training_report(path: Path, rna_chain: str) -> tuple[str, Path]:
    report = json.loads(path.read_text(encoding="utf-8"))
    generation = report.get("generation", {}).get("rna")
    if not isinstance(generation, dict):
        raise TypeError("training report has no RNA generation")
    structure_path = generation.get("structure_path")
    sequences = generation.get("sequences")
    if not isinstance(structure_path, str) or not isinstance(sequences, dict):
        raise TypeError("training report RNA generation lacks structure_path/sequences")
    sequence = sequences.get(rna_chain)
    if not isinstance(sequence, str):
        raise TypeError(f"training report has no sequence for RNA chain {rna_chain!r}")
    return _rna_sequence(sequence), Path(structure_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sequence = parser.add_mutually_exclusive_group(required=True)
    sequence.add_argument("--generated-sequence", type=_rna_sequence)
    sequence.add_argument("--generated-fasta", type=Path)
    sequence.add_argument("--training-report", type=Path)
    parser.add_argument("--generated-complex", type=Path)
    parser.add_argument("--native-complex", type=Path, required=True)
    parser.add_argument("--rna-chain", required=True)
    parser.add_argument(
        "--target-chain",
        action="append",
        required=True,
        dest="target_chains",
        help="target chain ID; repeat for a multi-chain target",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--rhofold-python", type=Path, required=True)
    parser.add_argument("--rhofold-inference-script", type=Path, required=True)
    parser.add_argument("--rhofold-checkpoint", type=Path, required=True)
    parser.add_argument("--rhofold-device")
    parser.add_argument("--usalign-executable", default="USalign")
    parser.add_argument("--dockq-executable", default="DockQ")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.training_report is not None:
        if args.generated_complex is not None:
            raise ValueError("--training-report cannot be combined with --generated-complex")
        sequence, generated_complex = _from_training_report(args.training_report, args.rna_chain)
    else:
        if args.generated_complex is None:
            raise ValueError("--generated-complex is required without --training-report")
        generated_complex = args.generated_complex
        sequence = (
            args.generated_sequence
            if args.generated_sequence is not None
            else _read_single_fasta(args.generated_fasta)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_fasta = args.output_dir / "generated_rna.fasta"
    generated_fasta.write_text(f">generated_rna\n{sequence}\n", encoding="utf-8")
    result = evaluate_rna(
        generated_fasta,
        generated_complex,
        args.native_complex,
        target_chains=args.target_chains,
        rna_chain=args.rna_chain,
        output_dir=args.output_dir,
        rhofold_python=args.rhofold_python,
        rhofold_inference_script=args.rhofold_inference_script,
        rhofold_checkpoint=args.rhofold_checkpoint,
        usalign_executable=args.usalign_executable,
        dockq_executable=args.dockq_executable,
        rhofold_device=args.rhofold_device,
    )
    payload = {
        "scTM": result.sctm,
        "scRMSD": result.scrmsd,
        "structure_confidence": result.structure_confidence,
        "DockQ": result.dockq,
    }
    result_json = args.result_json or args.output_dir / "metrics.json"
    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

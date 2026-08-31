#!/usr/bin/env python3
"""Run the frozen NanoDesign v0 RNA evaluation pipeline for one generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gemmi

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


def _catalog_row(catalog: Path, sample_id: str) -> dict[str, Any]:
    matches = [
        json.loads(line)
        for line in catalog.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("sample_id") == sample_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one catalog row for {sample_id!r}, found {len(matches)}")
    return matches[0]


def _catalog_chain_roles(row: dict[str, Any]) -> tuple[list[str], str]:
    target = [chain["chain_id"] for chain in row["chains"] if chain["role"] == "target"]
    rna = [
        chain["chain_id"]
        for chain in row["chains"]
        if chain["role"] in {"rna_aptamer", "rna_design_region"}
    ]
    if not target or len(rna) != 1 or len([*target, *rna]) != len({*target, *rna}):
        raise ValueError("RNA catalog roles require target chain(s) and exactly one RNA chain")
    return target, rna[0]


def _residue_key(residue: gemmi.Residue) -> tuple[int, str]:
    insertion = residue.seqid.icode
    return int(residue.seqid.num), "" if insertion in {" ", "\x00", "?", "."} else insertion


def _write_catalog_native(root: Path, row: dict[str, Any], destination: Path) -> None:
    output = gemmi.Structure()
    output.name = str(row["sample_id"])
    model = gemmi.Model("1")
    for record in row["chains"]:
        path = (root / record["raw_path"]).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"catalog chain path is invalid: {path}")
        source = gemmi.read_structure(str(path))[0]
        chain = source.find_chain(record["chain_id"])
        if chain is None and len(source) == 1:
            chain = source[0]
        if chain is None:
            raise ValueError(f"catalog chain {record['chain_id']!r} is not present in {path}")
        wanted = {(int(key[0]), str(key[1])) for key in record["residue_keys"]}
        selected = gemmi.Chain(record["chain_id"])
        for residue in chain:
            if _residue_key(residue) in wanted:
                selected.add_residue(residue.clone())
        if len(selected) != len(wanted):
            raise ValueError(f"catalog chain {record['chain_id']!r} is residue-incomplete")
        model.add_chain(selected)
    output.add_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.write_pdb(str(destination))


def _from_training_report(path: Path) -> tuple[dict[str, Any], Path, str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    generation = report.get("generation", {}).get("rna")
    if not isinstance(generation, dict):
        raise TypeError("training report has no RNA generation")
    structure_path = generation.get("structure_path")
    sequences = generation.get("sequences")
    if not isinstance(structure_path, str) or not isinstance(sequences, dict):
        raise TypeError("training report RNA generation lacks structure_path/sequences")
    sample_id = generation.get("sample_id")
    if not isinstance(sample_id, str):
        raise TypeError("training report RNA generation lacks sample_id")
    return sequences, Path(structure_path), sample_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sequence = parser.add_mutually_exclusive_group(required=True)
    sequence.add_argument("--generated-sequence", type=_rna_sequence)
    sequence.add_argument("--generated-fasta", type=Path)
    sequence.add_argument("--training-report", type=Path)
    parser.add_argument("--generated-complex", type=Path)
    parser.add_argument("--native-complex", type=Path)
    parser.add_argument("--rna-chain")
    parser.add_argument(
        "--target-chain",
        action="append",
        dest="target_chains",
        help="target chain ID; repeat for a multi-chain target",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/processed/v0/splits/rna_binding/test.jsonl"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.training_report is not None:
        if any(
            value is not None
            for value in (
                args.generated_complex,
                args.native_complex,
                args.rna_chain,
                args.target_chains,
            )
        ):
            raise ValueError(
                "--training-report cannot be combined with explicit complex or chain arguments"
            )
        sequences, generated_complex, sample_id = _from_training_report(args.training_report)
        root = args.repo_root.resolve()
        catalog = args.catalog if args.catalog.is_absolute() else root / args.catalog
        row = _catalog_row(catalog, sample_id)
        target_chains, rna_chain = _catalog_chain_roles(row)
        sequence = sequences.get(rna_chain)
        if not isinstance(sequence, str):
            raise TypeError(f"training report has no sequence for RNA chain {rna_chain!r}")
        sequence = _rna_sequence(sequence)
        native_complex = args.output_dir / "catalog_native_complex.pdb"
        _write_catalog_native(root, row, native_complex)
    else:
        if (
            args.generated_complex is None
            or args.native_complex is None
            or args.rna_chain is None
            or not args.target_chains
        ):
            raise ValueError("explicit evaluation requires complexes and target/RNA chain IDs")
        generated_complex = args.generated_complex
        native_complex = args.native_complex
        rna_chain = args.rna_chain
        target_chains = args.target_chains
        sequence = (
            args.generated_sequence
            if args.generated_sequence is not None
            else _read_single_fasta(args.generated_fasta)
        )
    generated_fasta = args.output_dir / "generated_rna.fasta"
    generated_fasta.write_text(f">generated_rna\n{sequence}\n", encoding="utf-8")
    result = evaluate_rna(
        generated_fasta,
        generated_complex,
        native_complex,
        target_chains=target_chains,
        rna_chain=rna_chain,
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

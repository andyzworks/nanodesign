#!/usr/bin/env python3
"""Build the concrete NanoDesign v0 data catalogs from downloaded structures.

This script is deliberately source-specific.  It does not fabricate examples when a
structure or required chain cannot be resolved: every exclusion is counted and written
to the release report.  The resulting JSONL catalogs contain the exact chain and residue
selections needed by the training loader, together with the raw-coordinate provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
RNA_ALPHABET = frozenset("ACGU")
BACKBONE = frozenset({"N", "CA", "C", "O"})
CONTACT_DISTANCE_ANGSTROM = 6.0
RNA_CONTACT_DISTANCE_ANGSTROM = 5.0
PPIREF_VERSION = "PPIRef-6A filtered_clustered_04; Zenodo 14845086; PDB January 2024"
SABDAB2_VERSION = "SAbDab2 ML dataset 0.1.0; Zenodo 20083995 (2026-05-08)"
RNASOLO2_VERSION = (
    "BGSU representatives; BGSU 4.54 (2026-08-27), PDB snapshot 2025-10-25, Rfam 15.1"
)
RIBOCENTRE_VERSION = "Ribocentre structures_merged.json commit 94ef203d7934 (2026-08-30 snapshot)"

# These are the Ribocentre entries whose annotated selected ligand is a protein or
# protein peptide.  Fab crystallisation chaperones for fluorogenic aptamers (41, 48)
# and RNA/RNA targets (50, 51) are intentionally excluded.
RIBOCENTRE_PROTEIN_TARGET_GROUPS = frozenset(
    {27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 49}
)


@dataclass(frozen=True)
class ChainInfo:
    chain_id: str
    sequence: str
    residue_keys: tuple[tuple[int, str], ...]
    heavy_atom_count: int
    backbone_complete_fraction: float
    coordinates: np.ndarray
    atom_residue_indices: np.ndarray


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"cannot serialise {type(value)!r}")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: item["sample_id"]):
            line = json.dumps(row, sort_keys=True, separators=(",", ":"), default=_json_default)
            handle.write(line + "\n")
            digest.update((line + "\n").encode("utf-8"))
            count += 1
    os.replace(temporary, path)
    return count, digest.hexdigest()


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _normalise_icode(value: str) -> str:
    return "" if value in {"", " ", "?", ".", "\x00"} else value


def _accepted_altloc(atom: gemmi.Atom) -> bool:
    return atom.altloc in {"\x00", " ", "A"}


def _heavy_atom_coordinates(residue: gemmi.Residue) -> list[tuple[str, np.ndarray]]:
    observed: set[str] = set()
    atoms: list[tuple[str, np.ndarray]] = []
    for atom in residue:
        name = atom.name.strip()
        if name in observed or not _accepted_altloc(atom) or atom.element.name == "H":
            continue
        position = np.asarray([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32)
        if np.isfinite(position).all():
            atoms.append((name, position))
            observed.add(name)
    return atoms


def _protein_letter(residue_name: str) -> str | None:
    info = gemmi.find_tabulated_residue(residue_name)
    letter = info.one_letter_code
    return letter if info.is_amino_acid() and letter in AA_ALPHABET else None


def _rna_letter(residue_name: str) -> str | None:
    name = residue_name.strip().upper()
    if name in RNA_ALPHABET:
        return name
    info = gemmi.find_tabulated_residue(name)
    letter = info.one_letter_code.upper()
    return letter if info.is_nucleic_acid() and letter in RNA_ALPHABET else None


def extract_chain(chain: gemmi.Chain, polymer: str) -> ChainInfo | None:
    letters: list[str] = []
    residue_keys: list[tuple[int, str]] = []
    coordinates: list[np.ndarray] = []
    atom_residue_indices: list[int] = []
    complete_backbones = 0
    for residue in chain:
        letter = (
            _protein_letter(residue.name) if polymer == "protein" else _rna_letter(residue.name)
        )
        if letter is None:
            continue
        atoms = _heavy_atom_coordinates(residue)
        atom_names = {name for name, _ in atoms}
        representative = "CA" if polymer == "protein" else "C1'"
        if representative not in atom_names:
            continue
        residue_index = len(letters)
        letters.append(letter)
        residue_keys.append((int(residue.seqid.num), _normalise_icode(residue.seqid.icode)))
        coordinates.extend(position for _, position in atoms)
        atom_residue_indices.extend([residue_index] * len(atoms))
        required = BACKBONE if polymer == "protein" else frozenset({"P", "C4'", "C1'"})
        complete_backbones += required.issubset(atom_names)
    if not letters or not coordinates:
        return None
    return ChainInfo(
        chain_id=chain.name,
        sequence="".join(letters),
        residue_keys=tuple(residue_keys),
        heavy_atom_count=len(coordinates),
        backbone_complete_fraction=complete_backbones / len(letters),
        coordinates=np.stack(coordinates),
        atom_residue_indices=np.asarray(atom_residue_indices, dtype=np.int32),
    )


def read_structure(path: Path) -> gemmi.Structure:
    structure = gemmi.read_structure(str(path))
    if len(structure) == 0:
        raise ValueError("structure has no coordinate models")
    return structure


def _chain_map(path: Path, polymer: str) -> dict[str, ChainInfo]:
    structure = read_structure(path)
    answer: dict[str, ChainInfo] = {}
    for chain in structure[0]:
        info = extract_chain(chain, polymer)
        if info is not None:
            answer[chain.name] = info
    return answer


def _residue_contact_counts(left: ChainInfo, right: ChainInfo, cutoff: float) -> tuple[int, int]:
    if len(left.coordinates) == 0 or len(right.coordinates) == 0:
        return 0, 0
    neighbours = cKDTree(right.coordinates).query_ball_point(left.coordinates, cutoff)
    left_atoms = np.fromiter((bool(items) for items in neighbours), dtype=bool)
    if not left_atoms.any():
        return 0, 0
    right_atom_indices = {index for items in neighbours for index in items}
    left_residues = np.unique(left.atom_residue_indices[left_atoms])
    right_residues = np.unique(right.atom_residue_indices[list(right_atom_indices)])
    return len(left_residues), len(right_residues)


def _record_chain(
    info: ChainInfo, role: str, *, include_residue_keys: bool = True
) -> dict[str, Any]:
    record = {
        "chain_id": info.chain_id,
        "role": role,
        "sequence": info.sequence,
        "resolved_residues": len(info.sequence),
        "heavy_atoms": info.heavy_atom_count,
        "backbone_complete_fraction": round(info.backbone_complete_fraction, 6),
    }
    if include_residue_keys:
        record["residue_keys"] = [list(key) for key in info.residue_keys]
    return record


def _locate_structure(structure_dir: Path, pdb_id: str) -> Path | None:
    for suffix in (".pdb.gz", ".cif.gz", ".pdb", ".cif"):
        candidate = structure_dir / f"{pdb_id.lower()}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _parse_ppiref_pdb(
    payload: tuple[str, list[str], str, str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    pdb_id, interface_ids, structure_dir_text, repo_root_text = payload
    structure_dir = Path(structure_dir_text)
    repo_root = Path(repo_root_text)
    rejected: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    structure_path = _locate_structure(structure_dir, pdb_id)
    if structure_path is None:
        return [], Counter({"missing_full_structure": len(interface_ids)})
    try:
        chains = _chain_map(structure_path, "protein")
    except Exception:  # noqa: BLE001 - malformed third-party structures are rejected
        return [], Counter({"unparseable_full_structure": len(interface_ids)})
    for interface_id in interface_ids:
        parts = interface_id.split("_")
        if len(parts) != 3:
            rejected["invalid_interface_id"] += 1
            continue
        chain_a, chain_b = (chains.get(parts[1]), chains.get(parts[2]))
        if chain_a is None or chain_b is None:
            rejected["chain_not_resolved"] += 1
            continue
        if min(len(chain_a.sequence), len(chain_b.sequence)) < 4:
            rejected["chain_shorter_than_4_residues"] += 1
            continue
        contact_a, contact_b = _residue_contact_counts(chain_a, chain_b, CONTACT_DISTANCE_ANGSTROM)
        if contact_a == 0 or contact_b == 0:
            rejected["no_6A_contact_in_full_structure"] += 1
            continue
        ordered = sorted((chain_a, chain_b), key=lambda item: (-len(item.sequence), item.chain_id))
        target, binder = ordered[0], ordered[1]
        records.append(
            {
                "sample_id": f"ppiref50k:{interface_id}",
                "task": "protein_binder",
                "source": "ppiref50k",
                "source_version": PPIREF_VERSION,
                "purpose": "binding_design",
                "raw_paths": [str(structure_path.relative_to(repo_root))],
                "pdb_id": pdb_id,
                # PPIRef chains cover the full resolved standard polymer chain.  The
                # loader deterministically reconstructs those keys from the pinned raw
                # structure, avoiding hundreds of MB of duplicated residue-key arrays.
                "chains": [
                    _record_chain(target, "target", include_residue_keys=False),
                    _record_chain(binder, "binder", include_residue_keys=False),
                ],
                "quality": {
                    "source_resolution_max_A": 3.5,
                    "source_bsa_min_A2": 500.0,
                    "source_idist_dedup_threshold": 0.04,
                    "contact_cutoff_A": CONTACT_DISTANCE_ANGSTROM,
                    "target_contact_residues": contact_a
                    if target.chain_id == chain_a.chain_id
                    else contact_b,
                    "binder_contact_residues": contact_b
                    if binder.chain_id == chain_b.chain_id
                    else contact_a,
                },
                "assignment": "longer_resolved_chain_is_target; ties_by_chain_id",
                "cluster_ids": {"ppiref_interface": interface_id},
            }
        )
    return records, rejected


def _parse_ppiref_interface(
    payload: tuple[str, str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Parse one official PPIRef interface coordinate file."""

    path_text, repo_root_text = payload
    path = Path(path_text)
    repo_root = Path(repo_root_text)
    interface_id = path.stem
    parts = interface_id.split("_")
    if len(parts) != 3:
        return [], Counter({"invalid_interface_filename": 1})
    pdb_id, chain_a_id, chain_b_id = parts
    try:
        chains = _chain_map(path, "protein")
    except Exception:  # noqa: BLE001 - malformed third-party structures are rejected
        return [], Counter({"unparseable_interface_structure": 1})
    chain_a, chain_b = chains.get(chain_a_id), chains.get(chain_b_id)
    if chain_a is None or chain_b is None:
        return [], Counter({"chain_not_resolved": 1})
    if min(len(chain_a.sequence), len(chain_b.sequence)) < 4:
        return [], Counter({"chain_shorter_than_4_residues": 1})
    contact_a, contact_b = _residue_contact_counts(chain_a, chain_b, CONTACT_DISTANCE_ANGSTROM)
    if contact_a == 0 or contact_b == 0:
        return [], Counter({"no_6A_contact_in_interface_structure": 1})
    target, binder = sorted(
        (chain_a, chain_b), key=lambda item: (-len(item.sequence), item.chain_id)
    )
    record = {
        "sample_id": f"ppiref50k:{interface_id}",
        "task": "protein_binder",
        "source": "ppiref50k",
        "source_version": PPIREF_VERSION,
        "purpose": "binding_design",
        "raw_paths": [str(path.relative_to(repo_root))],
        "pdb_id": pdb_id.lower(),
        "chains": [
            _record_chain(target, "target", include_residue_keys=False),
            _record_chain(binder, "binder", include_residue_keys=False),
        ],
        "quality": {
            "source_resolution_max_A": 3.5,
            "source_bsa_min_A2": 500.0,
            "source_idist_dedup_threshold": 0.04,
            "contact_cutoff_A": CONTACT_DISTANCE_ANGSTROM,
            "target_contact_residues": contact_a
            if target.chain_id == chain_a.chain_id
            else contact_b,
            "binder_contact_residues": contact_b
            if binder.chain_id == chain_b.chain_id
            else contact_a,
        },
        "assignment": "longer_resolved_chain_is_target; ties_by_chain_id",
        "cluster_ids": {"ppiref_interface": interface_id},
    }
    return [record], Counter()


def _parse_ppiref_candidate(
    payload: tuple[str, str, str, str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    interface_id, interface_dir_text, structure_dir_text, repo_root_text = payload
    pdb_id = interface_id.split("_", 1)[0].lower()
    interface_path = Path(interface_dir_text) / pdb_id[1:3] / f"{interface_id}.pdb"
    if interface_path.is_file():
        return _parse_ppiref_interface((str(interface_path), repo_root_text))
    rows, rejected = _parse_ppiref_pdb((pdb_id, [interface_id], structure_dir_text, repo_root_text))
    if not rows and rejected.get("missing_full_structure"):
        return [], Counter({"missing_interface_and_full_structure": 1})
    return rows, rejected


def prepare_ppiref(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    interface_dir = Path(args.interface_dir).resolve()
    structure_dir = Path(args.structure_dir).resolve()
    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    interface_ids = sorted(split["folds"]["whole"])
    payloads = [
        (
            interface_id,
            str(interface_dir),
            str(structure_dir),
            str(repo_root),
        )
        for interface_id in interface_ids
    ]
    rejections: Counter[str] = Counter()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    digest_state = hashlib.sha256()
    count = 0

    # Each catalog record contains complete residue selections and can be large.
    # Never retain the 50K release in RAM: parse entries in deterministic order and
    # stream records directly to the catalog.  A one-worker path also avoids keeping
    # a duplicate Python process alive on memory-constrained login/CPU nodes.
    executor = None
    if args.workers == 1:
        results = map(_parse_ppiref_candidate, payloads)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        results = executor.map(_parse_ppiref_candidate, payloads, chunksize=8)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for completed, (rows, rejected) in enumerate(results, start=1):
                for row in rows:
                    line = json.dumps(
                        row, sort_keys=True, separators=(",", ":"), default=_json_default
                    )
                    handle.write(line + "\n")
                    digest_state.update((line + "\n").encode("utf-8"))
                    count += 1
                rejections.update(rejected)
                if completed % 1000 == 0:
                    print(f"PPIRef: parsed {completed}/{len(payloads)} PDB entries", flush=True)
    finally:
        if executor is not None:
            executor.shutdown()
    os.replace(temporary, output)
    digest = digest_state.hexdigest()
    write_report(
        Path(args.report),
        {
            "source": "ppiref50k",
            "source_version": PPIREF_VERSION,
            "official_candidates": len(interface_ids),
            "usable_samples": count,
            "exclusive_rejection_counts": dict(sorted(rejections.items())),
            "catalog_sha256": digest,
            "filtering": [
                "official PPIRef X-ray/EM and resolution <= 3.5 A",
                "official buried surface area >= 500 A^2",
                "official iDist 0.04 structural deduplication",
                "both official interface chains resolve to >=4 standard residues with CA coordinates",
                "at least one inter-chain heavy-atom contact within 6 A in the official interface coordinates",
            ],
        },
    )
    print(f"PPIRef usable={count}; rejected={sum(rejections.values())}; sha256={digest}")


def _parse_numbering_list(value: str) -> list[tuple[str, tuple[int, str]]]:
    parsed = json.loads(value)
    return [
        (str(letter), (int(position[0]), _normalise_icode(str(position[1]))))
        for letter, position in parsed
    ]


def _split_slash(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split("/") if item.strip()]


def _parse_sabdab_row(
    row: dict[str, Any], structure_root_text: str, repo_root_text: str
) -> tuple[dict[str, Any] | None, str | None]:
    structure_root = Path(structure_root_text)
    repo_root = Path(repo_root_text)
    if not bool(row["holo"]):
        return None, "not_holo"
    if pd.isna(row["Hchain"]) or pd.isna(row["CDRH3"]):
        return None, "missing_heavy_chain_or_cdrh3"
    antigen_chains = _split_slash(row["agchains"])
    antigen_types = _split_slash(row["agtypes"])
    if not antigen_chains or len(antigen_chains) != len(antigen_types):
        return None, "missing_or_misaligned_antigen_annotation"
    if not set(antigen_types).issubset({"PROTEIN", "PEPTIDE"}):
        return None, "non_protein_or_peptide_antigen"
    path = structure_root / f"{row['INSTANCE']}.cif"
    if not path.is_file():
        return None, "missing_structure"
    try:
        chains = _chain_map(path, "protein")
        heavy = chains.get(str(row["Hchain"]))
        if heavy is None:
            return None, "heavy_chain_not_resolved"
        numbering = _parse_numbering_list(str(row["VH_numbering_list"]))
    except Exception:  # noqa: BLE001 - malformed third-party structures/metadata are rejected
        return None, "unparseable_structure_or_numbering"
    h3_numbering = [(letter, key) for letter, key in numbering if 105 <= key[0] <= 117]
    h3_keys = [key for _, key in h3_numbering]
    h3_sequence = "".join(letter for letter, _ in h3_numbering)
    if not h3_keys or h3_sequence != str(row["CDRH3"]):
        return None, "cdrh3_numbering_mismatch"
    heavy_key_set = set(heavy.residue_keys)
    if not set(h3_keys).issubset(heavy_key_set):
        return None, "cdrh3_not_fully_resolved"
    structure = read_structure(path)
    heavy_chain = structure[0].find_chain(str(row["Hchain"]))
    if heavy_chain is None:
        return None, "heavy_chain_not_resolved"
    h3_key_set = set(h3_keys)
    for residue in heavy_chain:
        key = (int(residue.seqid.num), _normalise_icode(residue.seqid.icode))
        if key in h3_key_set:
            names = {name for name, _ in _heavy_atom_coordinates(residue)}
            if not BACKBONE.issubset(names):
                return None, "cdrh3_backbone_incomplete"
    context: list[dict[str, Any]] = []
    heavy_record = _record_chain(heavy, "antibody_framework+cdr_h3")
    heavy_record["design_residue_keys"] = [list(key) for key in h3_keys]
    heavy_record["design_sequence"] = h3_sequence
    context.append(heavy_record)
    if not pd.isna(row["Lchain"]):
        light = chains.get(str(row["Lchain"]))
        if light is None:
            return None, "light_chain_not_resolved"
        context.append(_record_chain(light, "antibody_framework"))
    for chain_id in antigen_chains:
        antigen = chains.get(chain_id)
        if antigen is None:
            return None, "antigen_chain_not_resolved"
        context.append(_record_chain(antigen, "antigen"))
    split = str(row["ab_ag_split"])
    if split not in {"train", "test"} or pd.isna(row["ab_ag_cluster"]):
        return None, "missing_official_abag_split_or_cluster"
    cluster = str(int(row["ab_ag_cluster"]))
    return (
        {
            "sample_id": f"sabdab2:{row['INSTANCE']}",
            "task": "antibody_cdr",
            "source": "sabdab2",
            "source_version": SABDAB2_VERSION,
            "purpose": "binding_design",
            "raw_paths": [str(path.relative_to(repo_root))],
            "pdb_id": str(row["PDB_ID"]),
            "chains": context,
            "native_split": split,
            "cluster_ids": {
                "sabdab2_ab_ag": cluster,
                "sabdab2_cdrh3": str(row["cdrh3_cluster"]),
            },
            "quality": {
                "method": str(row["method"]),
                "resolution_A": None if pd.isna(row["resolution"]) else float(row["resolution"]),
                "cdrh3_length": len(h3_sequence),
                "antigen_types": antigen_types,
            },
        },
        None,
    )


def prepare_sabdab2(args: argparse.Namespace) -> None:
    metadata = pd.read_csv(args.metadata)
    rows = metadata.to_dict(orient="records")
    records: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    structure_root = str(Path(args.structure_root).resolve())
    repo_root = str(Path(args.repo_root).resolve())
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(
            _parse_sabdab_row,
            rows,
            repeat(structure_root),
            repeat(repo_root),
            chunksize=16,
        )
        for completed, (record, reason) in enumerate(results, start=1):
            if record is None:
                rejected[reason or "unknown"] += 1
            else:
                records.append(record)
            if completed % 2000 == 0:
                print(f"SAbDab2: parsed {completed}/{len(rows)} structures", flush=True)
    count, digest = write_jsonl(Path(args.output), records)
    native_counts = Counter(row["native_split"] for row in records)
    write_report(
        Path(args.report),
        {
            "source": "sabdab2",
            "source_version": SABDAB2_VERSION,
            "official_structures": len(rows),
            "usable_samples": count,
            "usable_native_split_counts": dict(sorted(native_counts.items())),
            "exclusive_rejection_counts": dict(sorted(rejected.items())),
            "catalog_sha256": digest,
            "filtering": [
                "official SAbDab2 X-ray/EM resolution <=3.5 A curation",
                "holo complex with a heavy chain and non-empty CDR-H3",
                "all annotated antigens are protein or peptide",
                "IMGT CDR-H3 positions 105-117 match metadata and have complete N/CA/C/O",
                "official antigen-aware ab_ag_cluster and 80/20 train/test split are present",
            ],
        },
    )
    print(f"SAbDab2 usable={count}; rejected={sum(rejected.values())}; sha256={digest}")


def _parse_rnasolo(path_text: str, repo_root_text: str) -> tuple[dict[str, Any] | None, str | None]:
    path = Path(path_text).resolve()
    repo_root = Path(repo_root_text)
    try:
        chains = _chain_map(path, "rna")
    except Exception:  # noqa: BLE001 - malformed third-party structures are rejected
        return None, "unparseable_structure"
    usable = [chain for chain in chains.values() if len(chain.sequence) >= 8]
    if not usable:
        return None, "no_resolved_rna_chain_length_at_least_8"
    return (
        {
            "sample_id": f"rnasolo2:{path.stem.lower()}",
            "task": "rna_aptamer",
            "source": "rnasolo2",
            "source_version": RNASOLO2_VERSION,
            "purpose": "rna_structure_prior",
            "raw_paths": [str(path.relative_to(repo_root))],
            "chains": [_record_chain(chain, "rna_structure_prior") for chain in usable],
            "cluster_ids": {"rnasolo_representative": path.stem},
        },
        None,
    )


def prepare_rnasolo2(args: argparse.Namespace) -> None:
    paths = sorted(Path(args.structure_root).rglob("*.cif"))
    records: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(_parse_rnasolo, str(path), str(Path(args.repo_root).resolve()))
            for path in paths
        ]
        for future in as_completed(futures):
            record, reason = future.result()
            if record is None:
                rejected[reason or "unknown"] += 1
            else:
                records.append(record)
    count, digest = write_jsonl(Path(args.output), records)
    write_report(
        Path(args.report),
        {
            "source": "rnasolo2",
            "source_version": RNASOLO2_VERSION,
            "downloaded_representative_structures": len(paths),
            "usable_auxiliary_samples": count,
            "exclusive_rejection_counts": dict(sorted(rejected.items())),
            "catalog_sha256": digest,
            "binding_ground_truth": False,
            "filtering": [
                "BGSU nonredundant representative set only",
                "solo-RNA molecule class only (not protein-RNA complexes)",
                "X-ray/EM resolution <=4 A or NMR",
                "at least one resolved RNA chain of length >=8 with C1' coordinates",
            ],
        },
    )
    print(f"RNAsolo2 usable={count}; rejected={sum(rejected.values())}; sha256={digest}")


def _ribocentre_pdb_ids(structures_json: Path) -> list[tuple[int, str, str]]:
    groups = json.loads(structures_json.read_text(encoding="utf-8"))
    answer: list[tuple[int, str, str]] = []
    for index in sorted(RIBOCENTRE_PROTEIN_TARGET_GROUPS):
        group = groups[index]
        raw_ids = str(group["NDB"]["id"]).replace(",", " ").split()
        for pdb_id in raw_ids:
            if len(pdb_id) == 4 and pdb_id.isalnum():
                answer.append((index, pdb_id.lower(), str(group["Name"])))
    return answer


def _connected_rna_protein_components(
    rna_chains: dict[str, ChainInfo], protein_chains: dict[str, ChainInfo]
) -> list[tuple[list[ChainInfo], list[ChainInfo], dict[str, tuple[int, int]]]]:
    edges: dict[tuple[str, str], tuple[int, int]] = {}
    graph: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for rna_id, rna in rna_chains.items():
        if not 8 <= len(rna.sequence) <= 256:
            continue
        for protein_id, protein in protein_chains.items():
            if len(protein.sequence) < 5:
                continue
            rna_contacts, protein_contacts = _residue_contact_counts(
                rna, protein, RNA_CONTACT_DISTANCE_ANGSTROM
            )
            if rna_contacts >= 3 and protein_contacts >= 3:
                edges[(rna_id, protein_id)] = (rna_contacts, protein_contacts)
                left, right = ("rna", rna_id), ("protein", protein_id)
                graph[left].add(right)
                graph[right].add(left)
    components: list[tuple[list[ChainInfo], list[ChainInfo], dict[str, tuple[int, int]]]] = []
    visited: set[tuple[str, str]] = set()
    for start in sorted(graph):
        if start in visited:
            continue
        stack = [start]
        nodes: set[tuple[str, str]] = set()
        while stack:
            node = stack.pop()
            if node in nodes:
                continue
            nodes.add(node)
            visited.add(node)
            stack.extend(graph[node] - nodes)
        rnas = [rna_chains[chain_id] for kind, chain_id in sorted(nodes) if kind == "rna"]
        proteins = [
            protein_chains[chain_id] for kind, chain_id in sorted(nodes) if kind == "protein"
        ]
        component_edges = {
            f"{rna_id}:{protein_id}": counts
            for (rna_id, protein_id), counts in edges.items()
            if ("rna", rna_id) in nodes and ("protein", protein_id) in nodes
        }
        if rnas and proteins:
            components.append((rnas, proteins, component_edges))
    return components


def _parse_ribocentre_entry(
    payload: tuple[int, str, str, str, str],
) -> tuple[list[dict[str, Any]], str | None]:
    group_index, pdb_id, aptamer_name, structure_root_text, repo_root_text = payload
    structure_root, repo_root = Path(structure_root_text), Path(repo_root_text)
    path = _locate_structure(structure_root, pdb_id)
    if path is None:
        return [], "missing_structure"
    try:
        rna_chains = _chain_map(path, "rna")
        protein_chains = _chain_map(path, "protein")
        components = _connected_rna_protein_components(rna_chains, protein_chains)
    except Exception:  # noqa: BLE001 - malformed third-party structures are rejected
        return [], "unparseable_structure"
    if not components:
        return [], "no_supported_rna_protein_contact_component"
    records: list[dict[str, Any]] = []
    for component_index, (rnas, proteins, contacts) in enumerate(components):
        records.append(
            {
                "sample_id": f"ribocentre:{pdb_id}:{component_index}",
                "task": "rna_aptamer",
                "source": "ribocentre_aptamer",
                "source_version": RIBOCENTRE_VERSION,
                "purpose": "binding_design",
                "raw_paths": [str(path.relative_to(repo_root))],
                "pdb_id": pdb_id,
                "ribocentre_group": group_index,
                "aptamer_name": aptamer_name,
                "chains": [_record_chain(chain, "target") for chain in proteins]
                + [_record_chain(chain, "rna_aptamer") for chain in rnas],
                "quality": {
                    "contact_cutoff_A": RNA_CONTACT_DISTANCE_ANGSTROM,
                    "minimum_contact_residues_per_partner": 3,
                    "chain_pair_contact_residues": contacts,
                },
                "cluster_ids": {"native_complex": pdb_id},
            }
        )
    return records, None


def prepare_ribocentre(args: argparse.Namespace) -> None:
    entries = _ribocentre_pdb_ids(Path(args.structures_json))
    payloads = [
        (
            index,
            pdb_id,
            name,
            str(Path(args.structure_root).resolve()),
            str(Path(args.repo_root).resolve()),
        )
        for index, pdb_id, name in entries
    ]
    records: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_parse_ribocentre_entry, payload) for payload in payloads]
        for future in as_completed(futures):
            rows, reason = future.result()
            if not rows:
                rejected[reason or "unknown"] += 1
            records.extend(rows)
    count, digest = write_jsonl(Path(args.output), records)
    write_report(
        Path(args.report),
        {
            "source": "ribocentre_aptamer",
            "source_version": RIBOCENTRE_VERSION,
            "curated_protein_target_groups": len(RIBOCENTRE_PROTEIN_TARGET_GROUPS),
            "candidate_pdb_entries": len(entries),
            "usable_contact_components": count,
            "exclusive_rejection_counts_by_pdb": dict(sorted(rejected.items())),
            "catalog_sha256": digest,
            "filtering": [
                "Ribocentre annotation identifies the selected target as protein/protein peptide",
                "exclude Fab crystallisation chaperones and RNA-target entries",
                "RNA length 8-256 resolved residues; protein/peptide length >=5",
                "at least 3 residues on each partner contact within 5 A",
            ],
        },
    )
    print(f"Ribocentre usable={count}; rejected PDBs={sum(rejected.values())}; sha256={digest}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1])
    subparsers = parser.add_subparsers(dest="command", required=True)

    ppiref = subparsers.add_parser("ppiref")
    ppiref.add_argument("--interface-dir", default="data/raw/ppiref/ppiref50k")
    ppiref.add_argument("--structure-dir", default="data/raw/ppiref/full_pdb")
    ppiref.add_argument(
        "--split-file",
        default="data/raw/ppiref/splits/ppiref_6A_filtered_clustered_04.json",
    )
    ppiref.add_argument("--output", default="data/processed/v0/catalogs/ppiref50k.jsonl")
    ppiref.add_argument("--report", default="data/processed/v0/reports/ppiref50k.json")
    ppiref.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    ppiref.set_defaults(func=prepare_ppiref)

    sabdab = subparsers.add_parser("sabdab2")
    sabdab.add_argument(
        "--metadata", default="data/raw/sabdab2/extracted/splits_final/abag_split.csv"
    )
    sabdab.add_argument("--structure-root", default="data/raw/sabdab2/extracted/splits_final")
    sabdab.add_argument("--output", default="data/processed/v0/catalogs/sabdab2.jsonl")
    sabdab.add_argument("--report", default="data/processed/v0/reports/sabdab2.json")
    sabdab.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    sabdab.set_defaults(func=prepare_sabdab2)

    rnasolo = subparsers.add_parser("rnasolo2")
    rnasolo.add_argument("--structure-root", default="data/raw/rnasolo2/structures")
    rnasolo.add_argument("--output", default="data/processed/v0/catalogs/rnasolo2.jsonl")
    rnasolo.add_argument("--report", default="data/processed/v0/reports/rnasolo2.json")
    rnasolo.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    rnasolo.set_defaults(func=prepare_rnasolo2)

    ribocentre = subparsers.add_parser("ribocentre")
    ribocentre.add_argument("--structures-json", required=True)
    ribocentre.add_argument("--structure-root", default="data/raw/ribocentre/structures")
    ribocentre.add_argument("--output", default="data/processed/v0/catalogs/ribocentre.jsonl")
    ribocentre.add_argument("--report", default="data/processed/v0/reports/ribocentre.json")
    ribocentre.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ribocentre.set_defaults(func=prepare_ribocentre)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    os.chdir(Path(args.repo_root).resolve())
    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the experimental PDB RNA-protein complex portion of NanoDesign v0.

The RCSB Search/Data APIs are frozen to a local metadata snapshot first.  Only eligible
RNA/protein chain coordinates are then fetched from the official RCSB ModelServer, so a
large ribosome entry does not require downloading every unrelated atom.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_v0_data import (
    _connected_rna_protein_components,
    _record_chain,
    extract_chain,
    read_structure,
    write_jsonl,
    write_report,
)

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"
RCSB_MODELS_URL = "https://models.rcsb.org/v1"
SNAPSHOT_DATE = "2026-08-30"
SOURCE_VERSION = f"RCSB PDB Search/Data API snapshot {SNAPSHOT_DATE}"
GRAPHQL_QUERY = """
query($ids:[String!]!) {
  entries(entry_ids:$ids) {
    rcsb_id
    rcsb_entry_info { experimental_method resolution_combined }
    exptl { method }
    polymer_entities {
      entity_poly { rcsb_entity_polymer_type pdbx_seq_one_letter_code_can }
      rcsb_polymer_entity_container_identifiers { entity_id asym_ids auth_asym_ids }
    }
  }
}
"""


def _request_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    retries: int = 6,
    **kwargs: Any,
) -> dict[str, Any]:
    error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.request(method, url, timeout=90, **kwargs)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as caught:
            error = caught
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"request failed after {retries} attempts: {url}") from error


def fetch_ids(output: Path) -> list[str]:
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_RNA",
                        "operator": "greater",
                        "value": 0,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "greater",
                        "value": 0,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }
    with requests.Session() as session:
        value = _request_json(session, "POST", RCSB_SEARCH_URL, json=query)
    ids = sorted(result["identifier"].lower() for result in value["result_set"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "snapshot_date": SNAPSHOT_DATE,
                "query": query,
                "total_count": int(value["total_count"]),
                "entry_ids": ids,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return ids


def fetch_metadata(ids: list[str], output: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with requests.Session() as session:
        for start in range(0, len(ids), 100):
            batch = ids[start : start + 100]
            value = _request_json(
                session,
                "POST",
                RCSB_GRAPHQL_URL,
                json={"query": GRAPHQL_QUERY, "variables": {"ids": batch}},
            )
            if value.get("errors"):
                raise RuntimeError(f"RCSB GraphQL error: {value['errors']}")
            entries.extend(item for item in value["data"]["entries"] if item is not None)
            print(f"RCSB metadata: {min(start + 100, len(ids))}/{len(ids)}", flush=True)
    write_jsonl(
        output,
        ({"sample_id": item["rcsb_id"].lower(), **item} for item in entries),
    )
    return entries


def load_metadata(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for row in rows:
        row.pop("sample_id", None)
    return rows


def _canonical_sequence(value: str | None) -> str:
    return "" if value is None else "".join(str(value).split()).upper()


def _method_and_resolution(entry: dict[str, Any]) -> tuple[str, float | None]:
    info = entry.get("rcsb_entry_info") or {}
    methods = info.get("experimental_method") or []
    if isinstance(methods, str):
        methods = [methods]
    exptl = entry.get("exptl") or []
    methods.extend(item.get("method", "") for item in exptl if item)
    method = ";".join(sorted({str(item).upper() for item in methods if item}))
    resolutions = info.get("resolution_combined") or []
    resolution = min(float(item) for item in resolutions) if resolutions else None
    return method, resolution


def eligible_entry(entry: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    method, resolution = _method_and_resolution(entry)
    is_nmr = "NMR" in method
    is_resolution_method = "X-RAY" in method or "ELECTRON MICROSCOPY" in method
    if not is_nmr and not is_resolution_method:
        return None, "unsupported_experimental_method"
    if is_resolution_method and (resolution is None or resolution > 4.0):
        return None, "resolution_above_4A_or_missing"
    rna: list[dict[str, str]] = []
    protein: list[dict[str, str]] = []
    for entity in entry.get("polymer_entities") or []:
        polymer = entity.get("entity_poly") or {}
        identifiers = entity.get("rcsb_polymer_entity_container_identifiers") or {}
        polymer_type = str(polymer.get("rcsb_entity_polymer_type", "")).upper()
        sequence = _canonical_sequence(polymer.get("pdbx_seq_one_letter_code_can"))
        asym_ids = identifiers.get("asym_ids") or []
        auth_ids = identifiers.get("auth_asym_ids") or []
        if polymer_type == "RNA" and 8 <= len(sequence) <= 256:
            target = rna
        elif polymer_type == "PROTEIN" and 20 <= len(sequence) <= 2048:
            target = protein
        else:
            continue
        for index, asym_id in enumerate(asym_ids):
            auth_id = auth_ids[index] if index < len(auth_ids) else ""
            target.append(
                {
                    "label_asym_id": str(asym_id),
                    "auth_asym_id": str(auth_id),
                    "entity_sequence": sequence,
                }
            )
    if not rna or not protein:
        return None, "no_eligible_rna_and_protein_entity_pair"
    # NanoDesign v0 is an aptamer-target benchmark, not a ribosome/virus assembly
    # benchmark.  Keep small experimentally resolved complexes while still allowing a
    # dimeric RNA and a tetrameric protein target.  Applying this before coordinate
    # downloads also prevents one large biological assembly from contributing hundreds
    # of nominal chain pairs.
    if len(rna) > 2 or len(protein) > 4:
        return None, "assembly_above_2_rna_or_4_protein_chains"
    return (
        {
            "pdb_id": entry["rcsb_id"].lower(),
            "method": method,
            "resolution_A": resolution,
            "rna": rna,
            "protein": protein,
        },
        None,
    )


def _download_chain(payload: tuple[str, str, Path]) -> tuple[str, str, Path, str | None]:
    pdb_id, asym_id, output_dir = payload
    path = output_dir / pdb_id[1:3] / f"{pdb_id}_{asym_id}.cif"
    if path.is_file() and path.stat().st_size > 0:
        return pdb_id, asym_id, path, None
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RCSB_MODELS_URL}/{pdb_id}/atoms?label_asym_id={quote(asym_id)}&encoding=cif"
    temporary = path.with_suffix(".cif.tmp")
    error: Exception | None = None
    for attempt in range(6):
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            if len(response.content) < 100:
                raise RuntimeError("empty ModelServer response")
            temporary.write_bytes(response.content)
            os.replace(temporary, path)
            return pdb_id, asym_id, path, None
        except (requests.RequestException, OSError, RuntimeError) as caught:
            error = caught
            time.sleep(min(2**attempt, 20))
    temporary.unlink(missing_ok=True)
    return pdb_id, asym_id, path, repr(error)


def _extract_modelserver_chain(path: Path, polymer: str, label_asym_id: str):
    structure = read_structure(path)
    candidates = []
    for chain in structure[0]:
        info = extract_chain(chain, polymer)
        if info is not None:
            candidates.append(info)
    if len(candidates) != 1:
        return None
    original = candidates[0]
    return type(original)(
        chain_id=label_asym_id,
        sequence=original.sequence,
        residue_keys=original.residue_keys,
        heavy_atom_count=original.heavy_atom_count,
        backbone_complete_fraction=original.backbone_complete_fraction,
        coordinates=original.coordinates,
        atom_residue_indices=original.atom_residue_indices,
    )


def prepare(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    ids_file = Path(args.ids_file)
    metadata_file = Path(args.metadata_file)
    if args.refresh or not ids_file.is_file():
        ids = fetch_ids(ids_file)
    else:
        ids = json.loads(ids_file.read_text(encoding="utf-8"))["entry_ids"]
    if args.refresh or not metadata_file.is_file():
        metadata = fetch_metadata(ids, metadata_file)
    else:
        metadata = load_metadata(metadata_file)

    ribocentre_ids = {
        line.strip().lower()
        for line in Path(args.ribocentre_ids).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    rejected: Counter[str] = Counter()
    if len(metadata) < len(ids):
        rejected["metadata_missing"] = len(ids) - len(metadata)
    eligible: list[dict[str, Any]] = []
    for entry in metadata:
        pdb_id = entry["rcsb_id"].lower()
        if pdb_id in ribocentre_ids:
            rejected["duplicate_of_ribocentre"] += 1
            continue
        item, reason = eligible_entry(entry)
        if item is None:
            rejected[reason or "unknown_metadata_filter"] += 1
        else:
            eligible.append(item)

    output_dir = Path(args.chain_dir).resolve()
    payloads: list[tuple[str, str, Path]] = []
    for item in eligible:
        for chain in item["rna"] + item["protein"]:
            payloads.append((item["pdb_id"], chain["label_asym_id"], output_dir))
    unique_payloads = sorted(set(payloads), key=lambda item: (item[0], item[1]))
    chain_paths: dict[tuple[str, str], Path] = {}
    failed_entries: set[str] = set()
    with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
        futures = [executor.submit(_download_chain, payload) for payload in unique_payloads]
        for completed, future in enumerate(as_completed(futures), start=1):
            pdb_id, asym_id, path, error = future.result()
            if error is None:
                chain_paths[(pdb_id, asym_id)] = path
            else:
                failed_entries.add(pdb_id)
            if completed % 1000 == 0:
                print(f"RCSB chains: {completed}/{len(futures)}", flush=True)

    records: list[dict[str, Any]] = []
    usable_entries = 0
    for completed, item in enumerate(eligible, start=1):
        pdb_id = item["pdb_id"]
        if pdb_id in failed_entries:
            rejected["chain_download_failure"] += 1
            continue
        rna_chains = {}
        protein_chains = {}
        raw_paths: dict[str, str] = {}
        parse_failed = False
        for polymer, destination in (("rna", rna_chains), ("protein", protein_chains)):
            for chain in item[polymer]:
                asym_id = chain["label_asym_id"]
                path = chain_paths[(pdb_id, asym_id)]
                try:
                    info = _extract_modelserver_chain(path, polymer, asym_id)
                except Exception:  # noqa: BLE001 - record malformed RCSB chains as rejected
                    info = None
                if info is None:
                    parse_failed = True
                    break
                destination[asym_id] = info
                raw_paths[asym_id] = str(path.relative_to(repo_root))
            if parse_failed:
                break
        if parse_failed:
            rejected["chain_parse_failure"] += 1
            continue
        components = _connected_rna_protein_components(rna_chains, protein_chains)
        if not components:
            rejected["no_supported_5A_contact_component"] += 1
            continue
        usable_entries += 1
        for component_index, (rnas, proteins, contacts) in enumerate(components):
            chains = []
            component_paths = []
            for chain in proteins:
                record = _record_chain(chain, "target")
                record["raw_path"] = raw_paths[chain.chain_id]
                chains.append(record)
                component_paths.append(raw_paths[chain.chain_id])
            for chain in rnas:
                record = _record_chain(chain, "rna_aptamer")
                record["raw_path"] = raw_paths[chain.chain_id]
                chains.append(record)
                component_paths.append(raw_paths[chain.chain_id])
            records.append(
                {
                    "sample_id": f"pdb_rna_target:{pdb_id}:{component_index}",
                    "task": "rna_aptamer",
                    "source": "pdb_rna_target_complex",
                    "source_version": SOURCE_VERSION,
                    "purpose": "binding_design",
                    "raw_paths": sorted(set(component_paths)),
                    "pdb_id": pdb_id,
                    "chains": chains,
                    "quality": {
                        "method": item["method"],
                        "resolution_A": item["resolution_A"],
                        "contact_cutoff_A": 5.0,
                        "minimum_contact_residues_per_partner": 3,
                        "chain_pair_contact_residues": contacts,
                    },
                    "cluster_ids": {"native_complex": pdb_id},
                }
            )
        if completed % 500 == 0:
            print(f"RCSB contacts: {completed}/{len(eligible)}", flush=True)

    count, digest = write_jsonl(Path(args.output), records)
    rejected_total = sum(rejected.values())
    if rejected_total + usable_entries != len(ids):
        raise RuntimeError(
            f"entry accounting mismatch: rejected={rejected_total}, usable={usable_entries}, "
            f"candidates={len(ids)}"
        )
    write_report(
        Path(args.report),
        {
            "source": "pdb_rna_target_complex",
            "source_version": SOURCE_VERSION,
            "candidate_pdb_entries": len(ids),
            "metadata_entries": len(metadata),
            "eligible_before_coordinate_contacts": len(eligible),
            "usable_pdb_entries": usable_entries,
            "usable_contact_components": count,
            "exclusive_rejection_counts_by_entry": dict(sorted(rejected.items())),
            "catalog_sha256": digest,
            "filtering": [
                "experimental entry contains both RNA and protein polymer entities",
                "X-ray/EM resolution <=4 A or NMR",
                "RNA entity length 8-256 and protein entity length 20-2048",
                "small-complex scope: at most 2 RNA and 4 protein chains",
                "at least 3 resolved residues per partner contact within 5 A",
                "Ribocentre PDB entries removed from the general PDB source",
            ],
        },
    )
    print(
        f"PDB RNA-target usable entries={usable_entries}; samples={count}; "
        f"rejected={rejected_total}; sha256={digest}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ids-file", default="data/raw/pdb_rna_target/rcsb_entry_ids.json")
    parser.add_argument("--metadata-file", default="data/raw/pdb_rna_target/rcsb_metadata.jsonl")
    parser.add_argument("--ribocentre-ids", required=True)
    parser.add_argument("--chain-dir", default="data/raw/pdb_rna_target/chains")
    parser.add_argument("--output", default="data/processed/v0/catalogs/pdb_rna_target.jsonl")
    parser.add_argument("--report", default="data/processed/v0/reports/pdb_rna_target.json")
    parser.add_argument("--download-workers", type=int, default=32)
    parser.add_argument("--refresh", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.chdir(Path(args.repo_root).resolve())
    prepare(args)


if __name__ == "__main__":
    main()

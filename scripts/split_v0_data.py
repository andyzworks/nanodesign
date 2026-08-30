#!/usr/bin/env python3
"""Cluster and split the concrete NanoDesign v0 catalogs without homology leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SPLIT_ORDER = ("train", "validation", "test")


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            if left_root > right_root:
                left_root, right_root = right_root, left_root
            self.parent[right_root] = left_root


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda value: value["sample_id"]):
            line = json.dumps(row, sort_keys=True, separators=(",", ":"))
            handle.write(line + "\n")
            digest.update((line + "\n").encode("utf-8"))
            count += 1
    os.replace(temporary, path)
    return count, digest.hexdigest()


def _sequence_id(sample_id: str, polymer: str, chain_index: int) -> str:
    key = f"{sample_id}|{polymer}|{chain_index}"
    return f"s{hashlib.sha1(key.encode()).hexdigest()}"


def cluster_sequences(
    name: str,
    sequences: dict[str, str],
    output_root: Path,
    mmseqs: Path,
    min_identity: float,
    coverage: float,
    nucleotide: bool = False,
) -> dict[str, str]:
    work = output_root / "clustering" / name
    work.mkdir(parents=True, exist_ok=True)
    fasta = work / "sequences.fasta"
    with fasta.open("w", encoding="utf-8") as handle:
        for sequence_id, sequence in sorted(sequences.items()):
            handle.write(f">{sequence_id}\n{sequence}\n")
    prefix = work / "clusters"
    cluster_file = prefix.with_name(prefix.name + "_cluster.tsv")
    if cluster_file.is_file():
        assignments: dict[str, str] = {}
        with cluster_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                representative, member = line.rstrip("\n").split("\t")[:2]
                assignments[member] = representative
        for sequence_id in sequences:
            assignments.setdefault(sequence_id, sequence_id)
        return assignments
    command = [
        str(mmseqs),
        "easy-linclust" if nucleotide else "easy-cluster",
        str(fasta),
        str(prefix),
        str(work / "tmp"),
        "--min-seq-id",
        str(min_identity),
        "-c",
        str(coverage),
        "--cov-mode",
        "0",
        "--cluster-mode",
        "2",
        "--threads",
        str(min(8 if nucleotide else 32, os.cpu_count() or 1)),
    ]
    if nucleotide:
        command.extend(["--dbtype", "2"])
    subprocess.run(command, check=True)
    assignments: dict[str, str] = {}
    with cluster_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            representative, member = line.rstrip("\n").split("\t")[:2]
            assignments[member] = representative
    for sequence_id in sequences:
        assignments.setdefault(sequence_id, sequence_id)
    return assignments


def _balanced_assign(
    components: dict[str, list[dict[str, Any]]], ratios: dict[str, float]
) -> dict[str, str]:
    total = sum(len(rows) for rows in components.values())
    target = {split: ratios[split] * total for split in SPLIT_ORDER}
    observed = {split: 0 for split in SPLIT_ORDER}
    assignments: dict[str, str] = {}
    ordered = sorted(
        components.items(),
        key=lambda item: (-len(item[1]), hashlib.sha256(item[0].encode()).hexdigest()),
    )
    for component, rows in ordered:
        deficits = {split: target[split] - observed[split] for split in SPLIT_ORDER}
        split = max(SPLIT_ORDER, key=lambda value: (deficits[value], -SPLIT_ORDER.index(value)))
        assignments[component] = split
        observed[split] += len(rows)
    return assignments


def _component_id(root: str, namespace: str) -> str:
    return f"{namespace}:{hashlib.sha256(root.encode()).hexdigest()[:16]}"


def split_ppiref(
    rows: list[dict[str, Any]], output_root: Path, mmseqs: Path
) -> list[dict[str, Any]]:
    sequences: dict[str, str] = {}
    chain_ids: dict[tuple[str, int], str] = {}
    for row in rows:
        for index, chain in enumerate(row["chains"]):
            sequence_id = _sequence_id(row["sample_id"], "protein", index)
            sequences[sequence_id] = chain["sequence"]
            chain_ids[(row["sample_id"], index)] = sequence_id
    clusters = cluster_sequences(
        "ppiref_protein_30pct_official51755",
        sequences,
        output_root,
        mmseqs,
        min_identity=0.30,
        coverage=0.80,
    )
    union = UnionFind()
    sample_nodes: dict[str, list[str]] = {}
    for row in rows:
        nodes = [
            f"protein:{clusters[chain_ids[(row['sample_id'], index)]]}"
            for index in range(len(row["chains"]))
        ]
        sample_nodes[row["sample_id"]] = nodes
        for node in nodes:
            union.union(nodes[0], node)
    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        root = union.find(sample_nodes[row["sample_id"]][0])
        components[root].append(row)
    assignments = _balanced_assign(components, {"train": 0.8, "validation": 0.1, "test": 0.1})
    answer: list[dict[str, Any]] = []
    for root, component_rows in components.items():
        component_id = _component_id(root, "ppiref_homology_component")
        for row in component_rows:
            chain_cluster_ids = [
                f"mmseqs30:{clusters[chain_ids[(row['sample_id'], index)]]}"
                for index in range(len(row["chains"]))
            ]
            role_to_cluster = {
                chain["role"]: chain_cluster_ids[index] for index, chain in enumerate(row["chains"])
            }
            row["split"] = assignments[root]
            row["cluster_ids"].update(
                {
                    "target": role_to_cluster["target"],
                    "design": role_to_cluster["binder"],
                    "complex_component": component_id,
                }
            )
            answer.append(row)
    return answer


def split_sabdab2(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    answer: list[dict[str, Any]] = []
    for row in rows:
        cluster = row["cluster_ids"]["sabdab2_ab_ag"]
        if row["native_split"] == "test":
            row["split"] = "test"
            answer.append(row)
        else:
            train_groups[cluster].append(row)
    validation_target = round(sum(map(len, train_groups.values())) * 0.10)
    selected_validation: set[str] = set()
    current = 0
    ordered = sorted(
        train_groups.items(),
        key=lambda item: hashlib.sha256(f"sabdab2:{item[0]}".encode()).hexdigest(),
    )
    for cluster, group in ordered:
        if current < validation_target:
            selected_validation.add(cluster)
            current += len(group)
    for cluster, group in train_groups.items():
        split = "validation" if cluster in selected_validation else "train"
        for row in group:
            row["split"] = split
            answer.append(row)
    for row in answer:
        composite = f"sabdab2_ab_ag:{row['cluster_ids']['sabdab2_ab_ag']}"
        row["cluster_ids"].update(
            {"target": composite, "design": composite, "complex_component": composite}
        )
    return answer


def split_rna(
    binding_rows: list[dict[str, Any]],
    prior_rows: list[dict[str, Any]],
    output_root: Path,
    mmseqs: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    protein_sequences: dict[str, str] = {}
    rna_sequences: dict[str, str] = {}
    protein_ids: dict[tuple[str, int], str] = {}
    rna_ids: dict[tuple[str, int], str] = {}
    all_rows = binding_rows + prior_rows
    for row in all_rows:
        for index, chain in enumerate(row["chains"]):
            role = chain["role"]
            if role == "target":
                sequence_id = _sequence_id(row["sample_id"], "protein", index)
                protein_sequences[sequence_id] = chain["sequence"]
                protein_ids[(row["sample_id"], index)] = sequence_id
            else:
                sequence_id = _sequence_id(row["sample_id"], "rna", index)
                rna_sequences[sequence_id] = chain["sequence"]
                rna_ids[(row["sample_id"], index)] = sequence_id
    protein_clusters = cluster_sequences(
        "rna_target_protein_30pct",
        protein_sequences,
        output_root,
        mmseqs,
        min_identity=0.30,
        coverage=0.80,
    )
    rna_clusters = cluster_sequences(
        "rna_design_and_prior_80pct_nucleotide_v2",
        rna_sequences,
        output_root,
        mmseqs,
        min_identity=0.80,
        coverage=0.80,
        nucleotide=True,
    )
    union = UnionFind()
    sample_nodes: dict[str, list[str]] = {}
    for row in all_rows:
        nodes: list[str] = []
        for index, chain in enumerate(row["chains"]):
            if chain["role"] == "target":
                cluster = protein_clusters[protein_ids[(row["sample_id"], index)]]
                nodes.append(f"protein:{cluster}")
            else:
                cluster = rna_clusters[rna_ids[(row["sample_id"], index)]]
                nodes.append(f"rna:{cluster}")
        if row.get("pdb_id"):
            nodes.append(f"native:{row['pdb_id']}")
        sample_nodes[row["sample_id"]] = nodes
        for node in nodes:
            union.union(nodes[0], node)
    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        root = union.find(sample_nodes[row["sample_id"]][0])
        components[root].append(row)
    assignments = _balanced_assign(components, {"train": 0.8, "validation": 0.1, "test": 0.1})
    binding_output: list[dict[str, Any]] = []
    prior_output: list[dict[str, Any]] = []
    for root, component_rows in components.items():
        component_id = _component_id(root, "rna_homology_component")
        for row in component_rows:
            target_clusters = []
            design_clusters = []
            for index, chain in enumerate(row["chains"]):
                if chain["role"] == "target":
                    cluster = protein_clusters[protein_ids[(row["sample_id"], index)]]
                    target_clusters.append(f"mmseqs30:{cluster}")
                else:
                    cluster = rna_clusters[rna_ids[(row["sample_id"], index)]]
                    design_clusters.append(f"mmseqs80:{cluster}")
            row["split"] = assignments[root]
            row["cluster_ids"].update(
                {
                    "target": "+".join(sorted(target_clusters)) if target_clusters else None,
                    "design": "+".join(sorted(design_clusters)),
                    "complex_component": component_id,
                }
            )
            if row["purpose"] == "binding_design":
                binding_output.append(row)
            else:
                prior_output.append(row)
    return binding_output, prior_output


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    observed = Counter(row["split"] for row in rows)
    return {split: observed[split] for split in SPLIT_ORDER}


def _assert_disjoint(rows: list[dict[str, Any]], cluster_name: str) -> None:
    assignments: dict[str, str] = {}
    for row in rows:
        cluster = row["cluster_ids"].get(cluster_name)
        if cluster is None:
            continue
        previous = assignments.setdefault(cluster, row["split"])
        if previous != row["split"]:
            raise RuntimeError(f"{cluster_name}={cluster} crosses {previous}/{row['split']} splits")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--catalog-root", default="data/processed/v0/catalogs")
    parser.add_argument("--output-root", default="data/processed/v0")
    parser.add_argument("--mmseqs", default="data/tools/mmseqs18/mmseqs/bin/mmseqs")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    os.chdir(repo_root)
    catalog_root = Path(args.catalog_root)
    output_root = Path(args.output_root)
    mmseqs = Path(args.mmseqs).resolve()

    ppiref = split_ppiref(load_jsonl(catalog_root / "ppiref50k.jsonl"), output_root, mmseqs)
    sabdab = split_sabdab2(load_jsonl(catalog_root / "sabdab2.jsonl"))
    ribocentre = load_jsonl(catalog_root / "ribocentre.jsonl")
    pdb_rna = load_jsonl(catalog_root / "pdb_rna_target.jsonl")
    rnasolo = load_jsonl(catalog_root / "rnasolo2.jsonl")
    rna_binding, rna_prior = split_rna(ribocentre + pdb_rna, rnasolo, output_root, mmseqs)

    all_rows = ppiref + sabdab + rna_binding + rna_prior
    for cluster_name in ("target", "design", "complex_component"):
        _assert_disjoint(ppiref, cluster_name)
        _assert_disjoint(sabdab, cluster_name)
        _assert_disjoint(rna_binding + rna_prior, cluster_name)

    task_groups = {
        "protein_binder": ppiref,
        "antibody_h3": sabdab,
        "rna_aptamer_binding": rna_binding,
        "rna_structure_prior_auxiliary": rna_prior,
    }
    manifest_hashes: dict[str, dict[str, Any]] = {}
    for task, rows in task_groups.items():
        for split in SPLIT_ORDER:
            selected = [row for row in rows if row["split"] == split]
            count, digest = write_jsonl(output_root / "splits" / task / f"{split}.jsonl", selected)
            manifest_hashes[f"{task}/{split}"] = {"count": count, "sha256": digest}

    report = {
        "release": "nanodesign-v0-data-2026-08-30",
        "split_counts": {task: _counts(rows) for task, rows in task_groups.items()},
        "total_samples": len(all_rows),
        "manifest_files": manifest_hashes,
        "split_protocols": {
            "protein_binder": (
                "MMseqs2 18-8cc5c, 30% identity/80% bidirectional coverage; connected "
                "components across target and binder clusters; deterministic 80/10/10"
            ),
            "antibody_h3": (
                "SAbDab2 official antigen-aware 80/20 train/test; 10% of official train "
                "ab_ag clusters reserved for validation"
            ),
            "rna": (
                "MMseqs2 target-protein 30% and RNA 80% identity, both 80% coverage; "
                "connected components include native PDB, binding, and RNAsolo2 prior records; "
                "deterministic 80/10/10"
            ),
        },
        "cluster_disjoint": True,
    }
    report_path = output_root / "reports" / "final_split_counts.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["split_counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

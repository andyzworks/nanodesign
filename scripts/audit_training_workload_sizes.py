#!/usr/bin/env python3
"""Audit frozen v0 train-catalog sizes against the existing read-only feature cache."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

TASKS = {
    "protein_binder": ("protein_binder", "protein_binder"),
    "antibody_h3": ("antibody_h3", "antibody_cdr"),
    "rna": ("rna_binding", "rna_aptamer"),
}
QUANTILES = (0.0, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _quantiles(values: list[int]) -> dict[str, float | int] | None:
    """Return deterministic linearly interpolated quantiles (NumPy method='linear')."""

    if not values:
        return None
    ordered = sorted(values)
    result: dict[str, float | int] = {}
    for quantile in QUANTILES:
        position = quantile * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        value = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
        key = "min" if quantile == 0 else "max" if quantile == 1 else f"p{quantile * 100:g}"
        result[key] = int(value) if value.is_integer() else round(value, 2)
    return result


def _load_catalog(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"duplicate sample IDs in {path}")
    return rows


def _catalog_sample(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "source": row["source"],
        "resolved_residues": sum(int(chain["resolved_residues"]) for chain in row["chains"]),
    }


def _cached_atom_counts(
    database: Path,
    rows_by_id: dict[str, dict[str, Any]],
    manifest_sha256: str,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Read matching payload shapes without mutating or rebuilding the SQLite cache."""

    counters = {
        "finalized": False,
        "finalization_status": "database_missing",
        "database_rows_at_start": 0,
        "catalog_id_matches": 0,
        "stale_identity_rows": 0,
        "payload_rows_read": 0,
    }
    if not database.is_file():
        return {}, counters
    sidecar = database.with_suffix(database.suffix + ".sha256.json")
    try:
        finalization = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        counters["finalization_status"] = "missing_or_invalid_sha256_sidecar"
        return {}, counters
    if finalization.get("size_bytes") != database.stat().st_size:
        counters["finalization_status"] = "database_size_does_not_match_sidecar"
        return {}, counters
    if not isinstance(finalization.get("sha256"), str) or len(finalization["sha256"]) != 64:
        counters["finalization_status"] = "invalid_sha256_in_sidecar"
        return {}, counters
    counters.update(
        {
            "finalized": True,
            "finalization_status": "finalized_sidecar_size_match",
            "database_sha256": finalization["sha256"],
        }
    )
    connection = sqlite3.connect(
        f"file:{database.resolve()}?mode=ro", uri=True, timeout=30, isolation_level=None
    )
    connection.execute("PRAGMA query_only=ON")
    try:
        identities = connection.execute(
            "SELECT sample_id, identity_json FROM features ORDER BY sample_id"
        ).fetchall()
        counters["database_rows_at_start"] = len(identities)
        valid_ids: list[str] = []
        for sample_id, identity_json in identities:
            row = rows_by_id.get(sample_id)
            if row is None:
                continue
            counters["catalog_id_matches"] += 1
            identity = json.loads(identity_json)
            row_sha = hashlib.sha256(_canonical_json(row).encode()).hexdigest()
            if (
                identity.get("row_sha256") != row_sha
                or identity.get("manifest_sha256") != manifest_sha256
            ):
                counters["stale_identity_rows"] += 1
                continue
            valid_ids.append(sample_id)

        atom_counts: dict[str, int] = {}
        for index, sample_id in enumerate(valid_ids, 1):
            record = connection.execute(
                "SELECT identity_json, payload FROM features WHERE sample_id = ?", (sample_id,)
            ).fetchone()
            if record is None:
                continue
            identity_json, payload = record
            identity = json.loads(identity_json)
            row = rows_by_id[sample_id]
            row_sha = hashlib.sha256(_canonical_json(row).encode()).hexdigest()
            if (
                identity.get("row_sha256") != row_sha
                or identity.get("manifest_sha256") != manifest_sha256
            ):
                counters["stale_identity_rows"] += 1
                continue
            batch = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
            positions = batch.get("ground_truth_positions")
            if not isinstance(positions, torch.Tensor) or positions.ndim != 3:
                raise ValueError(f"{database}:{sample_id}: invalid cached coordinate tensor")
            atom_counts[sample_id] = int(positions.shape[1])
            counters["payload_rows_read"] += 1
            del batch, positions, payload
            if index % 500 == 0 or index == len(valid_ids):
                print(
                    f"read {index:,}/{len(valid_ids):,} valid payloads from {database}",
                    file=sys.stderr,
                    flush=True,
                )
        return atom_counts, counters
    finally:
        connection.close()


def _task_report(
    task: str,
    catalog: Path,
    database: Path,
    root: Path,
    manifest_sha256: str,
    threshold: int,
) -> dict[str, Any]:
    rows = _load_catalog(catalog)
    samples = [_catalog_sample(row) for row in rows]
    rows_by_id = {row["sample_id"]: row for row in rows}
    atom_counts, cache_snapshot = _cached_atom_counts(database, rows_by_id, manifest_sha256)
    for sample in samples:
        atom_count = atom_counts.get(sample["sample_id"])
        sample["atom_count"] = atom_count
        sample["expected_route"] = (
            "unknown" if atom_count is None else "chunked" if atom_count > threshold else "standard"
        )
    cached = [sample for sample in samples if sample["atom_count"] is not None]
    chunked = [sample for sample in cached if sample["atom_count"] > threshold]
    standard = [sample for sample in cached if sample["atom_count"] <= threshold]
    unknown_count = len(samples) - len(cached)
    largest_residue = sorted(
        samples, key=lambda item: (-item["resolved_residues"], item["sample_id"])
    )[:10]
    largest_atom = sorted(cached, key=lambda item: (-item["atom_count"], item["sample_id"]))[:10]
    report: dict[str, Any] = {
        "catalog": str(catalog.relative_to(root)),
        "catalog_sha256": _sha256(catalog),
        "sample_count": len(samples),
        "resolved_residue_quantiles": _quantiles(
            [sample["resolved_residues"] for sample in samples]
        ),
        "cache": {
            "database": str(
                database.relative_to(root) if database.is_relative_to(root) else database
            ),
            **cache_snapshot,
            "coverage_count": len(cached),
            "coverage_fraction": round(len(cached) / len(samples), 6),
        },
        "cached_atom_count_quantiles": _quantiles([sample["atom_count"] for sample in cached]),
        "routing": {
            "threshold_rule": f"standard <= {threshold}; chunked > {threshold}",
            "standard_count": len(standard),
            "chunked_count": len(chunked),
            "unknown_uncached_count": unknown_count,
            "chunked_fraction_of_cached": round(len(chunked) / len(cached), 6) if cached else None,
            "chunked_fraction_of_catalog_lower_bound": round(len(chunked) / len(samples), 6),
            "chunked_fraction_of_catalog_upper_bound": round(
                (len(chunked) + unknown_count) / len(samples), 6
            ),
        },
        "largest_by_resolved_residues": largest_residue,
        "largest_by_cached_atom_count": largest_atom,
    }
    if task == "rna":
        report["rna_cached_samples_above_threshold"] = sorted(
            chunked, key=lambda item: (-item["atom_count"], item["sample_id"])
        )
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# NanoDesign v0 frozen training workload sizes",
        "",
        f"Generated at `{report['generated_at_utc']}` from the frozen train catalogs.",
        (
            "The audit opens feature-cache databases in SQLite read-only/query-only mode; it does "
            "not build cache entries or alter data, filters, splits, model, loss, or sampling. A "
            "database is opened only after its finalized SHA-256 sidecar exists and its recorded "
            "size matches; partial databases are never opened."
        ),
        "",
        (
            f"Routing uses the frozen rule: **standard <= {report['atom_threshold']}; chunked > "
            f"{report['atom_threshold']} model atoms**. Atom statistics cover only valid entries "
            "already present in a finalized cache. Uncached rows and rows in a non-finalized "
            "database remain `unknown`; lower/upper bounds therefore do not pretend that partial "
            "cache coverage is a full-catalog measurement."
        ),
        "",
        "Quantiles use linear interpolation (the NumPy `method=linear` convention).",
        "",
        f"| Task | Train samples | Resolved residues p50 / p95 / max | Cache coverage | Cached atoms p50 / p95 / max | Standard / chunked / unknown | >{report['atom_threshold']} among cached |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, item in report["tasks"].items():
        residues = item["resolved_residue_quantiles"]
        atoms = item["cached_atom_count_quantiles"]
        cache = item["cache"]
        routing = item["routing"]
        atom_text = "n/a" if atoms is None else f"{atoms['p50']} / {atoms['p95']} / {atoms['max']}"
        fraction = routing["chunked_fraction_of_cached"]
        fraction_text = "n/a" if fraction is None else f"{fraction:.2%}"
        lines.append(
            f"| {task} | {item['sample_count']:,} | {residues['p50']} / {residues['p95']} / "
            f"{residues['max']} | {cache['coverage_count']:,}/{item['sample_count']:,} "
            f"({cache['coverage_fraction']:.2%}) | {atom_text} | {routing['standard_count']:,} / "
            f"{routing['chunked_count']:,} / {routing['unknown_uncached_count']:,} | "
            f"{fraction_text} |"
        )
    lines.extend(["", "## Largest samples by catalog resolved residues", ""])
    for task, item in report["tasks"].items():
        lines.extend(
            [
                f"### {task}",
                "",
                "| Sample | Source | Resolved residues | Cached model atoms | Route |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for sample in item["largest_by_resolved_residues"]:
            atom_count = "unknown" if sample["atom_count"] is None else f"{sample['atom_count']:,}"
            lines.append(
                f"| `{sample['sample_id']}` | {sample['source']} | "
                f"{sample['resolved_residues']:,} | {atom_count} | {sample['expected_route']} |"
            )
        lines.append("")
    lines.extend(["## Largest cached samples by model atom count", ""])
    for task, item in report["tasks"].items():
        lines.extend(
            [
                f"### {task}",
                "",
                "| Sample | Source | Resolved residues | Model atoms | Route |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for sample in item["largest_by_cached_atom_count"]:
            lines.append(
                f"| `{sample['sample_id']}` | {sample['source']} | "
                f"{sample['resolved_residues']:,} | {sample['atom_count']:,} | "
                f"{sample['expected_route']} |"
            )
        lines.append("")
    rna_large = report["tasks"]["rna"]["rna_cached_samples_above_threshold"]
    lines.extend(
        [
            "## RNA samples routed to chunked mode in the cache snapshot",
            "",
            "All cached RNA train samples above the frozen threshold are listed here.",
            "",
            "| Sample | Source | Resolved residues | Model atoms |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for sample in rna_large:
        lines.append(
            f"| `{sample['sample_id']}` | {sample['source']} | "
            f"{sample['resolved_residues']:,} | {sample['atom_count']:,} |"
        )
    if not rna_large:
        lines.append("| _none in current cache snapshot_ | — | — | — |")
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "PYTHONPATH=src data/envs/rfd3na312/bin/python scripts/audit_training_workload_sizes.py",
            "```",
            "",
            (
                "The JSON file is the machine-readable source of truth and includes catalog SHA-256 "
                "digests, cache snapshot counters, bounds for the full-catalog chunked fraction, "
                "and the ten largest samples by resolved residues and cached model atoms."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/v0"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/data_v0_stats.json"))
    parser.add_argument("--atom-threshold", type=int, default=8008)
    parser.add_argument(
        "--json-output", type=Path, default=Path("docs/training_workload_sizes.json")
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=Path("docs/training_workload_sizes.md")
    )
    args = parser.parse_args()
    if args.atom_threshold < 1:
        raise ValueError("atom threshold must be positive")
    root = args.root.resolve()
    cache_root = args.cache_root if args.cache_root.is_absolute() else root / args.cache_root
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    manifest_sha = _sha256(manifest)
    tasks = {}
    for task, (catalog_name, cache_task) in TASKS.items():
        catalog = root / "data/processed/v0/splits" / catalog_name / "train.jsonl"
        database = cache_root / cache_task / "train.sqlite3"
        tasks[task] = _task_report(
            task,
            catalog,
            database,
            root,
            manifest_sha,
            args.atom_threshold,
        )
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "atom_threshold": args.atom_threshold,
        "manifest": str(manifest.relative_to(root)),
        "manifest_sha256": manifest_sha,
        "tasks": tasks,
    }
    json_output = args.json_output if args.json_output.is_absolute() else root / args.json_output
    markdown_output = (
        args.markdown_output if args.markdown_output.is_absolute() else root / args.markdown_output
    )
    json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_output.write_text(_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()

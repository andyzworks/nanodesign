"""Auditable RNA-target complex counts required before freezing the v0 RNA pool."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from nanodesign.v0.constants import DataSource
from nanodesign.v0.spec import SPEC_VERSION


class InventoryError(ValueError):
    pass


RNA_BINDING_SOURCES = frozenset(
    {
        DataSource.RIBOCENTRE_APTAMER,
        DataSource.PDB_RNA_TARGET_COMPLEX,
    }
)


@dataclass(frozen=True)
class RnaComplexInventoryRecord:
    source: DataSource
    source_version: str
    candidate_complexes: int
    usable_complexes: int
    exclusive_rejection_counts: Mapping[str, int]
    selection_protocol: str

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RnaComplexInventoryRecord:
        try:
            raw_rejections = value["exclusive_rejection_counts"]
            if not isinstance(raw_rejections, Mapping):
                raise TypeError("exclusive_rejection_counts must be a mapping")
            return cls(
                source=DataSource(str(value["source"])),
                source_version=str(value["source_version"]),
                candidate_complexes=int(value["candidate_complexes"]),
                usable_complexes=int(value["usable_complexes"]),
                exclusive_rejection_counts={
                    str(reason): int(count) for reason, count in raw_rejections.items()
                },
                selection_protocol=str(value["selection_protocol"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise InventoryError(f"invalid RNA inventory record: {value}") from error

    def validate(self) -> RnaComplexInventoryRecord:
        if self.source not in RNA_BINDING_SOURCES:
            raise InventoryError(f"{self.source.value} is not a NanoDesign v0 RNA binding source")
        if not self.source_version.strip() or not self.selection_protocol.strip():
            raise InventoryError("source_version and selection_protocol must be non-empty")
        if self.candidate_complexes < 0 or self.usable_complexes < 0:
            raise InventoryError("RNA complex counts must be non-negative")
        if not self.exclusive_rejection_counts:
            raise InventoryError("exclusive rejection counts must be recorded")
        if any(not reason.strip() for reason in self.exclusive_rejection_counts):
            raise InventoryError("RNA rejection reasons must be non-empty")
        if any(count < 0 for count in self.exclusive_rejection_counts.values()):
            raise InventoryError("RNA rejection counts must be non-negative")
        accounted = self.usable_complexes + sum(self.exclusive_rejection_counts.values())
        if accounted != self.candidate_complexes:
            raise InventoryError(
                f"{self.source.value}: usable + exclusive rejections ({accounted}) "
                f"does not equal candidates ({self.candidate_complexes})"
            )
        return self

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["source"] = self.source.value
        value["exclusive_rejection_counts"] = dict(sorted(self.exclusive_rejection_counts.items()))
        return value


def _canonical_payload(records: list[RnaComplexInventoryRecord]) -> dict[str, object]:
    return {
        "schema_version": SPEC_VERSION,
        "records": [
            record.to_dict() for record in sorted(records, key=lambda row: row.source.value)
        ],
    }


def audit_rna_complex_inventory(
    records: list[RnaComplexInventoryRecord],
) -> dict[str, object]:
    rows = [record.validate() for record in records]
    observed = [record.source for record in rows]
    if len(observed) != len(set(observed)):
        raise InventoryError("RNA inventory contains a duplicate source")
    missing = RNA_BINDING_SOURCES - set(observed)
    extra = set(observed) - RNA_BINDING_SOURCES
    if missing or extra:
        raise InventoryError(
            "RNA inventory must contain exactly Ribocentre Aptamer and PDB RNA-target "
            f"complexes; missing={sorted(source.value for source in missing)}, "
            f"extra={sorted(source.value for source in extra)}"
        )
    payload = _canonical_payload(rows)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": SPEC_VERSION,
        "sources": {
            record.source.value: {
                "source_version": record.source_version,
                "candidate_complexes": record.candidate_complexes,
                "usable_complexes": record.usable_complexes,
            }
            for record in sorted(rows, key=lambda row: row.source.value)
        },
        "candidate_complexes": sum(record.candidate_complexes for record in rows),
        "usable_complexes": sum(record.usable_complexes for record in rows),
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def load_rna_complex_inventory(path: str | Path) -> list[RnaComplexInventoryRecord]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as error:
        raise InventoryError(f"invalid JSON inventory: {path}") from error
    if not isinstance(value, dict) or value.get("schema_version") != SPEC_VERSION:
        raise InventoryError(f"RNA inventory schema_version must be {SPEC_VERSION!r}")
    raw_records = value.get("records")
    if not isinstance(raw_records, list):
        raise InventoryError("RNA inventory records must be a list")
    records: list[RnaComplexInventoryRecord] = []
    for record in raw_records:
        if not isinstance(record, Mapping):
            raise InventoryError(f"invalid RNA inventory record: {record}")
        records.append(RnaComplexInventoryRecord.from_dict(record))
    audit_rna_complex_inventory(records)
    return records

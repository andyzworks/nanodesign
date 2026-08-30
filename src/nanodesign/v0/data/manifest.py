"""Leakage-aware manifest contract for all v0 datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from nanodesign.v0.constants import DataSource, ExamplePurpose, Split, Task
from nanodesign.v0.spec import SPEC_VERSION, TASK_SPECS


class ManifestError(ValueError):
    pass


def _parse_task(value: object) -> Task:
    if isinstance(value, str) and not value.isdigit():
        return Task[value.upper()]
    return Task(int(value))


@dataclass(frozen=True)
class DatasetRecord:
    sample_id: str
    task: Task
    source: DataSource
    source_version: str
    purpose: ExamplePurpose
    split: Split
    path: str
    design_cluster_id: str
    complex_cluster_id: str | None = None
    target_cluster_id: str | None = None
    structure_cluster_id: str | None = None
    schema_version: str = SPEC_VERSION

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> DatasetRecord:
        try:
            payload = dict(value)
            payload["task"] = _parse_task(payload["task"])
            payload["source"] = DataSource(str(payload["source"]))
            payload["purpose"] = ExamplePurpose(str(payload["purpose"]))
            payload["split"] = Split(str(payload["split"]))
            return cls(**payload)  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError) as error:
            raise ManifestError(f"invalid manifest record: {value}") from error

    def validate(self) -> DatasetRecord:
        if self.schema_version != SPEC_VERSION:
            raise ManifestError(f"{self.sample_id}: invalid schema version")
        required_text = {
            "sample_id": self.sample_id,
            "source_version": self.source_version,
            "path": self.path,
            "design_cluster_id": self.design_cluster_id,
        }
        missing = [name for name, value in required_text.items() if not value.strip()]
        if missing:
            raise ManifestError(f"{self.sample_id}: empty required fields {missing}")
        path = Path(self.path)
        if path.is_absolute() or ".." in path.parts:
            raise ManifestError(f"{self.sample_id}: path must remain inside the dataset")
        optional_clusters = {
            "complex_cluster_id": self.complex_cluster_id,
            "target_cluster_id": self.target_cluster_id,
            "structure_cluster_id": self.structure_cluster_id,
        }
        empty_optional = [
            name
            for name, value in optional_clusters.items()
            if value is not None and not value.strip()
        ]
        if empty_optional:
            raise ManifestError(
                f"{self.sample_id}: optional cluster ids cannot be empty strings: {empty_optional}"
            )

        spec = TASK_SPECS[self.task]
        if self.purpose == ExamplePurpose.BINDING_DESIGN:
            if self.source not in spec.binding_sources:
                raise ManifestError(f"{self.sample_id}: invalid binding source {self.source.value}")
            missing_binding_clusters = [
                name
                for name in ("complex_cluster_id", "target_cluster_id")
                if getattr(self, name) is None
            ]
            if missing_binding_clusters:
                raise ManifestError(
                    f"{self.sample_id}: binding examples require {missing_binding_clusters}"
                )
        else:
            if self.task != Task.RNA_APTAMER or self.source != DataSource.RNASOLO2:
                raise ManifestError("only RNAsolo2 may supply RNA structure-prior examples")
            if self.complex_cluster_id is not None or self.target_cluster_id is not None:
                raise ManifestError(
                    f"{self.sample_id}: RNAsolo2 prior examples cannot claim "
                    "target/complex clusters"
                )
            if self.structure_cluster_id is None:
                raise ManifestError(
                    f"{self.sample_id}: RNAsolo2 prior examples require structure_cluster_id"
                )
        return self

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value.update(
            {
                "task": self.task.name.lower(),
                "source": self.source.value,
                "purpose": self.purpose.value,
                "split": self.split.value,
            }
        )
        return value


def load_manifest(path: str | Path) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ManifestError("record is not an object")
                records.append(DatasetRecord.from_dict(value).validate())
            except (json.JSONDecodeError, ManifestError) as error:
                raise ManifestError(f"{path}:{line_number}: {error}") from error
    if not records:
        raise ManifestError(f"empty manifest: {path}")
    return records


def _assert_cluster_disjoint(records: list[DatasetRecord], field: str) -> None:
    assignments: dict[str, Split] = {}
    for record in records:
        cluster = getattr(record, field)
        if cluster is None:
            continue
        previous = assignments.setdefault(cluster, record.split)
        if previous != record.split:
            raise ManifestError(
                f"{field}={cluster!r} crosses {previous.value}/{record.split.value} splits"
            )


def manifest_sha256(records: Iterable[DatasetRecord]) -> str:
    rows = sorted(
        (record.validate().to_dict() for record in records),
        key=lambda row: str(row["sample_id"]),
    )
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def audit_manifest(records: Iterable[DatasetRecord]) -> dict[str, object]:
    rows = [record.validate() for record in records]
    if not rows:
        raise ManifestError("cannot audit an empty manifest")
    sample_ids = [record.sample_id for record in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ManifestError("manifest contains duplicate sample_id values")
    for field in (
        "complex_cluster_id",
        "target_cluster_id",
        "design_cluster_id",
        "structure_cluster_id",
    ):
        _assert_cluster_disjoint(rows, field)
    task_counts = {task.name.lower(): sum(record.task == task for record in rows) for task in Task}
    split_counts = {split.value: sum(record.split == split for record in rows) for split in Split}
    return {
        "schema_version": SPEC_VERSION,
        "sha256": manifest_sha256(rows),
        "num_records": len(rows),
        "task_counts": task_counts,
        "split_counts": split_counts,
        "binding_records": sum(record.purpose == ExamplePurpose.BINDING_DESIGN for record in rows),
        "rna_prior_records": sum(
            record.purpose == ExamplePurpose.RNA_STRUCTURE_PRIOR for record in rows
        ),
        "cluster_disjoint": True,
    }

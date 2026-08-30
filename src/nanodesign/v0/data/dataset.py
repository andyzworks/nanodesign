"""NPZ serialization and a manifest-backed dataset for the unified v0 contract."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from nanodesign.v0.constants import DataSource, ExamplePurpose, Task
from nanodesign.v0.contracts import DesignExample, collate_examples
from nanodesign.v0.data.manifest import DatasetRecord
from nanodesign.v0.spec import SPEC_VERSION

ARRAY_FIELDS = (
    "token_ids",
    "polymer_type",
    "role_id",
    "chain_id",
    "residue_index",
    "design_mask",
    "atom_positions",
    "atom_mask",
    "atom_token_index",
    "atom_element",
)


def save_design_example(path: str | Path, example: DesignExample) -> None:
    """Write one validated example atomically in the interchange format."""

    example.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {name: np.asarray(getattr(example, name)) for name in ARRAY_FIELDS}
    payload.update(
        {
            "schema_version": np.asarray(example.schema_version),
            "sample_id": np.asarray(example.sample_id),
            "task": np.asarray(int(example.task), dtype=np.int64),
            "source": np.asarray(example.source.value),
            "purpose": np.asarray(example.purpose.value),
        }
    )
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, destination)


def _scalar_text(archive: np.lib.npyio.NpzFile, name: str) -> str:
    value = archive[name]
    if value.shape != ():
        raise ValueError(f"serialized field {name!r} must be scalar")
    return str(value.item())


def load_design_example(path: str | Path) -> DesignExample:
    """Load and validate one example without allowing pickled objects."""

    with np.load(path, allow_pickle=False) as archive:
        required = set(ARRAY_FIELDS) | {
            "schema_version",
            "sample_id",
            "task",
            "source",
            "purpose",
        }
        missing = required - set(archive.files)
        unknown = set(archive.files) - required
        if missing or unknown:
            raise ValueError(
                f"invalid serialized example fields: missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        example = DesignExample(
            sample_id=_scalar_text(archive, "sample_id"),
            task=Task(int(archive["task"].item())),
            source=DataSource(_scalar_text(archive, "source")),
            purpose=ExamplePurpose(_scalar_text(archive, "purpose")),
            schema_version=_scalar_text(archive, "schema_version"),
            **{name: np.array(archive[name], copy=True) for name in ARRAY_FIELDS},
        )
    if example.schema_version != SPEC_VERSION:
        raise ValueError(f"serialized schema {example.schema_version!r} must be {SPEC_VERSION!r}")
    return example.validate()


class ManifestDataset(Dataset[DesignExample]):
    """Load contract-checked examples from leakage-audited manifest records."""

    def __init__(self, dataset_root: str | Path, records: Sequence[DatasetRecord]):
        self.dataset_root = Path(dataset_root).resolve()
        self.records = tuple(record.validate() for record in records)
        if not self.records:
            raise ValueError("ManifestDataset requires at least one record")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> DesignExample:
        record = self.records[index]
        path = (self.dataset_root / record.path).resolve()
        if self.dataset_root not in path.parents:
            raise ValueError(f"manifest path escapes dataset root: {record.path}")
        example = load_design_example(path)
        observed = (example.sample_id, example.task, example.source, example.purpose)
        expected = (record.sample_id, record.task, record.source, record.purpose)
        if observed != expected:
            raise ValueError(
                f"manifest/example identity mismatch for {record.sample_id}: "
                f"expected={expected}, observed={observed}"
            )
        return example

    @staticmethod
    def collate_fn(examples: Sequence[DesignExample]):
        return collate_examples(examples)

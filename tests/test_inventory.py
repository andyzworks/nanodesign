import pytest

from nanodesign.v0.constants import DataSource
from nanodesign.v0.data.inventory import (
    InventoryError,
    RnaComplexInventoryRecord,
    audit_rna_complex_inventory,
)


def _record(source: DataSource, candidates: int, usable: int):
    return RnaComplexInventoryRecord(
        source=source,
        source_version="frozen-version",
        candidate_complexes=candidates,
        usable_complexes=usable,
        exclusive_rejection_counts={"quality_filter": candidates - usable},
        selection_protocol="frozen protocol",
    )


def test_rna_inventory_covers_both_binding_sources_and_counts_usable_complexes():
    result = audit_rna_complex_inventory(
        [
            _record(DataSource.RIBOCENTRE_APTAMER, 10, 7),
            _record(DataSource.PDB_RNA_TARGET_COMPLEX, 20, 12),
        ]
    )
    assert result["candidate_complexes"] == 30
    assert result["usable_complexes"] == 19
    assert len(result["sha256"]) == 64


def test_rna_inventory_rejects_missing_source_or_unaccounted_candidates():
    with pytest.raises(InventoryError, match="exactly Ribocentre"):
        audit_rna_complex_inventory(
            [_record(DataSource.RIBOCENTRE_APTAMER, 10, 7)]
        )
    bad = RnaComplexInventoryRecord(
        source=DataSource.PDB_RNA_TARGET_COMPLEX,
        source_version="frozen-version",
        candidate_complexes=20,
        usable_complexes=12,
        exclusive_rejection_counts={"quality_filter": 3},
        selection_protocol="frozen protocol",
    )
    with pytest.raises(InventoryError, match="does not equal candidates"):
        bad.validate()


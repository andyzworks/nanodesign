import pytest

from nanodesign.v0.constants import DataSource, ExamplePurpose, Split, Task
from nanodesign.v0.data.manifest import DatasetRecord, ManifestError, audit_manifest


def _record(sample_id: str, split: Split, cluster: str) -> DatasetRecord:
    return DatasetRecord(
        sample_id=sample_id,
        task=Task.PROTEIN_BINDER,
        source=DataSource.PPIREF50K,
        source_version="frozen-version",
        purpose=ExamplePurpose.BINDING_DESIGN,
        split=split,
        path=f"samples/{sample_id}.npz",
        design_cluster_id=f"design-{cluster}",
        complex_cluster_id=cluster,
        target_cluster_id=f"target-{cluster}",
    )


def test_manifest_requires_cluster_disjoint_splits():
    result = audit_manifest(
        [_record("train-a", Split.TRAIN, "a"), _record("test-b", Split.TEST, "b")]
    )
    assert result["cluster_disjoint"]
    assert len(result["sha256"]) == 64
    with pytest.raises(ManifestError, match="crosses train/test"):
        audit_manifest(
            [
                _record("train-a", Split.TRAIN, "same"),
                _record("test-a", Split.TEST, "same"),
            ]
        )


def test_rnasolo_prior_uses_structure_cluster_without_fake_target_cluster():
    record = DatasetRecord(
        sample_id="rna-prior",
        task=Task.RNA_APTAMER,
        source=DataSource.RNASOLO2,
        source_version="frozen-version",
        purpose=ExamplePurpose.RNA_STRUCTURE_PRIOR,
        split=Split.TRAIN,
        path="samples/rna-prior.npz",
        design_cluster_id="rna-sequence-1",
        structure_cluster_id="rna-structure-1",
    )
    assert record.validate() is record
    invalid = DatasetRecord(**{**record.__dict__, "target_cluster_id": "invented-target"})
    with pytest.raises(ManifestError, match="cannot claim"):
        invalid.validate()

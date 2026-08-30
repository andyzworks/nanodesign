from nanodesign.v0.constants import Split
from nanodesign.v0.data.dataset import (
    ManifestDataset,
    load_design_example,
    save_design_example,
)
from nanodesign.v0.data.manifest import DatasetRecord
from nanodesign.v0.testing import synthetic_binding_examples


def test_example_round_trip_and_manifest_dataset(tmp_path):
    example = synthetic_binding_examples()[0]
    relative_path = "samples/example.npz"
    save_design_example(tmp_path / relative_path, example)
    restored = load_design_example(tmp_path / relative_path)
    assert restored.sample_id == example.sample_id
    assert restored.atom_positions.shape == example.atom_positions.shape

    record = DatasetRecord(
        sample_id=example.sample_id,
        task=example.task,
        source=example.source,
        source_version="frozen-version",
        purpose=example.purpose,
        split=Split.TRAIN,
        path=relative_path,
        design_cluster_id="design-1",
        complex_cluster_id="complex-1",
        target_cluster_id="target-1",
    )
    dataset = ManifestDataset(tmp_path, [record])
    assert dataset[0].sample_id == example.sample_id


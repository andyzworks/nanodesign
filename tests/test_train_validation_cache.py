import random

import torch

from nanodesign.v0.data.cache import FeatureCacheSpec, SQLiteFeatureCache
from nanodesign.v0.distributed import DistributedContext
from scripts import train_v0


def _row(index: int) -> dict:
    return {
        "sample_id": f"validation-{index}",
        "task": "antibody_cdr",
        "split": "validation",
        "source": "sabdab2",
        "source_version": "SAbDab2 frozen validation",
        "raw_paths": ["must-not-be-read.cif"],
        "chains": [],
    }


def _batch(sample_id: str) -> dict:
    atom_count = 28
    return {
        "sample_id": sample_id,
        "task": "antibody_cdr",
        "f": {
            "restype": torch.zeros(2, 32, dtype=torch.long),
            "atom_to_token_map": torch.arange(atom_count, dtype=torch.int32) // 14,
        },
        "X_noisy_L": torch.zeros(1, atom_count, 3),
        "t": torch.tensor([0.5]),
        "ground_truth_positions": torch.zeros(1, atom_count, 3),
        "ground_truth_atom_mask": torch.ones(atom_count, dtype=torch.bool),
        "ground_truth_sequence": torch.zeros(2, 32),
        "ground_truth_sequence_mask": torch.tensor([False, True]),
        "coord_atom_lvl_to_be_noised": torch.zeros(1, atom_count, 3),
        "output_metadata": {
            "atom_names": ["CA"] * atom_count,
            "atom_to_token": [0] * 14 + [1] * 14,
            "atom_output_mask": [True] * atom_count,
            "token_chain_names": ["H", "H"],
            "token_residue_keys": [(1, ""), (105, "")],
        },
    }


def test_validation_prefers_cache_without_changing_samples_or_fixed_noise(tmp_path, monkeypatch):
    manifest_sha = "d" * 64
    rows = [_row(index) for index in range(3)]
    spec = FeatureCacheSpec(manifest_sha256=manifest_sha, max_context_tokens=384)
    with SQLiteFeatureCache(tmp_path, readonly=False) as cache:
        for row in rows:
            cache.put(row, spec, _batch(row["sample_id"]))

    def raw_loader_must_not_run(*_args, **_kwargs):
        raise AssertionError("validation reparsed a structure despite an exact cache row")

    observed: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    def fake_evaluate(_model, batch):
        observed.append((batch["sample_id"], batch["t"].clone(), batch["X_noisy_L"].clone()))
        return {"loss": 3.0, "coordinate_loss": 2.0, "sequence_loss": 1.0}

    monkeypatch.setattr(train_v0, "_batch", raw_loader_must_not_run)
    monkeypatch.setattr(train_v0, "evaluate_loss", fake_evaluate)
    arguments = {
        "model": object(),
        "root": tmp_path,
        "rows": {"antibody_h3": rows},
        "device": torch.device("cpu"),
        "max_context_tokens": 384,
        "samples_per_task": 2,
        "seed": 7,
        "distributed": DistributedContext(rank=0, world_size=1),
        "feature_cache_root": tmp_path,
        "feature_cache_fallback": False,
        "manifest_sha256": manifest_sha,
    }
    first = train_v0._validation(**arguments)
    first_observed = observed.copy()
    observed.clear()
    second = train_v0._validation(**arguments)

    expected_ids = [row["sample_id"] for row in random.Random(7 + 1000).sample(rows, 2)]
    assert [item[0] for item in first_observed] == expected_ids
    assert [item[0] for item in observed] == expected_ids
    assert all(torch.equal(item[1], torch.tensor([0.5])) for item in first_observed + observed)
    for first_item, second_item in zip(first_observed, observed, strict=True):
        assert torch.equal(first_item[2], second_item[2])
    assert (
        first
        == second
        == {"antibody_h3": {"loss": 3.0, "coordinate_loss": 2.0, "sequence_loss": 1.0}}
    )

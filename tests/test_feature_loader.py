import torch

import nanodesign.v0.data.loader as loader_module
from nanodesign.v0.data.cache import (
    FeatureCacheSpec,
    SQLiteFeatureCache,
    finalize_cache_database,
)
from nanodesign.v0.data.loader import (
    CachedFeatureDataset,
    build_async_feature_loader,
    recursive_to_device,
    stage_catalog_cache,
)


def _row() -> dict:
    return {
        "sample_id": "sabdab2:loader_H_L",
        "task": "antibody_cdr",
        "split": "train",
        "source": "sabdab2",
        "source_version": "SAbDab2 loader test version",
        "raw_paths": ["unused.cif"],
        "chains": [],
    }


def _spec() -> FeatureCacheSpec:
    return FeatureCacheSpec(
        manifest_sha256="d" * 64,
        max_context_tokens=384,
        diffusion_batch_size=1,
        noise_level=None,
        random_seed=None,
    )


def _batch() -> dict:
    token_count, atom_count = 2, 28
    return {
        "sample_id": _row()["sample_id"],
        "task": "antibody_cdr",
        "f": {
            "restype": torch.zeros(token_count, 32, dtype=torch.long),
            "atom_to_token_map": torch.arange(atom_count, dtype=torch.int32) // 14,
        },
        "X_noisy_L": torch.zeros(1, atom_count, 3),
        "t": torch.tensor([0.5]),
        "ground_truth_positions": torch.zeros(1, atom_count, 3),
        "ground_truth_atom_mask": torch.ones(atom_count, dtype=torch.bool),
        "ground_truth_sequence": torch.zeros(token_count, 32),
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


def test_two_persistent_workers_prefetch_cached_rows_with_fresh_noise(tmp_path):
    shared, staged = tmp_path / "shared", tmp_path / "node-local"
    row, spec = _row(), _spec()
    with SQLiteFeatureCache(shared, readonly=False) as cache:
        cache.put(row, spec, _batch())
    finalize_cache_database(shared, row["task"], row["split"])
    assert stage_catalog_cache(shared, staged, [row])
    dataset = CachedFeatureDataset(".", [row] * 6, spec, cache_root=staged, allow_fallback=False)
    loader = build_async_feature_loader(
        dataset,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=False,
        multiprocessing_context="spawn",
    )
    batches = list(loader)
    assert len(batches) == 6
    assert all(batch["sample_id"] == row["sample_id"] for batch in batches)
    assert len({tuple(batch["X_noisy_L"].flatten().tolist()) for batch in batches}) == 6
    deterministic = [
        {key: value for key, value in batch.items() if key not in {"t", "X_noisy_L"}}
        for batch in batches
    ]
    assert all(
        torch.equal(deterministic[0]["f"]["restype"], item["f"]["restype"])
        for item in deterministic[1:]
    )


def test_cache_miss_falls_back_and_recursive_transfer_preserves_metadata(tmp_path, monkeypatch):
    row, spec, fallback = _row(), _spec(), _batch()
    monkeypatch.setattr(loader_module, "preprocess_feature_batch", lambda *_: fallback)
    dataset = CachedFeatureDataset(".", [row], spec, cache_root=tmp_path, allow_fallback=True)
    loader = build_async_feature_loader(dataset, num_workers=0, pin_memory=False)
    batch = next(iter(loader))
    transferred = recursive_to_device(batch, "cpu", non_blocking=True)
    assert transferred["sample_id"] == row["sample_id"]
    assert transferred["output_metadata"] == fallback["output_metadata"]
    assert torch.equal(transferred["ground_truth_positions"], fallback["ground_truth_positions"])


def test_caller_supplied_sampling_seeds_make_prefetch_order_reproducible(tmp_path):
    row, spec = _row(), _spec()
    with SQLiteFeatureCache(tmp_path, readonly=False) as cache:
        cache.put(row, spec, _batch())
    rows, seeds = [row] * 3, [101, 102, 103]
    dataset = CachedFeatureDataset(
        ".", rows, spec, cache_root=tmp_path, allow_fallback=False, sampling_seeds=seeds
    )
    first = [dataset[index]["X_noisy_L"] for index in range(3)]
    second = [dataset[index]["X_noisy_L"] for index in range(3)]
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert len({tuple(value.flatten().tolist()) for value in first}) == 3

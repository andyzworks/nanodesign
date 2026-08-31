import io
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

import nanodesign.v0.data.cache as cache_module
from nanodesign.v0.data.cache import (
    FeatureCacheError,
    FeatureCacheSpec,
    SQLiteFeatureCache,
    cache_database_path,
    finalize_cache_database,
    load_cached_or_preprocess,
    model_ready_batches_equal,
    preprocess_feature_batch,
    stage_cache_database,
    verify_finalized_database,
)


def _row() -> dict:
    return {
        "sample_id": "sabdab2:fixed_H_L",
        "task": "antibody_cdr",
        "split": "test",
        "source": "sabdab2",
        "source_version": "SAbDab2 fixed test version",
        "raw_paths": ["unused.cif"],
        "chains": [],
    }


def _batch(sample_id: str = "sabdab2:fixed_H_L") -> dict:
    token_count, atom_count = 2, 28
    return {
        "sample_id": sample_id,
        "task": "antibody_cdr",
        "f": {
            "restype": torch.zeros(token_count, 32, dtype=torch.long),
            "atom_to_token_map": torch.arange(atom_count, dtype=torch.int32) // 14,
        },
        "X_noisy_L": torch.arange(atom_count * 3, dtype=torch.float32).reshape(1, atom_count, 3),
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


def _spec(**changes) -> FeatureCacheSpec:
    values = {
        "manifest_sha256": "a" * 64,
        "max_context_tokens": 384,
        "diffusion_batch_size": 1,
        "noise_level": 0.5,
        "random_seed": 7,
    }
    values.update(changes)
    return FeatureCacheSpec(**values)


def test_sqlite_cache_roundtrip_binds_identity_and_exact_tensors(tmp_path):
    row, batch, spec = _row(), _batch(), _spec()
    with SQLiteFeatureCache(tmp_path, readonly=False) as cache:
        database = cache.put(row, spec, batch)
    assert database == cache_database_path(tmp_path, "antibody_cdr", "test")
    connection = sqlite3.connect(database)
    payload = connection.execute("SELECT payload FROM features").fetchone()[0]
    connection.close()
    template = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    assert "t" not in template and "X_noisy_L" not in template
    with SQLiteFeatureCache(tmp_path, readonly=True, lru_size=2) as cache:
        cached = cache.get(row, spec)
        assert model_ready_batches_equal(cached, cache.get(row, spec))
        with pytest.raises(FeatureCacheError, match="stale"):
            cache.get(row, _spec(manifest_sha256="b" * 64))


def test_corrupt_payload_falls_back_to_unchanged_preprocessing(tmp_path, monkeypatch):
    row, batch, spec = _row(), _batch(), _spec()
    with SQLiteFeatureCache(tmp_path, readonly=False) as cache:
        database = cache.put(row, spec, batch)
    connection = sqlite3.connect(database)
    connection.execute(
        "UPDATE features SET payload_sha256 = ? WHERE sample_id = ?", ("0" * 64, row["sample_id"])
    )
    connection.commit()
    connection.close()
    fallback = _batch()
    fallback["t"] = torch.tensor([0.75])
    monkeypatch.setattr(cache_module, "preprocess_feature_batch", lambda *_: fallback)
    result = load_cached_or_preprocess(".", row, spec, cache_root=tmp_path)
    assert result.source == "preprocessing_fallback"
    assert model_ready_batches_equal(result.batch, fallback)
    with pytest.raises(FeatureCacheError, match="no valid"):
        load_cached_or_preprocess(".", row, spec, cache_root=tmp_path, allow_fallback=False)


def test_finalized_database_stages_as_one_verified_node_local_file(tmp_path):
    shared, staged = tmp_path / "shared", tmp_path / "node-local"
    row, batch, spec = _row(), _batch(), _spec()
    with SQLiteFeatureCache(shared, readonly=False) as cache:
        cache.put(row, spec, batch)
    finalize_cache_database(shared, row["task"], row["split"])
    source = cache_database_path(shared, row["task"], row["split"])
    verify_finalized_database(source)
    destination = stage_cache_database(shared, staged, row["task"], row["split"])
    verify_finalized_database(destination)
    with SQLiteFeatureCache(shared, readonly=True) as cache:
        shared_batch = cache.get(row, spec)
    with SQLiteFeatureCache(staged, readonly=True) as cache:
        assert model_ready_batches_equal(cache.get(row, spec), shared_batch)
    assert sorted(path.name for path in destination.parent.iterdir()) == [
        "test.sqlite3",
        "test.sqlite3.sha256.json",
    ]

    def independent_worker(seed: int) -> dict:
        with SQLiteFeatureCache(staged, readonly=True, lru_size=1) as worker_cache:
            return worker_cache.get(row, _spec(random_seed=None, noise_level=None))

    with ThreadPoolExecutor(max_workers=4) as executor:
        rank_batches = list(executor.map(independent_worker, range(4)))
    deterministic = [
        {key: value for key, value in batch.items() if key not in {"t", "X_noisy_L"}}
        for batch in rank_batches
    ]
    assert all(model_ready_batches_equal(deterministic[0], item) for item in deterministic[1:])
    assert len({tuple(batch["X_noisy_L"].flatten().tolist()) for batch in rank_batches}) == 4


def test_cache_resamples_diffusion_but_preserves_all_deterministic_features(tmp_path):
    row, batch = _row(), _batch()
    with SQLiteFeatureCache(tmp_path, readonly=False) as cache:
        cache.put(row, _spec(), batch)
    with SQLiteFeatureCache(tmp_path, readonly=True, lru_size=1) as cache:
        first = cache.get(row, _spec(noise_level=None, random_seed=11))
        second = cache.get(row, _spec(noise_level=None, random_seed=12))
    assert not torch.equal(first["t"], second["t"])
    assert not torch.equal(first["X_noisy_L"], second["X_noisy_L"])
    deterministic_first = {
        key: value for key, value in first.items() if key not in {"t", "X_noisy_L"}
    }
    deterministic_second = {
        key: value for key, value in second.items() if key not in {"t", "X_noisy_L"}
    }
    assert model_ready_batches_equal(deterministic_first, deterministic_second)


def test_fixed_real_sample_cached_and_uncached_are_bitwise_equal_when_present(tmp_path):
    root = Path(__file__).resolve().parents[1]
    catalog = root / "data/processed/v0/splits/antibody_h3/test.jsonl"
    if not catalog.is_file():
        pytest.skip("frozen real-data snapshot is not part of a source-only checkout")
    row = json.loads(catalog.read_text(encoding="utf-8").splitlines()[0])
    spec = _spec(manifest_sha256="c" * 64)
    uncached = preprocess_feature_batch(root, row, spec)
    with SQLiteFeatureCache(tmp_path, readonly=False) as cache:
        cache.put(row, spec, uncached)
    with SQLiteFeatureCache(tmp_path, readonly=True, lru_size=0) as cache:
        cached = cache.get(row, spec)
    assert model_ready_batches_equal(uncached, cached)

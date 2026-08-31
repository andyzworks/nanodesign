"""SQLite cache for exact outputs of the existing RFD3NA feature loader.

Each task/split is one database rather than thousands of small files. Entries bind the
serialized CPU tensors to sample ID, source dataset version, canonical catalog row,
manifest SHA, and loader parameters. Training workers use independent read-only SQLite
connections plus a bounded in-process LRU. Node-local staging copies and verifies a
whole finalized database.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sqlite3
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import torch

from nanodesign.v0.data.real import load_foundry_training_example

CACHE_FORMAT_VERSION = 1
PREPROCESSING_VERSION = "load_foundry_training_example.atom23.v1"
SCHEMA = """
CREATE TABLE IF NOT EXISTS features (
    sample_id TEXT PRIMARY KEY,
    identity_json TEXT NOT NULL,
    payload BLOB NOT NULL,
    payload_sha256 TEXT NOT NULL,
    payload_size INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS cache_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class FeatureCacheError(RuntimeError):
    """Raised when a feature-cache database or row is missing, stale, or corrupt."""


@dataclass(frozen=True)
class FeatureCacheSpec:
    manifest_sha256: str
    max_context_tokens: int | None = 384
    diffusion_batch_size: int = 1
    noise_level: float | None = None
    random_seed: int | None = None
    preprocessing_version: str = PREPROCESSING_VERSION

    def __post_init__(self) -> None:
        digest = self.manifest_sha256.lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("manifest_sha256 must be a 64-character hexadecimal digest")
        object.__setattr__(self, "manifest_sha256", digest)
        if self.max_context_tokens is not None and self.max_context_tokens < 0:
            raise ValueError("max_context_tokens must be non-negative or None")
        if self.diffusion_batch_size < 1:
            raise ValueError("diffusion_batch_size must be positive")
        if self.noise_level is not None and self.noise_level <= 0:
            raise ValueError("noise_level must be positive or None")


@dataclass(frozen=True)
class FeatureCacheResult:
    batch: dict[str, Any]
    source: str
    database_path: Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def cache_database_path(cache_root: str | Path, task: str, split: str) -> Path:
    return Path(cache_root) / task.replace("/", "_") / f"{split.replace('/', '_')}.sqlite3"


def _row_database_path(cache_root: str | Path, row: dict[str, Any]) -> Path:
    task, split = row.get("task"), row.get("split")
    if not isinstance(task, str) or not task or not isinstance(split, str) or not split:
        raise ValueError("catalog row requires non-empty task and split")
    return cache_database_path(cache_root, task, split)


def _identity(row: dict[str, Any], spec: FeatureCacheSpec) -> dict[str, Any]:
    sample_id, dataset_version = row.get("sample_id"), row.get("source_version")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("catalog row requires a non-empty sample_id")
    if not isinstance(dataset_version, str) or not dataset_version:
        raise ValueError(f"{sample_id}: catalog row requires source_version")
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "sample_id": sample_id,
        "task": row.get("task"),
        "split": row.get("split"),
        "source": row.get("source"),
        "dataset_version": dataset_version,
        "manifest_sha256": spec.manifest_sha256.lower(),
        "row_sha256": hashlib.sha256(_canonical_json(row).encode()).hexdigest(),
        # Diffusion batch size, t/noise settings, and RNG seed are intentionally not
        # cache identity: stochastic diffusion is freshly sampled after every read.
        "preprocessing": {
            "max_context_tokens": spec.max_context_tokens,
            "preprocessing_version": spec.preprocessing_version,
        },
    }


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise FeatureCacheError(f"unsupported cached value type: {type(value)!r}")


def _copy_containers(value: Any) -> Any:
    """Copy mutable containers while sharing immutable cached CPU tensor storage."""

    if isinstance(value, dict):
        return {key: _copy_containers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_containers(item) for item in value)
    return value


def _validate_batch(batch: Any, sample_id: str) -> dict[str, Any]:
    if not isinstance(batch, dict) or batch.get("sample_id") != sample_id:
        raise FeatureCacheError("cached batch sample_id does not match its identity")
    required = {
        "f",
        "X_noisy_L",
        "t",
        "ground_truth_positions",
        "ground_truth_atom_mask",
        "ground_truth_sequence",
        "ground_truth_sequence_mask",
        "coord_atom_lvl_to_be_noised",
        "output_metadata",
    }
    missing = required - set(batch)
    if missing:
        raise FeatureCacheError(f"cached batch is missing keys: {sorted(missing)}")
    tensors: list[torch.Tensor] = []

    def collect(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(batch)
    if not tensors or any(tensor.device.type != "cpu" for tensor in tensors):
        raise FeatureCacheError("cached model-ready tensors must all be on CPU")
    positions, sequence = batch["ground_truth_positions"], batch["ground_truth_sequence"]
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise FeatureCacheError("ground_truth_positions must have shape [D, A, 3]")
    if sequence.ndim != 2 or sequence.shape[-1] != 32:
        raise FeatureCacheError("ground_truth_sequence must have shape [T, 32]")
    metadata = batch["output_metadata"]
    if not isinstance(metadata, dict) or len(metadata.get("atom_names", [])) != positions.shape[1]:
        raise FeatureCacheError("atom metadata does not match coordinate shape")
    if len(metadata.get("token_chain_names", [])) != sequence.shape[0]:
        raise FeatureCacheError("token metadata does not match sequence shape")
    return batch


def _deterministic_template(batch: dict[str, Any], sample_id: str) -> dict[str, Any]:
    """Remove one-call diffusion state while preserving exact deterministic features."""

    batch = _validate_batch(_to_cpu(batch), sample_id)
    template = {key: value for key, value in batch.items() if key not in {"X_noisy_L", "t"}}
    template["ground_truth_positions"] = batch["ground_truth_positions"][:1].clone()
    template["coord_atom_lvl_to_be_noised"] = batch["coord_atom_lvl_to_be_noised"][:1].clone()
    return template


def _validate_template(template: Any, sample_id: str) -> dict[str, Any]:
    if not isinstance(template, dict) or template.get("sample_id") != sample_id:
        raise FeatureCacheError("cached deterministic template has the wrong sample_id")
    if "X_noisy_L" in template or "t" in template:
        raise FeatureCacheError("cache payload must not persist stochastic diffusion state")
    required = {
        "f",
        "ground_truth_positions",
        "ground_truth_atom_mask",
        "ground_truth_sequence",
        "ground_truth_sequence_mask",
        "coord_atom_lvl_to_be_noised",
        "output_metadata",
    }
    if required - set(template):
        raise FeatureCacheError("cached deterministic template is incomplete")
    if template["ground_truth_positions"].shape[0] != 1:
        raise FeatureCacheError("cached ground-truth coordinates must have one deterministic copy")
    return template


def _sample_diffusion(template: dict[str, Any], spec: FeatureCacheSpec) -> dict[str, Any]:
    """Recreate the loader's EDM t/noise fields from a deterministic cached template."""

    try:
        from atomworks.ml.transforms.diffusion.edm import sample_noise_edm, sample_t_edm
    except ImportError as error:
        raise ImportError("sampling cached features requires the project 'model' extra") from error

    def sample() -> tuple[torch.Tensor, torch.Tensor]:
        if spec.noise_level is None:
            timesteps = sample_t_edm(16.0, spec.diffusion_batch_size)
        else:
            timesteps = torch.full((spec.diffusion_batch_size,), float(spec.noise_level))
        base_positions = template["ground_truth_positions"][0]
        noise = sample_noise_edm(timesteps, len(base_positions))
        token_design = template["ground_truth_sequence_mask"]
        atom_to_token = template["f"]["atom_to_token_map"].long()
        atom_design = token_design[atom_to_token]
        noise[:, ~atom_design, :] = 0.0
        return timesteps, base_positions.unsqueeze(0) + noise

    if spec.random_seed is not None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec.random_seed)
            timesteps, noisy = sample()
    else:
        timesteps, noisy = sample()
    # Cached deterministic tensors are read-only model inputs. Reuse their storage in
    # the per-worker LRU and copy only containers; each call still owns fresh t/noise.
    batch = _copy_containers(template)
    positions = template["ground_truth_positions"][0]
    batch["t"] = timesteps
    batch["X_noisy_L"] = noisy
    # DataLoader pin-memory writes into a destination buffer and rejects overlapping
    # expand views. Materialize the identical values for asynchronous pinned H2D.
    batch["ground_truth_positions"] = (
        positions.unsqueeze(0).expand(spec.diffusion_batch_size, -1, -1).contiguous()
    )
    batch["coord_atom_lvl_to_be_noised"] = positions.unsqueeze(0)
    return _validate_batch(batch, str(template["sample_id"]))


def _serialize(batch: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(batch, buffer)
    return buffer.getvalue()


def _deserialize(payload: bytes) -> dict[str, Any]:
    try:
        return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as error:
        raise FeatureCacheError("cannot deserialize cached tensor payload") from error


class SQLiteFeatureCache:
    """Per-worker SQLite reader/writer with independent connections and batch LRU."""

    def __init__(self, cache_root: str | Path, *, readonly: bool = True, lru_size: int = 8) -> None:
        if lru_size < 0:
            raise ValueError("lru_size must be non-negative")
        self.cache_root = Path(cache_root)
        self.readonly = readonly
        self.lru_size = lru_size
        self._connections: dict[Path, sqlite3.Connection] = {}
        self._lru: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        for connection in self._connections.values():
            connection.close()
        self._connections.clear()
        self._lru.clear()

    def _connection(self, path: Path) -> sqlite3.Connection:
        connection = self._connections.get(path)
        if connection is not None:
            return connection
        if self.readonly:
            if not path.is_file():
                raise FeatureCacheError(f"feature-cache database miss: {path}")
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
            connection.execute("PRAGMA query_only=ON")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(path, timeout=30)
            connection.executescript(SCHEMA)
            connection.execute(
                "INSERT OR REPLACE INTO cache_metadata(key, value) VALUES (?, ?)",
                ("cache_format_version", str(CACHE_FORMAT_VERSION)),
            )
            connection.commit()
        self._connections[path] = connection
        return connection

    def put(self, row: dict[str, Any], spec: FeatureCacheSpec, batch: dict[str, Any]) -> Path:
        if self.readonly:
            raise FeatureCacheError("cannot write through a read-only feature cache")
        identity = _identity(row, spec)
        template = _deterministic_template(batch, identity["sample_id"])
        payload = _serialize(template)
        path = _row_database_path(self.cache_root, row)
        self._connection(path).execute(
            "INSERT OR REPLACE INTO features "
            "(sample_id, identity_json, payload, payload_sha256, payload_size) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                identity["sample_id"],
                _canonical_json(identity),
                sqlite3.Binary(payload),
                hashlib.sha256(payload).hexdigest(),
                len(payload),
            ),
        )
        self._connection(path).commit()
        self._lru.clear()
        return path

    def contains_valid(self, row: dict[str, Any], spec: FeatureCacheSpec) -> bool:
        """Return whether a resumable writer already contains this exact intact row."""

        if self.readonly:
            raise FeatureCacheError("resumable cache checks require a writable feature cache")
        expected = _canonical_json(_identity(row, spec))
        path = _row_database_path(self.cache_root, row)
        record = (
            self._connection(path)
            .execute(
                "SELECT identity_json, payload, payload_sha256, payload_size "
                "FROM features WHERE sample_id = ?",
                (row["sample_id"],),
            )
            .fetchone()
        )
        if record is None:
            return False
        identity_json, payload, payload_sha, payload_size = record
        payload = bytes(payload)
        return bool(
            identity_json == expected
            and payload_size == len(payload)
            and payload_sha == hashlib.sha256(payload).hexdigest()
        )

    def get(self, row: dict[str, Any], spec: FeatureCacheSpec) -> dict[str, Any]:
        expected = _identity(row, spec)
        path = _row_database_path(self.cache_root, row)
        lru_key = (str(path), hashlib.sha256(_canonical_json(expected).encode()).hexdigest())
        cached = self._lru.get(lru_key)
        if cached is not None:
            self._lru.move_to_end(lru_key)
            return _sample_diffusion(cached, spec)
        record = (
            self._connection(path)
            .execute(
                "SELECT identity_json, payload, payload_sha256, payload_size "
                "FROM features WHERE sample_id = ?",
                (expected["sample_id"],),
            )
            .fetchone()
        )
        if record is None:
            raise FeatureCacheError(f"feature cache miss: {expected['sample_id']}")
        identity_json, payload, payload_sha, payload_size = record
        if identity_json != _canonical_json(expected):
            raise FeatureCacheError(f"stale feature-cache identity: {expected['sample_id']}")
        payload = bytes(payload)
        if payload_size != len(payload) or payload_sha != hashlib.sha256(payload).hexdigest():
            raise FeatureCacheError(f"corrupt feature-cache payload: {expected['sample_id']}")
        template = _validate_template(_deserialize(payload), expected["sample_id"])
        if self.lru_size:
            self._lru[lru_key] = template
            while len(self._lru) > self.lru_size:
                self._lru.popitem(last=False)
        return _sample_diffusion(template, spec)

    def quick_check(self, task: str, split: str) -> None:
        path = cache_database_path(self.cache_root, task, split)
        result = self._connection(path).execute("PRAGMA quick_check").fetchone()
        if result != ("ok",):
            raise FeatureCacheError(f"SQLite quick_check failed for {path}: {result}")


def _database_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256.json")


def finalize_cache_database(cache_root: str | Path, task: str, split: str) -> Path:
    """Checkpoint and atomically record a whole-database checksum for staging."""

    path = cache_database_path(cache_root, task, split)
    if not path.is_file():
        raise FeatureCacheError(f"cannot finalize missing cache database: {path}")
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise FeatureCacheError(f"SQLite quick_check failed for {path}")
    finally:
        connection.close()
    sidecar = _database_sidecar(path)
    metadata = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=sidecar.parent, prefix=sidecar.name, delete=False
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, sidecar)
    return sidecar


def verify_finalized_database(path: str | Path) -> None:
    path = Path(path)
    sidecar = _database_sidecar(path)
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeatureCacheError(
            f"missing or invalid database checksum sidecar: {sidecar}"
        ) from error
    if metadata.get("size_bytes") != path.stat().st_size or metadata.get("sha256") != sha256_file(
        path
    ):
        raise FeatureCacheError(f"finalized database checksum mismatch: {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise FeatureCacheError(f"SQLite quick_check failed for {path}")
    finally:
        connection.close()


def stage_cache_database(
    shared_cache_root: str | Path,
    stage_root: str | Path,
    task: str,
    split: str,
) -> Path:
    """Copy one finalized task/split database to node-local storage and verify it."""

    source = cache_database_path(shared_cache_root, task, split)
    verify_finalized_database(source)
    destination = cache_database_path(stage_root, task, split)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for source_path, destination_path in (
        (source, destination),
        (_database_sidecar(source), _database_sidecar(destination)),
    ):
        temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
        shutil.copy2(source_path, temporary)
        os.replace(temporary, destination_path)
    verify_finalized_database(destination)
    return destination


def preprocess_feature_batch(
    dataset_root: str | Path, row: dict[str, Any], spec: FeatureCacheSpec
) -> dict[str, Any]:
    """Call unchanged preprocessing with the exact parameters bound by ``spec``."""

    def load() -> dict[str, Any]:
        return load_foundry_training_example(
            dataset_root,
            row,
            noise_level=spec.noise_level,
            diffusion_batch_size=spec.diffusion_batch_size,
            max_context_tokens=spec.max_context_tokens,
        )

    # A seeded cache build is deterministic and restores the caller's RNG. An unseeded
    # fallback deliberately calls the original loader directly, preserving its normal
    # RNG advancement semantics.
    if spec.random_seed is not None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec.random_seed)
            batch = load()
    else:
        batch = load()
    return _validate_batch(_to_cpu(batch), str(row["sample_id"]))


def load_cached_or_preprocess(
    dataset_root: str | Path,
    row: dict[str, Any],
    spec: FeatureCacheSpec,
    *,
    cache_root: str | Path,
    stage_root: str | Path | None = None,
    allow_fallback: bool = True,
    populate_on_fallback: bool = False,
    lru_size: int = 8,
) -> FeatureCacheResult:
    """Read a verified cache row or fall back to the unchanged original loader."""

    selected_root = Path(stage_root) if stage_root is not None else Path(cache_root)
    if stage_root is not None and not _row_database_path(stage_root, row).is_file():
        try:
            stage_cache_database(cache_root, stage_root, row["task"], row["split"])
        except FeatureCacheError:
            pass
    try:
        with SQLiteFeatureCache(selected_root, readonly=True, lru_size=lru_size) as cache:
            batch = cache.get(row, spec)
        source = "node_local_cache" if stage_root is not None else "shared_cache"
        return FeatureCacheResult(batch, source, _row_database_path(selected_root, row))
    except FeatureCacheError:
        if stage_root is not None:
            try:
                with SQLiteFeatureCache(cache_root, readonly=True, lru_size=lru_size) as cache:
                    batch = cache.get(row, spec)
                return FeatureCacheResult(
                    batch, "shared_cache", _row_database_path(cache_root, row)
                )
            except FeatureCacheError:
                pass
    if not allow_fallback:
        raise FeatureCacheError(f"no valid feature cache for {row.get('sample_id')}")
    batch = preprocess_feature_batch(dataset_root, row, spec)
    database = _row_database_path(cache_root, row)
    if populate_on_fallback:
        with SQLiteFeatureCache(cache_root, readonly=False, lru_size=0) as cache:
            database = cache.put(row, spec, batch)
    return FeatureCacheResult(batch, "preprocessing_fallback", database)


def model_ready_batches_equal(left: Any, right: Any) -> bool:
    """Recursively require bitwise-identical tensors and identical provenance values."""

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            model_ready_batches_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            model_ready_batches_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right

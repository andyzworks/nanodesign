import hashlib
import json
import random

import pytest
import torch

from nanodesign.v0.learnability import TASK_INDEX, _deterministic_linear_pool, load_frozen_panel


def _write_protocol(tmp_path):
    tasks = {}
    seed = 17
    for task, task_index in TASK_INDEX.items():
        rows = [
            {
                "sample_id": f"{task}-{index}",
                "task": task,
                "split": "validation",
                "source": "fixture",
                "source_version": "fixture-v1",
                "raw_paths": [],
                "chains": [],
            }
            for index in range(4)
        ]
        catalog = tmp_path / f"{task}.jsonl"
        catalog.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        selected = random.Random(seed + 1000 + task_index).sample(rows, 3)
        ids = "".join(f"{row['sample_id']}\n" for row in selected).encode()
        tasks[task] = {
            "catalog": catalog.name,
            "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
            "panel_size": 3,
            "selected_sample_ids_sha256": hashlib.sha256(ids).hexdigest(),
        }
    protocol = {
        "protocol": "nanodesign.learnability.v1",
        "seed": seed,
        "diffusion_t": 0.5,
        "diffusion_realizations_per_complex": 1,
        "coordinate_augmentation": False,
        "weight_source": "ema",
        "tasks": tasks,
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    return path


def test_frozen_panel_is_exact_and_repeatable(tmp_path):
    protocol_path = _write_protocol(tmp_path)
    _, first = load_frozen_panel(tmp_path, protocol_path)
    _, second = load_frozen_panel(tmp_path, protocol_path)

    assert first == second
    assert all(len(rows) == 3 for rows in first.values())


def test_frozen_panel_rejects_catalog_drift(tmp_path):
    protocol_path = _write_protocol(tmp_path)
    with (tmp_path / "rna.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(ValueError, match="catalog SHA256 changed"):
        load_frozen_panel(tmp_path, protocol_path)


def test_deterministic_pool_is_segment_mean_and_restores_official_forward():
    class Pool(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Identity()

        def forward(self, *_args, **_kwargs):
            raise RuntimeError("official path")

    pool = Pool()
    model = type("Model", (), {})()
    model.net = type("Net", (), {})()
    model.net.diffusion_module = type("Diffusion", (), {})()
    model.net.diffusion_module.process_a = pool
    coordinates = torch.tensor([[[1.0], [3.0], [2.0], [4.0], [6.0]]])
    token_indices = torch.tensor([0, 0, 1, 2, 2])

    with _deterministic_linear_pool(model):
        pooled = pool(coordinates, tok_idx=token_indices)
    assert torch.equal(pooled, torch.tensor([[[2.0], [2.0], [5.0]]]))
    with pytest.raises(RuntimeError, match="official path"):
        pool(coordinates, tok_idx=token_indices)

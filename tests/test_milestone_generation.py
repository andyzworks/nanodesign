import argparse
import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts import generate_milestone as runner


class _FakeModel:
    parameter_count = 123
    last_execution_mode = "standard"

    def __init__(self, config):
        self.config = config

    def to(self, device):
        return self

    def eval(self):
        return self


def _row(task):
    roles = {
        "protein_binder": [("target", 5), ("binder", 2)],
        "antibody_h3": [("antigen", 5), ("antibody_framework+cdr_h3", 2)],
        "rna": [("target", 5), ("rna_design_region", 2)],
    }[task]
    chains = []
    for index, (role, count) in enumerate(roles):
        chain = {"chain_id": chr(65 + index), "role": role, "resolved_residues": count}
        if role == "antibody_framework+cdr_h3":
            chain["design_residue_keys"] = [[105, ""]]
        chains.append(chain)
    return {"sample_id": f"{task}:test", "chains": chains}


def _args(tmp_path, samples_seen=3000):
    return argparse.Namespace(
        checkpoint=tmp_path / "milestone.pt",
        samples_seen=samples_seen,
        config="configs/v0.yaml",
        seed=17,
        tasks=list(runner.SPLITS),
        device="cpu",
        weight_source="ema",
        output_root=tmp_path / "generations",
    )


def test_generic_cuda_device_resolves_to_current_logical_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert runner._resolve_device("cuda") == torch.device("cuda:0")


def test_milestone_selection_matches_final_complete_context_rule():
    incomplete, complete = _row("protein_binder"), _row("protein_binder")
    incomplete["sample_id"] = "incomplete"
    incomplete["chains"][0]["resolved_residues"] = 500
    complete["sample_id"] = "complete"
    selected, is_complete = runner._generation_row([incomplete, complete], max_context_tokens=384)
    assert selected["sample_id"] == "complete"
    assert is_complete


def test_milestone_runner_validates_checkpoint_and_writes_all_three_tasks(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    resolved = runner.load_config(root / "configs/v0.yaml")
    config_sha = runner._config_sha256(resolved)
    expected_manifest = hashlib.sha256((root / "docs/data_v0_stats.json").read_bytes()).hexdigest()
    calls = {}

    def fake_load_checkpoint(path, *, model, expected_manifest_sha256, prefer_ema):
        calls["checkpoint"] = path
        calls["manifest"] = expected_manifest_sha256
        calls["prefer_ema"] = prefer_ema
        return {
            "samples_seen": 3000,
            "step": 750,
            "config_sha256": config_sha,
            "loaded_weight_source": "ema",
        }

    def fake_catalog(path):
        task = next(task for task, suffix in runner.SPLITS.items() if str(path).endswith(suffix))
        return [_row(task)]

    def fake_batch(root, row, **kwargs):
        return {"sample_id": row["sample_id"]}

    def fake_generate(model, batch):
        return {
            "X_L": torch.zeros(1, 1, 3),
            "sequence_logits_I": torch.zeros(1, 1, 32),
        }

    def fake_write(output, batch, path):
        Path(path).write_text("END\n", encoding="utf-8")
        return {"A": "A"}

    monkeypatch.setattr(runner, "NanoDesignTiny", _FakeModel)
    monkeypatch.setattr(runner, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(runner, "load_split_catalog", fake_catalog)
    monkeypatch.setattr(runner, "_batch", fake_batch)
    monkeypatch.setattr(runner, "generate", fake_generate)
    monkeypatch.setattr(runner, "write_generation_structure", fake_write)

    metadata = runner.run(_args(tmp_path), root=root)
    output = tmp_path / "generations/samples-00003000"
    assert calls == {
        "checkpoint": tmp_path / "milestone.pt",
        "manifest": expected_manifest,
        "prefer_ema": True,
    }
    assert metadata["samples_seen"] == 3000
    assert metadata["optimizer_step"] == 750
    assert metadata["weight_source"] == "ema"
    assert set(metadata["tasks"]) == set(runner.SPLITS)
    assert metadata["generation"] == metadata["tasks"]
    assert [metadata["tasks"][task]["seed"] for task in runner.SPLITS] == [17, 18, 19]
    assert metadata["tasks"]["protein_binder"]["target_chains"] == ["A"]
    assert metadata["tasks"]["protein_binder"]["binder_chain"] == "B"
    assert all((output / f"{task}.pdb").is_file() for task in runner.SPLITS)
    assert json.loads((output / "metadata.json").read_text()) == metadata


def test_milestone_runner_single_task_keeps_unified_generation_seed(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    resolved = runner.load_config(root / "configs/v0.yaml")
    config_sha = runner._config_sha256(resolved)
    args = _args(tmp_path)
    args.tasks = ["rna"]

    monkeypatch.setattr(runner, "NanoDesignTiny", _FakeModel)
    monkeypatch.setattr(
        runner,
        "load_checkpoint",
        lambda *args, **kwargs: {
            "samples_seen": 3000,
            "step": 3000,
            "config_sha256": config_sha,
            "loaded_weight_source": "ema",
        },
    )
    monkeypatch.setattr(runner, "load_split_catalog", lambda _path: [_row("rna")])
    monkeypatch.setattr(
        runner, "_batch", lambda _root, row, **_kwargs: {"sample_id": row["sample_id"]}
    )
    monkeypatch.setattr(
        runner,
        "generate",
        lambda _model, _batch: {
            "X_L": torch.zeros(1, 1, 3),
            "sequence_logits_I": torch.zeros(1, 1, 32),
        },
    )
    monkeypatch.setattr(
        runner,
        "write_generation_structure",
        lambda _output, _batch, path: Path(path).write_text("END\n", encoding="utf-8") or {},
    )

    metadata = runner.run(args, root=root)

    assert list(metadata["tasks"]) == ["rna"]
    assert metadata["tasks"]["rna"]["seed"] == 19


@pytest.mark.parametrize(
    ("checkpoint_samples", "checkpoint_config", "message"),
    [(2999, "current", "samples_seen"), (3000, "wrong", "frozen configuration")],
)
def test_milestone_runner_rejects_wrong_budget_or_config(
    tmp_path, monkeypatch, checkpoint_samples, checkpoint_config, message
):
    root = Path(__file__).resolve().parents[1]
    resolved = runner.load_config(root / "configs/v0.yaml")
    config_sha = runner._config_sha256(resolved)
    monkeypatch.setattr(runner, "NanoDesignTiny", _FakeModel)
    monkeypatch.setattr(
        runner,
        "load_checkpoint",
        lambda *args, **kwargs: {
            "samples_seen": checkpoint_samples,
            "step": 750,
            "config_sha256": config_sha if checkpoint_config == "current" else "0" * 64,
        },
    )
    with pytest.raises(ValueError, match=message):
        runner.run(_args(tmp_path), root=root)

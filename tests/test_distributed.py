from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from nanodesign.v0.distributed import (
    PROTEIN_ATOM_SLOTS,
    DistributedContext,
    all_gather_objects,
    catalog_model_atom_count,
    catalog_model_token_count,
    reduce_scalar_metrics,
    row_for_rank,
    samples_seen,
    size_aware_rank_packing,
    synchronize_training_execution_mode,
    task_for_step,
    validation_indices,
)


def _rows(prefix: str, count: int):
    return [{"sample_id": f"{prefix}-{index}"} for index in range(count)]


def _sized_row(sample_id: str, residues: int):
    return {
        "sample_id": sample_id,
        "chains": [
            {"role": "target", "resolved_residues": residues},
            {"role": "binder", "resolved_residues": 10},
        ],
    }


def test_catalog_size_matches_foundry_crop_and_size_packing_is_a_permutation():
    antibody = {
        "sample_id": "antibody",
        "chains": [
            {
                "role": "antibody_framework+cdr_h3",
                "resolved_residues": 120,
                "design_residue_keys": [[index, ""] for index in range(12)],
            },
            {"role": "antibody_framework", "resolved_residues": 110},
            {"role": "antigen", "resolved_residues": 800},
        ],
    }
    assert catalog_model_token_count(antibody, max_context_tokens=384) == 396
    assert catalog_model_atom_count(antibody, max_context_tokens=384) == 396 * PROTEIN_ATOM_SLOTS

    rows = [
        _sized_row(f"sample-{index}", residues) for index, residues in enumerate(range(8, 168, 10))
    ]
    packed = size_aware_rank_packing(rows, world_size=4, seed=7, max_context_tokens=384)
    assert sorted(row["sample_id"] for row in packed) == sorted(row["sample_id"] for row in rows)
    assert packed == size_aware_rank_packing(rows, world_size=4, seed=7, max_context_tokens=384)
    scheduled = [
        row_for_rank(
            {"protein_binder": packed},
            ["protein_binder"],
            optimizer_step=step,
            rank=rank,
            world_size=4,
        )["sample_id"]
        for step in range(4)
        for rank in range(4)
    ]
    assert len(scheduled) == len(set(scheduled)) == len(rows)
    resumed_suffix = [
        row_for_rank(
            {"protein_binder": packed},
            ["protein_binder"],
            optimizer_step=step,
            rank=rank,
            world_size=4,
        )["sample_id"]
        for step in range(2, 4)
        for rank in range(4)
    ]
    assert resumed_suffix == scheduled[8:]
    spans = []
    for start in range(0, len(packed), 4):
        sizes = [
            catalog_model_atom_count(row, max_context_tokens=384)
            for row in packed[start : start + 4]
        ]
        spans.append(max(sizes) - min(sizes))
    assert max(spans) == 30 * PROTEIN_ATOM_SLOTS


def test_distributed_assignment_preserves_task_rotation_and_shards_samples():
    tasks = ["protein_binder", "antibody_h3", "rna"]
    rows = {task: _rows(task, 12) for task in tasks}

    assert [task_for_step(tasks, step) for step in range(6)] == tasks * 2
    first = [
        row_for_rank(rows, tasks, optimizer_step=0, rank=rank, world_size=4)["sample_id"]
        for rank in range(4)
    ]
    second = [
        row_for_rank(rows, tasks, optimizer_step=3, rank=rank, world_size=4)["sample_id"]
        for rank in range(4)
    ]
    assert first == [f"protein_binder-{rank}" for rank in range(4)]
    assert second == [f"protein_binder-{rank}" for rank in range(4, 8)]
    assert not set(first) & set(second)


def test_single_task_diagnostic_matches_unified_per_task_sample_exposure():
    tasks = ["protein_binder", "antibody_h3", "rna"]
    rows = {task: _rows(task, 7) for task in tasks}
    for task_index, task in enumerate(tasks):
        unified = [
            row_for_rank(
                rows,
                tasks,
                optimizer_step=len(tasks) * occurrence + task_index,
                rank=0,
                world_size=1,
            )["sample_id"]
            for occurrence in range(12)
        ]
        single = [
            row_for_rank(
                {task: rows[task]},
                [task],
                optimizer_step=occurrence,
                rank=0,
                world_size=1,
            )["sample_id"]
            for occurrence in range(12)
        ]
        assert single == unified


def test_fixed_four_way_packing_exposes_same_flattened_samples_at_1_2_4_ranks():
    rows = [_sized_row(f"sample-{index}", index + 8) for index in range(16)]
    packed = size_aware_rank_packing(rows, world_size=4, seed=7, max_context_tokens=384)
    expected = [row["sample_id"] for row in packed[:8]]
    for world_size in (1, 2, 4):
        observed = []
        for step in range(8 // world_size):
            observed.extend(
                row_for_rank(
                    {"protein_binder": packed},
                    ["protein_binder"],
                    optimizer_step=step,
                    rank=rank,
                    world_size=world_size,
                )["sample_id"]
                for rank in range(world_size)
            )
        assert observed == expected


def test_validation_shards_have_no_duplicates_or_omissions():
    shards = [list(validation_indices(11, rank=rank, world_size=4)) for rank in range(4)]
    flattened = [index for shard in shards for index in shard]
    assert sorted(flattened) == list(range(11))
    assert len(flattened) == len(set(flattened))


def test_global_samples_seen_and_context_validation():
    assert samples_seen(25, 4) == 100
    assert [samples_seen(step, 4) for step in (750, 2250, 4500, 9000)] == [
        3000,
        9000,
        18000,
        36000,
    ]
    tasks = ["protein_binder", "antibody_h3", "rna"]
    counts = {
        task: sum(task_for_step(tasks, step) == task for step in range(750)) * 4 for task in tasks
    }
    assert counts == {task: 1000 for task in tasks}
    assert DistributedContext(rank=0, world_size=1).is_primary
    with pytest.raises(ValueError, match="inside"):
        DistributedContext(rank=2, world_size=2)


def _two_rank_gloo_worker(rank: int, rendezvous: str, output_dir: str) -> None:
    world_size = 2
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size
    )
    torch.manual_seed(31)
    model = torch.nn.Linear(1, 1, bias=False)
    ddp = DistributedDataParallel(model)
    optimizer = torch.optim.SGD(ddp.parameters(), lr=0.1)
    rows = {"protein_binder": _rows("protein_binder", 8)}
    row = row_for_rank(rows, ["protein_binder"], optimizer_step=0, rank=rank, world_size=world_size)
    optimizer.zero_grad(set_to_none=True)
    local_loss = ddp(torch.tensor([[float(rank + 1)]])).square().mean()
    local_loss.backward()
    optimizer.step()
    local_metrics = {"loss": float(local_loss)}
    if rank == 1:
        local_metrics["mask_dependent_aux"] = 3.0
    reduced = reduce_scalar_metrics(
        local_metrics, device=torch.device("cpu"), world_size=world_size
    )
    sample_ids = all_gather_objects(row["sample_id"], world_size)
    torch.save(
        {
            "weight": model.weight.detach(),
            "local_loss": float(local_loss),
            "loss": reduced["loss"],
            "mask_dependent_aux": reduced["mask_dependent_aux"],
            "sample_ids": sample_ids,
            "validation_indices": list(validation_indices(5, rank=rank, world_size=world_size)),
        },
        Path(output_dir) / f"rank-{rank}.pt",
    )
    dist.destroy_process_group()


def test_two_rank_gloo_ddp_reduces_metrics_and_uses_distinct_samples(tmp_path):
    rendezvous = tmp_path / "gloo-init"
    mp.spawn(
        _two_rank_gloo_worker,
        args=(str(rendezvous), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    rank0 = torch.load(tmp_path / "rank-0.pt", weights_only=True)
    rank1 = torch.load(tmp_path / "rank-1.pt", weights_only=True)
    assert torch.equal(rank0["weight"], rank1["weight"])
    assert rank0["loss"] == pytest.approx(rank1["loss"])
    assert rank0["loss"] == pytest.approx((rank0["local_loss"] + rank1["local_loss"]) / 2)
    assert rank0["mask_dependent_aux"] == rank1["mask_dependent_aux"] == 3.0
    assert rank0["sample_ids"] == ["protein_binder-0", "protein_binder-1"]
    assert rank1["sample_ids"] == rank0["sample_ids"]
    validation = rank0["validation_indices"] + rank1["validation_indices"]
    assert sorted(validation) == list(range(5))


class _CheckpointedTaskBranches(torch.nn.Module):
    """Small analogue of mask-dependent RFD3NA process_pll parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = torch.nn.Linear(2, 2)
        self.protein_only = torch.nn.Linear(2, 2)
        self.rna_only = torch.nn.Linear(2, 2)

    def forward(self, value: torch.Tensor, task: str) -> torch.Tensor:
        def task_branch(hidden: torch.Tensor) -> torch.Tensor:
            if task == "protein":
                return self.protein_only(hidden)
            return self.rna_only(hidden)

        hidden = self.shared(value)
        return torch.utils.checkpoint.checkpoint(task_branch, hidden, use_reentrant=False)


def _dynamic_graph_worker(rank: int, rendezvous: str, output_dir: str) -> None:
    world_size = 2
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size
    )
    torch.manual_seed(41)
    model = _CheckpointedTaskBranches()
    ddp = DistributedDataParallel(
        model,
        find_unused_parameters=True,
        static_graph=False,
    )
    optimizer = torch.optim.SGD(ddp.parameters(), lr=0.01)
    for task in ("protein", "rna", "protein", "rna"):
        optimizer.zero_grad(set_to_none=True)
        loss = ddp(torch.ones(1, 2), task).square().mean()
        loss.backward()
        optimizer.step()
    torch.save(
        {
            "state": model.state_dict(),
            "find_unused_parameters": ddp.find_unused_parameters,
            "static_graph": ddp.static_graph,
        },
        Path(output_dir) / f"dynamic-rank-{rank}.pt",
    )
    dist.destroy_process_group()


def test_dynamic_task_graph_with_non_reentrant_checkpointing_is_ddp_safe(tmp_path):
    rendezvous = tmp_path / "dynamic-gloo-init"
    mp.spawn(
        _dynamic_graph_worker,
        args=(str(rendezvous), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    rank0 = torch.load(tmp_path / "dynamic-rank-0.pt", weights_only=True)
    rank1 = torch.load(tmp_path / "dynamic-rank-1.pt", weights_only=True)
    assert rank0["find_unused_parameters"] is rank1["find_unused_parameters"] is True
    assert rank0["static_graph"] is rank1["static_graph"] is False
    assert rank0["state"].keys() == rank1["state"].keys()
    for name in rank0["state"]:
        assert torch.equal(rank0["state"][name], rank1["state"][name])


def _mixed_size_mode_worker(rank: int, rendezvous: str, output_dir: str) -> None:
    world_size = 2
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size
    )
    mode = synchronize_training_execution_mode(
        8008 if rank == 0 else 8009,
        device=torch.device("cpu"),
        world_size=world_size,
        standard_max_atoms=8008,
    )
    Path(output_dir, f"mode-{rank}.txt").write_text(mode, encoding="utf-8")
    dist.destroy_process_group()


def test_mixed_size_ddp_ranks_use_one_chunked_execution_graph(tmp_path):
    rendezvous = tmp_path / "mixed-size-init"
    mp.spawn(
        _mixed_size_mode_worker,
        args=(str(rendezvous), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    assert [Path(tmp_path, f"mode-{rank}.txt").read_text() for rank in range(2)] == [
        "chunked",
        "chunked",
    ]

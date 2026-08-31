from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from nanodesign.v0.distributed import (
    DistributedContext,
    all_gather_objects,
    reduce_scalar_metrics,
    row_for_rank,
    samples_seen,
    task_for_step,
    validation_indices,
)


def _rows(prefix: str, count: int):
    return [{"sample_id": f"{prefix}-{index}"} for index in range(count)]


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
    reduced = reduce_scalar_metrics(
        {"loss": float(local_loss)}, device=torch.device("cpu"), world_size=world_size
    )
    sample_ids = all_gather_objects(row["sample_id"], world_size)
    torch.save(
        {
            "weight": model.weight.detach(),
            "local_loss": float(local_loss),
            "loss": reduced["loss"],
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
    assert rank0["sample_ids"] == ["protein_binder-0", "protein_binder-1"]
    assert rank1["sample_ids"] == rank0["sample_ids"]
    validation = rank0["validation_indices"] + rank1["validation_indices"]
    assert sorted(validation) == list(range(5))

import pytest

from nanodesign.v0.distributed import (
    DistributedContext,
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
    assert DistributedContext(rank=0, world_size=1).is_primary
    with pytest.raises(ValueError, match="inside"):
        DistributedContext(rank=2, world_size=2)

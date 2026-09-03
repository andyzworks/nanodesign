import json

from nanodesign.v0.results import training_result_rows, write_training_result_rows


def _report():
    return {
        "model_parameter_count": 6_849_538,
        "samples_seen": 6,
        "global_samples_per_task": {"rna": 6},
        "seed": 17,
        "training_mechanics": {"optimizer": "AdamW", "learning_rate": 5e-4},
        "history": [
            {"task": "rna", "loss": 2.0},
            {"task": "rna", "loss": 1.0},
        ],
        "validation_after": {
            "rna": {
                "loss": 0.9,
                "sequence_loss": 0.2,
                "coordinate_loss": 0.7,
                "seq_recovery": 0.25,
            }
        },
        "generation": {"rna": {"finite": True}},
        "optimization_wall_seconds": 12.0,
        "optimization_gpu_hours": 12.0 / 3600,
    }


def test_training_result_rows_are_task_comparable():
    row = training_result_rows(_report(), experiment="rna-overfit")[0]
    assert row["experiment"] == "rna-overfit"
    assert row["task_samples_seen"] == 6
    assert row["train_loss"] == 1.5
    assert row["sequence_recovery"] == 0.25
    assert row["generation_metrics"] == {"finite": True}


def test_training_result_rows_are_written_atomically(tmp_path):
    destination = tmp_path / "experiment_results.json"
    write_training_result_rows(_report(), experiment="rna-overfit", destination=destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema"] == "nanodesign.experiment_rows.v1"
    assert len(payload["rows"]) == 1

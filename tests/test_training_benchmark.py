import gzip
import importlib.util
import json
import sys
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark_training.py"
SPEC = importlib.util.spec_from_file_location("nanodesign_training_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)
profile_training_stages = BENCHMARK.profile_training_stages
render_markdown = BENCHMARK.render_markdown
size_bucket = BENCHMARK.size_bucket
parse_bytes = BENCHMARK._parse_bytes

COMPARE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_training_modes.py"
COMPARE_SPEC = importlib.util.spec_from_file_location(
    "nanodesign_training_mode_comparison", COMPARE_SCRIPT
)
assert COMPARE_SPEC is not None and COMPARE_SPEC.loader is not None
COMPARE = importlib.util.module_from_spec(COMPARE_SPEC)
sys.modules[COMPARE_SPEC.name] = COMPARE
COMPARE_SPEC.loader.exec_module(COMPARE)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, batch):
        return {"prediction": self.linear(batch["input"])}


def _loss(batch, output):
    loss = torch.nn.functional.mse_loss(output["prediction"], batch["target"])
    return loss, {"loss": loss}


def test_cpu_stage_profiler_times_complete_training_step():
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters())
    result = profile_training_stages(
        model,
        optimizer,
        {"input": torch.ones(4, 3), "target": torch.zeros(4, 2), "metadata": "kept"},
        device=torch.device("cpu"),
        repeats=2,
        loss_function=_loss,
    )
    for name in (
        "h2d_ms",
        "forward_ms",
        "loss_ms",
        "backward_ms",
        "optimizer_ms",
        "training_step_ms",
    ):
        assert result[name] >= 0
    assert result["cuda_peak_allocated_bytes"] == 0
    assert result["cuda_peak_reserved_bytes"] == 0


def test_buckets_and_markdown_are_deterministic():
    assert size_bucket(64, small_max=64, medium_max=256) == "small"
    assert size_bucket(65, small_max=64, medium_max=256) == "medium"
    assert size_bucket(257, small_max=64, medium_max=256) == "large"
    report = {
        "records": [
            {
                "mode": "chunked",
                "task": "protein_binder",
                "sample_id": "fixed",
                "bucket": "small/small",
                "tokens": 8,
                "atoms": 112,
                "timing_ms": {
                    "raw_file_io_ms": 1.0,
                    "parse_ms": 2.0,
                    "feature_construction_ms": 3.0,
                    "h2d_ms": 4.0,
                    "forward_ms": 5.0,
                    "loss_ms": 6.0,
                    "backward_ms": 7.0,
                    "optimizer_ms": 8.0,
                },
                "cuda_peak_allocated_bytes": 1_000_000_000,
                "status": "ok",
            }
        ]
    }
    markdown = render_markdown(json.loads(json.dumps(report)))
    assert "| chunked | protein_binder | fixed | small/small |" in markdown
    assert "| 1.000 | 0.000 | ok |" in markdown


def test_parse_bytes_supports_gzip_pdb(tmp_path):
    path = tmp_path / "complex.pdb.gz"
    payload = gzip.compress(
        b"ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 20.00           C\nEND\n"
    )
    structure = parse_bytes(path, payload)
    assert len(structure) == 1
    assert structure[0][0][0][0].name == "CA"


def test_tensor_difference_reports_absolute_and_relative_error():
    difference = COMPARE.tensor_difference(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.5]))
    assert difference["max_absolute"] == 0.5
    assert difference["mean_absolute"] == 0.25
    assert difference["relative_l2"] > 0

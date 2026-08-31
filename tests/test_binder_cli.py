import json

from nanodesign.v0 import cli
from nanodesign.v0.evaluators import ProteinBinderResult


def test_binder_cli_runs_frozen_evaluator_and_writes_json(tmp_path, monkeypatch):
    generated = tmp_path / "generated.pdb"
    generated.write_text("END\n", encoding="utf-8")
    calls = []

    def fake_evaluate(generated_complex, **kwargs):
        calls.append((generated_complex, kwargs))
        return ProteinBinderResult(
            metrics={"iptm": 0.81, "self_consistency_rmsd": 1.25},
            passed=True,
        )

    monkeypatch.setattr(cli, "evaluate_protein_binder", fake_evaluate)
    result_json = tmp_path / "reports" / "binder.json"
    args = cli.build_parser().parse_args(
        [
            "evaluate-protein-binder",
            "--generated-complex",
            str(generated),
            "--target-chains",
            "A",
            "C",
            "--binder-chain",
            "B",
            "--output-dir",
            str(tmp_path / "evaluation"),
            "--result-json",
            str(result_json),
            "--colabfold-executable",
            "/tools/colabfold_batch",
            "--pyrosetta-python",
            "/tools/python",
            "--pyrosetta-analyzer-script",
            "/repo/scripts/pyrosetta_interface_analyzer.py",
        ]
    )

    assert args.function(args) == 0
    assert calls == [
        (
            str(generated),
            {
                "target_chains": ["A", "C"],
                "binder_chain": "B",
                "output_dir": str(tmp_path / "evaluation"),
                "colabfold_executable": "/tools/colabfold_batch",
                "rosetta_executable": "InterfaceAnalyzer.linuxgccrelease",
                "pyrosetta_python": "/tools/python",
                "pyrosetta_analyzer_script": "/repo/scripts/pyrosetta_interface_analyzer.py",
            },
        )
    ]
    assert json.loads(result_json.read_text(encoding="utf-8")) == {
        "generated_complex": str(generated.resolve()),
        "target_chains": ["A", "C"],
        "binder_chain": "B",
        "metrics": {"iptm": 0.81, "self_consistency_rmsd": 1.25},
        "passed": True,
        "in_silico_success_rate": 1.0,
    }


def test_binder_cli_accepts_binary_rosetta_backend(tmp_path, monkeypatch):
    generated = tmp_path / "generated.pdb"
    generated.write_text("END\n", encoding="utf-8")
    captured = {}

    def fake_evaluate(generated_complex, **kwargs):
        captured.update(kwargs)
        return ProteinBinderResult(metrics={}, passed=False)

    monkeypatch.setattr(cli, "evaluate_protein_binder", fake_evaluate)
    args = cli.build_parser().parse_args(
        [
            "evaluate-protein-binder",
            "--generated-complex",
            str(generated),
            "--target-chains",
            "A",
            "--binder-chain",
            "B",
            "--output-dir",
            str(tmp_path / "evaluation"),
            "--result-json",
            str(tmp_path / "result.json"),
            "--rosetta-executable",
            "/tools/InterfaceAnalyzer.linuxgccrelease",
        ]
    )

    assert args.function(args) == 0
    assert captured["rosetta_executable"] == "/tools/InterfaceAnalyzer.linuxgccrelease"
    assert captured["pyrosetta_python"] is None
    assert captured["pyrosetta_analyzer_script"] is None

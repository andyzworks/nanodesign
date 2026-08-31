import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _antibody_pdb(*, include_light: bool) -> str:
    lines = []
    serial = 1
    chain_residues = {"H": [1, 2, 3, 4, 105, 106]}
    if include_light:
        chain_residues["L"] = [1, 2, 3, 4]
    chain_residues["I"] = [1, 2, 3]
    for chain_index, (chain, residue_numbers) in enumerate(chain_residues.items()):
        for residue_index, residue_number in enumerate(residue_numbers):
            for atom_name, dx, element in (
                ("N", 0.0, "N"),
                ("CA", 1.2, "C"),
                ("C", 2.4, "C"),
                ("O", 3.0, "O"),
            ):
                x = residue_index * 3.8 + dx
                y = chain_index * 3.0
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:^4s} ALA {chain}{residue_number:4d}    "
                    f"{x:8.3f}{y:8.3f}{0.0:8.3f}  1.00 90.00          {element:>2s}"
                )
                serial += 1
        lines.append("TER")
    return "\n".join(lines) + "\nEND\n"


@pytest.mark.parametrize(("light_chain", "dockq_mapping"), [("L", "HLI:HLI"), (None, "HI:HI")])
def test_antibody_cli_runs_complete_catalog_resolved_path(tmp_path, light_chain, dockq_mapping):
    root = Path(__file__).resolve().parents[1]
    reference = tmp_path / "reference.pdb"
    prediction = tmp_path / "prediction.pdb"
    reference.write_text(_antibody_pdb(include_light=light_chain is not None), encoding="utf-8")
    prediction.write_text(_antibody_pdb(include_light=light_chain is not None), encoding="utf-8")
    sample_id = "sabdab2:synthetic_H_L"
    catalog = tmp_path / "data/processed/v0/splits/antibody_h3/test.jsonl"
    catalog.parent.mkdir(parents=True)
    for split in ("train", "validation"):
        (catalog.parent / f"{split}.jsonl").write_text("", encoding="utf-8")
    catalog.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "task": "antibody_cdr",
                "raw_paths": [str(reference)],
                "chains": [
                    {"chain_id": "H", "role": "antibody_framework+cdr_h3"},
                    *(
                        [{"chain_id": light_chain, "role": "antibody_framework"}]
                        if light_chain is not None
                        else []
                    ),
                    {"chain_id": "I", "role": "antigen"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "training_report.json"
    report.write_text(
        json.dumps(
            {
                "generation": {
                    "antibody_h3": {
                        "sample_id": sample_id,
                        "structure_path": str(prediction),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    dockq = tmp_path / "fake_dockq.py"
    dockq.write_text(
        f"""#!/usr/bin/env python3
import json, pathlib, sys
assert sys.argv[sys.argv.index('--mapping') + 1] == {dockq_mapping!r}
path = pathlib.Path(sys.argv[sys.argv.index('--json') + 1])
path.write_text(json.dumps({{'GlobalDockQ': 0.75}}))
""",
        encoding="utf-8",
    )
    os.chmod(dockq, 0o755)
    output = tmp_path / "antibody_metrics.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/evaluate_antibody_h3.py"),
            "--training-report",
            str(report),
            "--repo-root",
            str(tmp_path),
            "--dockq-executable",
            str(dockq),
            "--output",
            str(output),
        ],
        check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["chains"] == {"heavy": "H", "light": light_chain, "antigen": ["I"]}
    assert result["metrics"]["h3_aar"] == pytest.approx(1.0)
    assert result["metrics"]["h3_rmsd"] == pytest.approx(0.0, abs=1e-12)
    assert result["metrics"]["dockq"] == pytest.approx(0.75)

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from nanodesign.v0.evaluators import (
    aggregate_binder_results,
    evaluate_antibody_h3,
    evaluate_protein_binder,
    evaluate_rna,
    framework_aligned_h3_rmsd,
    run_dockq,
    run_pyrosetta_interface_analyzer,
    run_usalign_rna,
)


def _protein_pdb(sequences: list[str]) -> str:
    lines = []
    serial = 1
    for chain_index, sequence in enumerate(sequences):
        chain = chr(ord("A") + chain_index)
        for residue_index, _ in enumerate(sequence, start=1):
            for atom_name, dx, element in (
                ("N", 0.0, "N"),
                ("CA", 1.2, "C"),
                ("C", 2.4, "C"),
                ("O", 3.0, "O"),
            ):
                x = (residue_index - 1) * 3.8 + dx
                y = chain_index * 3.0
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:^4s} ALA {chain}{residue_index:4d}    "
                    f"{x:8.3f}{y:8.3f}{0.0:8.3f}  1.00 90.00          {element:>2s}"
                )
                serial += 1
        lines.append("TER")
    return "\n".join(lines) + "\nEND\n"


def _rna_target_pdb() -> str:
    protein = _protein_pdb(["AAA"]).splitlines()[:-1]
    lines = [line for line in protein if line != "END"]
    serial = sum(line.startswith("ATOM") for line in lines) + 1
    for residue_index in range(1, 4):
        for atom_name, dx, element in (("P", 0.0, "P"), ("C4'", 1.2, "C"), ("C1'", 2.4, "C")):
            x = (residue_index - 1) * 3.8 + dx
            lines.append(
                f"ATOM  {serial:5d} {atom_name:^4s}   A B{residue_index:4d}    "
                f"{x:8.3f}{3.0:8.3f}{0.0:8.3f}  1.00 90.00          {element:>2s}"
            )
            serial += 1
    lines.extend(("TER", "END"))
    return "\n".join(lines) + "\n"


def test_h3_rmsd_uses_framework_alignment():
    native_framework = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
    )
    native_h3 = np.asarray([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.asarray([7.0, -3.0, 2.0])
    predicted_framework = native_framework @ rotation + translation
    predicted_h3 = native_h3 @ rotation + translation
    assert framework_aligned_h3_rmsd(
        predicted_framework, native_framework, predicted_h3, native_h3
    ) == pytest.approx(0.0, abs=1e-12)


def test_complete_binder_evaluation_path_runs_tools_and_frozen_filters(tmp_path):
    generated = tmp_path / "generated.pdb"
    generated.write_text(_protein_pdb(["AAAAAAAA", "AAAAAAAA"]), encoding="utf-8")
    colabfold = tmp_path / "fake_colabfold.py"
    colabfold.write_text(
        """#!/usr/bin/env python3
import json, pathlib, sys
fasta = pathlib.Path(sys.argv[-2])
out = pathlib.Path(sys.argv[-1]); out.mkdir(parents=True, exist_ok=True)
sequence = ''.join(line.strip() for line in fasta.read_text().splitlines() if not line.startswith('>'))
chains = sequence.split(':')
lines=[]; serial=1
for ci, seq in enumerate(chains):
    chain=chr(ord('A')+ci)
    for ri, _ in enumerate(seq, 1):
        for name, dx, elem in [('N',0.,'N'),('CA',1.2,'C'),('C',2.4,'C'),('O',3.,'O')]:
            x=(ri-1)*3.8+dx; y=ci*3.
            lines.append(f'ATOM  {serial:5d} {name:^4s} ALA {chain}{ri:4d}    {x:8.3f}{y:8.3f}{0.:8.3f}  1.00 90.00          {elem:>2s}')
            serial += 1
    lines.append('TER')
(out/'result_rank_001.pdb').write_text('\\n'.join(lines)+'\\nEND\\n')
n=sum(map(len, chains))
scores={'plddt':[90.]*n,'pae':[[0.]*n for _ in range(n)],'ptm':0.8,'iptm':0.8}
(out/'result_scores_rank_001.json').write_text(json.dumps(scores))
""",
        encoding="utf-8",
    )
    rosetta = tmp_path / "fake_rosetta.py"
    rosetta.write_text(
        """#!/usr/bin/env python3
import pathlib, sys
path=pathlib.Path(sys.argv[sys.argv.index('-out:file:score_only')+1])
path.write_text('SCORE: dG_separated sc_value dSASA_int nres_int hbonds_int delta_unsatHbonds description\\nSCORE: -10 0.8 100 10 4 2 design\\n')
""",
        encoding="utf-8",
    )
    os.chmod(colabfold, 0o755)
    os.chmod(rosetta, 0o755)
    result = evaluate_protein_binder(
        generated,
        target_chains=["A"],
        binder_chain="B",
        output_dir=tmp_path / "evaluation",
        colabfold_executable=colabfold,
        rosetta_executable=rosetta,
    )
    assert result.passed
    assert result.metrics["self_consistency_rmsd"] == pytest.approx(0.0, abs=1e-6)
    assert result.metrics["binder_rmsd"] == pytest.approx(0.0, abs=1e-6)
    assert result.metrics["iptm"] == pytest.approx(0.8)
    aggregate = aggregate_binder_results([result, result], ["cluster-a", "cluster-a"])
    assert aggregate == {
        "in_silico_success_rate": 1.0,
        "diversity": 0.5,
        "cluster_level_success": 1.0,
    }


def test_pyrosetta_interface_wrapper_parses_native_json(tmp_path):
    analyzer = tmp_path / "fake_pyrosetta.py"
    analyzer.write_text(
        """import argparse, json, pathlib
p=argparse.ArgumentParser(); p.add_argument('complex'); p.add_argument('--interface'); p.add_argument('--output'); a=p.parse_args()
pathlib.Path(a.output).write_text(json.dumps({'rosetta_interface_delta_g':-5,'shape_complementarity':.7,'interface_dsasa':100,'interface_residue_count':9,'interface_hbond_count':4,'interface_unsatisfied_hbonds':2}))
""",
        encoding="utf-8",
    )
    result = run_pyrosetta_interface_analyzer(
        tmp_path / "complex.pdb",
        target_chains="A",
        binder_chains="B",
        python_executable=sys.executable,
        analyzer_script=analyzer,
    )
    assert result["rosetta_interface_delta_g"] == -5.0
    assert result["shape_complementarity"] == 0.7


def test_complete_rna_evaluation_path_runs_refold_alignment_and_dockq(tmp_path):
    designed = tmp_path / "designed_complex.pdb"
    designed.write_text(_rna_target_pdb(), encoding="utf-8")
    fasta = tmp_path / "rna.fasta"
    fasta.write_text(">rna\nAAA\n", encoding="utf-8")
    checkpoint = tmp_path / "rhofold.pt"
    checkpoint.write_bytes(b"frozen-test-checkpoint")
    rhofold = tmp_path / "fake_rhofold.py"
    rhofold.write_text(
        """import argparse, pathlib, numpy as np
p=argparse.ArgumentParser(); p.add_argument('--input_fas'); p.add_argument('--output_dir'); p.add_argument('--single_seq_pred'); p.add_argument('--relax_steps'); p.add_argument('--ckpt'); p.add_argument('--device', default=None); a=p.parse_args()
out=pathlib.Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
lines=[]; serial=1
for ri in range(1,4):
  for name,dx,elem in [('P',0.,'P'),("C4'",1.2,'C'),("C1'",2.4,'C')]:
    x=(ri-1)*3.8+dx
    lines.append(f'ATOM  {serial:5d} {name:^4s}   A A{ri:4d}    {x:8.3f}{10.:8.3f}{0.:8.3f}  1.00 90.00          {elem:>2s}')
    serial += 1
(out/'unrelaxed_model.pdb').write_text('\\n'.join(lines)+'\\nTER\\nEND\\n')
np.savez(out/'results.npz', plddt=np.array([0.9,0.8,1.0]))
""",
        encoding="utf-8",
    )
    usalign = tmp_path / "fake_usalign.py"
    usalign.write_text(
        "#!/usr/bin/env python3\nprint('Aligned length= 3, RMSD= 0.000, Seq_ID=n_identical/n_aligned= 1.0')\nprint('TM-score= 1.000')\n",
        encoding="utf-8",
    )
    dockq = tmp_path / "fake_dockq.py"
    dockq.write_text(
        """#!/usr/bin/env python3
import json, pathlib, sys
path=pathlib.Path(sys.argv[sys.argv.index('--json')+1]); path.write_text(json.dumps({'GlobalDockQ':0.9}))
""",
        encoding="utf-8",
    )
    os.chmod(usalign, 0o755)
    os.chmod(dockq, 0o755)
    result = evaluate_rna(
        fasta,
        designed,
        designed,
        target_chains=["A"],
        rna_chain="B",
        output_dir=tmp_path / "rna_evaluation",
        rhofold_python=sys.executable,
        rhofold_inference_script=rhofold,
        rhofold_checkpoint=checkpoint,
        usalign_executable=usalign,
        dockq_executable=dockq,
    )
    assert result.sctm == pytest.approx(1.0)
    assert result.scrmsd == pytest.approx(0.0)
    assert result.structure_confidence == pytest.approx(0.9)
    assert result.dockq == pytest.approx(0.9)


def test_real_usalign_rna_smoke_when_frozen_data_is_present():
    root = Path(__file__).resolve().parents[1]
    executable = root / "data/tools/usalign/USalign"
    catalog = root / "data/processed/v0/catalogs/ribocentre.jsonl"
    if not executable.is_file() or not catalog.is_file():
        pytest.skip("frozen real-data/tool snapshot is not part of a source-only checkout")
    row = json.loads(catalog.read_text(encoding="utf-8").splitlines()[0])
    structure = root / row["raw_paths"][0]
    result = run_usalign_rna(structure, structure, executable=executable)
    assert result["sctm"] == pytest.approx(1.0, abs=1e-6)
    assert result["scrmsd"] == pytest.approx(0.0, abs=1e-6)


def test_real_dockq_smoke_when_frozen_data_is_present():
    root = Path(__file__).resolve().parents[1]
    executable = root / "data/envs/evaluation/bin/DockQ"
    catalog = root / "data/processed/v0/catalogs/sabdab2.jsonl"
    if not executable.is_file() or not catalog.is_file():
        pytest.skip("frozen real-data/tool snapshot is not part of a source-only checkout")
    row = json.loads(catalog.read_text(encoding="utf-8").splitlines()[0])
    structure = root / row["raw_paths"][0]
    result = run_dockq(structure, structure, executable=executable)
    assert result["total_dockq"] == pytest.approx(1.0, abs=1e-6)


def test_complete_antibody_evaluation_handles_native_missing_light_chain():
    root = Path(__file__).resolve().parents[1]
    executable = root / "data/envs/evaluation/bin/DockQ"
    catalog = root / "data/processed/v0/catalogs/sabdab2.jsonl"
    if not executable.is_file() or not catalog.is_file():
        pytest.skip("frozen real-data/tool snapshot is not part of a source-only checkout")
    rows = (json.loads(line) for line in catalog.open(encoding="utf-8") if line.strip())
    row = next(
        value
        for value in rows
        if not any(chain["role"] == "antibody_framework" for chain in value["chains"])
    )
    structure = root / row["raw_paths"][0]
    heavy_chain = next(
        chain["chain_id"] for chain in row["chains"] if chain["role"] == "antibody_framework+cdr_h3"
    )
    result = evaluate_antibody_h3(
        structure,
        structure,
        heavy_chain=heavy_chain,
        light_chain=None,
        dockq_executable=executable,
    )
    assert result.h3_aar == pytest.approx(1.0)
    assert result.h3_rmsd == pytest.approx(0.0, abs=1e-6)
    assert result.dockq == pytest.approx(1.0, abs=1e-6)


def test_real_pyrosetta_interface_analyzer_when_official_install_is_present():
    root = Path(__file__).resolve().parents[1]
    python = root / "data/envs/pyrosetta312/bin/python"
    analyzer = root / "scripts/pyrosetta_interface_analyzer.py"
    structure = root / "data/raw/ppiref/ppiref50k/04/104l_A_B.pdb"
    if os.environ.get("NANODESIGN_RUN_HEAVY_EVAL") != "1":
        pytest.skip("set NANODESIGN_RUN_HEAVY_EVAL=1 to run licensed heavyweight tools")
    if not python.is_file() or not structure.is_file():
        pytest.skip("official PyRosetta/data snapshot is not part of a source-only checkout")
    result = run_pyrosetta_interface_analyzer(
        structure,
        target_chains="A",
        binder_chains="B",
        python_executable=python,
        analyzer_script=analyzer,
    )
    assert set(result) == {
        "rosetta_interface_delta_g",
        "shape_complementarity",
        "interface_dsasa",
        "interface_residue_count",
        "interface_hbond_count",
        "interface_unsatisfied_hbonds",
    }
    assert all(np.isfinite(value) for value in result.values())

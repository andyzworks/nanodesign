import json
from pathlib import Path

import numpy as np
import pytest

from nanodesign.v0.evaluators import framework_aligned_h3_rmsd, run_dockq, run_usalign_rna


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

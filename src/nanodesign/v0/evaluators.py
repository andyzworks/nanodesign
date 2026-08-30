"""Executable NanoDesign v0 evaluators.

These functions run the frozen third-party programs and parse their native outputs.  They
do not accept pre-computed metric dictionaries, which prevents a benchmark caller from
silently changing an alignment or substituting a different score implementation.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gemmi
import numpy as np

H3_IMGT_START = 105
H3_IMGT_END = 117
BACKBONE_ATOMS = ("N", "CA", "C", "O")

# Frozen from BindCraft's public default_filters.json.  The normalized iPAE cutoff
# 0.35 corresponds to 10.85 A on AlphaFold's 31 A PAE scale and is consistent with
# the RFdiffusion recommendation of interaction PAE <10 A.  These values are not
# fitted on NanoDesign's test split.
BINDCRAFT_DEFAULT_FILTER_SOURCE = (
    "https://github.com/martinpacesa/BindCraft/blob/main/settings_filters/default_filters.json"
)
BINDER_GENERATION_BUDGET = {
    "backbones_per_target": 1000,
    "sequences_per_backbone": 2,
    "source": "RFdiffusion public binder-design scale recommendation",
}
BINDCRAFT_DEFAULT_FILTERS: dict[str, tuple[str, float]] = {
    "plddt": (">=", 0.80),
    "ptm": (">=", 0.55),
    "iptm": (">=", 0.50),
    "ipae_normalized": ("<=", 0.35),
    "shape_complementarity": (">=", 0.60),
    "rosetta_interface_delta_g": ("<=", 0.0),
    "interface_dsasa": (">=", 1.0),
    "interface_residue_count": (">=", 7.0),
    "interface_hbond_count": (">=", 3.0),
    "interface_unsatisfied_hbonds": ("<=", 4.0),
    "hotspot_rmsd": ("<=", 6.0),
    "binder_plddt": (">=", 0.80),
    "binder_rmsd": ("<=", 3.5),
}


class EvaluationError(RuntimeError):
    """Raised when a real evaluator fails or emits an unsupported result."""


def binder_passes_frozen_filters(metrics: dict[str, float]) -> bool:
    """Apply the public BindCraft default filter without caller-supplied thresholds."""

    missing = set(BINDCRAFT_DEFAULT_FILTERS) - set(metrics)
    if missing:
        raise EvaluationError(f"binder evaluation is missing {sorted(missing)}")
    for name, (operation, threshold) in BINDCRAFT_DEFAULT_FILTERS.items():
        value = float(metrics[name])
        if operation == ">=" and value < threshold:
            return False
        if operation == "<=" and value > threshold:
            return False
    return True


def binder_success_rate(records: Sequence[dict[str, float]]) -> float:
    if not records:
        raise ValueError("binder success rate requires at least one design")
    return float(np.mean([binder_passes_frozen_filters(record) for record in records]))


def _run(command: Sequence[str], *, timeout: int = 3600) -> str:
    try:
        result = subprocess.run(
            [str(value) for value in command],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise EvaluationError(f"evaluator failed: {' '.join(map(str, command))}") from error
    return result.stdout + result.stderr


def _kabsch(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("Kabsch inputs must have matching [N, 3] shapes")
    if len(mobile) < 3:
        raise ValueError("at least three framework atoms are required for alignment")
    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (reference - reference_center)
    left, _, right = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right))
    rotation = left @ correction @ right
    translation = reference_center - mobile_center @ rotation
    return rotation, translation


def framework_aligned_h3_rmsd(
    predicted_framework: np.ndarray,
    native_framework: np.ndarray,
    predicted_h3: np.ndarray,
    native_h3: np.ndarray,
) -> float:
    """Align the fixed framework, then calculate H3 backbone RMSD.

    The alignment is deliberately part of this evaluator rather than a caller option.
    """

    predicted_framework = np.asarray(predicted_framework, dtype=np.float64)
    native_framework = np.asarray(native_framework, dtype=np.float64)
    predicted_h3 = np.asarray(predicted_h3, dtype=np.float64)
    native_h3 = np.asarray(native_h3, dtype=np.float64)
    if predicted_h3.shape != native_h3.shape or predicted_h3.ndim != 2:
        raise ValueError("H3 backbone coordinates must have matching [N, 3] shapes")
    rotation, translation = _kabsch(predicted_framework, native_framework)
    aligned_h3 = predicted_h3 @ rotation + translation
    return float(np.sqrt(np.mean(np.sum((aligned_h3 - native_h3) ** 2, axis=1))))


def _model(path: str | Path) -> gemmi.Model:
    structure = gemmi.read_structure(str(path))
    if not structure:
        raise EvaluationError(f"no coordinate model in {path}")
    return structure[0]


def _residue_key(residue: gemmi.Residue) -> tuple[int, str]:
    insertion = residue.seqid.icode
    return int(residue.seqid.num), "" if insertion in {" ", "\x00", "?", "."} else insertion


def _chain(model: gemmi.Model, chain_id: str) -> gemmi.Chain:
    chain = model.find_chain(chain_id)
    if chain is None:
        raise EvaluationError(f"chain {chain_id!r} not present")
    return chain


def _atom_map(chain: gemmi.Chain) -> dict[tuple[int, str, str], np.ndarray]:
    result: dict[tuple[int, str, str], np.ndarray] = {}
    for residue in chain:
        number, insertion = _residue_key(residue)
        for atom_name in BACKBONE_ATOMS:
            atom = residue.find_atom(atom_name, "*")
            if atom is not None:
                result[(number, insertion, atom_name)] = np.asarray(
                    [atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64
                )
    return result


def _residue_letters(chain: gemmi.Chain) -> dict[tuple[int, str], str]:
    result = {}
    for residue in chain:
        info = gemmi.find_tabulated_residue(residue.name)
        if info.is_amino_acid():
            result[_residue_key(residue)] = info.one_letter_code
    return result


@dataclass(frozen=True)
class AntibodyH3Result:
    h3_aar: float
    h3_rmsd: float
    dockq: float


def evaluate_antibody_h3(
    predicted_complex: str | Path,
    native_complex: str | Path,
    *,
    heavy_chain: str,
    light_chain: str | None = None,
    dockq_executable: str | Path = "DockQ",
) -> AntibodyH3Result:
    """Run the fixed v0 H3 protocol and DockQ v2 on an antibody complex."""

    predicted_model, native_model = _model(predicted_complex), _model(native_complex)
    predicted_heavy = _chain(predicted_model, heavy_chain)
    native_heavy = _chain(native_model, heavy_chain)
    predicted_atoms = _atom_map(predicted_heavy)
    native_atoms = _atom_map(native_heavy)
    common = sorted(set(predicted_atoms) & set(native_atoms))
    framework_keys = [key for key in common if not H3_IMGT_START <= key[0] <= H3_IMGT_END]
    h3_keys = [key for key in common if H3_IMGT_START <= key[0] <= H3_IMGT_END]
    if light_chain is not None:
        predicted_light = _atom_map(_chain(predicted_model, light_chain))
        native_light = _atom_map(_chain(native_model, light_chain))
        light_keys = sorted(set(predicted_light) & set(native_light))
        predicted_framework = [predicted_atoms[key] for key in framework_keys] + [
            predicted_light[key] for key in light_keys
        ]
        native_framework = [native_atoms[key] for key in framework_keys] + [
            native_light[key] for key in light_keys
        ]
    else:
        predicted_framework = [predicted_atoms[key] for key in framework_keys]
        native_framework = [native_atoms[key] for key in framework_keys]
    if len(h3_keys) < 4 or len(predicted_framework) < 12:
        raise EvaluationError("insufficient common framework or H3 backbone atoms")
    h3_rmsd = framework_aligned_h3_rmsd(
        np.asarray(predicted_framework),
        np.asarray(native_framework),
        np.asarray([predicted_atoms[key] for key in h3_keys]),
        np.asarray([native_atoms[key] for key in h3_keys]),
    )
    predicted_letters = _residue_letters(predicted_heavy)
    native_letters = _residue_letters(native_heavy)
    h3_residues = sorted(
        key
        for key in set(predicted_letters) & set(native_letters)
        if H3_IMGT_START <= key[0] <= H3_IMGT_END
    )
    if not h3_residues:
        raise EvaluationError("no common IMGT H3 residues")
    h3_aar = float(np.mean([predicted_letters[key] == native_letters[key] for key in h3_residues]))
    dockq = run_dockq(predicted_complex, native_complex, executable=dockq_executable)["total_dockq"]
    return AntibodyH3Result(h3_aar=h3_aar, h3_rmsd=h3_rmsd, dockq=dockq)


def run_dockq(
    model: str | Path,
    native: str | Path,
    *,
    executable: str | Path = "DockQ",
    mapping: str | None = None,
) -> dict[str, Any]:
    """Execute DockQ v2 and return its JSON result, including RNA interfaces."""

    with tempfile.TemporaryDirectory(prefix="nanodesign-dockq-") as directory:
        output = Path(directory) / "dockq.json"
        command = [str(executable), str(model), str(native), "--json", str(output)]
        if mapping:
            command.extend(["--mapping", mapping])
        _run(command)
        value = json.loads(output.read_text(encoding="utf-8"))
    total = value.get("GlobalDockQ", value.get("total_dockq", value.get("DockQ")))
    if total is None:
        # DockQ 2.x JSON stores per-interface scores plus a top-level total in
        # different keys across patch releases.  Average only as a last, explicit
        # compatibility path for a result containing interface dictionaries.
        scores = []
        containers = []
        if isinstance(value, dict):
            containers.extend(value.values())
            best_result = value.get("best_result")
            if isinstance(best_result, dict):
                containers.extend(best_result.values())
        for item in containers:
            if isinstance(item, dict) and "DockQ" in item:
                scores.append(float(item["DockQ"]))
        if not scores:
            raise EvaluationError("DockQ JSON does not contain a DockQ score")
        total = float(np.mean(scores))
    return {"total_dockq": float(total), "raw": value}


def run_usalign_rna(
    predicted_rna: str | Path,
    reference_rna: str | Path,
    *,
    executable: str | Path = "USalign",
) -> dict[str, float]:
    """Run sequence-correspondence RNA US-align and parse scTM/scRMSD."""

    output = _run(
        [str(executable), str(predicted_rna), str(reference_rna), "-mol", "RNA", "-TMscore", "5"]
    )
    rmsd_match = re.search(r"Aligned length=\s*\d+,\s*RMSD=\s*([0-9.eE+-]+)", output)
    tm_scores = re.findall(r"TM-score=\s*([0-9.eE+-]+)", output)
    if rmsd_match is None or not tm_scores:
        raise EvaluationError("could not parse US-align RNA output")
    # With model first and reference second, the final value is normalized by the
    # reference length; this is NanoDesign's frozen scTM definition.
    return {"sctm": float(tm_scores[-1]), "scrmsd": float(rmsd_match.group(1))}


def run_rosetta_interface_analyzer(
    complex_path: str | Path,
    *,
    target_chains: str,
    binder_chains: str,
    executable: str | Path = "InterfaceAnalyzer.linuxgccrelease",
) -> dict[str, float]:
    """Execute Rosetta InterfaceAnalyzer and parse physical interface metrics."""

    with tempfile.TemporaryDirectory(prefix="nanodesign-rosetta-") as directory:
        scorefile = Path(directory) / "interface.sc"
        _run(
            [
                str(executable),
                "-s",
                str(complex_path),
                "-interface",
                f"{target_chains}_{binder_chains}",
                "-compute_packstat",
                "true",
                "-pack_separated",
                "true",
                "-out:file:score_only",
                str(scorefile),
            ]
        )
        lines = [line.split() for line in scorefile.read_text(encoding="utf-8").splitlines()]
    headers = next(
        (line[1:] for line in lines if line and line[0] == "SCORE:" and "description" in line), None
    )
    values = next(
        (
            line[1:]
            for line in reversed(lines)
            if line and line[0] == "SCORE:" and "description" not in line
        ),
        None,
    )
    if headers is None or values is None:
        raise EvaluationError("Rosetta InterfaceAnalyzer scorefile is malformed")
    row = dict(zip(headers, values, strict=False))
    aliases = {
        "rosetta_interface_delta_g": "dG_separated",
        "shape_complementarity": "sc_value",
        "interface_dsasa": "dSASA_int",
        "interface_residue_count": "nres_int",
        "interface_hbond_count": "hbonds_int",
        "interface_unsatisfied_hbonds": "delta_unsatHbonds",
    }
    missing = [source for source in aliases.values() if source not in row]
    if missing:
        raise EvaluationError(f"Rosetta scorefile is missing {missing}")
    return {destination: float(row[source]) for destination, source in aliases.items()}


def run_colabfold_multimer(
    fasta: str | Path,
    output_dir: str | Path,
    *,
    executable: str | Path = "colabfold_batch",
) -> Path:
    """Run the frozen AF2-multimer verifier used for protein binder evaluation."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(executable),
            "--model-type",
            "alphafold2_multimer_v3",
            "--num-models",
            "5",
            "--num-recycle",
            "3",
            str(fasta),
            str(output_dir),
        ],
        timeout=24 * 3600,
    )
    ranked = sorted(output_dir.glob("*rank_001*.pdb"))
    if not ranked:
        raise EvaluationError("ColabFold did not produce a rank-001 complex")
    return ranked[0]


def run_rhofold_plus(
    fasta: str | Path,
    output_dir: str | Path,
    *,
    python_executable: str | Path,
    inference_script: str | Path,
) -> Path:
    """Run RhoFold+ as the independent RNA sequence refolder."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(python_executable),
            str(inference_script),
            "--input_fas",
            str(fasta),
            "--output_dir",
            str(output_dir),
        ],
        timeout=24 * 3600,
    )
    predictions = sorted(output_dir.rglob("*.pdb"))
    if not predictions:
        raise EvaluationError("RhoFold+ did not produce a PDB prediction")
    return predictions[0]

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
from collections import defaultdict
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


@dataclass(frozen=True)
class ProteinBinderResult:
    metrics: dict[str, float]
    passed: bool


@dataclass(frozen=True)
class RnaEvaluationResult:
    sctm: float
    scrmsd: float
    structure_confidence: float
    dockq: float


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


def cluster_binder_sequences(
    sequences: Sequence[str],
    *,
    executable: str | Path = "mmseqs",
) -> list[str]:
    """Assign generated binders to the frozen protein30/cov80 diversity clusters."""

    if not sequences:
        raise ValueError("binder diversity requires at least one sequence")
    with tempfile.TemporaryDirectory(prefix="nanodesign-binder-clusters-") as directory:
        root = Path(directory)
        fasta = root / "binders.fasta"
        fasta.write_text(
            "".join(f">design_{index}\n{sequence}\n" for index, sequence in enumerate(sequences)),
            encoding="utf-8",
        )
        prefix = root / "clusters"
        _run(
            [
                str(executable),
                "easy-cluster",
                str(fasta),
                str(prefix),
                str(root / "tmp"),
                "--min-seq-id",
                "0.30",
                "-c",
                "0.80",
                "--cov-mode",
                "0",
                "--cluster-mode",
                "2",
            ]
        )
        cluster_file = root / "clusters_cluster.tsv"
        if not cluster_file.is_file():
            raise EvaluationError("MMseqs2 did not produce binder cluster assignments")
        assignments = {}
        for line in cluster_file.read_text(encoding="utf-8").splitlines():
            representative, member = line.split("\t")[:2]
            assignments[member] = representative
    missing = [
        f"design_{index}" for index in range(len(sequences)) if f"design_{index}" not in assignments
    ]
    if missing:
        raise EvaluationError(f"MMseqs2 omitted generated binders: {missing}")
    return [assignments[f"design_{index}"] for index in range(len(sequences))]


def aggregate_binder_results(
    results: Sequence[ProteinBinderResult], cluster_ids: Sequence[str]
) -> dict[str, float]:
    """Compute success, diversity, and success-by-sequence-cluster for one target."""

    if not results or len(results) != len(cluster_ids):
        raise ValueError("binder results and cluster IDs must have the same non-zero length")
    by_cluster: dict[str, list[bool]] = defaultdict(list)
    for result, cluster_id in zip(results, cluster_ids, strict=True):
        by_cluster[cluster_id].append(result.passed)
    return {
        "in_silico_success_rate": float(np.mean([result.passed for result in results])),
        "diversity": len(by_cluster) / len(results),
        "cluster_level_success": float(
            np.mean([any(cluster_results) for cluster_results in by_cluster.values()])
        ),
    }


def _run(command: Sequence[str], *, timeout: int = 3600) -> str:
    try:
        result = subprocess.run(
            [str(value) for value in command],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as error:
        details = "\n".join(part for part in (error.stdout, error.stderr) if part)
        if len(details) > 4000:
            details = details[-4000:]
        raise EvaluationError(
            f"evaluator failed: {' '.join(map(str, command))}\n{details}"
        ) from error
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


def _ordered_backbone(model: gemmi.Model, chain_ids: Sequence[str]) -> np.ndarray:
    coordinates = []
    for chain_id in chain_ids:
        for residue in _chain(model, chain_id):
            info = gemmi.find_tabulated_residue(residue.name)
            if not info.is_amino_acid():
                continue
            for atom_name in BACKBONE_ATOMS:
                atom = residue.find_atom(atom_name, "*")
                if atom is None:
                    raise EvaluationError(
                        f"chain {chain_id!r} residue {residue.seqid} lacks {atom_name}"
                    )
                coordinates.append([atom.pos.x, atom.pos.y, atom.pos.z])
    if len(coordinates) < 3:
        raise EvaluationError("fewer than three protein backbone atoms were resolved")
    return np.asarray(coordinates, dtype=np.float64)


def _protein_sequence(chain: gemmi.Chain) -> str:
    letters = []
    for residue in chain:
        info = gemmi.find_tabulated_residue(residue.name)
        if info.is_amino_acid() and info.one_letter_code not in {"", "X"}:
            letters.append(info.one_letter_code)
    if not letters:
        raise EvaluationError(f"chain {chain.name!r} has no standard amino-acid sequence")
    return "".join(letters)


def target_aligned_binder_rmsd(
    designed_complex: str | Path,
    predicted_complex: str | Path,
    *,
    designed_target_chains: Sequence[str],
    designed_binder_chain: str,
    predicted_target_chains: Sequence[str],
    predicted_binder_chain: str,
) -> float:
    """Align predicted target backbone to the design, then score binder backbone RMSD."""

    designed, predicted = _model(designed_complex), _model(predicted_complex)
    designed_target = _ordered_backbone(designed, designed_target_chains)
    predicted_target = _ordered_backbone(predicted, predicted_target_chains)
    designed_binder = _ordered_backbone(designed, [designed_binder_chain])
    predicted_binder = _ordered_backbone(predicted, [predicted_binder_chain])
    if designed_target.shape != predicted_target.shape:
        raise EvaluationError("designed and predicted target backbones have different shapes")
    if designed_binder.shape != predicted_binder.shape:
        raise EvaluationError("designed and predicted binder backbones have different shapes")
    rotation, translation = _kabsch(predicted_target, designed_target)
    aligned_binder = predicted_binder @ rotation + translation
    return float(np.sqrt(np.mean(np.sum((aligned_binder - designed_binder) ** 2, axis=1))))


def aligned_binder_rmsd(
    designed_complex: str | Path,
    predicted_binder: str | Path,
    *,
    designed_binder_chain: str,
    predicted_binder_chain: str,
) -> float:
    """Score binder conformation after optimal binder-backbone alignment."""

    designed = _ordered_backbone(_model(designed_complex), [designed_binder_chain])
    predicted = _ordered_backbone(_model(predicted_binder), [predicted_binder_chain])
    if designed.shape != predicted.shape:
        raise EvaluationError("designed and independently folded binder shapes differ")
    rotation, translation = _kabsch(predicted, designed)
    aligned = predicted @ rotation + translation
    return float(np.sqrt(np.mean(np.sum((aligned - designed) ** 2, axis=1))))


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
    predicted_h3_keys = {key for key in predicted_atoms if H3_IMGT_START <= key[0] <= H3_IMGT_END}
    native_h3_keys = {key for key in native_atoms if H3_IMGT_START <= key[0] <= H3_IMGT_END}
    if predicted_h3_keys != native_h3_keys:
        raise EvaluationError(
            "predicted H3 backbone is not residue/atom-complete against reference"
        )
    predicted_framework_keys = {
        key for key in predicted_atoms if not H3_IMGT_START <= key[0] <= H3_IMGT_END
    }
    native_framework_keys = {
        key for key in native_atoms if not H3_IMGT_START <= key[0] <= H3_IMGT_END
    }
    if predicted_framework_keys != native_framework_keys:
        raise EvaluationError("predicted heavy framework backbone differs from fixed reference")
    framework_keys = sorted(native_framework_keys)
    h3_keys = sorted(native_h3_keys)
    if light_chain is not None:
        predicted_light = _atom_map(_chain(predicted_model, light_chain))
        native_light = _atom_map(_chain(native_model, light_chain))
        if set(predicted_light) != set(native_light):
            raise EvaluationError("predicted light framework backbone differs from fixed reference")
        light_keys = sorted(native_light)
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
    predicted_h3_residues = {
        key for key in predicted_letters if H3_IMGT_START <= key[0] <= H3_IMGT_END
    }
    native_h3_residues = {key for key in native_letters if H3_IMGT_START <= key[0] <= H3_IMGT_END}
    if predicted_h3_residues != native_h3_residues:
        raise EvaluationError("predicted H3 sequence positions differ from reference")
    h3_residues = sorted(native_h3_residues)
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


def run_colabfold_prediction(
    fasta: str | Path,
    output_dir: str | Path,
    *,
    executable: str | Path = "colabfold_batch",
    model_type: str = "alphafold2_multimer_v3",
) -> tuple[Path, Path]:
    """Run ColabFold and return the rank-1 structure and its native score JSON."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(executable),
            "--model-type",
            model_type,
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
    score_files = sorted(output_dir.glob("*scores_rank_001*.json"))
    if not score_files:
        score_files = sorted(output_dir.glob("*rank_001*.json"))
    if not score_files:
        raise EvaluationError("ColabFold did not produce rank-001 score JSON")
    return ranked[0], score_files[0]


def run_colabfold_multimer(
    fasta: str | Path,
    output_dir: str | Path,
    *,
    executable: str | Path = "colabfold_batch",
) -> Path:
    """Compatibility wrapper returning the frozen AF2-multimer rank-1 structure."""

    structure, _ = run_colabfold_prediction(fasta, output_dir, executable=executable)
    return structure


def parse_colabfold_scores(
    score_json: str | Path,
    *,
    chain_lengths: Sequence[int],
    binder_chain_index: int,
) -> dict[str, float]:
    """Parse ColabFold confidence without accepting caller-provided metric values."""

    value = json.loads(Path(score_json).read_text(encoding="utf-8"))
    plddt = np.asarray(value.get("plddt"), dtype=np.float64)
    pae = np.asarray(value.get("pae"), dtype=np.float64)
    total_length = sum(chain_lengths)
    if plddt.shape != (total_length,) or pae.shape != (total_length, total_length):
        raise EvaluationError("ColabFold score arrays do not match FASTA chain lengths")
    if not 0 <= binder_chain_index < len(chain_lengths):
        raise ValueError("binder_chain_index is outside the FASTA chain list")
    offsets = np.cumsum([0, *chain_lengths])
    binder = slice(offsets[binder_chain_index], offsets[binder_chain_index + 1])
    target_ranges = [
        np.arange(offsets[index], offsets[index + 1])
        for index in range(len(chain_lengths))
        if index != binder_chain_index
    ]
    if not target_ranges:
        ptm = value.get("ptm")
        if ptm is None:
            raise EvaluationError("ColabFold monomer score JSON lacks pTM")
        return {
            "plddt": float(plddt.mean() / 100.0),
            "binder_plddt": float(plddt.mean() / 100.0),
            "ptm": float(ptm),
            "iptm": float(ptm),
            "ipae_normalized": 0.0,
        }
    target_indices = np.concatenate(target_ranges)
    binder_indices = np.arange(binder.start, binder.stop)
    cross_pae = np.concatenate(
        (
            pae[np.ix_(target_indices, binder_indices)].ravel(),
            pae[np.ix_(binder_indices, target_indices)].ravel(),
        )
    )
    ptm = value.get("ptm")
    iptm = value.get("iptm", value.get("ipTM"))
    if ptm is None or iptm is None or cross_pae.size == 0:
        raise EvaluationError("ColabFold score JSON lacks pTM, ipTM, or interface PAE")
    return {
        "plddt": float(plddt.mean() / 100.0),
        "binder_plddt": float(plddt[binder].mean() / 100.0),
        "ptm": float(ptm),
        "iptm": float(iptm),
        "ipae_normalized": float(cross_pae.mean() / 31.0),
    }


def count_bindcraft_clashes(structure_path: str | Path, *, threshold: float = 2.4) -> int:
    """Count inter-chain heavy-atom pairs below BindCraft's 2.4 A cutoff."""

    model = _model(structure_path)
    atoms: list[tuple[str, np.ndarray]] = []
    for chain in model:
        for residue in chain:
            for atom in residue:
                if atom.element.name == "H":
                    continue
                atoms.append(
                    (
                        chain.name,
                        np.asarray([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64),
                    )
                )
    clashes = 0
    squared = threshold**2
    for left in range(len(atoms)):
        chain_left, position_left = atoms[left]
        for chain_right, position_right in atoms[left + 1 :]:
            if (
                chain_left != chain_right
                and np.sum((position_left - position_right) ** 2) < squared
            ):
                clashes += 1
    return clashes


def evaluate_protein_binder(
    generated_complex: str | Path,
    *,
    target_chains: Sequence[str],
    binder_chain: str,
    output_dir: str | Path,
    colabfold_executable: str | Path = "colabfold_batch",
    rosetta_executable: str | Path = "InterfaceAnalyzer.linuxgccrelease",
) -> ProteinBinderResult:
    """Run the frozen NanoDesign v0 binder protocol from structure to pass/fail."""

    if not target_chains or binder_chain in target_chains:
        raise ValueError("target_chains must be non-empty and exclude the binder chain")
    generated_model = _model(generated_complex)
    chain_order = [*target_chains, binder_chain]
    sequences = [_protein_sequence(_chain(generated_model, chain_id)) for chain_id in chain_order]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    complex_fasta = output_dir / "complex.fasta"
    complex_fasta.write_text(">nanodesign_complex\n" + ":".join(sequences) + "\n", encoding="utf-8")
    binder_fasta = output_dir / "binder.fasta"
    binder_fasta.write_text(">nanodesign_binder\n" + sequences[-1] + "\n", encoding="utf-8")
    predicted_complex, complex_scores = run_colabfold_prediction(
        complex_fasta,
        output_dir / "complex_prediction",
        executable=colabfold_executable,
        model_type="alphafold2_multimer_v3",
    )
    predicted_binder, binder_scores = run_colabfold_prediction(
        binder_fasta,
        output_dir / "binder_prediction",
        executable=colabfold_executable,
        model_type="alphafold2_ptm",
    )
    predicted_chain_ids = [chr(ord("A") + index) for index in range(len(chain_order))]
    confidence = parse_colabfold_scores(
        complex_scores,
        chain_lengths=[len(sequence) for sequence in sequences],
        binder_chain_index=len(sequences) - 1,
    )
    binder_confidence = parse_colabfold_scores(
        binder_scores,
        chain_lengths=[len(sequences[-1])],
        binder_chain_index=0,
    )
    self_consistency = target_aligned_binder_rmsd(
        generated_complex,
        predicted_complex,
        designed_target_chains=target_chains,
        designed_binder_chain=binder_chain,
        predicted_target_chains=predicted_chain_ids[:-1],
        predicted_binder_chain=predicted_chain_ids[-1],
    )
    binder_rmsd = aligned_binder_rmsd(
        generated_complex,
        predicted_binder,
        designed_binder_chain=binder_chain,
        predicted_binder_chain="A",
    )
    rosetta = run_rosetta_interface_analyzer(
        predicted_complex,
        target_chains="".join(predicted_chain_ids[:-1]),
        binder_chains=predicted_chain_ids[-1],
        executable=rosetta_executable,
    )
    metrics = {
        **confidence,
        **rosetta,
        "binder_plddt": binder_confidence["plddt"],
        "binder_rmsd": binder_rmsd,
        "hotspot_rmsd": self_consistency,
        "interface_confidence": confidence["iptm"],
        "self_consistency_rmsd": self_consistency,
        "clashes": float(count_bindcraft_clashes(predicted_complex)),
    }
    return ProteinBinderResult(metrics=metrics, passed=binder_passes_frozen_filters(metrics))


def run_rhofold_plus(
    fasta: str | Path,
    output_dir: str | Path,
    *,
    python_executable: str | Path,
    inference_script: str | Path,
    checkpoint: str | Path | None = None,
    device: str | None = None,
) -> Path:
    """Run RhoFold+ as the independent RNA sequence refolder."""

    inference_script = Path(inference_script)
    if not inference_script.is_file():
        raise EvaluationError(f"RhoFold+ inference script does not exist: {inference_script}")
    if checkpoint is not None and not Path(checkpoint).is_file():
        raise EvaluationError(f"RhoFold+ checkpoint does not exist: {checkpoint}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(python_executable),
        str(inference_script),
        "--input_fas",
        str(fasta),
        "--output_dir",
        str(output_dir),
        "--single_seq_pred",
        "True",
        "--relax_steps",
        "0",
    ]
    if checkpoint is not None:
        command.extend(["--ckpt", str(checkpoint)])
    if device is not None:
        command.extend(["--device", device])
    _run(command, timeout=24 * 3600)
    predictions = sorted(output_dir.rglob("*.pdb"))
    if not predictions:
        raise EvaluationError("RhoFold+ did not produce a PDB prediction")
    return predictions[0]


def parse_rhofold_confidence(output_dir: str | Path, predicted_rna: str | Path) -> float:
    """Read RhoFold's pLDDT output, preferring the native results.npz array."""

    archives = sorted(Path(output_dir).rglob("results.npz"))
    if archives:
        with np.load(archives[0], allow_pickle=False) as value:
            if "plddt" not in value:
                raise EvaluationError("RhoFold+ results.npz lacks plddt")
            confidence = float(np.asarray(value["plddt"], dtype=np.float64).mean())
    else:
        factors = [
            atom.b_iso
            for chain in _model(predicted_rna)
            for residue in chain
            for atom in residue
            if atom.element.name != "H"
        ]
        if not factors:
            raise EvaluationError("RhoFold+ output contains no confidence values")
        confidence = float(np.mean(factors) / 100.0)
    if confidence > 1.0:
        confidence /= 100.0
    if not 0.0 <= confidence <= 1.0:
        raise EvaluationError(f"RhoFold+ confidence is outside [0,1]: {confidence}")
    return confidence


def _ordered_rna_alignment_atoms(model: gemmi.Model, chain_id: str) -> np.ndarray:
    coordinates = []
    for residue in _chain(model, chain_id):
        residue_coordinates = []
        # Terminal phosphates are often absent experimentally; C4'/C1' are the
        # sequence-corresponding sugar-frame atoms present in the frozen filter.
        for atom_name in ("C4'", "C1'"):
            atom = residue.find_atom(atom_name, "*")
            if atom is None:
                raise EvaluationError(
                    f"RNA chain {chain_id!r} residue {residue.seqid} lacks {atom_name}"
                )
            residue_coordinates.append([atom.pos.x, atom.pos.y, atom.pos.z])
        coordinates.extend(residue_coordinates)
    if len(coordinates) < 3:
        raise EvaluationError("RNA alignment requires at least one complete residue")
    return np.asarray(coordinates, dtype=np.float64)


def _write_selected_chains(
    source: str | Path, destination: str | Path, chain_ids: Sequence[str]
) -> None:
    structure = gemmi.read_structure(str(source)).clone()
    keep = set(chain_ids)
    for chain_name in [chain.name for chain in structure[0]]:
        if chain_name not in keep:
            structure[0].remove_chain(chain_name)
    if {chain.name for chain in structure[0]} != keep:
        raise EvaluationError(f"could not select every requested chain {sorted(keep)}")
    structure.write_pdb(str(destination))


def _build_rna_target_complex(
    designed_complex: str | Path,
    predicted_rna: str | Path,
    destination: str | Path,
    *,
    target_chains: Sequence[str],
    designed_rna_chain: str,
) -> None:
    designed_structure = gemmi.read_structure(str(designed_complex)).clone()
    predicted_structure = gemmi.read_structure(str(predicted_rna)).clone()
    if not predicted_structure or len(predicted_structure[0]) != 1:
        raise EvaluationError("RhoFold+ prediction must contain exactly one RNA chain")
    predicted_chain = predicted_structure[0][0]
    designed_coordinates = _ordered_rna_alignment_atoms(designed_structure[0], designed_rna_chain)
    predicted_coordinates = _ordered_rna_alignment_atoms(
        predicted_structure[0], predicted_chain.name
    )
    if designed_coordinates.shape != predicted_coordinates.shape:
        raise EvaluationError("RhoFold+ and designed RNA alignment atoms have different shapes")
    rotation, translation = _kabsch(predicted_coordinates, designed_coordinates)
    for residue in predicted_chain:
        for atom in residue:
            position = np.asarray([atom.pos.x, atom.pos.y, atom.pos.z]) @ rotation + translation
            atom.pos = gemmi.Position(*position)
    keep = set(target_chains)
    for chain_name in [chain.name for chain in designed_structure[0]]:
        if chain_name not in keep:
            designed_structure[0].remove_chain(chain_name)
    if {chain.name for chain in designed_structure[0]} != keep:
        raise EvaluationError("designed complex is missing a requested target chain")
    predicted_chain.name = designed_rna_chain
    designed_structure[0].add_chain(predicted_chain.clone())
    designed_structure.write_pdb(str(destination))


def evaluate_rna(
    generated_fasta: str | Path,
    designed_complex: str | Path,
    native_complex: str | Path,
    *,
    target_chains: Sequence[str],
    rna_chain: str,
    output_dir: str | Path,
    rhofold_python: str | Path,
    rhofold_inference_script: str | Path,
    rhofold_checkpoint: str | Path,
    usalign_executable: str | Path = "USalign",
    dockq_executable: str | Path = "DockQ",
    rhofold_device: str | None = None,
) -> RnaEvaluationResult:
    """Run RhoFold+ -> US-align -> target-complex DockQ without hand-filled metrics."""

    if not target_chains or rna_chain in target_chains:
        raise ValueError("target chains must be non-empty and exclude the RNA chain")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "rhofold_prediction"
    predicted_rna = run_rhofold_plus(
        generated_fasta,
        prediction_dir,
        python_executable=rhofold_python,
        inference_script=rhofold_inference_script,
        checkpoint=rhofold_checkpoint,
        device=rhofold_device,
    )
    designed_rna = output_dir / "designed_rna.pdb"
    _write_selected_chains(designed_complex, designed_rna, [rna_chain])
    structural = run_usalign_rna(predicted_rna, designed_rna, executable=usalign_executable)
    predicted_complex = output_dir / "predicted_rna_target_complex.pdb"
    _build_rna_target_complex(
        designed_complex,
        predicted_rna,
        predicted_complex,
        target_chains=target_chains,
        designed_rna_chain=rna_chain,
    )
    dockq = run_dockq(predicted_complex, native_complex, executable=dockq_executable)["total_dockq"]
    return RnaEvaluationResult(
        sctm=structural["sctm"],
        scrmsd=structural["scrmsd"],
        structure_confidence=parse_rhofold_confidence(prediction_dir, predicted_rna),
        dockq=dockq,
    )

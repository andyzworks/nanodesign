#!/usr/bin/env python3
"""Run geometry-level Perfect/Perturbed/Broken checks for true evaluators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gemmi
import numpy as np

from nanodesign.v0.evaluators import (
    BINDCRAFT_DEFAULT_FILTERS,
    H3_IMGT_END,
    H3_IMGT_START,
    binder_passes_frozen_filters,
    evaluate_antibody_h3,
    run_dockq,
    run_usalign_rna,
    target_aligned_binder_rmsd,
)


def _mutate_structure(
    source: Path,
    destination: Path,
    *,
    chain_id: str,
    residue_selector,
    noise_scale: float,
    translation: np.ndarray,
    mutation_stride: int | None,
    seed: int,
) -> None:
    structure = gemmi.read_structure(str(source)).clone()
    chain = structure[0].find_chain(chain_id)
    if chain is None:
        raise ValueError(f"chain {chain_id!r} not present in {source}")
    rng = np.random.default_rng(seed)
    selected_index = 0
    for residue in chain:
        if not residue_selector(residue):
            continue
        if mutation_stride is not None and selected_index % mutation_stride == 0:
            residue.name = "GLY" if residue.name != "GLY" else "ALA"
        selected_index += 1
        for atom in residue:
            delta = translation + rng.normal(0.0, noise_scale, size=3)
            atom.pos = gemmi.Position(
                atom.pos.x + float(delta[0]),
                atom.pos.y + float(delta[1]),
                atom.pos.z + float(delta[2]),
            )
    if selected_index == 0:
        raise ValueError(f"no selected residues in chain {chain_id!r}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    structure.write_pdb(str(destination))


def _copy_as_pdb(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    gemmi.read_structure(str(source)).write_pdb(str(destination))


def _binder_filter_sanity() -> dict[str, object]:
    perfect = {}
    broken = {}
    for name, (operation, threshold) in BINDCRAFT_DEFAULT_FILTERS.items():
        margin = max(abs(threshold) * 0.1, 0.1)
        if operation == ">=":
            perfect[name] = threshold + margin
            broken[name] = threshold - margin
        else:
            perfect[name] = threshold - margin
            broken[name] = threshold + margin
    return {
        "perfect_passes": binder_passes_frozen_filters(perfect),
        "broken_rejected": not binder_passes_frozen_filters(broken),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dockq", type=Path, default=Path("data/envs/evaluation/bin/DockQ"))
    parser.add_argument("--usalign", type=Path, default=Path("data/tools/usalign/USalign"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dockq = args.dockq if args.dockq.is_absolute() else root / args.dockq
    usalign = args.usalign if args.usalign.is_absolute() else root / args.usalign

    # Binder: isolate the designed chain while the target is held exactly fixed.
    binder_reference = root / (
        "data/runs/signal-maskfix-matrix-9000-seeds17-23/seed-17/lr5e4-d16-c10/"
        "generation-ema/samples-00018000/protein_binder.pdb"
    )
    binder_paths = {"perfect": output_dir / "binder-perfect.pdb"}
    _copy_as_pdb(binder_reference, binder_paths["perfect"])
    for quality, noise, translation in (
        ("perturbed", 0.5, np.zeros(3)),
        ("broken", 3.0, np.asarray([30.0, -20.0, 15.0])),
    ):
        binder_paths[quality] = output_dir / f"binder-{quality}.pdb"
        _mutate_structure(
            binder_reference,
            binder_paths[quality],
            chain_id="B",
            residue_selector=lambda _residue: True,
            noise_scale=noise,
            translation=translation,
            mutation_stride=None,
            seed=17,
        )
    binder_rmsd = {
        quality: target_aligned_binder_rmsd(
            binder_reference,
            path,
            designed_target_chains=["A"],
            designed_binder_chain="B",
            predicted_target_chains=["A"],
            predicted_binder_chain="B",
        )
        for quality, path in binder_paths.items()
    }
    binder_checks = {
        "rmsd_order": binder_rmsd["perfect"] < binder_rmsd["perturbed"] < binder_rmsd["broken"]
    } | _binder_filter_sanity()

    # H3: all framework and antigen coordinates remain bit-identical.
    h3_reference = root / "data/raw/sabdab2/extracted/splits_final/pdb_000010zo_A_+.cif"
    h3_paths = {"perfect": output_dir / "h3-perfect.pdb"}
    _copy_as_pdb(h3_reference, h3_paths["perfect"])
    for quality, noise, translation, stride in (
        ("perturbed", 0.5, np.zeros(3), 4),
        ("broken", 3.0, np.asarray([30.0, -20.0, 15.0]), 1),
    ):
        h3_paths[quality] = output_dir / f"h3-{quality}.pdb"
        _mutate_structure(
            h3_reference,
            h3_paths[quality],
            chain_id="A",
            residue_selector=lambda residue: H3_IMGT_START
            <= int(residue.seqid.num)
            <= H3_IMGT_END,
            noise_scale=noise,
            translation=translation,
            mutation_stride=stride,
            seed=18,
        )
    h3 = {
        quality: evaluate_antibody_h3(
            path,
            h3_reference,
            heavy_chain="A",
            light_chain=None,
            antigen_chains=["B"],
            dockq_executable=dockq,
        )
        for quality, path in h3_paths.items()
    }
    h3_values = {
        quality: {"h3_aar": result.h3_aar, "h3_rmsd": result.h3_rmsd, "dockq": result.dockq}
        for quality, result in h3.items()
    }
    h3_checks = {
        "aar_order": h3["perfect"].h3_aar > h3["perturbed"].h3_aar > h3["broken"].h3_aar,
        "rmsd_order": h3["perfect"].h3_rmsd
        < h3["perturbed"].h3_rmsd
        < h3["broken"].h3_rmsd,
        "global_dockq_order": h3["perfect"].dockq
        > h3["perturbed"].dockq
        > h3["broken"].dockq,
    }

    # RNA self-consistency: X' is compared with the model-designed X, not native X.
    rna_work = root / "data/runs/single-task-transfer-seed17/evaluation/unified/rna/work"
    designed_rna = rna_work / "designed_rna.pdb"
    rna_paths = {"perfect": output_dir / "rna-perfect.pdb"}
    _copy_as_pdb(designed_rna, rna_paths["perfect"])
    rna_chain = gemmi.read_structure(str(designed_rna))[0][0].name
    for quality, noise in (("perturbed", 0.75), ("broken", 10.0)):
        rna_paths[quality] = output_dir / f"rna-{quality}.pdb"
        _mutate_structure(
            designed_rna,
            rna_paths[quality],
            chain_id=rna_chain,
            residue_selector=lambda _residue: True,
            noise_scale=noise,
            translation=np.zeros(3),
            mutation_stride=None,
            seed=19,
        )
    rna_structural = {
        quality: run_usalign_rna(path, designed_rna, executable=usalign)
        for quality, path in rna_paths.items()
    }
    native_complex = rna_work / "native_rna_target_complex.pdb"
    native_model = gemmi.read_structure(str(native_complex))[0]
    native_chains = [chain.name for chain in native_model]
    native_rna_chain = next(
        chain.name
        for chain in native_model
        if any(gemmi.find_tabulated_residue(residue.name).is_nucleic_acid() for residue in chain)
    )
    rna_complex_paths = {"perfect": output_dir / "rna-complex-perfect.pdb"}
    _copy_as_pdb(native_complex, rna_complex_paths["perfect"])
    for quality, noise, translation in (
        ("perturbed", 0.75, np.zeros(3)),
        ("broken", 3.0, np.asarray([30.0, -20.0, 15.0])),
    ):
        rna_complex_paths[quality] = output_dir / f"rna-complex-{quality}.pdb"
        _mutate_structure(
            native_complex,
            rna_complex_paths[quality],
            chain_id=native_rna_chain,
            residue_selector=lambda _residue: True,
            noise_scale=noise,
            translation=translation,
            mutation_stride=None,
            seed=20,
        )
    mapping = "".join(native_chains)
    rna_dockq = {
        quality: run_dockq(
            path,
            native_complex,
            executable=dockq,
            mapping=f"{mapping}:{mapping}",
        )["total_dockq"]
        for quality, path in rna_complex_paths.items()
    }
    rna_checks = {
        "sctm_order": rna_structural["perfect"]["sctm"]
        > rna_structural["perturbed"]["sctm"]
        > rna_structural["broken"]["sctm"],
        "scrmsd_order": rna_structural["perfect"]["scrmsd"]
        < rna_structural["perturbed"]["scrmsd"]
        < rna_structural["broken"]["scrmsd"],
        "dockq_order": rna_dockq["perfect"] > rna_dockq["perturbed"] > rna_dockq["broken"],
    }

    payload = {
        "binder": {
            "self_consistency_rmsd": binder_rmsd,
            "checks": binder_checks,
            "passed": all(binder_checks.values()),
        },
        "antibody_h3": {
            "metrics": h3_values,
            "checks": h3_checks,
            "primary_metrics_passed": h3_checks["aar_order"] and h3_checks["rmsd_order"],
            "dockq_is_auxiliary": True,
        },
        "rna": {
            "self_consistency": rna_structural,
            "dockq": rna_dockq,
            "checks": rna_checks,
            "passed": all(rna_checks.values()),
        },
        "semantics": {
            "rna_sc_reference": "generated design structure X; independent/refold structure is X-prime",
            "h3_alignment": "fixed antibody framework, then H3 backbone only",
            "h3_global_dockq": "auxiliary because fixed framework/antigen can dominate",
        },
    }
    payload["passed"] = bool(
        payload["binder"]["passed"]
        and payload["antibody_h3"]["primary_metrics_passed"]
        and payload["rna"]["passed"]
    )
    destination = output_dir / "true-evaluator-sanity.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit("true evaluator sanity ordering failed")


if __name__ == "__main__":
    main()

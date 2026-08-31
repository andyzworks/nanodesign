#!/usr/bin/env python3
"""Run Rosetta InterfaceAnalyzerMover and emit NanoDesign's frozen metrics as JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pyrosetta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("complex")
    parser.add_argument("--interface", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pyrosetta.init("-mute all")
    pose = pyrosetta.pose_from_pdb(args.complex)
    mover = pyrosetta.rosetta.protocols.analysis.InterfaceAnalyzerMover()
    partners = pyrosetta.rosetta.core.pose.DockingPartners.docking_partners_from_string(
        args.interface
    )
    mover.set_interface(partners)
    mover.set_scorefunction(pyrosetta.get_fa_scorefxn())
    mover.set_compute_packstat(True)
    mover.set_compute_interface_energy(True)
    mover.set_calc_dSASA(True)
    mover.set_calc_hbond_sasaE(True)
    mover.set_compute_interface_sc(True)
    mover.set_pack_separated(True)
    mover.apply(pose)
    data = mover.get_all_data()
    metrics = {
        "rosetta_interface_delta_g": float(mover.get_interface_dG()),
        "shape_complementarity": float(data.sc_value),
        "interface_dsasa": float(mover.get_interface_delta_sasa()),
        # InterfaceData exposes per-partner counts as a Rosetta vector.  The
        # end-to-end evaluator replaces this with BindCraft's binder-side 4 A
        # contact count; retaining the total here keeps the backend self-describing.
        "interface_residue_count": float(sum(data.interface_nres)),
        "interface_hbond_count": float(data.interface_hbonds),
        "interface_unsatisfied_hbonds": float(data.delta_unsat_hbonds),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"Rosetta emitted non-finite interface metrics: {metrics}")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

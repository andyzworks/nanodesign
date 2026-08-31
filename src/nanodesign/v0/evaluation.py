"""Frozen metric inventory backed by the executable v0 evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from nanodesign.v0.constants import Task
from nanodesign.v0.evaluators import (
    BINDCRAFT_DEFAULT_FILTERS,
    BINDER_GENERATION_BUDGET,
    aggregate_binder_results,
    binder_success_rate,
    evaluate_antibody_h3,
    evaluate_protein_binder,
    evaluate_rna,
    framework_aligned_h3_rmsd,
    run_dockq,
    run_rosetta_interface_analyzer,
    run_usalign_rna,
)


class MetricDirection(StrEnum):
    HIGHER = "higher"
    LOWER = "lower"


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: MetricDirection
    tier: str
    conditional: str | None = None


@dataclass(frozen=True)
class EvaluationProtocol:
    task: Task
    metrics: tuple[MetricSpec, ...]
    implementation: str


PROTOCOLS = {
    Task.PROTEIN_BINDER: EvaluationProtocol(
        Task.PROTEIN_BINDER,
        (
            MetricSpec("in_silico_success_rate", MetricDirection.HIGHER, "primary"),
            MetricSpec("interface_confidence", MetricDirection.HIGHER, "auxiliary"),
            MetricSpec("self_consistency_rmsd", MetricDirection.LOWER, "auxiliary"),
            MetricSpec("rosetta_interface_delta_g", MetricDirection.LOWER, "auxiliary"),
            MetricSpec("shape_complementarity", MetricDirection.HIGHER, "auxiliary"),
            MetricSpec("clashes", MetricDirection.LOWER, "auxiliary"),
            MetricSpec("diversity", MetricDirection.HIGHER, "auxiliary"),
            MetricSpec("cluster_level_success", MetricDirection.HIGHER, "auxiliary"),
        ),
        "ColabFold AF2-multimer v3 + Rosetta InterfaceAnalyzer + frozen BindCraft defaults",
    ),
    Task.ANTIBODY_CDR: EvaluationProtocol(
        Task.ANTIBODY_CDR,
        (
            MetricSpec("h3_aar", MetricDirection.HIGHER, "primary"),
            MetricSpec("h3_rmsd", MetricDirection.LOWER, "primary"),
            MetricSpec("dockq", MetricDirection.HIGHER, "primary"),
        ),
        "fixed-framework Kabsch alignment + DockQ v2",
    ),
    Task.RNA_APTAMER: EvaluationProtocol(
        Task.RNA_APTAMER,
        (
            MetricSpec("sctm", MetricDirection.HIGHER, "primary"),
            MetricSpec("scrmsd", MetricDirection.LOWER, "primary"),
            MetricSpec("structure_confidence", MetricDirection.HIGHER, "primary"),
            MetricSpec(
                "dockq",
                MetricDirection.HIGHER,
                "primary",
                conditional="native RNA-target complex available",
            ),
        ),
        "RhoFold+ refolding + US-align RNA + DockQ v2",
    ),
}


def get_protocol(task: Task | int | str) -> EvaluationProtocol:
    if isinstance(task, str):
        task = Task[task.upper()]
    return PROTOCOLS[Task(int(task))]


def amino_acid_recovery(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(predicted)
    target = np.asarray(target)
    if predicted.shape != target.shape or predicted.size == 0:
        raise ValueError("AAR arrays must have the same non-empty shape")
    return float(np.mean(predicted == target))


__all__ = [
    "BINDCRAFT_DEFAULT_FILTERS",
    "BINDER_GENERATION_BUDGET",
    "PROTOCOLS",
    "aggregate_binder_results",
    "amino_acid_recovery",
    "binder_success_rate",
    "evaluate_antibody_h3",
    "evaluate_protein_binder",
    "evaluate_rna",
    "framework_aligned_h3_rmsd",
    "get_protocol",
    "run_dockq",
    "run_rosetta_interface_analyzer",
    "run_usalign_rna",
]

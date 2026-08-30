"""Task-specific evaluation contracts fixed by the NanoDesign v0 plan."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from nanodesign.v0.constants import Task


class MetricDirection(StrEnum):
    HIGHER = "higher"
    LOWER = "lower"


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: MetricDirection
    tier: str
    required: bool = True
    conditional: str | None = None


@dataclass(frozen=True)
class EvaluationProtocol:
    task: Task
    metrics: tuple[MetricSpec, ...]
    scientific_note: str

    @property
    def metric_names(self) -> frozenset[str]:
        return frozenset(metric.name for metric in self.metrics)


PROTOCOLS: dict[Task, EvaluationProtocol] = {
    Task.PROTEIN_BINDER: EvaluationProtocol(
        task=Task.PROTEIN_BINDER,
        metrics=(
            MetricSpec("in_silico_success_rate", MetricDirection.HIGHER, "primary"),
            MetricSpec("interface_confidence", MetricDirection.HIGHER, "auxiliary"),
            MetricSpec("self_consistency_rmsd", MetricDirection.LOWER, "auxiliary"),
            MetricSpec("rosetta_interface_delta_g", MetricDirection.LOWER, "auxiliary"),
            MetricSpec("shape_complementarity", MetricDirection.HIGHER, "auxiliary"),
            MetricSpec("clashes", MetricDirection.LOWER, "auxiliary"),
            MetricSpec("diversity", MetricDirection.HIGHER, "auxiliary"),
            MetricSpec("cluster_level_success", MetricDirection.HIGHER, "auxiliary"),
        ),
        scientific_note=(
            "Success thresholds must come from prior work or a frozen calibration set, "
            "not hand-picked after test evaluation."
        ),
    ),
    Task.ANTIBODY_CDR: EvaluationProtocol(
        task=Task.ANTIBODY_CDR,
        metrics=(
            MetricSpec("cdr_aar", MetricDirection.HIGHER, "primary"),
            MetricSpec("cdr_rmsd", MetricDirection.LOWER, "primary"),
            MetricSpec("dockq", MetricDirection.HIGHER, "primary"),
            MetricSpec("cdr_h3_aar", MetricDirection.HIGHER, "required_report"),
            MetricSpec("cdr_h3_rmsd", MetricDirection.LOWER, "required_report"),
            MetricSpec("rosetta_interface_delta_g", MetricDirection.LOWER, "auxiliary"),
            MetricSpec("clashes", MetricDirection.LOWER, "auxiliary"),
            MetricSpec("geometry_validity", MetricDirection.HIGHER, "auxiliary"),
        ),
        scientific_note=(
            "H3 AAR and H3 RMSD are always reported. If all six CDRs are designed, "
            "each CDR must also be reported separately."
        ),
    ),
    Task.RNA_APTAMER: EvaluationProtocol(
        task=Task.RNA_APTAMER,
        metrics=(
            MetricSpec("sctm", MetricDirection.HIGHER, "structure_quality"),
            MetricSpec("scrmsd", MetricDirection.LOWER, "structure_quality"),
            MetricSpec("structure_confidence", MetricDirection.HIGHER, "structure_quality"),
            MetricSpec(
                "dockq",
                MetricDirection.HIGHER,
                "target_interaction",
                conditional="native RNA-target complex is available",
            ),
        ),
        scientific_note=(
            "DockQ and self-consistency metrics measure computational geometry, not binding "
            "affinity. Experimental Kd is outside NanoDesign v0."
        ),
    ),
}


def get_protocol(task: Task | int | str) -> EvaluationProtocol:
    if isinstance(task, str):
        task = Task[task.upper()]
    return PROTOCOLS[Task(int(task))]


@dataclass(frozen=True)
class ThresholdRule:
    metric: str
    operation: str
    threshold: float

    def __post_init__(self) -> None:
        if self.operation not in {">=", "<="}:
            raise ValueError(f"unsupported threshold operation {self.operation!r}")
        if not np.isfinite(self.threshold):
            raise ValueError("success threshold must be finite")

    def passes(self, value: float) -> bool:
        if self.operation == ">=":
            return value >= self.threshold
        return value <= self.threshold


@dataclass(frozen=True)
class BinderSuccessProfile:
    name: str
    source: str
    rules: tuple[ThresholdRule, ...]

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.source.strip() or not self.rules:
            raise ValueError("a success profile needs a name, source, and at least one rule")
        protocol = PROTOCOLS[Task.PROTEIN_BINDER]
        filter_metrics = {
            metric.name: metric
            for metric in protocol.metrics
            if metric.name not in {"in_silico_success_rate", "diversity", "cluster_level_success"}
        }
        unknown = {rule.metric for rule in self.rules} - set(filter_metrics)
        if unknown:
            raise ValueError(f"success profile contains unknown filter metrics: {sorted(unknown)}")
        names = [rule.metric for rule in self.rules]
        if len(names) != len(set(names)):
            raise ValueError("success profile contains duplicate metric rules")
        for rule in self.rules:
            expected = (
                ">=" if filter_metrics[rule.metric].direction == MetricDirection.HIGHER else "<="
            )
            if rule.operation != expected:
                raise ValueError(
                    f"{rule.metric} is {filter_metrics[rule.metric].direction.value}; "
                    f"expected operation {expected}"
                )

    def design_passes(self, metrics: Mapping[str, float]) -> bool:
        missing = {rule.metric for rule in self.rules} - set(metrics)
        if missing:
            raise ValueError(f"design is missing success-filter metrics: {sorted(missing)}")
        return all(rule.passes(float(metrics[rule.metric])) for rule in self.rules)


def in_silico_success_rate(
    records: list[Mapping[str, float]], profile: BinderSuccessProfile
) -> float:
    if not records:
        raise ValueError("cannot calculate success rate from zero designs")
    return float(np.mean([profile.design_passes(record) for record in records]))


def amino_acid_recovery(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    predicted = np.asarray(predicted)
    target = np.asarray(target)
    mask = np.asarray(mask).astype(bool)
    if predicted.shape != target.shape or mask.shape != target.shape or not mask.any():
        raise ValueError("AAR arrays must share shape and select at least one residue")
    return float(np.mean(predicted[mask] == target[mask]))


def coordinate_rmsd(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Calculate RMSD after the caller applies the protocol-specific alignment."""

    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask).astype(bool)
    if predicted.shape != target.shape or predicted.shape[-1] != 3:
        raise ValueError("RMSD coordinates must have matching [..., 3] shapes")
    if mask.shape != predicted.shape[:-1] or not mask.any():
        raise ValueError("RMSD mask has the wrong shape or selects nothing")
    squared_distance = np.sum((predicted[mask] - target[mask]) ** 2, axis=-1)
    return float(np.sqrt(np.mean(squared_distance)))


def validate_evaluation_record(
    task: Task,
    metrics: Mapping[str, float],
    *,
    cdr_design: str | None = None,
    has_native_complex: bool = True,
) -> None:
    protocol = PROTOCOLS[task]
    required = {
        metric.name
        for metric in protocol.metrics
        if metric.required and (metric.conditional is None or has_native_complex)
    }
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"{task.name} evaluation is missing {sorted(missing)}")
    values = np.asarray([float(value) for value in metrics.values()])
    if not np.isfinite(values).all():
        raise ValueError("evaluation metrics must be finite")
    if task == Task.ANTIBODY_CDR and cdr_design == "all_six":
        per_cdr = {
            f"cdr_{cdr}_{metric}"
            for cdr in ("h1", "h2", "h3", "l1", "l2", "l3")
            for metric in ("aar", "rmsd")
        }
        missing_per_cdr = per_cdr - set(metrics)
        if missing_per_cdr:
            raise ValueError(
                f"all-six-CDR evaluation is missing per-CDR metrics: {sorted(missing_per_cdr)}"
            )

import pytest

from nanodesign.v0.constants import Task
from nanodesign.v0.evaluation import (
    PROTOCOLS,
    BinderSuccessProfile,
    ThresholdRule,
    in_silico_success_rate,
    validate_evaluation_record,
)


def test_tasks_have_separate_protocols_with_exact_primary_metrics():
    assert set(PROTOCOLS) == set(Task)
    binder_primary = {
        metric.name for metric in PROTOCOLS[Task.PROTEIN_BINDER].metrics if metric.tier == "primary"
    }
    antibody_primary = {
        metric.name for metric in PROTOCOLS[Task.ANTIBODY_CDR].metrics if metric.tier == "primary"
    }
    assert binder_primary == {"in_silico_success_rate"}
    assert antibody_primary == {"cdr_aar", "cdr_rmsd", "dockq"}


def test_binder_success_requires_frozen_sourced_thresholds():
    with pytest.raises(ValueError, match="name, source"):
        BinderSuccessProfile("", "", ())
    profile = BinderSuccessProfile(
        "calibrated-v1",
        "frozen calibration protocol",
        (
            ThresholdRule("interface_confidence", ">=", 0.8),
            ThresholdRule("self_consistency_rmsd", "<=", 2.0),
        ),
    )
    rate = in_silico_success_rate(
        [
            {"interface_confidence": 0.9, "self_consistency_rmsd": 1.5},
            {"interface_confidence": 0.7, "self_consistency_rmsd": 1.0},
        ],
        profile,
    )
    assert rate == 0.5


def test_rna_dockq_is_conditional_but_structure_metrics_are_required():
    validate_evaluation_record(
        Task.RNA_APTAMER,
        {"sctm": 0.7, "scrmsd": 2.0, "structure_confidence": 0.8},
        has_native_complex=False,
    )
    with pytest.raises(ValueError, match="dockq"):
        validate_evaluation_record(
            Task.RNA_APTAMER,
            {"sctm": 0.7, "scrmsd": 2.0, "structure_confidence": 0.8},
            has_native_complex=True,
        )


def test_antibody_always_requires_h3_reporting():
    with pytest.raises(ValueError, match="cdr_h3"):
        validate_evaluation_record(
            Task.ANTIBODY_CDR,
            {
                "cdr_aar": 0.5,
                "cdr_rmsd": 1.5,
                "dockq": 0.6,
                "rosetta_interface_delta_g": -5.0,
                "clashes": 0.0,
                "geometry_validity": 1.0,
            },
            cdr_design="h3_only",
        )

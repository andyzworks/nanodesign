import pytest

from nanodesign.v0.constants import Task
from nanodesign.v0.evaluation import (
    BINDCRAFT_DEFAULT_FILTERS,
    BINDER_GENERATION_BUDGET,
    PROTOCOLS,
    binder_success_rate,
)


def test_tasks_have_exact_real_primary_metrics():
    assert set(PROTOCOLS) == set(Task)
    assert {
        metric.name for metric in PROTOCOLS[Task.PROTEIN_BINDER].metrics if metric.tier == "primary"
    } == {"in_silico_success_rate"}
    assert {
        metric.name for metric in PROTOCOLS[Task.ANTIBODY_CDR].metrics if metric.tier == "primary"
    } == {"h3_aar", "h3_rmsd", "dockq"}
    assert {
        metric.name for metric in PROTOCOLS[Task.RNA_APTAMER].metrics if metric.tier == "primary"
    } == {"sctm", "scrmsd", "structure_confidence", "dockq"}


def test_binder_success_uses_frozen_bindcraft_defaults():
    passing = {name: threshold for name, (_, threshold) in BINDCRAFT_DEFAULT_FILTERS.items()}
    failing = dict(passing, iptm=0.49)
    assert binder_success_rate([passing, failing]) == 0.5
    assert BINDER_GENERATION_BUDGET["backbones_per_target"] == 1000
    with pytest.raises(Exception, match="missing"):
        binder_success_rate([{}])


def test_rna_note_does_not_claim_binding_affinity():
    implementation = PROTOCOLS[Task.RNA_APTAMER].implementation
    assert "RhoFold+" in implementation
    assert "DockQ" in implementation

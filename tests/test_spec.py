from nanodesign.v0.constants import DataSource, Polymer, Role, Task
from nanodesign.v0.spec import MODEL_ARCHITECTURE, TASK_SPECS, get_v0_spec


def test_v0_has_exactly_three_fixed_tasks():
    spec = get_v0_spec()
    assert spec.model_architecture == MODEL_ARCHITECTURE == "rfd3na_tiny"
    assert [task.task for task in spec.tasks] == list(Task)
    assert set(TASK_SPECS) == {
        Task.PROTEIN_BINDER,
        Task.ANTIBODY_CDR,
        Task.RNA_APTAMER,
    }


def test_rna_aptamer_is_target_conditioned_and_rnasolo_is_auxiliary():
    spec = TASK_SPECS[Task.RNA_APTAMER]
    assert spec.fixed_roles == {Role.TARGET}
    assert spec.design_roles == {Role.RNA_APTAMER}
    assert spec.fixed_polymers == {Polymer.PROTEIN}
    assert spec.design_polymer == Polymer.RNA
    assert spec.binding_sources == (
        DataSource.RIBOCENTRE_APTAMER,
        DataSource.PDB_RNA_TARGET_COMPLEX,
    )
    assert spec.auxiliary_sources == (DataSource.RNASOLO2,)


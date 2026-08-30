"""Single source of truth for the immutable NanoDesign v0 specification."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from nanodesign.v0.constants import DataSource, Polymer, Role, Task


SPEC_VERSION = "nanodesign.v0"
MODEL_ARCHITECTURE = "rfd3na_tiny"
MIN_MODEL_PARAMETERS = 5_000_000
MAX_MODEL_PARAMETERS = 20_000_000


@dataclass(frozen=True)
class TaskSpec:
    task: Task
    input_description: str
    output_description: str
    fixed_roles: frozenset[Role]
    design_roles: frozenset[Role]
    fixed_polymers: frozenset[Polymer]
    design_polymer: Polymer
    binding_sources: tuple[DataSource, ...]
    auxiliary_sources: tuple[DataSource, ...] = ()


TASK_SPECS: dict[Task, TaskSpec] = {
    Task.PROTEIN_BINDER: TaskSpec(
        task=Task.PROTEIN_BINDER,
        input_description="target protein",
        output_description="protein binder sequence + structure",
        fixed_roles=frozenset({Role.TARGET}),
        design_roles=frozenset({Role.BINDER}),
        fixed_polymers=frozenset({Polymer.PROTEIN}),
        design_polymer=Polymer.PROTEIN,
        binding_sources=(DataSource.PPIREF, DataSource.PPIREF50K),
    ),
    Task.ANTIBODY_CDR: TaskSpec(
        task=Task.ANTIBODY_CDR,
        input_description="antigen + fixed antibody framework",
        output_description="CDR sequence + structure",
        fixed_roles=frozenset({Role.ANTIGEN, Role.ANTIBODY_FRAMEWORK}),
        design_roles=frozenset({Role.CDR}),
        fixed_polymers=frozenset({Polymer.PROTEIN}),
        design_polymer=Polymer.PROTEIN,
        binding_sources=(DataSource.SABDAB2,),
    ),
    Task.RNA_APTAMER: TaskSpec(
        task=Task.RNA_APTAMER,
        input_description="target protein",
        output_description="RNA aptamer sequence + structure",
        fixed_roles=frozenset({Role.TARGET}),
        design_roles=frozenset({Role.RNA_APTAMER}),
        fixed_polymers=frozenset({Polymer.PROTEIN}),
        design_polymer=Polymer.RNA,
        binding_sources=(
            DataSource.RIBOCENTRE_APTAMER,
            DataSource.PDB_RNA_TARGET_COMPLEX,
        ),
        auxiliary_sources=(DataSource.RNASOLO2,),
    ),
}


@dataclass(frozen=True)
class V0Spec:
    version: str
    objective: str
    model_architecture: str
    parameter_range: tuple[int, int]
    tasks: tuple[TaskSpec, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["tasks"] = [
            {
                **asdict(task),
                "task": task.task.name.lower(),
                "fixed_roles": sorted(role.name.lower() for role in task.fixed_roles),
                "design_roles": sorted(role.name.lower() for role in task.design_roles),
                "fixed_polymers": sorted(p.name.lower() for p in task.fixed_polymers),
                "design_polymer": task.design_polymer.name.lower(),
                "binding_sources": [source.value for source in task.binding_sources],
                "auxiliary_sources": [source.value for source in task.auxiliary_sources],
            }
            for task in self.tasks
        ]
        return value


def get_v0_spec() -> V0Spec:
    return V0Spec(
        version=SPEC_VERSION,
        objective=(
            "Given molecular context, design a new molecule that can interact with or bind "
            "the target."
        ),
        model_architecture=MODEL_ARCHITECTURE,
        parameter_range=(MIN_MODEL_PARAMETERS, MAX_MODEL_PARAMETERS),
        tasks=tuple(TASK_SPECS[task] for task in Task),
    )


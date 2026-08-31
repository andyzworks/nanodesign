"""Explicit adapter registry; source-specific conversion is intentionally not guessed."""

from __future__ import annotations

from dataclasses import dataclass

from nanodesign.v0.constants import DataSource, ExamplePurpose, Task


@dataclass(frozen=True)
class AdapterDefinition:
    source: DataSource
    task: Task
    purpose: ExamplePurpose
    required_outputs: tuple[str, ...]
    implementation_status: str = "contract_only"


_BINDING_CLUSTERS = (
    "complex_cluster_id",
    "target_cluster_id",
    "design_cluster_id",
)


ADAPTERS = {
    DataSource.PPIREF: AdapterDefinition(
        DataSource.PPIREF,
        Task.PROTEIN_BINDER,
        ExamplePurpose.BINDING_DESIGN,
        ("target_chain", "binder_chain", *_BINDING_CLUSTERS),
    ),
    DataSource.PPIREF50K: AdapterDefinition(
        DataSource.PPIREF50K,
        Task.PROTEIN_BINDER,
        ExamplePurpose.BINDING_DESIGN,
        ("target_chain", "binder_chain", *_BINDING_CLUSTERS),
    ),
    DataSource.SABDAB2: AdapterDefinition(
        DataSource.SABDAB2,
        Task.ANTIBODY_CDR,
        ExamplePurpose.BINDING_DESIGN,
        ("antigen", "framework", "cdr_labels", *_BINDING_CLUSTERS),
    ),
    DataSource.RIBOCENTRE_APTAMER: AdapterDefinition(
        DataSource.RIBOCENTRE_APTAMER,
        Task.RNA_APTAMER,
        ExamplePurpose.BINDING_DESIGN,
        ("target_protein", "rna_aptamer", *_BINDING_CLUSTERS),
    ),
    DataSource.PDB_RNA_TARGET_COMPLEX: AdapterDefinition(
        DataSource.PDB_RNA_TARGET_COMPLEX,
        Task.RNA_APTAMER,
        ExamplePurpose.BINDING_DESIGN,
        ("target_protein", "rna_design_region", *_BINDING_CLUSTERS),
    ),
    DataSource.RNASOLO2: AdapterDefinition(
        DataSource.RNASOLO2,
        Task.RNA_APTAMER,
        ExamplePurpose.RNA_STRUCTURE_PRIOR,
        ("rna_chain", "design_cluster_id", "structure_cluster_id"),
    ),
}


def get_adapter_definition(source: DataSource | str) -> AdapterDefinition:
    return ADAPTERS[DataSource(source)]

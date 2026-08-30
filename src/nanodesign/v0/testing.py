"""Small synthetic fixtures for contract and plumbing tests only."""

from __future__ import annotations

import numpy as np

from nanodesign.v0.constants import (
    AA_TOKEN_IDS,
    RNA_TOKEN_IDS,
    DataSource,
    ExamplePurpose,
    Polymer,
    Role,
    Task,
)
from nanodesign.v0.contracts import DesignExample


def _example(
    sample_id: str,
    task: Task,
    source: DataSource,
    roles: list[Role],
    polymers: list[Polymer],
) -> DesignExample:
    protein_token = min(AA_TOKEN_IDS)
    rna_token = min(RNA_TOKEN_IDS)
    token_ids = np.asarray(
        [protein_token if polymer == Polymer.PROTEIN else rna_token for polymer in polymers],
        dtype=np.int64,
    )
    design_roles = {
        Task.PROTEIN_BINDER: {Role.BINDER},
        Task.ANTIBODY_CDR: {Role.CDR},
        Task.RNA_APTAMER: {Role.RNA_APTAMER},
    }[task]
    n = len(roles)
    positions = np.stack([np.arange(n, dtype=np.float32) * 3.5, np.zeros(n), np.zeros(n)], axis=-1)
    return DesignExample(
        sample_id=sample_id,
        task=task,
        source=source,
        purpose=ExamplePurpose.BINDING_DESIGN,
        token_ids=token_ids,
        polymer_type=np.asarray([int(polymer) for polymer in polymers], dtype=np.int64),
        role_id=np.asarray([int(role) for role in roles], dtype=np.int64),
        chain_id=np.asarray([1 if index < n // 2 else 2 for index in range(n)]),
        residue_index=np.arange(1, n + 1, dtype=np.int64),
        design_mask=np.asarray([role in design_roles for role in roles], dtype=np.float32),
        atom_positions=positions,
        atom_mask=np.ones(n, dtype=np.float32),
        atom_token_index=np.arange(n, dtype=np.int64),
        atom_element=np.full(n, 6, dtype=np.int64),
    ).validate()


def synthetic_binding_examples() -> list[DesignExample]:
    return [
        _example(
            "synthetic_binder",
            Task.PROTEIN_BINDER,
            DataSource.PPIREF50K,
            [Role.TARGET, Role.TARGET, Role.TARGET, Role.BINDER, Role.BINDER],
            [Polymer.PROTEIN] * 5,
        ),
        _example(
            "synthetic_antibody",
            Task.ANTIBODY_CDR,
            DataSource.SABDAB2,
            [
                Role.ANTIGEN,
                Role.ANTIGEN,
                Role.ANTIBODY_FRAMEWORK,
                Role.ANTIBODY_FRAMEWORK,
                Role.CDR,
                Role.CDR,
            ],
            [Polymer.PROTEIN] * 6,
        ),
        _example(
            "synthetic_rna_aptamer",
            Task.RNA_APTAMER,
            DataSource.PDB_RNA_TARGET_COMPLEX,
            [
                Role.TARGET,
                Role.TARGET,
                Role.RNA_APTAMER,
                Role.RNA_APTAMER,
                Role.RNA_APTAMER,
            ],
            [Polymer.PROTEIN, Polymer.PROTEIN, Polymer.RNA, Polymer.RNA, Polymer.RNA],
        ),
    ]

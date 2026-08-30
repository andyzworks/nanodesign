"""Versioned token/atom contracts shared by all NanoDesign v0 tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from nanodesign.v0.constants import (
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
    DataSource,
    ExamplePurpose,
    Polymer,
    Role,
    Task,
    allowed_token_ids,
)
from nanodesign.v0.spec import SPEC_VERSION, TASK_SPECS


class ContractError(ValueError):
    """Raised when an example violates the frozen v0 formulation."""


_PROTEIN_ROLES = {
    Role.TARGET,
    Role.BINDER,
    Role.ANTIGEN,
    Role.ANTIBODY_FRAMEWORK,
    Role.CDR,
}


@dataclass
class DesignExample:
    sample_id: str
    task: Task
    source: DataSource
    purpose: ExamplePurpose
    token_ids: np.ndarray
    polymer_type: np.ndarray
    role_id: np.ndarray
    chain_id: np.ndarray
    residue_index: np.ndarray
    design_mask: np.ndarray
    atom_positions: np.ndarray
    atom_mask: np.ndarray
    atom_token_index: np.ndarray
    atom_element: np.ndarray
    schema_version: str = SPEC_VERSION

    @property
    def num_tokens(self) -> int:
        return int(np.asarray(self.token_ids).shape[0])

    @property
    def num_atoms(self) -> int:
        return int(np.asarray(self.atom_positions).shape[0])

    def validate(self) -> "DesignExample":
        if self.schema_version != SPEC_VERSION:
            raise ContractError(
                f"{self.sample_id}: schema {self.schema_version!r} must be {SPEC_VERSION!r}"
            )
        if not self.sample_id or any(character.isspace() for character in self.sample_id):
            raise ContractError("sample_id must be non-empty and contain no whitespace")
        try:
            task = Task(int(self.task))
            source = DataSource(self.source)
            purpose = ExamplePurpose(self.purpose)
        except (TypeError, ValueError) as error:
            raise ContractError(f"{self.sample_id}: invalid task/source/purpose") from error

        n = self.num_tokens
        a = self.num_atoms
        if n < 1 or a < 1:
            raise ContractError(f"{self.sample_id}: examples require tokens and atoms")
        token_fields = {
            "token_ids": self.token_ids,
            "polymer_type": self.polymer_type,
            "role_id": self.role_id,
            "chain_id": self.chain_id,
            "residue_index": self.residue_index,
            "design_mask": self.design_mask,
        }
        for name, value in token_fields.items():
            if np.asarray(value).shape != (n,):
                raise ContractError(f"{self.sample_id}: {name} must have shape {(n,)}")
        atom_fields = {
            "atom_mask": self.atom_mask,
            "atom_token_index": self.atom_token_index,
            "atom_element": self.atom_element,
        }
        for name, value in atom_fields.items():
            if np.asarray(value).shape != (a,):
                raise ContractError(f"{self.sample_id}: {name} must have shape {(a,)}")
        positions = np.asarray(self.atom_positions)
        if positions.shape != (a, 3) or not np.isfinite(positions).all():
            raise ContractError(
                f"{self.sample_id}: atom_positions must be finite with shape {(a, 3)}"
            )

        design = np.asarray(self.design_mask)
        atom_mask = np.asarray(self.atom_mask)
        if not np.isin(design, (0, 1)).all() or not np.isin(atom_mask, (0, 1)).all():
            raise ContractError(f"{self.sample_id}: design_mask and atom_mask must be binary")
        if design.sum() < 1:
            raise ContractError(f"{self.sample_id}: no design tokens")
        atom_to_token = np.asarray(self.atom_token_index, dtype=np.int64)
        if atom_to_token.min(initial=0) < 0 or atom_to_token.max(initial=-1) >= n:
            raise ContractError(f"{self.sample_id}: atom_token_index is out of range")
        if np.any(np.asarray(self.atom_element) < 0):
            raise ContractError(f"{self.sample_id}: atom_element ids must be non-negative")
        active_atom_tokens = set(atom_to_token[atom_mask.astype(bool)].tolist())
        missing_atom_tokens = sorted(set(range(n)) - active_atom_tokens)
        if missing_atom_tokens:
            raise ContractError(
                f"{self.sample_id}: every token needs at least one active atom; "
                f"missing={missing_atom_tokens}"
            )

        polymers = np.asarray(self.polymer_type, dtype=np.int64)
        roles = np.asarray(self.role_id, dtype=np.int64)
        tokens = np.asarray(self.token_ids, dtype=np.int64)
        chains = np.asarray(self.chain_id, dtype=np.int64)
        residues = np.asarray(self.residue_index, dtype=np.int64)
        if np.any(chains < 1):
            raise ContractError(f"{self.sample_id}: unpadded chain_id values must be positive")
        if np.any(residues < 0):
            raise ContractError(f"{self.sample_id}: residue_index values must be non-negative")
        for index in range(n):
            try:
                polymer = Polymer(int(polymers[index]))
                role = Role(int(roles[index]))
            except ValueError as error:
                raise ContractError(
                    f"{self.sample_id}: invalid polymer/role at token {index}"
                ) from error
            if polymer == Polymer.PAD:
                raise ContractError(
                    f"{self.sample_id}: unpadded examples cannot contain PAD polymer"
                )
            if int(tokens[index]) not in allowed_token_ids(polymer):
                raise ContractError(
                    f"{self.sample_id}: token {tokens[index]} is invalid for {polymer.name}"
                )
            if role in _PROTEIN_ROLES and polymer != Polymer.PROTEIN:
                raise ContractError(f"{self.sample_id}: {role.name} must be protein")
            if role == Role.RNA_APTAMER and polymer != Polymer.RNA:
                raise ContractError(f"{self.sample_id}: RNA_APTAMER role must be RNA")

        spec = TASK_SPECS[task]
        observed_roles = {Role(int(value)) for value in np.unique(roles)}
        if purpose == ExamplePurpose.BINDING_DESIGN:
            if source not in spec.binding_sources:
                raise ContractError(
                    f"{self.sample_id}: {source.value} is not a binding source for {task.name}"
                )
            expected_roles = spec.fixed_roles | spec.design_roles
            if observed_roles != expected_roles:
                raise ContractError(
                    f"{self.sample_id}: roles must be exactly "
                    f"{sorted(role.name for role in expected_roles)}; observed "
                    f"{sorted(role.name for role in observed_roles)}"
                )
            expected_design = np.isin(roles, [int(role) for role in spec.design_roles])
        else:
            if task != Task.RNA_APTAMER or source != DataSource.RNASOLO2:
                raise ContractError("RNA structure-prior examples must come from RNAsolo2")
            if observed_roles != {Role.RNA_APTAMER}:
                raise ContractError(
                    "RNAsolo2 prior examples may contain only RNA_APTAMER tokens"
                )
            expected_design = np.ones(n, dtype=bool)

        if not np.array_equal(design.astype(bool), expected_design):
            raise ContractError(
                f"{self.sample_id}: design_mask does not match the task/purpose contract"
            )
        for role in spec.design_roles & observed_roles:
            role_polymer = {Polymer(int(value)) for value in polymers[roles == int(role)]}
            if role_polymer != {spec.design_polymer}:
                raise ContractError(
                    f"{self.sample_id}: {role.name} must use {spec.design_polymer.name}"
                )
        return self


def collate_examples(examples: Sequence[DesignExample]) -> dict[str, torch.Tensor | list[str]]:
    if not examples:
        raise ValueError("cannot collate an empty example list")
    for example in examples:
        example.validate()
    batch_size = len(examples)
    max_tokens = max(example.num_tokens for example in examples)
    max_atoms = max(example.num_atoms for example in examples)

    batch: dict[str, torch.Tensor | list[str]] = {
        "sample_id": [example.sample_id for example in examples],
        "task_id": torch.zeros(batch_size, dtype=torch.long),
        "token_ids_0": torch.full((batch_size, max_tokens), PAD_TOKEN_ID, dtype=torch.long),
        "token_ids_t": torch.full((batch_size, max_tokens), MASK_TOKEN_ID, dtype=torch.long),
        "polymer_type": torch.zeros((batch_size, max_tokens), dtype=torch.long),
        "role_id": torch.zeros((batch_size, max_tokens), dtype=torch.long),
        "chain_id": torch.zeros((batch_size, max_tokens), dtype=torch.long),
        "residue_index": torch.zeros((batch_size, max_tokens), dtype=torch.long),
        "design_mask": torch.zeros((batch_size, max_tokens), dtype=torch.float32),
        "token_mask": torch.zeros((batch_size, max_tokens), dtype=torch.float32),
        "atom_positions_0": torch.zeros((batch_size, max_atoms, 3), dtype=torch.float32),
        "atom_positions_t": torch.zeros((batch_size, max_atoms, 3), dtype=torch.float32),
        "atom_mask": torch.zeros((batch_size, max_atoms), dtype=torch.float32),
        "atom_token_index": torch.zeros((batch_size, max_atoms), dtype=torch.long),
        "atom_element": torch.zeros((batch_size, max_atoms), dtype=torch.long),
        "diffusion_time": torch.zeros((batch_size,), dtype=torch.float32),
    }
    for row, example in enumerate(examples):
        n, a = example.num_tokens, example.num_atoms
        batch["task_id"][row] = int(example.task)  # type: ignore[index]
        for target, source in (
            ("token_ids_0", example.token_ids),
            ("polymer_type", example.polymer_type),
            ("role_id", example.role_id),
            ("chain_id", example.chain_id),
            ("residue_index", example.residue_index),
        ):
            batch[target][row, :n] = torch.as_tensor(source, dtype=torch.long)  # type: ignore[index]
        batch["token_ids_t"][row, :n] = batch["token_ids_0"][row, :n]  # type: ignore[index]
        batch["design_mask"][row, :n] = torch.as_tensor(  # type: ignore[index]
            example.design_mask, dtype=torch.float32
        )
        batch["token_mask"][row, :n] = 1.0  # type: ignore[index]
        batch["atom_positions_0"][row, :a] = torch.as_tensor(  # type: ignore[index]
            example.atom_positions, dtype=torch.float32
        )
        batch["atom_positions_t"][row, :a] = batch["atom_positions_0"][row, :a]  # type: ignore[index]
        batch["atom_mask"][row, :a] = torch.as_tensor(  # type: ignore[index]
            example.atom_mask, dtype=torch.float32
        )
        batch["atom_token_index"][row, :a] = torch.as_tensor(  # type: ignore[index]
            example.atom_token_index, dtype=torch.long
        )
        batch["atom_element"][row, :a] = torch.as_tensor(  # type: ignore[index]
            example.atom_element, dtype=torch.long
        )
    return batch


def batch_to_device(
    batch: dict[str, torch.Tensor | list[str]], device: torch.device | str
) -> dict[str, torch.Tensor | list[str]]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


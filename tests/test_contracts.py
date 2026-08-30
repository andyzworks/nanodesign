import numpy as np
import pytest

from nanodesign.v0.constants import DataSource, ExamplePurpose, Polymer, Role, Task
from nanodesign.v0.contracts import ContractError, collate_examples
from nanodesign.v0.testing import synthetic_binding_examples


def test_all_three_binding_contracts_and_collation():
    examples = synthetic_binding_examples()
    assert [example.task for example in examples] == list(Task)
    batch = collate_examples(examples)
    assert tuple(batch["task_id"].shape) == (3,)
    assert set(batch) >= {
        "atom_positions_0",
        "atom_token_index",
        "design_mask",
        "token_ids_0",
    }


def test_rna_binding_rejects_missing_target_protein():
    example = synthetic_binding_examples()[-1]
    selected = example.role_id == int(Role.RNA_APTAMER)
    example.role_id = example.role_id[selected]
    example.polymer_type = example.polymer_type[selected]
    example.token_ids = example.token_ids[selected]
    example.chain_id = example.chain_id[selected]
    example.residue_index = example.residue_index[selected]
    example.design_mask = example.design_mask[selected]
    example.atom_positions = example.atom_positions[selected]
    example.atom_mask = example.atom_mask[selected]
    example.atom_element = example.atom_element[selected]
    example.atom_token_index = np.arange(selected.sum())
    with pytest.raises(ContractError, match="roles must be exactly"):
        example.validate()


def test_rnasolo_cannot_be_labeled_as_binding_data():
    example = synthetic_binding_examples()[-1]
    example.source = DataSource.RNASOLO2
    with pytest.raises(ContractError, match="not a binding source"):
        example.validate()


def test_rnasolo_prior_is_separate_from_binding_contract():
    example = synthetic_binding_examples()[-1]
    selected = example.role_id == int(Role.RNA_APTAMER)
    example.source = DataSource.RNASOLO2
    example.purpose = ExamplePurpose.RNA_STRUCTURE_PRIOR
    example.role_id = example.role_id[selected]
    example.polymer_type = np.full(selected.sum(), int(Polymer.RNA))
    example.token_ids = example.token_ids[selected]
    example.chain_id = example.chain_id[selected]
    example.residue_index = example.residue_index[selected]
    example.design_mask = np.ones(selected.sum())
    example.atom_positions = example.atom_positions[selected]
    example.atom_mask = example.atom_mask[selected]
    example.atom_element = example.atom_element[selected]
    example.atom_token_index = np.arange(selected.sum())
    example.validate()


def test_binding_contract_rejects_roles_from_another_task():
    example = synthetic_binding_examples()[0]
    example.role_id[0] = int(Role.ANTIGEN)
    with pytest.raises(ContractError, match="roles must be exactly"):
        example.validate()


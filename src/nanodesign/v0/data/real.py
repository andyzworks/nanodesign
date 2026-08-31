"""Training loader for the concrete, split NanoDesign v0 structure catalogs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import torch
from torch.utils.data import Dataset

from nanodesign.v0.constants import (
    AA_ORDER,
    RNA_ORDER,
    DataSource,
    ExamplePurpose,
    Polymer,
    Role,
    Task,
)
from nanodesign.v0.contracts import DesignExample, collate_examples

AA_TOKEN = {letter: index + 2 for index, letter in enumerate(AA_ORDER)}
RNA_TOKEN = {letter: index + 2 + len(AA_ORDER) for index, letter in enumerate(RNA_ORDER)}
AA_ALPHABET = frozenset(AA_TOKEN)
RNA_ALPHABET = frozenset(RNA_TOKEN)


def load_split_catalog(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"empty split catalog: {path}")
    return rows


def _normalise_icode(value: str) -> str:
    return "" if value in {"", " ", "?", ".", "\x00"} else value


def _accepted_altloc(atom: gemmi.Atom) -> bool:
    return atom.altloc in {"\x00", " ", "A"}


def _protein_letter(residue_name: str) -> str | None:
    info = gemmi.find_tabulated_residue(residue_name)
    letter = info.one_letter_code
    return letter if info.is_amino_acid() and letter in AA_ALPHABET else None


def _rna_letter(residue_name: str) -> str | None:
    name = residue_name.strip().upper()
    if name in RNA_ALPHABET:
        return name
    info = gemmi.find_tabulated_residue(name)
    letter = info.one_letter_code.upper()
    return letter if info.is_nucleic_acid() and letter in RNA_ALPHABET else None


@lru_cache(maxsize=16)
def _read_model(path: str) -> gemmi.Model:
    structure = gemmi.read_structure(path)
    if len(structure) == 0:
        raise ValueError(f"structure contains no coordinate model: {path}")
    return structure[0]


def _select_chain(model: gemmi.Model, chain_id: str) -> gemmi.Chain:
    chain = model.find_chain(chain_id)
    if chain is not None:
        return chain
    chains = list(model)
    if len(chains) == 1:
        # RCSB ModelServer selections retain label_asym_id in the catalog but may write
        # auth_asym_id into the coordinate file.  A one-chain selection is unambiguous.
        return chains[0]
    raise ValueError(f"chain {chain_id!r} not found among {[item.name for item in chains]}")


def _catalog_residue_keys(
    chain_record: dict[str, Any], chain: gemmi.Chain, polymer: Polymer
) -> list[tuple[int, str]]:
    """Return the pinned selection, or reconstruct a declared full-chain selection."""

    if "residue_keys" in chain_record:
        return [(int(value[0]), str(value[1])) for value in chain_record["residue_keys"]]
    keys: list[tuple[int, str]] = []
    for residue in chain:
        letter = (
            _protein_letter(residue.name)
            if polymer == Polymer.PROTEIN
            else _rna_letter(residue.name)
        )
        representative = "CA" if polymer == Polymer.PROTEIN else "C1'"
        if letter is not None and any(
            atom.name.strip() == representative and _accepted_altloc(atom) for atom in residue
        ):
            keys.append((int(residue.seqid.num), _normalise_icode(residue.seqid.icode)))
    if len(keys) != int(chain_record["resolved_residues"]):
        raise ValueError(
            f"full-chain selection changed: expected {chain_record['resolved_residues']}, "
            f"observed {len(keys)}"
        )
    return keys


def _catalog_residues_by_key(
    chain: gemmi.Chain,
) -> dict[tuple[int, str], list[gemmi.Residue]]:
    """Keep microheterogeneous candidates instead of silently overwriting them."""

    residues: dict[tuple[int, str], list[gemmi.Residue]] = {}
    for residue in chain:
        key = (int(residue.seqid.num), _normalise_icode(residue.seqid.icode))
        residues.setdefault(key, []).append(residue)
    return residues


def _select_catalog_residue(
    candidates: dict[tuple[int, str], list[gemmi.Residue]],
    key: tuple[int, str],
    expected_letter: str,
    polymer: Polymer,
    *,
    sample_id: str,
) -> gemmi.Residue:
    """Resolve one catalog token using the same letter/altloc rules as preprocessing."""

    central_name = "CA" if polymer == Polymer.PROTEIN else "C1'"
    matching = []
    for residue in candidates.get(key, []):
        letter = (
            _protein_letter(residue.name)
            if polymer == Polymer.PROTEIN
            else _rna_letter(residue.name)
        )
        if letter == expected_letter and any(
            atom.name.strip() == central_name and _accepted_altloc(atom) for atom in residue
        ):
            matching.append(residue)
    if not matching:
        raise ValueError(
            f"{sample_id}: no catalog-matching {expected_letter} residue at {key} "
            f"with accepted atom23 center {central_name}"
        )
    # Preprocessing traverses Gemmi residues in file order. If an exact duplicate is
    # present, retain that deterministic order rather than allowing a dict overwrite.
    return matching[0]


def _role_for_token(chain: dict[str, Any], key: tuple[int, str]) -> Role:
    role = chain["role"]
    if role == "target":
        return Role.TARGET
    if role == "binder":
        return Role.BINDER
    if role == "antigen":
        return Role.ANTIGEN
    if role == "antibody_framework":
        return Role.ANTIBODY_FRAMEWORK
    if role == "antibody_framework+cdr_h3":
        design_keys = {(int(item[0]), str(item[1])) for item in chain["design_residue_keys"]}
        return Role.CDR if key in design_keys else Role.ANTIBODY_FRAMEWORK
    if role in {"rna_aptamer", "rna_design_region", "rna_structure_prior"}:
        return Role.RNA_APTAMER
    raise ValueError(f"unsupported concrete data role: {role!r}")


def _polymer_for_chain(chain: dict[str, Any]) -> Polymer:
    if chain["role"] in {"rna_aptamer", "rna_design_region", "rna_structure_prior"}:
        return Polymer.RNA
    return Polymer.PROTEIN


def _task(value: str) -> Task:
    return {
        "protein_binder": Task.PROTEIN_BINDER,
        "antibody_cdr": Task.ANTIBODY_CDR,
        "rna_aptamer": Task.RNA_APTAMER,
    }[value]


def load_catalog_example(dataset_root: str | Path, row: dict[str, Any]) -> DesignExample:
    root = Path(dataset_root).resolve()
    token_ids: list[int] = []
    polymer_types: list[int] = []
    roles: list[int] = []
    chain_ids: list[int] = []
    residue_indices: list[int] = []
    design_mask: list[int] = []
    atom_positions: list[tuple[float, float, float]] = []
    atom_masks: list[int] = []
    atom_token_indices: list[int] = []
    atom_elements: list[int] = []

    default_path = row["raw_paths"][0] if len(row["raw_paths"]) == 1 else None
    for numeric_chain_id, chain_record in enumerate(row["chains"], start=1):
        raw_path = chain_record.get("raw_path", default_path)
        if raw_path is None:
            raise ValueError(f"{row['sample_id']}: chain has no unambiguous raw_path")
        path = (root / raw_path).resolve()
        if root not in path.parents:
            raise ValueError(f"{row['sample_id']}: raw path escapes dataset root")
        model = _read_model(str(path))
        chain = _select_chain(model, str(chain_record["chain_id"]))
        polymer = _polymer_for_chain(chain_record)
        expected_keys = _catalog_residue_keys(chain_record, chain, polymer)
        residues_by_key = _catalog_residues_by_key(chain)
        observed_sequence: list[str] = []
        for within_chain_index, key in enumerate(expected_keys):
            expected_letter = chain_record["sequence"][within_chain_index]
            residue = _select_catalog_residue(
                residues_by_key,
                key,
                expected_letter,
                polymer,
                sample_id=row["sample_id"],
            )
            letter = (
                _protein_letter(residue.name)
                if polymer == Polymer.PROTEIN
                else _rna_letter(residue.name)
            )
            if letter is None:
                raise ValueError(f"{row['sample_id']}: unsupported residue {residue.name}")
            role = _role_for_token(chain_record, key)
            token_index = len(token_ids)
            observed_sequence.append(letter)
            token_ids.append(AA_TOKEN[letter] if polymer == Polymer.PROTEIN else RNA_TOKEN[letter])
            polymer_types.append(int(polymer))
            roles.append(int(role))
            chain_ids.append(numeric_chain_id)
            residue_indices.append(within_chain_index)
            design_mask.append(int(role in {Role.BINDER, Role.CDR, Role.RNA_APTAMER}))
            observed_atom_names: set[str] = set()
            for atom in residue:
                atom_name = atom.name.strip()
                if (
                    atom_name in observed_atom_names
                    or not _accepted_altloc(atom)
                    or atom.element.name == "H"
                ):
                    continue
                coordinate = (float(atom.pos.x), float(atom.pos.y), float(atom.pos.z))
                if not np.isfinite(coordinate).all():
                    continue
                observed_atom_names.add(atom_name)
                atom_positions.append(coordinate)
                atom_masks.append(1)
                atom_token_indices.append(token_index)
                atom_elements.append(int(atom.element.atomic_number))
            if token_index not in atom_token_indices:
                raise ValueError(f"{row['sample_id']}: residue {key} has no heavy atoms")
        if "".join(observed_sequence) != chain_record["sequence"]:
            raise ValueError(f"{row['sample_id']}: coordinate/catalog sequence mismatch")

    purpose = ExamplePurpose(row["purpose"])
    return DesignExample(
        sample_id=row["sample_id"].replace(":", "_"),
        task=_task(row["task"]),
        source=DataSource(row["source"]),
        purpose=purpose,
        token_ids=np.asarray(token_ids, dtype=np.int64),
        polymer_type=np.asarray(polymer_types, dtype=np.int64),
        role_id=np.asarray(roles, dtype=np.int64),
        chain_id=np.asarray(chain_ids, dtype=np.int64),
        residue_index=np.asarray(residue_indices, dtype=np.int64),
        design_mask=np.asarray(design_mask, dtype=np.int8),
        atom_positions=np.asarray(atom_positions, dtype=np.float32),
        atom_mask=np.asarray(atom_masks, dtype=np.int8),
        atom_token_index=np.asarray(atom_token_indices, dtype=np.int64),
        atom_element=np.asarray(atom_elements, dtype=np.int64),
    ).validate()


class RealCatalogDataset(Dataset[DesignExample]):
    """Lazy, directly trainable dataset backed by a frozen real-data split catalog."""

    def __init__(self, dataset_root: str | Path, catalog: str | Path):
        self.dataset_root = Path(dataset_root).resolve()
        self.rows = load_split_catalog(catalog)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> DesignExample:
        return load_catalog_example(self.dataset_root, self.rows[index])

    @staticmethod
    def collate_fn(examples: Sequence[DesignExample]):
        return collate_examples(examples)


def load_foundry_training_example(
    dataset_root: str | Path,
    row: dict[str, Any],
    *,
    noise_level: float | None = None,
    diffusion_batch_size: int = 1,
    max_context_tokens: int | None = 384,
) -> dict[str, Any]:
    """Convert one catalog row into the public RFD3NA atom23 training format.

    Protein tokens always have 14 slots and RNA tokens 23.  Unknown design sequence
    therefore changes neither atom count nor slot identity; absent side-chain/base atoms
    are virtual atoms placed on CA (protein) or C1' (RNA), exactly matching Foundry's
    atom23 padding policy.
    """

    try:
        from atomworks.ml.encoding_definitions import AF3SequenceEncoding
        from atomworks.ml.transforms.af3_reference_molecule import (
            _encode_atom_names_like_af3,
        )
        from atomworks.ml.transforms.diffusion.edm import sample_noise_edm, sample_t_edm
        from rfd3na.constants import association_schemes
    except ImportError as error:
        raise ImportError("Foundry-format data requires the project 'model' extra") from error

    root = Path(dataset_root).resolve()
    encoding = AF3SequenceEncoding()
    token_restypes: list[int] = []
    token_native_restypes: list[int] = []
    residue_indices: list[int] = []
    asym_ids: list[int] = []
    design_tokens: list[bool] = []
    protected_context_tokens: list[bool] = []
    protein_tokens: list[bool] = []
    rna_tokens: list[bool] = []
    polar_tokens: list[bool] = []
    atom_to_token: list[int] = []
    atom_names: list[str] = []
    atom_positions: list[np.ndarray] = []
    atom_is_real: list[bool] = []
    atom_is_central: list[bool] = []
    atom_is_backbone: list[bool] = []
    token_chain_names: list[str] = []
    token_residue_keys: list[tuple[int, str]] = []
    output_atom_mask: list[bool] = []

    default_path = row["raw_paths"][0] if len(row["raw_paths"]) == 1 else None
    for numeric_chain_id, chain_record in enumerate(row["chains"], start=1):
        raw_path = chain_record.get("raw_path", default_path)
        if raw_path is None:
            raise ValueError(f"{row['sample_id']}: chain has no unambiguous raw_path")
        path = (root / raw_path).resolve()
        if root not in path.parents:
            raise ValueError(f"{row['sample_id']}: raw path escapes dataset root")
        chain = _select_chain(_read_model(str(path)), str(chain_record["chain_id"]))
        polymer = _polymer_for_chain(chain_record)
        residues_by_key = _catalog_residues_by_key(chain)
        expected_keys = _catalog_residue_keys(chain_record, chain, polymer)
        for within_chain_index, key in enumerate(expected_keys):
            expected_letter = chain_record["sequence"][within_chain_index]
            residue = _select_catalog_residue(
                residues_by_key,
                key,
                expected_letter,
                polymer,
                sample_id=row["sample_id"],
            )
            role = _role_for_token(chain_record, key)
            is_design = role in {Role.BINDER, Role.CDR, Role.RNA_APTAMER}
            if polymer == Polymer.PROTEIN:
                native_name = residue.name.strip().upper()
                central_name = "CA"
                protein_tokens.append(True)
                rna_tokens.append(False)
                polar_tokens.append(
                    native_name
                    in {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "HIS", "LYS", "ARG", "ASP", "GLU"}
                )
            else:
                native_name = (_rna_letter(residue.name) or "N").upper()
                central_name = "C1'"
                protein_tokens.append(False)
                rna_tokens.append(True)
                polar_tokens.append(False)
            native_index = int(encoding.encode([native_name])[0])
            token_native_restypes.append(native_index)
            token_restypes.append(31 if is_design else native_index)
            residue_indices.append(within_chain_index)
            asym_ids.append(numeric_chain_id)
            design_tokens.append(is_design)
            protected_context_tokens.append(role == Role.ANTIBODY_FRAMEWORK)
            token_chain_names.append(str(chain_record["chain_id"]))
            token_residue_keys.append(key)

            observed = {
                atom.name.strip(): np.asarray(
                    [atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32
                )
                for atom in residue
                if _accepted_altloc(atom) and atom.element.name != "H"
            }
            if central_name not in observed:
                raise ValueError(
                    f"{row['sample_id']}: {native_name} lacks atom23 center {central_name}"
                )
            # A masked design token must not reveal its native residue through
            # residue-specific side-chain/base atom names.  Foundry's UNK/X atom23
            # schemes retain only sequence-independent protein backbone/CB or RNA
            # phosphate-ribose atoms; predicted sequence is expanded after sampling.
            input_name = (
                "UNK"
                if is_design and polymer == Polymer.PROTEIN
                else "X"
                if is_design and polymer == Polymer.RNA
                else native_name
            )
            scheme = association_schemes["atom23"].get(input_name)
            if scheme is None:
                scheme = association_schemes["atom23"]["UNK" if polymer == Polymer.PROTEIN else "X"]
            token_index = len(token_restypes) - 1
            for slot_index, scheme_name in enumerate(scheme):
                clean_name = None if scheme_name is None else str(scheme_name).strip()
                is_real = clean_name in observed
                atom_to_token.append(token_index)
                atom_names.append(clean_name if clean_name is not None else f"V{slot_index}")
                atom_positions.append(
                    observed[clean_name] if is_real else observed[central_name].copy()
                )
                atom_is_real.append(is_real)
                output_atom_mask.append((is_design and clean_name is not None) or is_real)
                atom_is_central.append(clean_name == central_name)
                atom_is_backbone.append(
                    clean_name
                    in (
                        {"N", "CA", "C", "O"}
                        if polymer == Polymer.PROTEIN
                        else {
                            "P",
                            "OP1",
                            "OP2",
                            "O5'",
                            "C5'",
                            "C4'",
                            "O4'",
                            "C3'",
                            "O3'",
                            "C2'",
                            "O2'",
                            "C1'",
                        }
                    )
                )

    if max_context_tokens is not None:
        if max_context_tokens < 0:
            raise ValueError("max_context_tokens must be non-negative or None")
        design_indices = [index for index, value in enumerate(design_tokens) if value]
        context_indices = [index for index, value in enumerate(design_tokens) if not value]
        protected_indices = [index for index, value in enumerate(protected_context_tokens) if value]
        optional_context_indices = [
            index for index in context_indices if not protected_context_tokens[index]
        ]
        if design_indices and len(context_indices) > max_context_tokens:
            centers = np.zeros((len(design_tokens), 3), dtype=np.float32)
            center_seen = np.zeros(len(design_tokens), dtype=bool)
            for token_index, position, central in zip(
                atom_to_token, atom_positions, atom_is_central, strict=True
            ):
                if central:
                    centers[token_index] = position
                    center_seen[token_index] = True
            if not center_seen.all():
                raise ValueError(f"{row['sample_id']}: token missing atom23 center")
            design_centers = centers[design_indices]
            ranked_context = sorted(
                optional_context_indices,
                key=lambda index: (
                    float(np.linalg.norm(design_centers - centers[index], axis=1).min()),
                    index,
                ),
            )
            optional_budget = max(0, max_context_tokens - len(protected_indices))
            selected = sorted(design_indices + protected_indices + ranked_context[:optional_budget])
            remap = {old: new for new, old in enumerate(selected)}
            token_restypes = [token_restypes[index] for index in selected]
            token_native_restypes = [token_native_restypes[index] for index in selected]
            residue_indices = [residue_indices[index] for index in selected]
            asym_ids = [asym_ids[index] for index in selected]
            design_tokens = [design_tokens[index] for index in selected]
            protected_context_tokens = [protected_context_tokens[index] for index in selected]
            protein_tokens = [protein_tokens[index] for index in selected]
            rna_tokens = [rna_tokens[index] for index in selected]
            polar_tokens = [polar_tokens[index] for index in selected]
            token_chain_names = [token_chain_names[index] for index in selected]
            token_residue_keys = [token_residue_keys[index] for index in selected]
            atom_selected = [index in remap for index in atom_to_token]
            atom_to_token = [
                remap[token_index]
                for token_index, keep in zip(atom_to_token, atom_selected, strict=True)
                if keep
            ]
            atom_names = [
                value for value, keep in zip(atom_names, atom_selected, strict=True) if keep
            ]
            atom_positions = [
                value for value, keep in zip(atom_positions, atom_selected, strict=True) if keep
            ]
            atom_is_real = [
                value for value, keep in zip(atom_is_real, atom_selected, strict=True) if keep
            ]
            output_atom_mask = [
                value for value, keep in zip(output_atom_mask, atom_selected, strict=True) if keep
            ]
            atom_is_central = [
                value for value, keep in zip(atom_is_central, atom_selected, strict=True) if keep
            ]
            atom_is_backbone = [
                value for value, keep in zip(atom_is_backbone, atom_selected, strict=True) if keep
            ]

    token_count = len(token_restypes)
    atom_count = len(atom_positions)
    token_design = torch.as_tensor(design_tokens, dtype=torch.bool)
    atom_token_tensor = torch.as_tensor(atom_to_token, dtype=torch.long)
    atom_design = token_design[atom_token_tensor]
    positions = torch.as_tensor(np.asarray(atom_positions), dtype=torch.float32)
    real_atom_mask = torch.as_tensor(atom_is_real, dtype=torch.bool)
    fixed_atom = ~atom_design
    motif_positions = torch.where(fixed_atom[:, None], positions, torch.zeros_like(positions))
    if diffusion_batch_size < 1:
        raise ValueError("diffusion_batch_size must be positive")
    if noise_level is None:
        timesteps = sample_t_edm(16.0, diffusion_batch_size)
    else:
        if noise_level <= 0:
            raise ValueError("noise_level must be positive")
        timesteps = torch.full((diffusion_batch_size,), float(noise_level))
    diffusion_noise = sample_noise_edm(timesteps, len(positions))
    diffusion_noise[:, ~atom_design, :] = 0.0
    noisy_positions = positions.unsqueeze(0) + diffusion_noise

    restype = torch.nn.functional.one_hot(torch.as_tensor(token_restypes), num_classes=32).long()
    native_restype = torch.nn.functional.one_hot(
        torch.as_tensor(token_native_restypes), num_classes=32
    ).float()
    protein = torch.as_tensor(protein_tokens, dtype=torch.bool)
    rna = torch.as_tensor(rna_tokens, dtype=torch.bool)
    ref_name_indices = torch.as_tensor(_encode_atom_names_like_af3(np.asarray(atom_names)))
    feats: dict[str, torch.Tensor] = {
        "unindexing_pair_mask": torch.zeros((token_count, token_count), dtype=torch.bool),
        "is_motif_token_unindexed": torch.zeros(token_count, dtype=torch.bool),
        "residue_index": torch.as_tensor(residue_indices, dtype=torch.int32),
        "token_index": torch.arange(token_count),
        "asym_id": torch.as_tensor(asym_ids, dtype=torch.long),
        "entity_id": torch.as_tensor(asym_ids, dtype=torch.long),
        "sym_id": torch.zeros(token_count, dtype=torch.int32),
        "restype": restype,
        "is_protein": protein,
        "is_rna": rna,
        "is_dna": torch.zeros(token_count, dtype=torch.bool),
        "is_ligand": torch.zeros(token_count, dtype=torch.bool),
        "is_polar": torch.as_tensor(polar_tokens, dtype=torch.bool),
        "is_protein_token": protein,
        "is_rna_token": rna,
        "is_dna_token": torch.zeros(token_count, dtype=torch.bool),
        "ref_atom_name_chars": torch.nn.functional.one_hot(
            ref_name_indices.long(), num_classes=64
        ).float(),
        "ref_pos": torch.zeros((atom_count, 3)),
        "ref_mask": torch.zeros(atom_count, dtype=torch.bool),
        "ref_element": torch.nn.functional.one_hot(
            torch.zeros(atom_count, dtype=torch.long), num_classes=128
        ).float(),
        "ref_charge": torch.zeros(atom_count, dtype=torch.int8),
        "ref_space_uid": atom_token_tensor.clone(),
        "has_zero_occupancy": ~real_atom_mask,
        "ref_is_motif_atom_with_fixed_coord": fixed_atom,
        "ref_is_motif_atom_unindexed": torch.zeros(atom_count, dtype=torch.bool),
        "ref_motif_token_type": torch.nn.functional.one_hot(
            (~token_design).long(), num_classes=3
        ).to(torch.int8),
        "motif_pos": motif_positions,
        "is_backbone": torch.as_tensor(atom_is_backbone, dtype=torch.bool),
        "is_sidechain": ~torch.as_tensor(atom_is_backbone, dtype=torch.bool),
        "is_virtual": ~real_atom_mask,
        "is_motif_atom_with_fixed_coord": fixed_atom,
        "is_motif_atom_with_fixed_seq": fixed_atom,
        "is_motif_atom_unindexed": torch.zeros(atom_count, dtype=torch.bool),
        "is_motif_token_with_fully_fixed_coord": ~token_design,
        "is_central": torch.as_tensor(atom_is_central, dtype=torch.bool),
        "is_ca": torch.as_tensor(atom_is_central, dtype=torch.bool),
        "ref_atomwise_rasa": torch.zeros((atom_count, 3), dtype=torch.long),
        "active_donor": torch.zeros(atom_count, dtype=torch.long),
        "active_acceptor": torch.zeros(atom_count, dtype=torch.long),
        "ref_plddt": torch.zeros(token_count, dtype=torch.long),
        "is_non_loopy": torch.zeros((token_count, 1)),
        "is_atom_level_hotspot": torch.zeros((atom_count, 1)),
        "bp_partners": torch.zeros((token_count, token_count, 3)),
        "token_bonds": torch.zeros((token_count, token_count), dtype=torch.bool),
        "atom_to_token_map": atom_token_tensor.to(torch.int32),
    }
    return {
        "sample_id": row["sample_id"],
        "task": row["task"],
        "f": feats,
        "X_noisy_L": noisy_positions,
        "t": timesteps,
        "ground_truth_positions": positions.unsqueeze(0).expand(diffusion_batch_size, -1, -1),
        "ground_truth_atom_mask": real_atom_mask & atom_design,
        "ground_truth_sequence": native_restype,
        "ground_truth_sequence_mask": token_design,
        "coord_atom_lvl_to_be_noised": positions.unsqueeze(0),
        # Non-tensor provenance required to turn the official sampler output into a
        # standards-compliant structure for the frozen evaluators.  RFD3 ignores
        # unknown top-level keys; train/inference receive the same feature tensors.
        "output_metadata": {
            "atom_names": atom_names,
            "atom_to_token": atom_to_token,
            "atom_output_mask": output_atom_mask,
            "token_chain_names": token_chain_names,
            "token_residue_keys": token_residue_keys,
        },
    }


class FoundryCatalogDataset(Dataset[dict[str, Any]]):
    """Lazy real-data loader that emits native RFD3NA atom23 inputs."""

    def __init__(
        self,
        dataset_root: str | Path,
        catalog: str | Path,
        *,
        max_context_tokens: int | None = 384,
        diffusion_batch_size: int = 1,
    ):
        self.dataset_root = Path(dataset_root).resolve()
        self.rows = load_split_catalog(catalog)
        self.max_context_tokens = max_context_tokens
        self.diffusion_batch_size = diffusion_batch_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return load_foundry_training_example(
            self.dataset_root,
            self.rows[index],
            max_context_tokens=self.max_context_tokens,
            diffusion_batch_size=self.diffusion_batch_size,
        )

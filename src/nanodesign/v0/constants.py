"""Closed vocabularies and identifiers fixed by the NanoDesign v0 contract."""

from __future__ import annotations

from enum import IntEnum, StrEnum


class Task(IntEnum):
    PROTEIN_BINDER = 0
    ANTIBODY_CDR = 1
    RNA_APTAMER = 2


class Polymer(IntEnum):
    PAD = 0
    PROTEIN = 1
    RNA = 2


class Role(IntEnum):
    PAD = 0
    TARGET = 1
    BINDER = 2
    ANTIGEN = 3
    ANTIBODY_FRAMEWORK = 4
    CDR = 5
    RNA_APTAMER = 6


class ExamplePurpose(StrEnum):
    BINDING_DESIGN = "binding_design"
    RNA_STRUCTURE_PRIOR = "rna_structure_prior"


class DataSource(StrEnum):
    PPIREF = "ppiref"
    PPIREF50K = "ppiref50k"
    SABDAB2 = "sabdab2"
    RIBOCENTRE_APTAMER = "ribocentre_aptamer"
    PDB_RNA_TARGET_COMPLEX = "pdb_rna_target_complex"
    RNASOLO2 = "rnasolo2"


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


PAD_TOKEN_ID = 0
MASK_TOKEN_ID = 1
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
RNA_ORDER = "ACGU"
AA_TOKEN_IDS = frozenset(range(2, 2 + len(AA_ORDER)))
RNA_TOKEN_IDS = frozenset(range(2 + len(AA_ORDER), 2 + len(AA_ORDER) + len(RNA_ORDER)))
VOCAB_SIZE = 2 + len(AA_ORDER) + len(RNA_ORDER)


def allowed_token_ids(polymer: Polymer) -> frozenset[int]:
    if polymer == Polymer.PROTEIN:
        return AA_TOKEN_IDS
    if polymer == Polymer.RNA:
        return RNA_TOKEN_IDS
    if polymer == Polymer.PAD:
        return frozenset({PAD_TOKEN_ID})
    raise ValueError(f"unsupported polymer: {polymer}")

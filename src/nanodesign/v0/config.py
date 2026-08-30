"""Strict config loader separating frozen decisions from unresolved science choices."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nanodesign.v0.constants import DataSource, Task
from nanodesign.v0.model import FOUNDRY_COMMIT, NanoDesignTinyConfig
from nanodesign.v0.spec import (
    MAX_MODEL_PARAMETERS,
    MIN_MODEL_PARAMETERS,
    MODEL_ARCHITECTURE,
    SPEC_VERSION,
)


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ConfigReport:
    valid: bool
    ready: bool
    blockers: tuple[str, ...]

    def require_ready(self) -> None:
        if not self.valid:
            raise ConfigError("configuration violates the NanoDesign v0 specification")
        if self.blockers:
            raise ConfigError("unresolved v0 decisions: " + "; ".join(self.blockers))


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ConfigError("configuration root must be a mapping")
    return value


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping")
    return value


def _list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"{path} must be a list")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ConfigError(f"{path}: missing={sorted(missing)}, unknown={sorted(unknown)}")


def _need(value: object, path: str, blockers: list[str]) -> None:
    if value is None or value == "TBD" or value == []:
        blockers.append(path)


def _validate_sha256(value: object, path: str) -> None:
    if value is None or value == "TBD":
        return
    if not isinstance(value, str) or len(value) != 64:
        raise ConfigError(f"{path} must contain 64 hexadecimal characters")
    try:
        int(value, 16)
    except ValueError as error:
        raise ConfigError(f"{path} must contain 64 hexadecimal characters") from error


def validate_v0_config(config: Mapping[str, Any]) -> ConfigReport:
    _exact_keys(config, {"schema_version", "tasks", "data", "model", "evaluation"}, "root")
    if config["schema_version"] != SPEC_VERSION:
        raise ConfigError(f"schema_version must be {SPEC_VERSION!r}")
    tasks = _list(config["tasks"], "tasks")
    expected_tasks = [task.name.lower() for task in Task]
    if tasks != expected_tasks:
        raise ConfigError(f"tasks must be exactly {expected_tasks}")

    data = _mapping(config["data"], "data")
    _exact_keys(data, set(expected_tasks), "data")
    blockers: list[str] = []

    binder = _mapping(data["protein_binder"], "data.protein_binder")
    _exact_keys(
        binder,
        {"source", "version", "split_method", "redundancy_filter", "chain_assignment"},
        "data.protein_binder",
    )
    if binder["source"] not in {
        None,
        "TBD",
        DataSource.PPIREF.value,
        DataSource.PPIREF50K.value,
    }:
        raise ConfigError("protein_binder source must be ppiref, ppiref50k, or unresolved")
    for key in (
        "source",
        "version",
        "split_method",
        "redundancy_filter",
        "chain_assignment",
    ):
        _need(binder[key], f"data.protein_binder.{key}", blockers)

    antibody = _mapping(data["antibody_cdr"], "data.antibody_cdr")
    _exact_keys(
        antibody,
        {"source", "version", "cdr_design", "split_method", "quality_filter"},
        "data.antibody_cdr",
    )
    if antibody["source"] != DataSource.SABDAB2.value:
        raise ConfigError("antibody_cdr source must be sabdab2")
    if antibody["cdr_design"] not in {None, "TBD", "h3_only", "all_six"}:
        raise ConfigError("cdr_design must be h3_only, all_six, or unresolved")
    for key in ("version", "cdr_design", "split_method", "quality_filter"):
        _need(antibody[key], f"data.antibody_cdr.{key}", blockers)

    rna = _mapping(data["rna_aptamer"], "data.rna_aptamer")
    _exact_keys(
        rna,
        {
            "binding_sources",
            "auxiliary_sources",
            "versions",
            "usable_complex_inventory",
            "split_method",
            "quality_filter",
        },
        "data.rna_aptamer",
    )
    binding_sources = _list(rna["binding_sources"], "data.rna_aptamer.binding_sources")
    auxiliary_sources = _list(rna["auxiliary_sources"], "data.rna_aptamer.auxiliary_sources")
    expected_binding = [
        DataSource.RIBOCENTRE_APTAMER.value,
        DataSource.PDB_RNA_TARGET_COMPLEX.value,
    ]
    if binding_sources != expected_binding:
        raise ConfigError("RNA binding sources must be Ribocentre + PDB RNA-target complexes")
    if auxiliary_sources != [DataSource.RNASOLO2.value]:
        raise ConfigError("RNAsolo2 must be the only auxiliary RNA-prior source")
    versions = _mapping(rna["versions"], "data.rna_aptamer.versions")
    expected_rna_sources = set(expected_binding) | {DataSource.RNASOLO2.value}
    _exact_keys(versions, expected_rna_sources, "data.rna_aptamer.versions")
    for source in (*binding_sources, *auxiliary_sources):
        _need(versions[source], f"data.rna_aptamer.versions.{source}", blockers)
    inventory = _mapping(
        rna["usable_complex_inventory"],
        "data.rna_aptamer.usable_complex_inventory",
    )
    _exact_keys(
        inventory,
        {"path", "sha256"},
        "data.rna_aptamer.usable_complex_inventory",
    )
    for key in ("path", "sha256"):
        _need(
            inventory[key],
            f"data.rna_aptamer.usable_complex_inventory.{key}",
            blockers,
        )
    _validate_sha256(inventory["sha256"], "data.rna_aptamer.usable_complex_inventory.sha256")
    for key in ("split_method", "quality_filter"):
        _need(rna[key], f"data.rna_aptamer.{key}", blockers)

    model = _mapping(config["model"], "model")
    required_model_keys = {
        "architecture",
        "foundry_commit",
        "c_s",
        "c_z",
        "c_atom",
        "c_atompair",
        "c_token",
        "c_time",
        "initializer_pairformer_blocks",
        "diffusion_pairformer_blocks",
        "diffusion_transformer_blocks",
        "atom_encoder_blocks",
        "atom_decoder_blocks",
        "atom_attention_keys",
        "recycle_steps",
        "sampling_steps",
        "atom_slot_schema",
        "max_context_tokens",
        "parameter_count",
    }
    _exact_keys(model, required_model_keys, "model")
    if model["architecture"] != MODEL_ARCHITECTURE:
        raise ConfigError(f"model.architecture must be {MODEL_ARCHITECTURE}")
    if model["foundry_commit"] != FOUNDRY_COMMIT:
        raise ConfigError(f"model.foundry_commit must be {FOUNDRY_COMMIT}")
    if model["atom_slot_schema"] != "atom23_unk_x_sequence_independent":
        raise ConfigError("model.atom_slot_schema must hide design side-chain/base identity")
    try:
        model_config = NanoDesignTinyConfig.from_mapping(
            {key: model[key] for key in NanoDesignTinyConfig.__dataclass_fields__}
        )
        if not MIN_MODEL_PARAMETERS <= int(model["parameter_count"]) <= MAX_MODEL_PARAMETERS:
            raise ValueError("parameter_count is outside the v0 budget")
        if int(model["max_context_tokens"]) < 1:
            raise ValueError("max_context_tokens must be positive")
        del model_config
    except (TypeError, ValueError) as error:
        raise ConfigError(f"invalid NanoDesign-Tiny model configuration: {error}") from error

    evaluation = _mapping(config["evaluation"], "evaluation")
    _exact_keys(evaluation, set(expected_tasks), "evaluation")
    binder_eval = _mapping(evaluation["protein_binder"], "evaluation.protein_binder")
    _exact_keys(
        binder_eval,
        {"independent_structure_predictor", "success_filter_profile", "threshold_source"},
        "evaluation.protein_binder",
    )
    for key in binder_eval:
        _need(binder_eval[key], f"evaluation.protein_binder.{key}", blockers)
    antibody_eval = _mapping(evaluation["antibody_cdr"], "evaluation.antibody_cdr")
    _exact_keys(
        antibody_eval,
        {"structure_predictor", "dockq_implementation", "rosetta_protocol"},
        "evaluation.antibody_cdr",
    )
    for key in antibody_eval:
        _need(antibody_eval[key], f"evaluation.antibody_cdr.{key}", blockers)
    rna_eval = _mapping(evaluation["rna_aptamer"], "evaluation.rna_aptamer")
    _exact_keys(
        rna_eval,
        {"rna_structure_predictor", "dockq_implementation"},
        "evaluation.rna_aptamer",
    )
    for key in rna_eval:
        _need(rna_eval[key], f"evaluation.rna_aptamer.{key}", blockers)
    return ConfigReport(valid=True, ready=not blockers, blockers=tuple(blockers))


def validate_resolved_assets(
    config: Mapping[str, Any], *, config_directory: str | Path
) -> dict[str, object]:
    """Verify the frozen RNA inventory referenced by an otherwise ready config."""

    validate_v0_config(config).require_ready()
    from nanodesign.v0.data.inventory import (  # Avoid a data/config import cycle.
        audit_rna_complex_inventory,
        load_rna_complex_inventory,
    )

    rna = config["data"]["rna_aptamer"]
    inventory_reference = rna["usable_complex_inventory"]
    path = Path(str(inventory_reference["path"]))
    if not path.is_absolute():
        path = Path(config_directory) / path
    records = load_rna_complex_inventory(path)
    report = audit_rna_complex_inventory(records)
    if report["sha256"] != inventory_reference["sha256"]:
        raise ConfigError("RNA usable-complex inventory SHA-256 mismatch")
    expected_versions = rna["versions"]
    for record in records:
        if record.source_version != expected_versions[record.source.value]:
            raise ConfigError(
                f"RNA inventory version mismatch for {record.source.value}: "
                f"{record.source_version!r} != {expected_versions[record.source.value]!r}"
            )
    return report

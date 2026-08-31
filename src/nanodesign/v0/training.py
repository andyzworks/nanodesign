"""Minimal training/checkpoint infrastructure shared by every v0 task."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import gemmi
import torch

from nanodesign.v0.config import validate_v0_config
from nanodesign.v0.model import NanoDesignTiny
from nanodesign.v0.spec import SPEC_VERSION


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("invalid optimizer or gradient-clipping configuration")


def train_step(
    model: NanoDesignTiny,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, object],
    config: TrainingConfig | None = None,
) -> dict[str, float]:
    """Run one step with the public RFD3NA diffusion and sequence losses."""

    config = config or TrainingConfig()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    loss, metrics = _compute_rfd3na_loss(batch, output)
    core_losses = torch.stack((loss, metrics["coordinate_loss"], metrics["sequence_loss"]))
    if not torch.isfinite(core_losses).all():
        raise FloatingPointError("non-finite RFD3NA coordinate or sequence loss")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("non-finite RFD3NA gradient norm")
    optimizer.step()
    return {
        key: float(value.detach().item())
        for key, value in metrics.items()
        if isinstance(value, torch.Tensor) and value.numel() == 1 and torch.isfinite(value)
    } | {"loss": float(loss.detach().item()), "gradient_norm": float(gradient_norm)}


def _compute_rfd3na_loss(
    batch: Mapping[str, object], output: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    try:
        from rfd3na.metrics.losses import DiffusionLoss, SequenceLoss
    except ImportError as error:
        raise ImportError("training requires the project 'model' extra") from error
    features = batch["f"]
    if not isinstance(features, Mapping):
        raise TypeError("batch.f must be an RFD3NA feature mapping")
    gt_positions = batch["ground_truth_positions"]
    gt_atom_mask = batch["ground_truth_atom_mask"]
    gt_sequence = batch["ground_truth_sequence"]
    gt_sequence_mask = batch["ground_truth_sequence_mask"]
    if not all(
        isinstance(value, torch.Tensor)
        for value in (gt_positions, gt_atom_mask, gt_sequence, gt_sequence_mask)
    ):
        raise TypeError("batch is missing tensor ground truth")
    loss_input = {
        "X_gt_L_in_input_frame": gt_positions,
        "crd_mask_L": gt_atom_mask,
        "is_original_unindexed_token": torch.zeros(
            gt_sequence.shape[0], dtype=torch.bool, device=gt_sequence.device
        ),
        "seq_token_lvl": gt_sequence.argmax(dim=-1),
        "sequence_valid_mask": gt_sequence_mask.float(),
    }
    coordinate_loss_module = DiffusionLoss(
        weight=4.0,
        sigma_data=16.0,
        lddt_weight=0.25,
        alpha_virtual_atom=1.0,
        alpha_polar_residues=1.0,
        alpha_ligand=10.0,
        lp_weight=0.0,
    )
    sequence_loss_module = SequenceLoss(weight=0.1, max_t=1.0)
    coordinate_loss, coordinate_metrics = coordinate_loss_module(batch, output, loss_input)
    sequence_loss, sequence_metrics = sequence_loss_module(batch, output, loss_input)
    loss = coordinate_loss + sequence_loss
    metrics = {
        "coordinate_loss": coordinate_loss,
        "sequence_loss": sequence_loss,
        **coordinate_metrics,
        **sequence_metrics,
    }
    return loss, metrics


@torch.no_grad()
def evaluate_loss(model: NanoDesignTiny, batch: Mapping[str, object]) -> dict[str, float]:
    """Evaluate the same official denoising losses without an optimizer update."""

    was_training = model.training
    model.train()
    output = model(batch)
    loss, metrics = _compute_rfd3na_loss(batch, output)
    model.train(was_training)
    core = {
        "loss": loss,
        "coordinate_loss": metrics["coordinate_loss"],
        "sequence_loss": metrics["sequence_loss"],
    }
    if not torch.isfinite(torch.stack(tuple(core.values()))).all():
        raise FloatingPointError("non-finite validation loss")
    return {name: float(value.detach().item()) for name, value in core.items()}


@torch.no_grad()
def generate(model: NanoDesignTiny, batch: Mapping[str, object]) -> dict[str, torch.Tensor]:
    """Generate sequence and atom23 coordinates with RFD3NA's official EDM sampler."""

    coordinates = batch.get("coord_atom_lvl_to_be_noised")
    if not isinstance(coordinates, torch.Tensor):
        raise TypeError("batch is missing coord_atom_lvl_to_be_noised")
    model.eval()
    return model(batch, coord_atom_lvl_to_be_noised=coordinates)


_AF3_RESIDUE_NAMES = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
    "A",
    "C",
    "G",
    "U",
    "N",
    "DA",
    "DC",
    "DG",
    "DT",
    "DN",
    "<G>",
)


def _generated_token_types(
    output: Mapping[str, torch.Tensor], batch: Mapping[str, object]
) -> torch.Tensor:
    logits = output.get("sequence_logits_I")
    if not isinstance(logits, torch.Tensor):
        raise TypeError("generation output is missing sequence_logits_I")
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError("structure export requires exactly one diffusion realization")
        logits = logits[0]
    if logits.ndim != 2 or logits.shape[-1] != len(_AF3_RESIDUE_NAMES):
        raise ValueError("sequence_logits_I must have shape [tokens, 32]")
    features = batch.get("f")
    native = batch.get("ground_truth_sequence")
    design = batch.get("ground_truth_sequence_mask")
    if not isinstance(features, Mapping) or not all(
        isinstance(value, torch.Tensor) for value in (native, design)
    ):
        raise TypeError("batch lacks sequence export features")
    protein, rna = features.get("is_protein"), features.get("is_rna")
    if not isinstance(protein, torch.Tensor) or not isinstance(rna, torch.Tensor):
        raise TypeError("batch lacks polymer masks")
    if logits.shape[0] != native.shape[0]:
        raise ValueError("generation logits and input token counts differ")
    selected = native.argmax(dim=-1).clone()
    for index in torch.nonzero(design, as_tuple=False).flatten().tolist():
        if bool(protein[index]):
            selected[index] = logits[index, :20].argmax()
        elif bool(rna[index]):
            selected[index] = logits[index, 21:25].argmax() + 21
        else:
            raise ValueError("NanoDesign v0 supports only protein or RNA design tokens")
    return selected.detach().cpu()


def write_generation_structure(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, object],
    path: str | Path,
) -> dict[str, str]:
    """Write one official RFD3NA sample as an evaluator-ready PDB.

    Sequence choices are restricted to the token's declared polymer alphabet.  Design
    residues contain only the sequence-independent atom23 UNK/X atoms that the frozen
    baseline actually predicts; no residue-specific side-chain coordinates are invented.
    """

    coordinates = output.get("X_L")
    metadata = batch.get("output_metadata")
    if not isinstance(coordinates, torch.Tensor) or not isinstance(metadata, Mapping):
        raise TypeError("generation output/batch lacks structure export data")
    if coordinates.ndim == 3:
        if coordinates.shape[0] != 1:
            raise ValueError("structure export requires exactly one diffusion realization")
        coordinates = coordinates[0]
    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
        raise ValueError("X_L must have shape [atoms, 3]")
    coordinates = coordinates.detach().float().cpu()
    if not torch.isfinite(coordinates).all():
        raise ValueError("cannot export non-finite generated coordinates")

    required = {
        "atom_names",
        "atom_to_token",
        "atom_output_mask",
        "token_chain_names",
        "token_residue_keys",
    }
    if required - set(metadata):
        raise TypeError(f"output metadata is missing {sorted(required - set(metadata))}")
    atom_names = list(metadata["atom_names"])
    atom_to_token = list(metadata["atom_to_token"])
    atom_output_mask = list(metadata["atom_output_mask"])
    chain_names = list(metadata["token_chain_names"])
    residue_keys = list(metadata["token_residue_keys"])
    if not (len(atom_names) == len(atom_to_token) == len(atom_output_mask) == len(coordinates)):
        raise ValueError("atom export metadata has inconsistent lengths")
    if len(chain_names) != len(residue_keys):
        raise ValueError("token export metadata has inconsistent lengths")

    token_types = _generated_token_types(output, batch)
    if len(token_types) != len(chain_names):
        raise ValueError("sequence and token export metadata have inconsistent lengths")
    structure = gemmi.Structure()
    structure.name = str(batch.get("sample_id", "nanodesign"))
    model = gemmi.Model("1")
    chains: dict[str, gemmi.Chain] = {}
    residues: list[gemmi.Residue] = []
    for token_index, (chain_name, key, token_type) in enumerate(
        zip(chain_names, residue_keys, token_types.tolist(), strict=True)
    ):
        if chain_name not in chains:
            chains[chain_name] = gemmi.Chain(chain_name)
        residue = gemmi.Residue()
        residue.name = _AF3_RESIDUE_NAMES[token_type]
        number, insertion = int(key[0]), str(key[1])
        residue.seqid = gemmi.SeqId(number, insertion or " ")
        residues.append(residue)
    for atom_index, (name, token_index, keep) in enumerate(
        zip(atom_names, atom_to_token, atom_output_mask, strict=True)
    ):
        if not keep:
            continue
        atom = gemmi.Atom()
        atom.name = str(name)
        atom.element = gemmi.Element(str(name).strip()[0])
        x, y, z = coordinates[atom_index].tolist()
        atom.pos = gemmi.Position(x, y, z)
        atom.occ = 1.0
        residues[int(token_index)].add_atom(atom)
    for chain_name, residue in zip(chain_names, residues, strict=True):
        chains[chain_name].add_residue(residue)
    for chain in chains.values():
        model.add_chain(chain)
    structure.add_model(model)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    structure.write_pdb(str(destination))
    sequences: dict[str, str] = {}
    for chain_name in chains:
        sequence = []
        for index, name in enumerate(chain_names):
            if name != chain_name:
                continue
            residue_name = _AF3_RESIDUE_NAMES[int(token_types[index])]
            sequence.append(gemmi.find_tabulated_residue(residue_name).one_letter_code)
        sequences[chain_name] = "".join(sequence)
    return sequences


def build_optimizer(
    model: NanoDesignTiny, config: TrainingConfig | None = None
) -> torch.optim.Optimizer:
    config = config or TrainingConfig()
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{name} must contain 64 hexadecimal characters")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must contain 64 hexadecimal characters") from error


def _assert_model_matches_resolved_config(
    model: NanoDesignTiny, resolved_config: Mapping[str, object]
) -> None:
    model_section = resolved_config.get("model")
    if not isinstance(model_section, Mapping):
        raise TypeError("resolved configuration is missing its model section")
    expected = {key: model_section.get(key) for key in model.config.__dataclass_fields__}
    if expected != asdict(model.config):
        raise ValueError(
            "runtime model configuration does not match resolved train/inference config"
        )


def save_checkpoint(
    path: str | Path,
    *,
    model: NanoDesignTiny,
    optimizer: torch.optim.Optimizer,
    step: int,
    manifest_sha256: str,
    resolved_config: Mapping[str, object],
) -> None:
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    _require_sha256(manifest_sha256, "manifest_sha256")
    validate_v0_config(resolved_config).require_ready()
    _assert_model_matches_resolved_config(model, resolved_config)
    model.validate_parameter_budget()
    config_json = json.dumps(resolved_config, sort_keys=True, separators=(",", ":"))
    state = {
        "schema_version": SPEC_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "manifest_sha256": manifest_sha256,
        "resolved_config": dict(resolved_config),
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "model_config": asdict(model.config),
        "parameter_count": model.parameter_count,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, destination)


def load_checkpoint(
    path: str | Path,
    *,
    model: NanoDesignTiny,
    optimizer: torch.optim.Optimizer | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != SPEC_VERSION:
        raise ValueError("checkpoint does not implement NanoDesign v0")
    if checkpoint.get("model_config") != asdict(model.config):
        raise ValueError("checkpoint model configuration mismatch")
    if int(checkpoint.get("parameter_count", -1)) != model.parameter_count:
        raise ValueError("checkpoint parameter count mismatch")
    resolved_config = checkpoint.get("resolved_config")
    if not isinstance(resolved_config, Mapping):
        raise TypeError("checkpoint is missing its resolved configuration")
    config_json = json.dumps(resolved_config, sort_keys=True, separators=(",", ":"))
    config_sha256 = hashlib.sha256(config_json.encode()).hexdigest()
    if checkpoint.get("config_sha256") != config_sha256:
        raise ValueError("checkpoint resolved configuration fingerprint mismatch")
    validate_v0_config(resolved_config).require_ready()
    _assert_model_matches_resolved_config(model, resolved_config)
    if expected_manifest_sha256 is not None:
        _require_sha256(expected_manifest_sha256, "expected_manifest_sha256")
        if checkpoint.get("manifest_sha256") != expected_manifest_sha256:
            raise ValueError("checkpoint dataset manifest mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint

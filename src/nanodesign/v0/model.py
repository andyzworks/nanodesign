"""NanoDesign-Tiny built directly from the public Foundry RFD3NA implementation.

No geometry or attention block is reimplemented here. This module instantiates the
published RFD3NA classes at Foundry commit aad357b776e3 and reduces only channel counts,
block counts, neighbourhood size, recycling, and EDM sampling steps.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from nanodesign.v0.spec import MAX_MODEL_PARAMETERS, MIN_MODEL_PARAMETERS

FOUNDRY_COMMIT = "aad357b776e3c0d6b973080f8f8c4bcf3ed21e40"
ATOM_ASSOCIATION_SCHEME = "atom23"


@dataclass(frozen=True)
class NanoDesignTinyConfig:
    c_s: int = 128
    c_z: int = 64
    c_atom: int = 64
    c_atompair: int = 16
    c_token: int = 240
    c_time: int = 128
    initializer_pairformer_blocks: int = 1
    diffusion_pairformer_blocks: int = 1
    diffusion_transformer_blocks: int = 6
    atom_encoder_blocks: int = 2
    atom_decoder_blocks: int = 2
    atom_attention_keys: int = 64
    recycle_steps: int = 1
    sampling_steps: int = 50

    def __post_init__(self) -> None:
        if min(self.__dict__.values()) < 1:
            raise ValueError("all NanoDesign-Tiny dimensions and block counts must be positive")
        if self.c_s % 4 or self.c_z % 4 or self.c_atom % 4 or self.c_token % 24:
            raise ValueError("channels must be divisible by unchanged RFD3NA head counts")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> NanoDesignTinyConfig:
        known = set(cls.__dataclass_fields__)
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown official RFD3NA tiny config keys: {sorted(unknown)}")
        return cls(**value)  # type: ignore[arg-type]


def _cross_attention(config: NanoDesignTinyConfig) -> dict[str, Any]:
    return {
        "method": "cross_attention",
        "cross_attention_block": {
            "n_head": 4,
            "c_model": config.c_atom,
            "dropout": 0.0,
            "kq_norm": True,
        },
    }


def _pairformer_block() -> dict[str, Any]:
    return {
        "use_triangle_attn": False,
        "use_triangle_mult": False,
        "attention_pair_bias": {"n_head": 4, "kq_norm": True},
    }


def _official_arguments(config: NanoDesignTinyConfig) -> dict[str, Any]:
    downcast = _cross_attention(config)
    atom_block = {
        "n_head": 4,
        "kq_norm": True,
        "no_residual_connection_between_attention_and_transition": False,
        "dropout": 0.0,
    }
    token_initializer = {
        "relative_position_encoding": {"r_max": 32, "s_max": 2},
        "n_pairformer_blocks": config.initializer_pairformer_blocks,
        "pairformer_block": _pairformer_block(),
        "token_1d_features": {
            "ref_motif_token_type": 3,
            "restype": 32,
            "ref_plddt": 1,
            "is_non_loopy": 1,
            "is_dna_token": 1,
            "is_rna_token": 1,
            "is_protein_token": 1,
        },
        "token_2d_features": {"bp_partners": 3},
        "downcast": downcast,
        "atom_1d_features": {
            "ref_atom_name_chars": 256,
            "ref_element": 128,
            "ref_charge": 1,
            "ref_mask": 1,
            "ref_is_motif_atom_with_fixed_coord": 1,
            "ref_is_motif_atom_unindexed": 1,
            "has_zero_occupancy": 1,
            "ref_pos": 3,
            "ref_atomwise_rasa": 3,
            "active_donor": 1,
            "active_acceptor": 1,
            "is_atom_level_hotspot": 1,
        },
        "atom_transformer": {
            "n_blocks": 0,
            "atom_transformer_block": atom_block
            | {
                "n_attn_seq_neighbours": 4,
                "n_attn_keys": config.atom_attention_keys,
            },
        },
    }
    diffusion_module = {
        "_target_": "rfd3na.model.RFD3_diffusion_module.RFD3DiffusionModule",
        "c_token": config.c_token,
        "c_t_embed": config.c_time,
        "sigma_data": 16,
        "f_pred": "edm",
        "n_attn_seq_neighbours": 2,
        "n_attn_keys": config.atom_attention_keys,
        "n_recycle": config.recycle_steps,
        "use_local_token_attention": True,
        "downcast": downcast,
        "atom_attention_encoder": {
            "n_blocks": config.atom_encoder_blocks,
            "atom_transformer_block": atom_block,
        },
        "diffusion_token_encoder": {
            "use_distogram": True,
            "use_self": True,
            "use_sinusoidal_distogram_embedder": False,
            "sigma_data": 16,
            "n_pairformer_blocks": config.diffusion_pairformer_blocks,
            "pairformer_block": _pairformer_block(),
        },
        "diffusion_transformer": {
            "n_block": config.diffusion_transformer_blocks,
            "n_registers": 0,
            "n_local_tokens": 32,
            "n_keys": config.atom_attention_keys,
            "diffusion_transformer_block": {
                "n_head": 8,
                "kq_norm": True,
                "no_residual_connection_between_attention_and_transition": False,
                "dropout": 0.0,
            },
        },
        "atom_attention_decoder": {
            "n_blocks": config.atom_decoder_blocks,
            "upcast": {
                "method": "cross_attention",
                "n_split": 3,
                "cross_attention_block": {
                    "n_head": 4,
                    "c_model": config.c_atom,
                    "dropout": 0.0,
                    "kq_norm": True,
                },
            },
            "downcast": downcast,
            "atom_transformer_block": atom_block,
        },
    }
    inference_sampler = {
        "kind": "default",
        "solver": "af3",
        "num_timesteps": config.sampling_steps,
        "min_t": 0,
        "max_t": 1,
        "sigma_data": 16,
        "s_min": 4e-4,
        "s_max": 160,
        "p": 7,
        "gamma_0": 0.8,
        "gamma_min": 1.0,
        "noise_scale": 1.003,
        "step_scale": 1.5,
        "allow_realignment": False,
        "use_classifier_free_guidance": False,
        "cfg_scale": 1.0,
        "cfg_features": [],
    }
    return {
        "c_s": config.c_s,
        "c_z": config.c_z,
        "c_atom": config.c_atom,
        "c_atompair": config.c_atompair,
        "token_initializer": token_initializer,
        "diffusion_module": diffusion_module,
        "inference_sampler": inference_sampler,
    }


class NanoDesignTiny(nn.Module):
    """Thin, provenance-pinned wrapper around the public RFD3NA model."""

    def __init__(self, config: NanoDesignTinyConfig | Mapping[str, object] | None = None):
        super().__init__()
        if config is None:
            config = NanoDesignTinyConfig()
        elif isinstance(config, Mapping):
            config = NanoDesignTinyConfig.from_mapping(config)
        self.config = config
        try:
            from rfd3na.model.RFD3 import RFD3
        except ImportError as error:
            raise ImportError(
                "NanoDesign-Tiny requires Python >=3.12 and the project 'model' extra "
                f"pinned to Foundry commit {FOUNDRY_COMMIT}"
            ) from error
        self.net = RFD3(**_official_arguments(config))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def validate_parameter_budget(self) -> None:
        if not MIN_MODEL_PARAMETERS <= self.parameter_count <= MAX_MODEL_PARAMETERS:
            raise ValueError(
                f"NanoDesign-Tiny has {self.parameter_count:,} parameters; v0 requires "
                f"{MIN_MODEL_PARAMETERS:,}-{MAX_MODEL_PARAMETERS:,}"
            )

    def forward(
        self,
        batch: Mapping[str, Any],
        coord_atom_lvl_to_be_noised: torch.Tensor | None = None,
        *,
        n_cycle: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the official denoise step (train) or EDM rollout (evaluation)."""

        if self.training and n_cycle is None:
            n_cycle = self.config.recycle_steps
        return self.net(
            input=dict(batch),
            coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised,
            n_cycle=n_cycle,
        )

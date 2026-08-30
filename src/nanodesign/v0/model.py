"""NanoDesign-Tiny: an independent small RFD3NA-style atom/token model.

The implementation follows the public high-level RFD3 pattern without copying its
code: atom encoding/downsampling -> sparse token transformer -> atom upsampling.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from nanodesign.v0.constants import VOCAB_SIZE, Polymer, Role, Task
from nanodesign.v0.spec import (
    MAX_MODEL_PARAMETERS,
    MIN_MODEL_PARAMETERS,
    MODEL_ARCHITECTURE,
)


@dataclass(frozen=True)
class NanoDesignTinyConfig:
    architecture: str = MODEL_ARCHITECTURE
    atom_dim: int = 128
    token_dim: int = 384
    num_layers: int = 6
    num_heads: int = 8
    ff_multiplier: int = 4
    max_neighbors: int = 64
    max_elements: int = 32
    max_chain_id: int = 128
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.architecture != MODEL_ARCHITECTURE:
            raise ValueError(f"architecture must be {MODEL_ARCHITECTURE}")
        if self.token_dim % self.num_heads:
            raise ValueError("token_dim must be divisible by num_heads")
        dimensions = (
            self.atom_dim,
            self.token_dim,
            self.num_layers,
            self.num_heads,
            self.ff_multiplier,
            self.max_neighbors,
            self.max_elements,
            self.max_chain_id,
        )
        if min(dimensions) < 1:
            raise ValueError("model dimensions and counts must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> NanoDesignTinyConfig:
        names = set(cls.__dataclass_fields__)
        model_only = {key: item for key, item in value.items() if key in names}
        unknown = (
            set(value)
            - names
            - {
                "num_diffusion_steps",
                "coordinate_loss_weight",
                "sequence_loss_weight",
                "atom_slot_schema",
                "capacity_benchmark",
            }
        )
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        return cls(**model_only)  # type: ignore[arg-type]


def sinusoidal_time_embedding(time: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=time.device, dtype=time.dtype)
        / max(half - 1, 1)
    )
    angles = time[:, None] * frequency[None, :]
    embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
    if embedding.shape[-1] < dimension:
        embedding = torch.nn.functional.pad(embedding, (0, dimension - embedding.shape[-1]))
    return embedding


def _scatter_atom_mean(
    atom_features: torch.Tensor,
    atom_token_index: torch.Tensor,
    atom_mask: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    batch_size, _, dimension = atom_features.shape
    output = atom_features.new_zeros((batch_size, num_tokens, dimension))
    count = atom_features.new_zeros((batch_size, num_tokens, 1))
    index = atom_token_index.clamp(0, num_tokens - 1)
    output.scatter_add_(
        1,
        index[..., None].expand(-1, -1, dimension),
        atom_features * atom_mask[..., None],
    )
    count.scatter_add_(1, index[..., None], atom_mask[..., None])
    return output / count.clamp_min(1.0)


def _token_centers(
    atom_positions: torch.Tensor,
    atom_token_index: torch.Tensor,
    atom_mask: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    return _scatter_atom_mean(atom_positions, atom_token_index, atom_mask, num_tokens)


def _reference_center(
    centers: torch.Tensor,
    token_mask: torch.Tensor,
    design_mask: torch.Tensor,
) -> torch.Tensor:
    context_mask = token_mask * (1.0 - design_mask)
    fallback = token_mask
    has_context = context_mask.sum(dim=1, keepdim=True) > 0
    weights = torch.where(has_context, context_mask, fallback)
    return (centers * weights[..., None]).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(
        1.0
    )


class SparseTokenBlock(nn.Module):
    def __init__(self, config: NanoDesignTinyConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.max_neighbors = config.max_neighbors
        self.attention_norm = nn.LayerNorm(config.token_dim)
        self.attention = nn.MultiheadAttention(
            config.token_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.feedforward_norm = nn.LayerNorm(config.token_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(config.token_dim, config.token_dim * config.ff_multiplier),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.token_dim * config.ff_multiplier, config.token_dim),
        )
        self.dropout = nn.Dropout(config.dropout)

    def _attention_mask(self, centers: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = centers.shape
        valid = token_mask.bool()
        distance = torch.cdist(centers.float(), centers.float())
        distance = distance.masked_fill(~valid[:, None, :], torch.inf)
        neighbors = min(self.max_neighbors, num_tokens)
        neighbor_index = distance.topk(neighbors, dim=-1, largest=False).indices
        allowed = torch.zeros(
            (batch_size, num_tokens, num_tokens),
            dtype=torch.bool,
            device=centers.device,
        )
        allowed.scatter_(2, neighbor_index, True)
        allowed &= valid[:, None, :]
        identity = torch.eye(num_tokens, dtype=torch.bool, device=centers.device)[None]
        allowed |= identity & valid[:, :, None]
        # Padded queries are discarded after attention but need a valid key to avoid NaN.
        allowed = torch.where(valid[:, :, None], allowed, valid[:, None, :])
        return (
            (~allowed)[:, None]
            .expand(batch_size, self.num_heads, num_tokens, num_tokens)
            .reshape(batch_size * self.num_heads, num_tokens, num_tokens)
        )

    def forward(
        self,
        token_features: torch.Tensor,
        centers: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.attention_norm(token_features)
        attention, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=self._attention_mask(centers, token_mask),
            key_padding_mask=~token_mask.bool(),
            need_weights=False,
        )
        token_features = token_features + self.dropout(attention)
        token_features = token_features + self.dropout(
            self.feedforward(self.feedforward_norm(token_features))
        )
        return token_features * token_mask[..., None]


class NanoDesignTiny(nn.Module):
    """One model pipeline for protein binder, antibody CDR, and RNA aptamer."""

    def __init__(self, config: NanoDesignTinyConfig | Mapping[str, object] | None = None):
        super().__init__()
        if config is None:
            config = NanoDesignTinyConfig()
        elif isinstance(config, Mapping):
            config = NanoDesignTinyConfig.from_mapping(config)
        self.config = config
        d = config.token_dim
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d, padding_idx=0)
        self.polymer_embedding = nn.Embedding(len(Polymer), d, padding_idx=0)
        self.role_embedding = nn.Embedding(len(Role), d, padding_idx=0)
        self.task_embedding = nn.Embedding(len(Task), d)
        self.chain_embedding = nn.Embedding(config.max_chain_id + 1, d, padding_idx=0)
        self.position_projection = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.time_projection = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.design_projection = nn.Linear(1, d)
        self.center_projection = nn.Sequential(nn.Linear(3, d), nn.SiLU(), nn.Linear(d, d))

        self.element_embedding = nn.Embedding(config.max_elements, config.atom_dim)
        self.coordinate_projection = nn.Linear(3, config.atom_dim, bias=False)
        self.atom_encoder = nn.Sequential(
            nn.Linear(config.atom_dim * 2, d),
            nn.SiLU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )
        self.input_norm = nn.LayerNorm(d)
        self.blocks = nn.ModuleList(SparseTokenBlock(config) for _ in range(config.num_layers))
        self.final_norm = nn.LayerNorm(d)
        self.atom_decoder = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.SiLU(),
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, 3),
        )
        self.sequence_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Linear(d, VOCAB_SIZE),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def validate_parameter_budget(self) -> None:
        if not MIN_MODEL_PARAMETERS <= self.parameter_count <= MAX_MODEL_PARAMETERS:
            raise ValueError(
                f"NanoDesign-Tiny has {self.parameter_count:,} parameters; v0 requires "
                f"{MIN_MODEL_PARAMETERS:,}-{MAX_MODEL_PARAMETERS:,}"
            )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        required = {
            "task_id",
            "token_ids_t",
            "polymer_type",
            "role_id",
            "chain_id",
            "residue_index",
            "design_mask",
            "token_mask",
            "atom_positions_t",
            "atom_mask",
            "atom_token_index",
            "atom_element_t",
            "diffusion_time",
        }
        missing = required - set(batch)
        if missing:
            raise KeyError(f"NanoDesign-Tiny batch is missing {sorted(missing)}")
        token_mask = batch["token_mask"].float()
        atom_mask = batch["atom_mask"].float()
        atom_to_token = batch["atom_token_index"].long()
        positions = batch["atom_positions_t"].float()
        _, num_tokens = token_mask.shape
        centers = _token_centers(positions, atom_to_token, atom_mask, num_tokens)
        reference = _reference_center(centers, token_mask, batch["design_mask"].float())
        relative_centers = (centers - reference[:, None, :]) * token_mask[..., None]
        gathered_centers = centers.gather(1, atom_to_token[..., None].expand(-1, -1, 3))
        relative_positions = (positions - gathered_centers) * atom_mask[..., None]
        atom_features = (
            self.atom_encoder(
                torch.cat(
                    (
                        self.element_embedding(
                            batch["atom_element_t"].long().clamp(0, self.config.max_elements - 1)
                        ),
                        self.coordinate_projection(relative_positions),
                    ),
                    dim=-1,
                )
            )
            * atom_mask[..., None]
        )
        pooled_atoms = _scatter_atom_mean(atom_features, atom_to_token, atom_mask, num_tokens)
        normalized_position = batch["residue_index"].float()[..., None] / 1024.0
        time = sinusoidal_time_embedding(batch["diffusion_time"].float(), self.config.token_dim)
        token_features = (
            self.token_embedding(batch["token_ids_t"].long())
            + self.polymer_embedding(batch["polymer_type"].long())
            + self.role_embedding(batch["role_id"].long())
            + self.task_embedding(batch["task_id"].long())[:, None, :]
            + self.chain_embedding(batch["chain_id"].long().clamp(0, self.config.max_chain_id))
            + self.position_projection(normalized_position)
            + self.time_projection(time)[:, None, :]
            + self.design_projection(batch["design_mask"].float()[..., None])
            + self.center_projection(relative_centers)
            + pooled_atoms
        )
        token_features = self.input_norm(token_features) * token_mask[..., None]
        for block in self.blocks:
            token_features = block(token_features, centers, token_mask)
        token_features = self.final_norm(token_features) * token_mask[..., None]
        gathered_tokens = token_features.gather(
            1,
            atom_to_token[..., None].expand(-1, -1, token_features.shape[-1]),
        )
        coordinate_noise = (
            self.atom_decoder(torch.cat((atom_features, gathered_tokens), dim=-1))
            * atom_mask[..., None]
        )
        return {
            "pred_coordinate_noise": coordinate_noise,
            "token_logits": self.sequence_head(token_features),
            "design_mask": batch["design_mask"].float(),
        }

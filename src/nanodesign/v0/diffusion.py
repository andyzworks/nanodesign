"""Shared coordinate and masked-token diffusion for all three v0 tasks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from nanodesign.v0.constants import (
    AA_TOKEN_IDS,
    MASK_TOKEN_ID,
    RNA_TOKEN_IDS,
    Polymer,
    Task,
)


@dataclass(frozen=True)
class DiffusionConfig:
    num_steps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.02
    coordinate_loss_weight: float = 1.0
    sequence_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.num_steps < 2 or not 0 < self.beta_start < self.beta_end < 1:
            raise ValueError("invalid diffusion schedule")
        if self.coordinate_loss_weight <= 0 or self.sequence_loss_weight <= 0:
            raise ValueError("coordinate and sequence loss weights must both be positive")


class UnifiedDiffusion:
    def __init__(self, config: DiffusionConfig | None = None):
        self.config = config or DiffusionConfig()
        betas = torch.linspace(self.config.beta_start, self.config.beta_end, self.config.num_steps)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def _schedule_value(self, values: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return values.to(timestep.device)[timestep.long()]

    @staticmethod
    def atom_design_mask(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return (
            batch["design_mask"].gather(1, batch["atom_token_index"].long())
            * batch["atom_mask"].float()
        )

    @staticmethod
    def _context_center(
        positions: torch.Tensor,
        atom_design: torch.Tensor,
        atom_mask: torch.Tensor,
        *,
        fallback_to_all_atoms: bool,
    ) -> torch.Tensor:
        context_mask = (1.0 - atom_design) * atom_mask
        has_context = context_mask.sum(dim=1, keepdim=True) > 0
        if fallback_to_all_atoms:
            weights = torch.where(has_context, context_mask, atom_mask)
        else:
            weights = context_mask
        center = (positions * weights[..., None]).sum(dim=1) / weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        return torch.where(has_context, center, center if fallback_to_all_atoms else 0.0)

    def corrupt(
        self,
        clean_batch: Mapping[str, torch.Tensor],
        timestep: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = dict(clean_batch)
        positions_0 = batch["atom_positions_0"].float()
        atom_mask = batch["atom_mask"].float()
        batch_size = positions_0.shape[0]
        if timestep is None:
            timestep = torch.randint(
                0, self.config.num_steps, (batch_size,), device=positions_0.device
            )
        timestep = timestep.to(positions_0.device).long().reshape(batch_size)
        if torch.any(timestep < 0) or torch.any(timestep >= self.config.num_steps):
            raise ValueError("diffusion timestep is outside the configured schedule")
        alpha_bar = self._schedule_value(self.alpha_bar, timestep).to(positions_0.dtype)
        atom_design = self.atom_design_mask(batch)
        center = self._context_center(
            positions_0,
            atom_design,
            atom_mask,
            fallback_to_all_atoms=True,
        )
        centered_0 = positions_0 - center[:, None, :]
        noise = torch.randn_like(positions_0)
        noisy = (
            center[:, None, :]
            + alpha_bar.sqrt()[:, None, None] * centered_0
            + (1.0 - alpha_bar).sqrt()[:, None, None] * noise
        )
        positions_t = (
            noisy * atom_design[..., None] + positions_0 * (1.0 - atom_design[..., None])
        ) * atom_mask[..., None]

        tokens_0 = batch["token_ids_0"].long()
        mask_probability = ((timestep.float() + 1.0) / self.config.num_steps)[:, None]
        token_loss_mask = (
            (torch.rand_like(batch["design_mask"].float()) < mask_probability)
            & batch["design_mask"].bool()
            & batch["token_mask"].bool()
        )
        for row in range(batch_size):
            if not token_loss_mask[row].any():
                candidate = torch.nonzero(
                    batch["design_mask"][row].bool() & batch["token_mask"][row].bool(),
                    as_tuple=False,
                ).flatten()
                token_loss_mask[row, candidate[0]] = True
        tokens_t = tokens_0.clone()
        tokens_t[token_loss_mask] = MASK_TOKEN_ID
        atom_elements_0 = batch["atom_element_0"].long()
        atom_elements_t = torch.where(
            atom_design.bool(), torch.zeros_like(atom_elements_0), atom_elements_0
        )
        batch.update(
            {
                "atom_positions_t": positions_t,
                "token_ids_t": tokens_t,
                "diffusion_time": (timestep.float() + 1.0) / self.config.num_steps,
                "diffusion_timestep": timestep,
                "coordinate_noise_target": noise * atom_design[..., None],
                "token_loss_mask": token_loss_mask.float(),
                "atom_element_t": atom_elements_t,
            }
        )
        return batch

    @staticmethod
    def mask_invalid_token_logits(logits: torch.Tensor, polymer_type: torch.Tensor) -> torch.Tensor:
        valid = torch.zeros_like(logits, dtype=torch.bool)
        protein_ids = torch.as_tensor(sorted(AA_TOKEN_IDS), device=logits.device)
        rna_ids = torch.as_tensor(sorted(RNA_TOKEN_IDS), device=logits.device)
        valid[..., protein_ids] |= (polymer_type == int(Polymer.PROTEIN))[..., None]
        valid[..., rna_ids] |= (polymer_type == int(Polymer.RNA))[..., None]
        return logits.masked_fill(~valid, torch.finfo(logits.dtype).min)

    @staticmethod
    def _per_example_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        axes = tuple(range(1, value.ndim))
        denominator = mask.expand_as(value).sum(dim=axes).clamp_min(1)
        return (value * mask).sum(dim=axes) / denominator

    @staticmethod
    def _task_macro_mean(value: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        return torch.stack([value[task_id == task].mean() for task in torch.unique(task_id)]).mean()

    def loss(
        self,
        output: Mapping[str, torch.Tensor],
        batch: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        atom_design = self.atom_design_mask(batch)
        coordinate_error = (
            output["pred_coordinate_noise"] - batch["coordinate_noise_target"]
        ).square()
        per_example_coordinate = self._per_example_mean(coordinate_error, atom_design)
        logits = self.mask_invalid_token_logits(output["token_logits"], batch["polymer_type"])
        token_error = F.cross_entropy(
            logits.transpose(1, 2), batch["token_ids_0"].long(), reduction="none"
        )
        token_mask = batch["token_loss_mask"].float() * batch["design_mask"].float()
        per_example_sequence = self._per_example_mean(token_error, token_mask)
        per_example = (
            self.config.coordinate_loss_weight * per_example_coordinate
            + self.config.sequence_loss_weight * per_example_sequence
        )
        task_id = batch["task_id"].long()
        result = {
            "loss": self._task_macro_mean(per_example, task_id),
            "coordinate_loss": self._task_macro_mean(per_example_coordinate, task_id),
            "sequence_loss": self._task_macro_mean(per_example_sequence, task_id),
            "per_example_loss": per_example,
        }
        for task in Task:
            selected = task_id == int(task)
            if selected.any():
                result[f"{task.name.lower()}_loss"] = per_example[selected].mean()
        return result

    def _partially_reveal_tokens(
        self,
        logits: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        next_timestep: int,
    ) -> torch.Tensor:
        predicted = logits.argmax(dim=-1)
        confidence = logits.softmax(dim=-1).amax(dim=-1)
        design = batch["design_mask"].bool() & batch["token_mask"].bool()
        result = torch.where(
            design,
            torch.full_like(batch["token_ids_0"], MASK_TOKEN_ID),
            batch["token_ids_0"],
        )
        reveal_fraction = 1.0 - (next_timestep + 1) / self.config.num_steps
        for row in range(result.shape[0]):
            indices = torch.nonzero(design[row], as_tuple=False).flatten()
            count = min(len(indices), max(1, math.ceil(len(indices) * reveal_fraction)))
            selected = indices[confidence[row, indices].topk(count).indices]
            result[row, selected] = predicted[row, selected]
        return result

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        condition_batch: Mapping[str, torch.Tensor],
        *,
        num_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = dict(condition_batch)
        positions_0 = batch["atom_positions_0"].float()
        tokens_0 = batch["token_ids_0"].long()
        atom_mask = batch["atom_mask"].float()
        atom_design = self.atom_design_mask(batch)
        center = self._context_center(
            positions_0,
            atom_design,
            atom_mask,
            fallback_to_all_atoms=False,
        )
        positions = (
            (torch.randn_like(positions_0) + center[:, None, :]) * atom_design[..., None]
            + positions_0 * (1.0 - atom_design[..., None])
        ) * atom_mask[..., None]
        tokens = torch.where(
            batch["design_mask"].bool(),
            torch.full_like(tokens_0, MASK_TOKEN_ID),
            tokens_0,
        )
        batch["atom_element_t"] = torch.where(
            atom_design.bool(),
            torch.zeros_like(batch["atom_element_0"]),
            batch["atom_element_0"],
        )
        requested_steps = num_steps or self.config.num_steps
        if requested_steps < 1:
            raise ValueError("sampling requires at least one step")
        schedule = (
            torch.linspace(
                self.config.num_steps - 1,
                0,
                requested_steps,
                device=positions.device,
            )
            .round()
            .long()
            .unique_consecutive()
        )
        last_logits: torch.Tensor | None = None
        for index, timestep in enumerate(schedule):
            t = timestep.expand(positions.shape[0])
            batch.update(
                {
                    "atom_positions_t": positions,
                    "token_ids_t": tokens,
                    "diffusion_time": (t.float() + 1.0) / self.config.num_steps,
                }
            )
            output = model(batch)
            last_logits = self.mask_invalid_token_logits(
                output["token_logits"], batch["polymer_type"]
            )
            alpha_bar = self._schedule_value(self.alpha_bar, t).to(positions.dtype)
            centered_t = positions - center[:, None, :]
            predicted_centered_0 = (
                centered_t
                - (1.0 - alpha_bar).sqrt()[:, None, None] * output["pred_coordinate_noise"]
            ) / alpha_bar.sqrt()[:, None, None].clamp_min(1e-6)
            predicted_0 = predicted_centered_0 + center[:, None, :]
            if index + 1 < len(schedule):
                next_value = int(schedule[index + 1].item())
                next_t = schedule[index + 1].expand_as(t)
                next_alpha_bar = self._schedule_value(self.alpha_bar, next_t).to(positions.dtype)
                positions = (
                    center[:, None, :]
                    + next_alpha_bar.sqrt()[:, None, None] * predicted_centered_0
                    + (1.0 - next_alpha_bar).sqrt()[:, None, None] * output["pred_coordinate_noise"]
                )
                tokens = self._partially_reveal_tokens(last_logits, batch, next_value)
            else:
                positions = predicted_0
            positions = (
                positions * atom_design[..., None] + positions_0 * (1.0 - atom_design[..., None])
            ) * atom_mask[..., None]
        assert last_logits is not None
        predicted_tokens = torch.where(
            batch["design_mask"].bool(), last_logits.argmax(dim=-1), tokens_0
        )
        return {
            "pred_atom_positions": positions,
            "pred_token_ids": predicted_tokens,
            "token_logits": last_logits,
            "design_mask": batch["design_mask"].float(),
        }

import torch

from nanodesign.v0.contracts import collate_examples
from nanodesign.v0.diffusion import DiffusionConfig, UnifiedDiffusion
from nanodesign.v0.model import NanoDesignTiny, NanoDesignTinyConfig
from nanodesign.v0.spec import MAX_MODEL_PARAMETERS, MIN_MODEL_PARAMETERS
from nanodesign.v0.testing import synthetic_binding_examples


def _small_model():
    return NanoDesignTiny(
        NanoDesignTinyConfig(
            atom_dim=32,
            token_dim=64,
            num_layers=2,
            num_heads=4,
            ff_multiplier=2,
            max_neighbors=8,
        )
    )


def _tensor_batch():
    return {
        key: value
        for key, value in collate_examples(synthetic_binding_examples()).items()
        if isinstance(value, torch.Tensor)
    }


def test_default_model_is_inside_v0_parameter_range():
    model = NanoDesignTiny()
    assert MIN_MODEL_PARAMETERS <= model.parameter_count <= MAX_MODEL_PARAMETERS
    model.validate_parameter_budget()


def test_one_model_runs_all_three_tasks_and_backpropagates():
    model = _small_model()
    diffusion = UnifiedDiffusion(DiffusionConfig(num_steps=40))
    corrupted = diffusion.corrupt(_tensor_batch(), timestep=torch.tensor([5, 10, 20]))
    output = model(corrupted)
    assert output["pred_coordinate_noise"].shape == corrupted["atom_positions_t"].shape
    assert output["token_logits"].shape[:2] == corrupted["token_ids_t"].shape
    losses = diffusion.loss(output, corrupted)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert all(
        name in losses
        for name in ("protein_binder_loss", "antibody_cdr_loss", "rna_aptamer_loss")
    )


def test_corruption_and_sampling_keep_context_exact():
    batch = _tensor_batch()
    diffusion = UnifiedDiffusion(DiffusionConfig(num_steps=10))
    corrupted = diffusion.corrupt(batch, timestep=torch.tensor([9, 9, 9]))
    atom_design = diffusion.atom_design_mask(batch)
    context = (1.0 - atom_design) * batch["atom_mask"]
    assert torch.equal(
        corrupted["atom_positions_t"] * context[..., None],
        batch["atom_positions_0"] * context[..., None],
    )
    sampled = diffusion.sample(_small_model().eval(), batch, num_steps=3)
    assert torch.allclose(
        sampled["pred_atom_positions"] * context[..., None],
        batch["atom_positions_0"] * context[..., None],
    )


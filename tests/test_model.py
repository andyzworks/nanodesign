from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("rfd3na")

from nanodesign.v0.model import STANDARD_MODE_MAX_ATOMS, NanoDesignTiny, NanoDesignTinyConfig
from nanodesign.v0.spec import MAX_MODEL_PARAMETERS, MIN_MODEL_PARAMETERS


def test_default_model_is_official_rfd3na_and_inside_v0_parameter_range():
    model = NanoDesignTiny()
    assert model.net.__class__.__module__ == "rfd3na.model.RFD3"
    assert MIN_MODEL_PARAMETERS <= model.parameter_count <= MAX_MODEL_PARAMETERS
    model.validate_parameter_budget()


def test_token_channel_must_satisfy_official_decoder_and_attention_divisibility():
    with pytest.raises(ValueError, match="divisible"):
        NanoDesignTinyConfig(c_token=256)


def test_official_sampler_requires_at_least_two_steps():
    with pytest.raises(ValueError, match="at least two"):
        NanoDesignTinyConfig(sampling_steps=1)


def test_low_memory_training_wires_official_chunked_initializer_outputs():
    model = NanoDesignTiny.__new__(NanoDesignTiny)
    torch.nn.Module.__init__(model)
    model.config = NanoDesignTinyConfig()
    model.execution_mode = "chunked"
    model.last_execution_mode = None
    model.net = MagicMock()
    model.net.token_initializer.use_chunked_pll = True
    chunked_embedder = object()
    model.net.token_initializer.return_value = {
        "Q_L_init": "q",
        "C_L": "c",
        "S_I": "s",
        "Z_II": "z",
        "chunked_pairwise_embedder": chunked_embedder,
    }
    model.net.diffusion_module.return_value = {"X_L": torch.zeros(1, 2, 3)}
    features = {"feature": torch.ones(1)}
    batch = {
        "f": features,
        "X_noisy_L": torch.zeros(1, 2, 3),
        "t": torch.ones(1),
    }

    result = model(batch)

    assert result["X_L"].shape == (1, 2, 3)
    assert model.last_execution_mode == "chunked"
    model.net.diffusion_module.assert_called_once_with(
        X_noisy_L=batch["X_noisy_L"],
        t=batch["t"],
        f=features,
        n_recycle=model.config.recycle_steps,
        P_LL=None,
        chunked_pairwise_embedder=chunked_embedder,
        initializer_outputs={"Q_L_init": "q", "C_L": "c", "S_I": "s", "Z_II": "z"},
        Q_L_init="q",
        C_L="c",
        S_I="s",
        Z_II="z",
    )


def test_auto_execution_mode_uses_profiled_atom_threshold():
    model = NanoDesignTiny.__new__(NanoDesignTiny)
    torch.nn.Module.__init__(model)
    model.execution_mode = "auto"
    standard = {"f": {"atom_to_token_map": torch.zeros(STANDARD_MODE_MAX_ATOMS)}}
    chunked = {"f": {"atom_to_token_map": torch.zeros(STANDARD_MODE_MAX_ATOMS + 1)}}

    assert model.execution_mode_for(standard) == "standard"
    assert model.execution_mode_for(chunked) == "chunked"


def test_explicit_execution_mode_overrides_atom_threshold():
    model = NanoDesignTiny.__new__(NanoDesignTiny)
    torch.nn.Module.__init__(model)
    batch = {"f": {"atom_to_token_map": torch.zeros(STANDARD_MODE_MAX_ATOMS + 1)}}
    model.execution_mode = "standard"
    assert model.execution_mode_for(batch) == "standard"
    model.execution_mode = "chunked"
    assert model.execution_mode_for(batch) == "chunked"


def test_low_memory_environment_preserves_benchmark_override(monkeypatch):
    monkeypatch.setenv("RFD3_LOW_MEMORY_MODE", "1")
    model = NanoDesignTiny()
    assert model.execution_mode == "chunked"

import pytest

from nanodesign.v0.config import ConfigError, load_config
from nanodesign.v0.model import NanoDesignTiny, NanoDesignTinyConfig
from nanodesign.v0.training import build_optimizer, save_checkpoint


def test_checkpoint_cannot_claim_formal_run_with_unresolved_config(tmp_path):
    model = NanoDesignTiny(
        NanoDesignTinyConfig(
            atom_dim=16,
            token_dim=32,
            num_layers=1,
            num_heads=4,
            ff_multiplier=2,
            max_neighbors=4,
        )
    )
    with pytest.raises(ConfigError, match="unresolved v0 decisions"):
        save_checkpoint(
            tmp_path / "invalid.pt",
            model=model,
            optimizer=build_optimizer(model),
            step=0,
            manifest_sha256="0" * 64,
            resolved_config=load_config("configs/v0.yaml"),
        )


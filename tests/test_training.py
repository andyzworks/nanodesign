import pytest

from nanodesign.v0.training import TrainingConfig


def test_training_config_rejects_invalid_optimizer_values():
    with pytest.raises(ValueError, match="invalid"):
        TrainingConfig(learning_rate=0)

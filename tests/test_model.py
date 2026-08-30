import pytest

pytest.importorskip("rfd3na")

from nanodesign.v0.model import NanoDesignTiny, NanoDesignTinyConfig
from nanodesign.v0.spec import MAX_MODEL_PARAMETERS, MIN_MODEL_PARAMETERS


def test_default_model_is_official_rfd3na_and_inside_v0_parameter_range():
    model = NanoDesignTiny()
    assert model.net.__class__.__module__ == "rfd3na.model.RFD3"
    assert MIN_MODEL_PARAMETERS <= model.parameter_count <= MAX_MODEL_PARAMETERS
    model.validate_parameter_budget()


def test_token_channel_must_satisfy_official_decoder_and_attention_divisibility():
    with pytest.raises(ValueError, match="divisible"):
        NanoDesignTinyConfig(c_token=256)

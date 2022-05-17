import pytest
import functools
from autogluon.text.text_prediction.presets import (
    get_text_preset,
    list_text_presets,
)
from autogluon.text.automm.utils import get_config


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split("."))


def test_presets():
    available_presets = list_text_presets(verbose=True)

    for preset, to_be_verified in available_presets.items():
        automm_preset, overrides = get_text_preset(preset)
        config = get_config(
            config=automm_preset,
            overrides=overrides,
        )
        for k, v in to_be_verified.items():
            assert v == rgetattr(config, k)
        assert sorted(config.model.names) == sorted(["hf_text", "numerical_mlp", "categorical_mlp", "fusion_mlp"])
    # test invalid presets
    presets = ["hello", "haha"]
    for preset in presets:
        with pytest.raises(ValueError):
            config = get_text_preset(preset)

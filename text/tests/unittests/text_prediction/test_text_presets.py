import pytest
import functools
from autogluon.text.text_prediction.presets import (
    text_preset_to_config,
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
        config, overrides = text_preset_to_config(preset)
        config = get_config(
            config=config,
            overrides=overrides,
        )
        for k, v in to_be_verified.items():
            assert v == rgetattr(config, k)

    # test invalid presets
    presets = ["hello", "haha"]
    for preset in presets:
        with pytest.raises(ValueError):
            config = text_preset_to_config(preset)

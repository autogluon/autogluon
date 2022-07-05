import pytest
import functools
from autogluon.multimodal.utils import get_config
from autogluon.text.text_prediction.presets import (
    get_text_preset,
    list_text_presets,
)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split("."))


def test_presets():
    available_presets = list_text_presets(verbose=True)

    for preset, to_be_verified in available_presets.items():
        overrides = get_text_preset(preset)
        config = get_config(
            overrides=overrides,
        )
        for k, v in to_be_verified.items():
            assert v == rgetattr(config, k)

    # test invalid presets
    presets = ["hello", "haha"]
    for preset in presets:
        with pytest.raises(ValueError):
            config = get_text_preset(preset)

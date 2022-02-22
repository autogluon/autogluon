import pytest
from autogluon.text.automm.presets import (
    preset_to_config,
    list_model_presets,
)


def test_presets():
    all_model_presets = list_model_presets()
    all_model_presets += ["text", "image"]
    for preset in all_model_presets:
        config = preset_to_config(preset)

    # test cases
    config1 = preset_to_config("Text")
    config2 = preset_to_config("TEXT")
    assert config1 == config2
    config1 = preset_to_config("IMAGE")
    config2 = preset_to_config("IMaGE")
    assert config1 == config2

    # test non-existing types
    non_exist_types = ["hello", "haha"]
    for per_type in non_exist_types:
        with pytest.raises(ValueError):
            config = preset_to_config(per_type)

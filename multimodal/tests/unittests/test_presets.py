import pytest
from autogluon.multimodal.presets import (
    automm_preset_to_config,
    list_automm_presets,
)


def test_presets():
    all_model_presets = list_automm_presets()
    for preset in all_model_presets:
        config, overrides = automm_preset_to_config(preset)

    # test non-existing types
    non_exist_types = ["hello", "haha"]
    for per_type in non_exist_types:
        with pytest.raises(ValueError):
            config, overrides = automm_preset_to_config(per_type)

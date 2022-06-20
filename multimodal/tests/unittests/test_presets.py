import pytest
from autogluon.multimodal.presets import (
    get_automm_preset,
    list_automm_presets,
)


def test_presets():
    all_model_presets = list_automm_presets()
    for preset in all_model_presets:
        config, overrides = get_automm_preset(preset)

    # test non-existing types
    non_exist_types = ["hello", "haha"]
    for per_type in non_exist_types:
        with pytest.raises(ValueError):
            config = get_automm_preset(per_type)

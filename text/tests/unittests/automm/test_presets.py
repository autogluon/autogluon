import pytest
from autogluon.text.automm.presets import (
    get_preset,
    list_model_presets,
)


def test_presets():
    all_model_presets = list_model_presets()
    for preset in all_model_presets:
        config = get_preset(preset)

    # test non-existing types
    non_exist_types = ["hello", "haha"]
    for per_type in non_exist_types:
        with pytest.raises(ValueError):
            config = get_preset(per_type)

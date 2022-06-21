import pytest
from omegaconf import OmegaConf
from autogluon.multimodal.presets import (
    get_automm_preset,
    list_automm_presets,
)
from autogluon.multimodal.utils import get_config


def test_presets():
    all_presets = list_automm_presets()
    for preset in all_presets:
        basic_config, overrides = get_automm_preset(preset)

    # test non-existing types
    non_exist_types = ["hello", "haha"]
    for per_type in non_exist_types:
        with pytest.raises(ValueError):
            basic_config, overrides = get_automm_preset(per_type)


def test_preset_to_config():
    all_presets = list_automm_presets()
    for preset in all_presets:
        basic_config, overrides = get_automm_preset(preset)
        config = get_config(preset=preset)
        for k, v in overrides.items():
            assert OmegaConf.select(config, k) == v

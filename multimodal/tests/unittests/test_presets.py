import pytest
from omegaconf import OmegaConf

from autogluon.multimodal.constants import DATA, DISTILLER, ENVIRONMENT, MODEL, OPTIMIZATION
from autogluon.multimodal.presets import get_automm_presets, get_basic_automm_config, list_automm_presets
from autogluon.multimodal.utils import get_config


def test_presets():
    all_presets = list_automm_presets()
    for preset in all_presets:
        overrides = get_automm_presets(preset)

    # test non-existing types
    non_exist_types = ["hello", "haha"]
    for per_type in non_exist_types:
        with pytest.raises(ValueError):
            overrides = get_automm_presets(per_type)


def test_preset_to_config():
    all_presets = list_automm_presets()
    for preset in all_presets:
        overrides = get_automm_presets(preset)
        config = get_config(presets=preset)
        for k, v in overrides.items():
            assert OmegaConf.select(config, k) == v


def test_basic_config():
    basic_config = get_basic_automm_config()
    assert list(basic_config.keys()).sort() == [MODEL, DATA, OPTIMIZATION, ENVIRONMENT].sort()

    basic_config = get_basic_automm_config(extra=[DISTILLER])
    assert list(basic_config.keys()).sort() == [MODEL, DATA, OPTIMIZATION, ENVIRONMENT, DISTILLER].sort()

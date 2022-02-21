import pytest
import functools
from autogluon.text.text_prediction.text_presets import (
    text_preset_to_config,
    list_text_presets,
)
from autogluon.text.automm.utils import get_config


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split("."))


def test_presets():
    available_presets = list_text_presets()
    to_be_verified = [
        {
            "model.hf_text.checkpoint_name": "google/electra-base-discriminator",
        },
        {
            "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
            "optimization.learning_rate": 4e-4,
        },
        {
            "model.hf_text.checkpoint_name": "google/electra-base-discriminator",
        },
        {
            "model.hf_text.checkpoint_name": "microsoft/deberta-v3-base",
            "env.per_gpu_batch_size": 2,
        },
        {
            "model.hf_text.checkpoint_name": "microsoft/mdeberta-v3-base",
            "env.precision": "bf16",
            "env.per_gpu_batch_size": 2,
        },
    ]
    assert len(available_presets) == len(to_be_verified)

    for preset, tbv in zip(available_presets, to_be_verified):
        config, overrides = text_preset_to_config(preset)
        config = get_config(
            config=config,
            overrides=overrides,
        )
        for k, v in tbv.items():
            assert v == rgetattr(config, k)

    # test invalid presets
    presets = ["hello", "haha"]
    for preset in presets:
        with pytest.raises(ValueError):
            config = text_preset_to_config(preset)

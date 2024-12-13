import pytest
from omegaconf import OmegaConf

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import (
    ALL_MODEL_QUALITIES,
    DATA,
    DISTILLER,
    ENV,
    FEATURE_EXTRACTION,
    MODEL,
    OBJECT_DETECTION,
    OPTIM,
    ZERO_SHOT_IMAGE_CLASSIFICATION,
)
from autogluon.multimodal.utils import PROBLEM_TYPES_REG, get_basic_config, get_config, get_presets, list_presets


def test_get_presets():
    problem_types = list_presets()
    for per_type in problem_types:
        for model_quality in ALL_MODEL_QUALITIES:
            hyperparameters, hyperparameter_tune_kwargs = get_presets(per_type, model_quality)

    # test non-existing types
    non_exist_types = ["hello", "haha"]
    for per_type in non_exist_types:
        for model_quality in ALL_MODEL_QUALITIES:
            with pytest.raises(ValueError):
                hyperparameters, hyperparameter_tune_kwargs = get_presets(per_type, model_quality)


def test_preset_to_config():
    problem_types = list_presets()
    for per_type in problem_types:
        for model_quality in ALL_MODEL_QUALITIES:
            if per_type != "ensemble":
                hyperparameters, _ = get_presets(per_type, model_quality)
                config = get_config(
                    problem_type=per_type,
                    presets=model_quality,
                    extra=["matcher", "distiller"],
                )
                for k, v in hyperparameters.items():
                    assert OmegaConf.select(config, k) == v
            else:
                hyperparameters, _ = get_presets(per_type, model_quality)
                for per_name, per_hparams in hyperparameters.items():
                    config = get_config(
                        presets=model_quality,
                        extra=["matcher", "distiller"],
                        overrides=per_hparams,
                    )
                    for k, v in per_hparams.items():
                        assert OmegaConf.select(config, k) == v or (
                            OmegaConf.select(config, k) is None and v == "null"
                        )


@pytest.mark.parametrize("problem_type", list(PROBLEM_TYPES_REG.list_keys()))
@pytest.mark.parametrize("presets", ALL_MODEL_QUALITIES)
def test_presets_in_init(problem_type, presets):
    if problem_type != OBJECT_DETECTION:
        predictor = MultiModalPredictor(problem_type=problem_type, presets=presets)

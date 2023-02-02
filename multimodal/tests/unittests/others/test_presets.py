import pytest
from omegaconf import OmegaConf

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import (
    ALL_MODEL_QUALITIES,
    DATA,
    DISTILLER,
    ENVIRONMENT,
    FEATURE_EXTRACTION,
    FEW_SHOT_TEXT_CLASSIFICATION,
    MODEL,
    OCR_TEXT_DETECTION,
    OCR_TEXT_RECOGNITION,
    OPTIMIZATION,
    ZERO_SHOT_IMAGE_CLASSIFICATION,
)
from autogluon.multimodal.presets import get_automm_presets, get_basic_automm_config, list_automm_presets
from autogluon.multimodal.problem_types import PROBLEM_TYPES_REG
from autogluon.multimodal.utils import get_config


def test_presets():
    problem_types = list_automm_presets()
    for per_type in problem_types:
        for model_quality in ALL_MODEL_QUALITIES:
            hyperparameters, hyperparameter_tune_kwargs = get_automm_presets(per_type, model_quality)

    # test non-existing types
    non_exist_types = ["hello", "haha"]
    for per_type in non_exist_types:
        for model_quality in ALL_MODEL_QUALITIES:
            with pytest.raises(ValueError):
                hyperparameters, hyperparameter_tune_kwargs = get_automm_presets(per_type, model_quality)


def test_preset_to_config():
    problem_types = list_automm_presets()
    for per_type in problem_types:
        for model_quality in ALL_MODEL_QUALITIES:
            hyperparameters, _ = get_automm_presets(per_type, model_quality)
            config = get_config(
                problem_type=per_type,
                presets=model_quality,
                extra=["matcher", "distiller"],
            )
            for k, v in hyperparameters.items():
                assert OmegaConf.select(config, k) == v


def test_basic_config():
    basic_config = get_basic_automm_config()
    assert list(basic_config.keys()).sort() == [MODEL, DATA, OPTIMIZATION, ENVIRONMENT].sort()

    basic_config = get_basic_automm_config(extra=[DISTILLER])
    assert list(basic_config.keys()).sort() == [MODEL, DATA, OPTIMIZATION, ENVIRONMENT, DISTILLER].sort()


@pytest.mark.parametrize("problem_type", list(PROBLEM_TYPES_REG.list_keys()))
@pytest.mark.parametrize("presets", ALL_MODEL_QUALITIES)
def test_preset_in_init(problem_type, presets):
    if problem_type in [
        OCR_TEXT_DETECTION,
        OCR_TEXT_RECOGNITION,
    ]:
        pytest.skip(reason="Need to fix these presets before testing them.")

    predictor = MultiModalPredictor(problem_type=problem_type)
    if problem_type not in [
        FEATURE_EXTRACTION,
        FEW_SHOT_TEXT_CLASSIFICATION,
        ZERO_SHOT_IMAGE_CLASSIFICATION,
    ]:
        predictor = MultiModalPredictor(problem_type=problem_type, presets=presets)

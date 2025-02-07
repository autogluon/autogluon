from typing import Any, Callable, Dict, Optional, Type
from unittest.mock import patch

import pytest

from .common import (
    ALL_LOCAL_MODELS,
    CHRONOS_CLASSIC_MODEL_PATH,
    CHRONOS_BOLT_MODEL_PATH,
    GLUONTS_MODELS,
    GLUONTS_MODELS_WITH_KNOWN_COVARIATES,
    GLUONTS_MODELS_WITH_STATIC_FEATURES,
    GLUONTS_MODELS_WITH_STATIC_FEATURES_AND_KNOWN_COVARIATES,
    MLFORECAST_MODELS,
    NONSEASONAL_LOCAL_MODELS,
    SEASONAL_LOCAL_MODELS,
    SEASONAL_LOCAL_MODELS_EXTRA,
    AbstractTimeSeriesModel,
    ChronosModel,
    get_default_hyperparameters,
    get_multi_window_deepar,
)


def patch_constructor(model_class: Type[AbstractTimeSeriesModel], extra_hyperparameters: Optional[Dict[str, Any]] = None) -> Callable[..., AbstractTimeSeriesModel]:
    """Return a model constructor function that provides additional hyperparameters
    from this module in addition to the ones defined in the respective tests."""
    
    default_hyperparameters = get_default_hyperparameters(model_class)
    if extra_hyperparameters is None:
        extra_hyperparameters = {}

    def constructor(*args, **kwargs):
        hyperparameters = {
            **default_hyperparameters,
            **extra_hyperparameters,
            **kwargs.get("hyperparameters", {}),
        }
        return model_class(*args, **{**kwargs, "hyperparameters": hyperparameters})

    return constructor


@pytest.fixture(params=ALL_LOCAL_MODELS)
def local_model_class(request):
    with patch.object(request.param, "is_local_model_arg_allowed", return_value=True):
        yield patch_constructor(request.param)


@pytest.fixture(params=SEASONAL_LOCAL_MODELS + SEASONAL_LOCAL_MODELS_EXTRA)
def seasonal_local_model_class(request):
    with patch.object(request.param, "is_local_model_arg_allowed", return_value=True):
        yield patch_constructor(request.param)


@pytest.fixture(params=NONSEASONAL_LOCAL_MODELS)
def nonseasonal_local_model_class(request):
    with patch.object(request.param, "is_local_model_arg_allowed", return_value=True):
        yield patch_constructor(request.param)


@pytest.fixture(params=GLUONTS_MODELS)
def gluonts_model_class(request):
    yield patch_constructor(request.param)


@pytest.fixture(params=GLUONTS_MODELS_WITH_STATIC_FEATURES)
def gluonts_model_with_static_features_class(request):
    yield patch_constructor(request.param)


@pytest.fixture(params=GLUONTS_MODELS_WITH_KNOWN_COVARIATES)
def gluonts_model_with_known_covariates_class(request):
    yield patch_constructor(request.param)


@pytest.fixture(params=GLUONTS_MODELS_WITH_STATIC_FEATURES_AND_KNOWN_COVARIATES)
def gluonts_model_with_known_covariates_and_static_features_class(request):
    yield patch_constructor(request.param)


@pytest.fixture(params=MLFORECAST_MODELS)
def mlforecast_model_class(request):
    yield patch_constructor(request.param)


@pytest.fixture()
def multi_window_deepar_model_class():
    yield get_multi_window_deepar
    

@pytest.fixture(params=[CHRONOS_BOLT_MODEL_PATH, CHRONOS_CLASSIC_MODEL_PATH])
def chronos_zero_shot_model_class(request):
    yield patch_constructor(ChronosModel, extra_hyperparameters={"model_path": request.param})
    
    
@pytest.fixture(
    params=[  # model_path, fine_tune
        (CHRONOS_BOLT_MODEL_PATH, False), 
        (CHRONOS_CLASSIC_MODEL_PATH, False),
        (CHRONOS_BOLT_MODEL_PATH, True), 
        (CHRONOS_CLASSIC_MODEL_PATH, True),
    ]
)
def chronos_model_class(request):
    extra_hyperparameters = {"model_path": request.param[0]}
    if request.param[1]:
        extra_hyperparameters |= {"fine_tune": True, "fine_tune_steps": 10}
    
    yield patch_constructor(ChronosModel, extra_hyperparameters=extra_hyperparameters)


@pytest.fixture(
    scope="session",
    params=(
        GLUONTS_MODELS + 
        SEASONAL_LOCAL_MODELS + 
        NONSEASONAL_LOCAL_MODELS + 
        MLFORECAST_MODELS + [
            patch_constructor(ChronosModel, extra_hyperparameters={"model_path": CHRONOS_BOLT_MODEL_PATH}),
            patch_constructor(ChronosModel, extra_hyperparameters={"model_path": CHRONOS_CLASSIC_MODEL_PATH}),
            patch_constructor(ChronosModel, extra_hyperparameters={"model_path": CHRONOS_BOLT_MODEL_PATH, "fine_tune": True, "fine_tune_steps": 10}),
            patch_constructor(ChronosModel, extra_hyperparameters={"model_path": CHRONOS_CLASSIC_MODEL_PATH, "fine_tune": True, "fine_tune_steps": 10}),
        ]
    )
)
def model_class(request):
    yield patch_constructor(request.param)


@pytest.fixture(
    scope="session",
    params=(
        SEASONAL_LOCAL_MODELS + 
        NONSEASONAL_LOCAL_MODELS + 
        MLFORECAST_MODELS + [
            patch_constructor(ChronosModel, extra_hyperparameters={"model_path": CHRONOS_BOLT_MODEL_PATH}),
            patch_constructor(ChronosModel, extra_hyperparameters={"model_path": CHRONOS_CLASSIC_MODEL_PATH}),
        ]
    )
)
def inference_only_model_class(request):
    yield patch_constructor(request.param)
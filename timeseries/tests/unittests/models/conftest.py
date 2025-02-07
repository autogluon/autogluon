from typing import Any, Callable, Dict, Type
from unittest.mock import patch

import pytest

from . import (
    ALL_LOCAL_MODELS,
    GLUONTS_MODELS,
    GLUONTS_MODELS_WITH_KNOWN_COVARIATES,
    GLUONTS_MODELS_WITH_STATIC_FEATURES,
    GLUONTS_MODELS_WITH_STATIC_FEATURES_AND_KNOWN_COVARIATES,
    MLFORECAST_MODELS,
    NONSEASONAL_LOCAL_MODELS,
    SEASONAL_LOCAL_MODELS,
    SEASONAL_LOCAL_MODELS_EXTRA,
    AbstractGluonTSModel,
    AbstractLocalModel,
    AbstractMLForecastModel,
    AbstractTimeSeriesModel,
    AutoARIMAModel,
    AutoCESModel,
    AutoETSModel,
    DeepARModel,
    MultiWindowBacktestingModel,
)

DEFAULT_HYPERPARAMETERS: Dict[Type[AbstractTimeSeriesModel], Dict] = {
    # Supertypes should come first, so that the most specific hyperparameters are used
    # in case of an overlap
    AbstractLocalModel: {"n_jobs": 1, "use_fallback_model": False},
    AbstractGluonTSModel: {"max_epochs": 1, "num_batches_per_epoch": 1},
    AbstractMLForecastModel: {"tabular_hyperparameters": {"DUMMY": {}}},
    AutoARIMAModel: {
        "nmodels": 5,
        "max_p": 2,
        "max_P": 1,
        "max_q": 2,
        "max_Q": 1,
        "max_d": 1,
        "max_D": 1,
    },
    AutoETSModel: {
        "model": "ZNN",
    },
    AutoCESModel: {"model": "S"},
}


def get_default_hyperparameters(model_type: Type[AbstractTimeSeriesModel]) -> Dict[str, Any]:
    default_hyperparameters = {}

    for type_, hps in DEFAULT_HYPERPARAMETERS.items():
        if issubclass(model_type, type_) or model_type is type_:
            default_hyperparameters |= hps

    return default_hyperparameters


def patch_constructor(model_class: Type[AbstractTimeSeriesModel]) -> Callable[..., AbstractTimeSeriesModel]:
    default_hyperparameters = get_default_hyperparameters(model_class)

    def constructor(*args, **kwargs):
        hyperparameters = kwargs.get("hyperparameters", {})
        hyperparameters = {
            **default_hyperparameters,
            **hyperparameters,
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
    yield patch_constructor(
        request.param,
    )


def get_multi_window_deepar(hyperparameters=None, **kwargs):
    """Wrap DeepAR inside MultiWindowBacktestingModel."""
    if hyperparameters is None:
        hyperparameters = {"max_epochs": 1, "num_batches_per_epoch": 1}
    model_base_kwargs = {**kwargs, "hyperparameters": hyperparameters}
    return MultiWindowBacktestingModel(model_base=DeepARModel, model_base_kwargs=model_base_kwargs, **kwargs)


@pytest.fixture()
def multi_window_deepar_model_class():
    yield get_multi_window_deepar

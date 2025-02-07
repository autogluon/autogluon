from typing import Any, Callable, Dict, Type

import pytest
from unittest.mock import patch

from . import (
    AutoARIMAModel,
    AutoCESModel,
    AutoETSModel,
    ALL_LOCAL_MODELS,
    NONSEASONAL_LOCAL_MODELS,
    SEASONAL_LOCAL_MODELS, 
    SEASONAL_LOCAL_MODELS_EXTRA,
    GLUONTS_MODELS,
    GLUONTS_MODELS_WITH_STATIC_FEATURES,
    GLUONTS_MODELS_WITH_KNOWN_COVARIATES,
    GLUONTS_MODELS_WITH_STATIC_FEATURES_AND_KNOWN_COVARIATES,
)
from autogluon.timeseries.models.abstract.abstract_timeseries_model import AbstractTimeSeriesModel



DEFAULT_LOCAL_HYPERPARAMETERS = {"n_jobs": 1, "use_fallback_model": False}
DEFAULT_GLUONTS_HYPERPARAMETERS = {"max_epochs": 1, "num_batches_per_epoch": 1}

EXTRA_HYPERPARAMETERS: dict[Type[AbstractTimeSeriesModel], dict] = {
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
    AutoCESModel: {
        "model": "S"
    },
}


def patch_constructor(model_class: Type[AbstractTimeSeriesModel], default_hyperparameters: Dict[str, Any]) -> Callable[..., AbstractTimeSeriesModel]:
    def constructor(*args, **kwargs):
        hyperparameters = kwargs.get("hyperparameters", {})
        hyperparameters = {
            **EXTRA_HYPERPARAMETERS.get(model_class, {}),
            **hyperparameters,
        }
        kwargs["hyperparameters"] = hyperparameters
        return model_class(*args, **kwargs)
    
    return constructor


@pytest.fixture(params=ALL_LOCAL_MODELS)
def local_model_class(request):
    with patch.object(request.param, 'is_local_model_arg_allowed', return_value=True):
        yield patch_constructor(request.param, DEFAULT_LOCAL_HYPERPARAMETERS)


@pytest.fixture(params=SEASONAL_LOCAL_MODELS + SEASONAL_LOCAL_MODELS_EXTRA)
def seasonal_local_model_class(request):
    with patch.object(request.param, 'is_local_model_arg_allowed', return_value=True):
        yield patch_constructor(request.param, DEFAULT_LOCAL_HYPERPARAMETERS)
    
    
@pytest.fixture(params=NONSEASONAL_LOCAL_MODELS)
def nonseasonal_local_model_class(request):
    with patch.object(request.param, 'is_local_model_arg_allowed', return_value=True):
        yield patch_constructor(request.param, DEFAULT_LOCAL_HYPERPARAMETERS)


@pytest.fixture(params=GLUONTS_MODELS)
def gluonts_model_class(request):
    yield patch_constructor(request.param, DEFAULT_GLUONTS_HYPERPARAMETERS)


@pytest.fixture(params=GLUONTS_MODELS_WITH_STATIC_FEATURES)
def gluonts_model_with_static_features_class(request):
    yield patch_constructor(request.param, DEFAULT_GLUONTS_HYPERPARAMETERS)
    
    
@pytest.fixture(params=GLUONTS_MODELS_WITH_KNOWN_COVARIATES)
def gluonts_model_with_known_covariates_class(request):
    yield patch_constructor(request.param, DEFAULT_GLUONTS_HYPERPARAMETERS)
    

@pytest.fixture(params=GLUONTS_MODELS_WITH_STATIC_FEATURES_AND_KNOWN_COVARIATES)
def gluonts_model_with_known_covariates_and_static_features_class(request):
    yield patch_constructor(request.param, DEFAULT_GLUONTS_HYPERPARAMETERS)


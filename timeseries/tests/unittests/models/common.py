import inspect
from typing import Callable, Dict, Type, Any

from autogluon.timeseries.models import (
    ADIDAModel,
    ARIMAModel,
    AutoARIMAModel,
    AutoCESModel,
    AutoETSModel,
    AverageModel,
    ChronosModel,
    CrostonModel,
    DynamicOptimizedThetaModel,
    ETSModel,
    IMAPAModel,
    NaiveModel,
    NPTSModel,
    SeasonalAverageModel,
    SeasonalNaiveModel,
    ThetaModel,
    ZeroModel,
    DeepARModel,
    DLinearModel,
    PatchTSTModel,
    SimpleFeedForwardModel,
    TemporalFusionTransformerModel,
    TiDEModel,
    WaveNetModel,
    DirectTabularModel,
    RecursiveTabularModel,
)
from autogluon.timeseries.models.multi_window import MultiWindowBacktestingModel
from autogluon.timeseries.models.abstract.abstract_timeseries_model import AbstractTimeSeriesModel
from autogluon.timeseries.models.local.abstract_local_model import AbstractLocalModel
from autogluon.timeseries.models.gluonts.abstract_gluonts import AbstractGluonTSModel
from autogluon.timeseries.models.autogluon_tabular.mlforecast import AbstractMLForecastModel

# local models accepting seasonal_period
SEASONAL_LOCAL_MODELS = [
    AutoARIMAModel,
    AutoETSModel,
    AverageModel,
    DynamicOptimizedThetaModel,
    NaiveModel,
    NPTSModel,
    SeasonalAverageModel,
    SeasonalNaiveModel,
]
# these models will only be tested in local tests, and will not be exported 
# to model tests to decrease test running time
SEASONAL_LOCAL_MODELS_EXTRA = [
    AutoCESModel,
    ThetaModel,
    ETSModel,
    ARIMAModel,
]
# intermittent demand models do not accept seasonal_period
NONSEASONAL_LOCAL_MODELS = [
    ADIDAModel,
    ZeroModel,
    CrostonModel,
    IMAPAModel,
]
ALL_LOCAL_MODELS = SEASONAL_LOCAL_MODELS + SEASONAL_LOCAL_MODELS_EXTRA + NONSEASONAL_LOCAL_MODELS

# gluonts models
GLUONTS_MODELS_WITH_STATIC_FEATURES = [DeepARModel, TemporalFusionTransformerModel, TiDEModel, WaveNetModel]
GLUONTS_MODELS_WITH_KNOWN_COVARIATES = [DeepARModel, TemporalFusionTransformerModel, TiDEModel, PatchTSTModel, WaveNetModel]
GLUONTS_MODELS_WITH_STATIC_FEATURES_AND_KNOWN_COVARIATES = [
    m for m in GLUONTS_MODELS_WITH_STATIC_FEATURES if m in GLUONTS_MODELS_WITH_KNOWN_COVARIATES
]
GLUONTS_MODELS = [
    DeepARModel,
    DLinearModel,
    PatchTSTModel,
    SimpleFeedForwardModel,
    TemporalFusionTransformerModel,
    TiDEModel,
    WaveNetModel,
]

# tabular models supported by MLForecast
MLFORECAST_MODELS = [DirectTabularModel, RecursiveTabularModel]

CHRONOS_BOLT_MODEL_PATH = "autogluon/chronos-bolt-tiny"
CHRONOS_CLASSIC_MODEL_PATH = "autogluon/chronos-t5-tiny"

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


def get_default_hyperparameters(model_type: Callable[..., AbstractTimeSeriesModel]) -> Dict[str, Any]:
    if not inspect.isclass(model_type):
        return {}
    
    default_hyperparameters = {}

    for type_, hps in DEFAULT_HYPERPARAMETERS.items():
        if issubclass(model_type, type_) or model_type is type_:
            default_hyperparameters |= hps

    return default_hyperparameters


def get_multi_window_deepar(hyperparameters=None, **kwargs):
    """Wrap DeepAR inside MultiWindowBacktestingModel."""
    if hyperparameters is None:
        hyperparameters = {"max_epochs": 1, "num_batches_per_epoch": 1}
    model_base_kwargs = {**kwargs, "hyperparameters": hyperparameters}
    return MultiWindowBacktestingModel(model_base=DeepARModel, model_base_kwargs=model_base_kwargs, **kwargs)

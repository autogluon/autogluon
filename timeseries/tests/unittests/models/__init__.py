from autogluon.timeseries.models import (
    ADIDAModel,
    ARIMAModel,
    AutoARIMAModel,
    AutoCESModel,
    AutoETSModel,
    AverageModel,
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
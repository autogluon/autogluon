from autogluon.timeseries.models.local import (
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
)
from autogluon.timeseries.models.gluonts import (
    DeepARModel,
    DLinearModel,
    PatchTSTModel,
    SimpleFeedForwardModel,
    TemporalFusionTransformerModel,
    TiDEModel,
    WaveNetModel,
)


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


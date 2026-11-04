from __future__ import annotations

from typing import TypedDict

from .baseline_models import (
    AverageModel,
    NaiveModel,
    SeasonalAverageModel,
    SeasonalNaiveModel,
    ZeroModel,
)
from .deep_models import (
    Chronos2Model,
    ChronosModel,
    DeepARModel,
    DLinearModel,
    PatchTSTModel,
    SimpleFeedForwardModel,
    TemporalFusionTransformerModel,
    TiDEModel,
    TotoModel,
    WaveNetModel,
)
from .statistical_models import (
    ADIDAModel,
    AutoARIMAModel,
    AutoCESModel,
    AutoETSModel,
    CrostonModel,
    ETSModel,
    IMAPAModel,
    NPTSModel,
    ThetaModel,
)
from .tabular_models import (
    DirectTabularModel,
    PerStepTabularModel,
    RecursiveTabularModel,
)


class BaselineHyperparameters(TypedDict, total=False):
    """Hyperparameters for baseline models.

    See https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-model-zoo.html#baseline-models
    """

    Average: AverageModel | list[AverageModel]
    Naive: NaiveModel | list[NaiveModel]
    SeasonalAverage: SeasonalAverageModel | list[SeasonalAverageModel]
    SeasonalNaive: SeasonalNaiveModel | list[SeasonalNaiveModel]
    Zero: ZeroModel | list[ZeroModel]


class StatisticalHyperparameters(TypedDict, total=False):
    """Hyperparameters for statistical models.

    See https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-model-zoo.html#statistical-models
    """

    ADIDA: ADIDAModel | list[ADIDAModel]
    AutoARIMA: AutoARIMAModel | list[AutoARIMAModel]
    AutoCES: AutoCESModel | list[AutoCESModel]
    AutoETS: AutoETSModel | list[AutoETSModel]
    Croston: CrostonModel | list[CrostonModel]
    ETS: ETSModel | list[ETSModel]
    IMAPA: IMAPAModel | list[IMAPAModel]
    NPTS: NPTSModel | list[NPTSModel]
    Theta: ThetaModel | list[ThetaModel]


class TabularHyperparameters(TypedDict, total=False):
    """Hyperparameters for tabular models.

    See https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-model-zoo.html#tabular-models
    """

    DirectTabular: DirectTabularModel | list[DirectTabularModel]
    PerStepTabular: PerStepTabularModel | list[PerStepTabularModel]
    RecursiveTabular: RecursiveTabularModel | list[RecursiveTabularModel]


class DeepLearningHyperparameters(TypedDict, total=False):
    """Hyperparameters for deep learning and pretrained models.

    See https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-model-zoo.html#deep-learning-models
    See https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-model-zoo.html#pretrained-models
    """

    Chronos: ChronosModel | list[ChronosModel]
    Chronos2: Chronos2Model | list[Chronos2Model]
    DeepAR: DeepARModel | list[DeepARModel]
    DLinear: DLinearModel | list[DLinearModel]
    PatchTST: PatchTSTModel | list[PatchTSTModel]
    SimpleFeedForward: SimpleFeedForwardModel | list[SimpleFeedForwardModel]
    TemporalFusionTransformer: TemporalFusionTransformerModel | list[TemporalFusionTransformerModel]
    TiDE: TiDEModel | list[TiDEModel]
    Toto: TotoModel | list[TotoModel]
    WaveNet: WaveNetModel | list[WaveNetModel]


class TimeSeriesHyperparameters(
    BaselineHyperparameters,
    StatisticalHyperparameters,
    TabularHyperparameters,
    DeepLearningHyperparameters,
    total=False,
):
    """
    Hyperparameters for time series forecasting models.

    https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-model-zoo.html
    """

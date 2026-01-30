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
    Average: AverageModel
    Naive: NaiveModel
    SeasonalAverage: SeasonalAverageModel
    SeasonalNaive: SeasonalNaiveModel
    Zero: ZeroModel


class StatisticalHyperparameters(TypedDict, total=False):
    ADIDA: ADIDAModel
    AutoARIMA: AutoARIMAModel
    AutoCES: AutoCESModel
    AutoETS: AutoETSModel
    Croston: CrostonModel
    ETS: ETSModel
    IMAPA: IMAPAModel
    NPTS: NPTSModel
    Theta: ThetaModel


class TabularHyperparameters(TypedDict, total=False):
    DirectTabular: DirectTabularModel
    PerStepTabular: PerStepTabularModel
    RecursiveTabular: RecursiveTabularModel


class DeepLearningHyperparameters(TypedDict, total=False):
    Chronos: ChronosModel
    Chronos2: Chronos2Model
    DeepAR: DeepARModel
    DLinear: DLinearModel
    PatchTST: PatchTSTModel
    SimpleFeedForward: SimpleFeedForwardModel
    TemporalFusionTransformer: TemporalFusionTransformerModel
    TiDE: TiDEModel
    Toto: TotoModel
    WaveNet: WaveNetModel


class TimeSeriesHyperparameters(
    BaselineHyperparameters,
    StatisticalHyperparameters,
    TabularHyperparameters,
    DeepLearningHyperparameters,
    total=False,
): ...

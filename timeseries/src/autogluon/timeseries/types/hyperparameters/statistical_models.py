from typing import Literal

from autogluon.common import space

from ._base import SearchableBool, SearchableFloat, SearchableInt
from .mixins import DampedMixIn, MaxTsLengthMixIn, ModelMixIn, NJobsMixIn, SeasonalPeriodMixIn


class ETSModel(NJobsMixIn, SeasonalPeriodMixIn, MaxTsLengthMixIn, ModelMixIn, DampedMixIn, total=False): ...


class AutoETSModel(NJobsMixIn, SeasonalPeriodMixIn, MaxTsLengthMixIn, ModelMixIn, DampedMixIn, total=False): ...


class AutoCESModel(NJobsMixIn, SeasonalPeriodMixIn, MaxTsLengthMixIn, ModelMixIn, total=False): ...


class AutoARIMAModel(NJobsMixIn, SeasonalPeriodMixIn, MaxTsLengthMixIn, total=False):
    d: SearchableInt | None
    D: SearchableInt | None
    max_p: SearchableInt
    max_q: SearchableInt
    max_P: SearchableInt
    max_Q: SearchableInt
    max_d: SearchableInt
    max_D: SearchableInt
    start_p: SearchableInt
    start_q: SearchableInt
    start_P: SearchableInt
    start_Q: SearchableInt
    stationary: SearchableBool
    seasonal: SearchableBool
    approximation: SearchableBool
    allowdrift: SearchableBool
    allowmean: SearchableBool


class ThetaModel(NJobsMixIn, SeasonalPeriodMixIn, MaxTsLengthMixIn, total=False):
    decomposition_type: Literal["multiplicative", "additive"] | space.Categorical


class NPTSModel(NJobsMixIn, MaxTsLengthMixIn, total=False):
    kernel_type: Literal["exponential", "uniform"] | space.Categorical
    exp_kernel_weights: float | SearchableFloat
    use_seasonal_model: bool | SearchableBool
    num_samples: int | SearchableInt
    num_default_time_features: int | SearchableInt


class ADIDAModel(NJobsMixIn, MaxTsLengthMixIn, total=False): ...


class CrostonModel(NJobsMixIn, MaxTsLengthMixIn, total=False):
    variant: Literal["SBA", "classic", "optimized"] | space.Categorical


class IMAPAModel(NJobsMixIn, MaxTsLengthMixIn, total=False): ...

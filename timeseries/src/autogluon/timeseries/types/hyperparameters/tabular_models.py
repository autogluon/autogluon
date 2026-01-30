from typing import Callable

from autogluon.common import space
from autogluon.timeseries.types.hyperparameters._base import SearchableFloat

from .mixins import NJobsMixIn, TabularModelMixIn


class DirectTabularModel(TabularModelMixIn, total=False):
    lags: list[int] | space.Categorical | None
    differences: list[int] | space.Categorical | None


class PerStepTabularModel(TabularModelMixIn, NJobsMixIn, total=False):
    trailing_lags: list[int] | space.Categorical | None
    seasonal_lags: list[int] | space.Categorical | None
    validation_fraction: SearchableFloat | None


class RecursiveTabularModel(TabularModelMixIn, total=False):
    lags: list[int] | space.Categorical | None
    differences: list[int] | space.Categorical | None
    lag_transforms: dict[int, list[Callable]] | None

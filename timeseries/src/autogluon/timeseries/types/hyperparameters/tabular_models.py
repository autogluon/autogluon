from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from ._base import SearchableFloat
from .mixins import TabularModelMixIn

if TYPE_CHECKING:
    from autogluon.common import space


class DirectTabularModel(TabularModelMixIn, total=False):
    lags: list[int] | space.Categorical | None
    differences: list[int] | space.Categorical | None


class PerStepTabularModel(TabularModelMixIn, total=False):
    # NJobsMixIn is not used here because its n_jobs type (int | float) is too
    # wide — PerStepTabularModel only accepts int | None. TypedDict fields are
    # invariant, so narrowing via inheritance would be a type error.
    n_jobs: int | None
    trailing_lags: list[int] | space.Categorical | None
    seasonal_lags: list[int] | space.Categorical | None
    validation_fraction: SearchableFloat | None


class RecursiveTabularModel(TabularModelMixIn, total=False):
    lags: list[int] | space.Categorical | None
    differences: list[int] | space.Categorical | None
    lag_transforms: dict[int, list[Callable]] | None

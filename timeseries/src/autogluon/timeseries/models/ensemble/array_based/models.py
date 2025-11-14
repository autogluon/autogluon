import os
from abc import ABC
from typing import Any, Type

from .abstract import ArrayBasedTimeSeriesEnsembleModel
from .regressor import (
    EnsembleRegressor,
    MedianEnsembleRegressor,
    PerQuantileTabularEnsembleRegressor,
    TabularEnsembleRegressor,
)


class MedianEnsemble(ArrayBasedTimeSeriesEnsembleModel):
    def _get_ensemble_regressor(self) -> MedianEnsembleRegressor:
        return MedianEnsembleRegressor()


class BaseTabularEnsemble(ArrayBasedTimeSeriesEnsembleModel, ABC):
    ensemble_regressor_type: Type[EnsembleRegressor]

    def _get_default_hyperparameters(self) -> dict[str, Any]:
        default_hps = super()._get_default_hyperparameters()
        default_hps.update(
            {
                "tabular_hyperparameters": {"GBM": {}},
            }
        )
        return default_hps

    def _get_ensemble_regressor(self):
        return self.ensemble_regressor_type(
            path=os.path.join(self.path, "ensemble_regressor"),
            quantile_levels=list(self.quantile_levels),
            tabular_hyperparameters=self.get_hyperparameters()["tabular_hyperparameters"],
        )


class TabularEnsemble(BaseTabularEnsemble):
    """Time series ensemble model using single AutoGluon TabularPredictor for all quantiles."""

    ensemble_regressor_type = TabularEnsembleRegressor


class PerQuantileTabularEnsemble(BaseTabularEnsemble):
    """Time series ensemble model using separate `TabularPredictor` instances for each quantile in
    addition to a dedicated `TabularPredictor` for the mean (point) forecast.
    """

    ensemble_regressor_type = PerQuantileTabularEnsembleRegressor

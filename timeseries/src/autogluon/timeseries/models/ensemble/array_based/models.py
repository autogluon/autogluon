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
        default_hps.update({"model_name": "GBM", "model_hyperparameters": {}})
        return default_hps

    def _get_ensemble_regressor(self):
        hyperparameters = self.get_hyperparameters()
        return self.ensemble_regressor_type(
            quantile_levels=list(self.quantile_levels),
            model_name=hyperparameters["model_name"],
            model_hyperparameters=hyperparameters["model_hyperparameters"],
        )


class TabularEnsemble(BaseTabularEnsemble):
    """Time series ensemble model using a single AutoGluon-Tabular model for all quantiles."""

    ensemble_regressor_type = TabularEnsembleRegressor


class PerQuantileTabularEnsemble(BaseTabularEnsemble):
    """Time series ensemble model using separate AutoGluon-Tabular models for each quantile in
    addition to a dedicated model for the mean (point) forecast.
    """

    ensemble_regressor_type = PerQuantileTabularEnsembleRegressor

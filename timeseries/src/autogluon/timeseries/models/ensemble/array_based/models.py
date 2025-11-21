from abc import ABC
from typing import Any, Type

from .abstract import ArrayBasedTimeSeriesEnsembleModel
from .regressor import (
    EnsembleRegressor,
    LinearStackerEnsembleRegressor,
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


class LinearStackerEnsemble(ArrayBasedTimeSeriesEnsembleModel):
    """Time series ensemble model using linear stacker with PyTorch optimization."""

    def _get_default_hyperparameters(self) -> dict[str, Any]:
        default_hps = super()._get_default_hyperparameters()
        default_hps.update(
            {
                "weights_per": "m",
                "lr": 0.1,
                "max_epochs": 10000,
                "tolerance_change": 1e-7,
                "tolerance_grad": 1e-6,
            }
        )
        return default_hps

    def _get_ensemble_regressor(self) -> LinearStackerEnsembleRegressor:
        hps = self.get_hyperparameters()
        return LinearStackerEnsembleRegressor(
            quantile_levels=list(self.quantile_levels),
            weights_per=hps["weights_per"],
            lr=hps["lr"],
            max_epochs=hps["max_epochs"],
            relative_tolerance=hps["tolerance_change"],
            tolerance_grad=hps["tolerance_grad"],
        )

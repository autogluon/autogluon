import os
from typing import Any

from .abstract import ArrayBasedTimeSeriesEnsembleModel
from .regressor import MedianEnsembleRegressor, TabularEnsembleRegressor


class MedianEnsemble(ArrayBasedTimeSeriesEnsembleModel):
    def _get_ensemble_regressor(self) -> MedianEnsembleRegressor:
        return MedianEnsembleRegressor()


class TabularEnsemble(ArrayBasedTimeSeriesEnsembleModel):
    """Time series ensemble model using single AutoGluon TabularPredictor for all quantiles.

    This ensemble uses a single TabularPredictor to learn optimal combinations of base model
    predictions for all quantiles simultaneously.
    """

    def _get_default_hyperparameters(self) -> dict[str, Any]:
        default_hps = super()._get_default_hyperparameters()
        default_hps.update(
            {
                "tabular_hyperparameters": {"GBM": {}},
            }
        )
        return default_hps

    def _get_ensemble_regressor(self) -> TabularEnsembleRegressor:
        """Create and return a TabularEnsembleRegressor instance."""
        return TabularEnsembleRegressor(
            path=os.path.join(self.path, "ensemble_regressor"),
            quantile_levels=list(self.quantile_levels),
            tabular_hyperparameters=self.get_hyperparameters()["tabular_hyperparameters"],
        )

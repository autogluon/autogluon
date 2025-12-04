import logging

import numpy as np
import pandas as pd
from typing_extensions import Self

from autogluon.tabular.registry import ag_model_registry as tabular_ag_model_registry

from .abstract import EnsembleRegressor

logger = logging.getLogger(__name__)


class TabularEnsembleRegressor(EnsembleRegressor):
    """Ensemble regressor based on a single model from AutoGluon-Tabular that predicts all quantiles simultaneously."""

    def __init__(
        self,
        quantile_levels: list[float],
        model_name: str,
        model_hyperparameters: dict | None = None,
    ):
        super().__init__()
        self.quantile_levels = quantile_levels
        model_type = tabular_ag_model_registry.key_to_cls(model_name)
        model_hyperparameters = model_hyperparameters or {}
        self.model = model_type(
            problem_type="quantile",
            hyperparameters=model_hyperparameters | {"ag.quantile_levels": quantile_levels},
            path="",
            name=model_name,
        )

    def fit(
        self,
        base_model_mean_predictions: np.ndarray,
        base_model_quantile_predictions: np.ndarray,
        labels: np.ndarray,
        time_limit: float | None = None,
    ) -> Self:
        X = self._get_feature_df(base_model_mean_predictions, base_model_quantile_predictions)
        num_windows, num_items, prediction_length = base_model_mean_predictions.shape[:3]
        y = pd.Series(labels.reshape(num_windows * num_items * prediction_length))
        self.model.fit(X=X, y=y, time_limit=time_limit)
        return self

    def predict(
        self,
        base_model_mean_predictions: np.ndarray,
        base_model_quantile_predictions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.model.is_fit()
        num_windows, num_items, prediction_length = base_model_mean_predictions.shape[:3]
        assert num_windows == 1, "Prediction expects a single window to be provided"

        X = self._get_feature_df(base_model_mean_predictions, base_model_quantile_predictions)

        pred = self.model.predict(X)

        # Reshape back to (num_windows, num_items, prediction_length, num_quantiles)
        pred = pred.reshape(num_windows, num_items, prediction_length, len(self.quantile_levels))

        # Use median quantile as mean prediction
        median_idx = self._get_median_quantile_index()
        mean_pred = pred[:, :, :, median_idx : median_idx + 1]
        quantile_pred = pred

        return mean_pred, quantile_pred

    def _get_feature_df(
        self,
        base_model_mean_predictions: np.ndarray,
        base_model_quantile_predictions: np.ndarray,
    ) -> pd.DataFrame:
        num_windows, num_items, prediction_length, _, num_models = base_model_mean_predictions.shape
        num_tabular_items = num_windows * num_items * prediction_length
        features_array = np.hstack(
            [
                base_model_mean_predictions.reshape(num_tabular_items, -1),
                base_model_quantile_predictions.reshape(num_tabular_items, -1),
            ]
        )
        return pd.DataFrame(features_array, columns=self._get_feature_names(num_models))

    def _get_feature_names(self, num_models: int) -> list[str]:
        feature_names = []
        for mi in range(num_models):
            feature_names.append(f"model_{mi}_mean")
        for quantile in self.quantile_levels:
            for mi in range(num_models):
                feature_names.append(f"model_{mi}_q{quantile}")

        return feature_names

    def _get_median_quantile_index(self):
        """Get quantile index closest to 0.5"""
        quantile_array = np.array(self.quantile_levels)
        median_idx = int(np.argmin(np.abs(quantile_array - 0.5)))
        selected_quantile = quantile_array[median_idx]

        if selected_quantile != 0.5:
            logger.warning(
                f"Selected quantile {selected_quantile} is not exactly 0.5. "
                f"Using closest available quantile for median prediction."
            )

        return median_idx

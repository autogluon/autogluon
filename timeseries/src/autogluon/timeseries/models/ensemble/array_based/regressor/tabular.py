import logging
from typing import Optional

import numpy as np
import pandas as pd
from typing_extensions import Self

from autogluon.tabular import TabularPredictor

from .abstract import EnsembleRegressor

logger = logging.getLogger(__name__)


class TabularEnsembleRegressor(EnsembleRegressor):
    """TabularPredictor ensemble regressor using AutoGluon-Tabular as a single
    quantile regressor for the target.
    """

    def __init__(
        self,
        quantile_levels: list[float],
        tabular_hyperparameters: Optional[dict] = None,
    ):
        super().__init__()
        self.quantile_levels = quantile_levels
        self.tabular_hyperparameters = tabular_hyperparameters or {}
        self.predictor: Optional[TabularPredictor] = None

    def fit(
        self,
        base_model_mean_predictions: np.ndarray,
        base_model_quantile_predictions: np.ndarray,
        labels: np.ndarray,
        **kwargs,
    ) -> Self:
        time_limit = kwargs.get("time_limit", None)

        self.predictor = TabularPredictor(
            label="target",
            problem_type="quantile",
            quantile_levels=self.quantile_levels,
            verbosity=1,
        )

        # get features
        df = self._get_feature_df(base_model_mean_predictions, base_model_quantile_predictions)

        # get labels
        num_windows, num_items, prediction_length = base_model_mean_predictions.shape[:3]
        label_series = labels.reshape(num_windows * num_items * prediction_length)
        df["target"] = label_series

        self.predictor.fit(
            df,
            hyperparameters=self.tabular_hyperparameters,
            time_limit=time_limit,  # type: ignore
        )

        return self

    def predict(
        self,
        base_model_mean_predictions: np.ndarray,
        base_model_quantile_predictions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.predictor is None:
            raise ValueError("Model must be fitted before prediction")

        df = self._get_feature_df(base_model_mean_predictions, base_model_quantile_predictions)

        pred = self.predictor.predict(df)
        if isinstance(pred, pd.DataFrame):
            pred = pred.to_numpy()

        # Reshape back to (num_windows, num_items, prediction_length, num_quantiles)
        num_windows, num_items, prediction_length = base_model_mean_predictions.shape[:3]
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

        X = np.hstack(
            [
                base_model_mean_predictions.reshape(num_tabular_items, -1),
                base_model_quantile_predictions.reshape(num_tabular_items, -1),
            ]
        )

        df = pd.DataFrame(X, columns=self._get_feature_names(num_models))
        return df

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

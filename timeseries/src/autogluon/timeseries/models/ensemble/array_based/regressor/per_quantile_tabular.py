import logging

import numpy as np
import pandas as pd
from typing_extensions import Self

from autogluon.tabular.registry import ag_model_registry as tabular_ag_model_registry
from autogluon.timeseries.utils.timer import SplitTimer

from .abstract import EnsembleRegressor

logger = logging.getLogger(__name__)


class PerQuantileTabularEnsembleRegressor(EnsembleRegressor):
    """Ensemble regressor using separate models per quantile plus dedicated mean model."""

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
        self.mean_model = model_type(
            problem_type="regression",
            hyperparameters=model_hyperparameters,
            path="",
            name=f"{model_name}_mean",
        )
        self.quantile_models = [
            model_type(
                problem_type="quantile",
                hyperparameters=model_hyperparameters | {"ag.quantile_levels": [quantile]},
                path="",
                name=f"{model_name}_q{quantile}",
            )
            for quantile in quantile_levels
        ]

    def fit(
        self,
        base_model_mean_predictions: np.ndarray,
        base_model_quantile_predictions: np.ndarray,
        labels: np.ndarray,
        time_limit: float | None = None,
    ) -> Self:
        num_windows, num_items, prediction_length = base_model_mean_predictions.shape[:3]
        y = pd.Series(labels.reshape(num_windows * num_items * prediction_length))

        total_rounds = 1 + len(self.quantile_levels)
        timer = SplitTimer(time_limit, rounds=total_rounds).start()

        # Fit mean model
        X_mean = self._get_feature_df(base_model_mean_predictions, 0)
        self.mean_model.fit(X=X_mean, y=y, time_limit=timer.round_time_remaining())
        timer.next_round()

        # Fit quantile models
        for i, model in enumerate(self.quantile_models):
            X_q = self._get_feature_df(base_model_quantile_predictions, i)
            model.fit(X=X_q, y=y, time_limit=timer.round_time_remaining())
            timer.next_round()

        return self

    def _get_feature_df(self, predictions: np.ndarray, index: int) -> pd.DataFrame:
        num_windows, num_items, prediction_length, _, num_models = predictions.shape
        num_tabular_items = num_windows * num_items * prediction_length
        return pd.DataFrame(
            predictions[:, :, :, index].reshape(num_tabular_items, num_models),
            columns=[f"model_{mi}" for mi in range(num_models)],
        )

    def predict(
        self, base_model_mean_predictions: np.ndarray, base_model_quantile_predictions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.mean_model.is_fit()
        num_windows, num_items, prediction_length = base_model_mean_predictions.shape[:3]
        assert num_windows == 1, "Prediction expects a single window to be provided"

        X_mean = self._get_feature_df(base_model_mean_predictions, 0)
        mean_predictions = self.mean_model.predict(X_mean).reshape(num_windows, num_items, prediction_length, 1)

        quantile_predictions_list = []
        for i, model in enumerate(self.quantile_models):
            X_q = self._get_feature_df(base_model_quantile_predictions, i)
            quantile_predictions_list.append(model.predict(X_q).reshape(num_windows, num_items, prediction_length))
        quantile_predictions = np.stack(quantile_predictions_list, axis=-1)

        return mean_predictions, quantile_predictions

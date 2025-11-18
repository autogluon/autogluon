import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from typing_extensions import Self

from autogluon.tabular import TabularPredictor
from autogluon.timeseries.utils.timer import SplitTimer

from .abstract import EnsembleRegressor

logger = logging.getLogger(__name__)


class PerQuantileTabularEnsembleRegressor(EnsembleRegressor):
    """TabularPredictor ensemble regressor using separate models per quantile plus dedicated mean model."""

    def __init__(
        self,
        path: str,
        quantile_levels: list[float],
        tabular_hyperparameters: Optional[dict] = None,
    ):
        super().__init__()
        self.path = path
        self.quantile_levels = quantile_levels
        self.tabular_hyperparameters = tabular_hyperparameters or {}
        self.quantile_predictors: list[TabularPredictor] = []
        self.mean_predictor: Optional[TabularPredictor] = None

    def set_path(self, path: str) -> None:
        self.path = path

    def fit(
        self,
        base_model_mean_predictions: np.ndarray,
        base_model_quantile_predictions: np.ndarray,
        labels: np.ndarray,
        time_limit: Optional[float] = None,
    ) -> Self:
        num_windows, num_items, prediction_length = base_model_mean_predictions.shape[:3]
        target = labels.reshape(num_windows * num_items * prediction_length).ravel()

        # Split time between mean predictor (1 round) and quantile predictors
        # (len(quantile_levels) rounds)
        total_rounds = 1 + len(self.quantile_levels)
        timer = SplitTimer(time_limit, rounds=total_rounds).start()

        # fit mean predictor, based on mean predictions of base models
        mean_df = self._get_feature_df(base_model_mean_predictions, 0)
        mean_df["target"] = target
        self.mean_predictor = TabularPredictor(
            label="target",
            path=os.path.join(self.path, "mean"),
            verbosity=1,
            problem_type="regression",
        ).fit(
            mean_df,
            hyperparameters=self.tabular_hyperparameters,
            time_limit=timer.get(),  # type: ignore
        )
        timer.split()

        # fit quantile predictors, each quantile predictor is based on the
        # estimates of that quantile from base models
        for i, quantile in enumerate(self.quantile_levels):
            q_df = self._get_feature_df(base_model_quantile_predictions, i)
            q_df["target"] = target

            predictor = TabularPredictor(
                label="target",
                path=os.path.join(self.path, f"quantile_{quantile}"),
                verbosity=1,
                problem_type="quantile",
                quantile_levels=[quantile],
            ).fit(q_df, hyperparameters=self.tabular_hyperparameters, time_limit=timer.get())  # type: ignore
            self.quantile_predictors.append(predictor)
            timer.split()

        return self

    def _get_feature_df(self, predictions: np.ndarray, index: int) -> pd.DataFrame:
        num_windows, num_items, prediction_length, _, num_models = predictions.shape
        num_tabular_items = num_windows * num_items * prediction_length

        df = pd.DataFrame(
            predictions[:, :, :, index].reshape(num_tabular_items, num_models),
            columns=[f"model_{mi}" for mi in range(num_models)],
        )

        return df

    def load_predictors(self):
        if self.mean_predictor is None or len(self.quantile_predictors) < len(self.quantile_levels):
            try:
                self.mean_predictor = TabularPredictor.load(os.path.join(self.path, "mean"))

                self.quantile_predictors = []
                for quantile in self.quantile_levels:
                    predictor = TabularPredictor.load(os.path.join(self.path, f"quantile_{quantile}"))
                    self.quantile_predictors.append(predictor)

            except FileNotFoundError:
                raise ValueError("Model must be fitted before loading for prediction")

    def predict(
        self, base_model_mean_predictions: np.ndarray, base_model_quantile_predictions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self.load_predictors()

        num_windows, num_items, prediction_length, _, _ = base_model_mean_predictions.shape
        assert num_windows == 1, "Prediction expects a single window to be provided"

        # predict means
        assert self.mean_predictor is not None
        mean_predictions = self.mean_predictor.predict(
            self._get_feature_df(base_model_mean_predictions, 0),
            as_pandas=False,
        ).reshape(num_windows, num_items, prediction_length, 1)

        # predict quantiles
        quantile_predictions_list = []
        for i, predictor in enumerate(self.quantile_predictors):
            quantile_predictions_list.append(
                predictor.predict(self._get_feature_df(base_model_quantile_predictions, i), as_pandas=False).reshape(
                    num_windows, num_items, prediction_length
                )
            )
        quantile_predictions = np.stack(quantile_predictions_list, axis=-1)

        return mean_predictions, quantile_predictions

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove predictors to avoid pickling heavy TabularPredictor objects
        state["mean_predictor"] = None
        state["quantile_predictors"] = []
        return state

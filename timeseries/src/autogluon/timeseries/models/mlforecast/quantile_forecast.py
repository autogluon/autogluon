import logging
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

import autogluon.core as ag
from autogluon.tabular import TabularPredictor
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.seasonality import get_seasonality
from autogluon.timeseries.utils.warning_filters import statsmodels_warning_filter

from .base import BaseMLForecastModel


class QuantileRecursiveTabularModel(BaseMLForecastModel):
    def __init__(
        self,
        freq: Optional[str] = None,
        prediction_length: int = 1,
        path: Optional[str] = None,
        name: Optional[str] = None,
        eval_metric: str = None,
        hyperparameters: Dict[str, Any] = None,
        **kwargs,  # noqa
    ):
        super().__init__(
            path=path,
            freq=freq,
            prediction_length=prediction_length,
            name=name,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            **kwargs,
        )
        self.median_should_be_dropped = 0.5 not in self.quantile_levels
        self.quantile_levels = sorted(set([0.5] + self.quantile_levels))

    def _get_estimator(self, predictor_init_kwargs: dict, predictor_fit_kwargs: dict) -> BaseEstimator:
        from .utils import TabularQuantileRegressor

        predictor_init_kwargs["problem_type"] = ag.constants.QUANTILE
        predictor_init_kwargs["quantile_levels"] = self.quantile_levels
        predictor_init_kwargs["eval_metric"] = "pinball_loss"
        return TabularQuantileRegressor(
            predictor_init_kwargs=predictor_init_kwargs,
            predictor_fit_kwargs=predictor_fit_kwargs,
        )

    def _add_quantile_columns(self, predictions, mean_per_item, scale_per_item, stored_predictions):
        quantile_forecast = pd.concat(stored_predictions).sort_index(kind="stable")
        breakpoint()
        pass

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: TimeSeriesDataFrame = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        self.mlf.models_["mean"].reset_stored_predictions()
        predictions = self._predict_point_forecast(data, known_covariates)
        stored_predictions = self.mlf.models_["mean"].get_stored_predictions()
        mean_per_item, scale_per_item = self._get_mean_and_scale_per_item(data.item_ids)

        predictions = self._add_quantile_columns(predictions, mean_per_item, scale_per_item, stored_predictions)
        if self.median_should_be_dropped:
            predictions.drop("0.5", axis=1, inplace=True)
        return predictions

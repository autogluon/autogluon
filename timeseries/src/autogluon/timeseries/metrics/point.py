import logging
from typing import Optional

import numpy as np
import pandas as pd

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.dataset.ts_dataframe import ITEMID

from .abstract import TimeSeriesScorer
from .utils import _in_sample_abs_seasonal_error, _in_sample_squared_seasonal_error

logger = logging.getLogger(__name__)


class RMSE(TimeSeriesScorer):
    """Root mean squared error."""

    equivalent_tabular_regression_metric = "root_mean_squared_error"

    def compute_metric(
        self, data_future: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame, target: str = "target", **kwargs
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        return np.sqrt(self._safemean((y_true - y_pred) ** 2))


class MSE(TimeSeriesScorer):
    """Mean squared error."""

    equivalent_tabular_regression_metric = "mean_squared_error"

    def compute_metric(
        self, data_future: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame, target: str = "target", **kwargs
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        return self._safemean((y_true - y_pred) ** 2)


class MAE(TimeSeriesScorer):
    """Mean absolute error."""

    optimized_by_median = True
    equivalent_tabular_regression_metric = "mean_absolute_error"

    def compute_metric(
        self, data_future: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame, target: str = "target", **kwargs
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        return self._safemean((y_true - y_pred).abs())


class WAPE(TimeSeriesScorer):
    """Weighted absolute percentage error."""

    optimized_by_median = True
    equivalent_tabular_regression_metric = "mean_absolute_error"

    def compute_metric(
        self, data_future: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame, target: str = "target", **kwargs
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        return (y_true - y_pred).abs().sum() / y_true.abs().sum()


class sMAPE(TimeSeriesScorer):
    """Symmetric mean absolute percentage error."""

    optimized_by_median = True
    equivalent_tabular_regression_metric = "symmetric_mean_absolute_percentage_error"

    def compute_metric(
        self, data_future: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame, target: str = "target", **kwargs
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        return self._safemean(2 * ((y_true - y_pred).abs() / (y_true.abs() + y_pred.abs())))


class MAPE(TimeSeriesScorer):
    """Mean Absolute Percentage Error."""

    optimized_by_median = True
    equivalent_tabular_regression_metric = "mean_absolute_percentage_error"

    def compute_metric(
        self, data_future: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame, target: str = "target", **kwargs
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        return self._safemean((y_true - y_pred).abs() / y_true.abs())


class MASE(TimeSeriesScorer):
    """Mean absolute scaled error."""

    optimized_by_median = True
    equivalent_tabular_regression_metric = "mean_absolute_error"

    def __init__(self):
        self._past_abs_seasonal_error: Optional[pd.Series] = None

    def save_past_metrics(
        self, data_past: TimeSeriesDataFrame, target: str = "target", seasonal_period: int = 1, **kwargs
    ) -> None:
        self._past_abs_seasonal_error = _in_sample_abs_seasonal_error(
            y_past=data_past[target], seasonal_period=seasonal_period
        )

    def clear_past_metrics(self) -> None:
        self._past_abs_seasonal_error = None

    def compute_metric(
        self, data_future: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame, target: str = "target", **kwargs
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        if self._past_abs_seasonal_error is None:
            raise AssertionError("Call `save_past_metrics` before `compute_metric`")

        mae_per_item = (y_true - y_pred).abs().groupby(level=ITEMID, sort=False).mean()
        return self._safemean(mae_per_item / self._past_abs_seasonal_error)


class RMSSE(TimeSeriesScorer):
    """Root mean squared scaled error."""

    equivalent_tabular_regression_metric = "root_mean_squared_error"

    def __init__(self):
        self._past_squared_seasonal_error: Optional[pd.Series] = None

    def save_past_metrics(
        self, data_past: TimeSeriesDataFrame, target: str = "target", seasonal_period: int = 1, **kwargs
    ) -> None:
        self._past_squared_seasonal_error = _in_sample_squared_seasonal_error(
            y_past=data_past[target], seasonal_period=seasonal_period
        )

    def clear_past_metrics(self) -> None:
        self._past_squared_seasonal_error = None

    def compute_metric(
        self, data_future: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame, target: str = "target", **kwargs
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        if self._past_squared_seasonal_error is None:
            raise AssertionError("Call `save_past_metrics` before `compute_metric`")

        mse_per_item = (y_true - y_pred).pow(2.0).groupby(level=ITEMID, sort=False).mean()
        return np.sqrt(self._safemean(mse_per_item / self._past_squared_seasonal_error))

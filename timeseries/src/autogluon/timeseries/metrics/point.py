import logging
from typing import Optional

import numpy as np
import pandas as pd

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.dataset.ts_dataframe import ITEMID

from .abstract import TimeSeriesMetric
from .utils import _in_sample_abs_seasonal_error, _in_sample_squared_seasonal_error

logger = logging.getLogger(__name__)


class PointForecastMetric(TimeSeriesMetric):
    def compute_metric(
        self,
        data_future: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        **kwargs,
    ) -> float:
        return self._compute_point_metric(y_true=data_future[target], y_pred=predictions["mean"], **kwargs)

    def _compute_point_metric(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        raise NotImplementedError


class RMSE(PointForecastMetric):
    def _compute_point_metric(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return np.sqrt(self._safemean((y_true - y_pred) ** 2))


class MSE(PointForecastMetric):
    def _compute_point_metric(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return self._safemean((y_true - y_pred) ** 2)


class MAE(PointForecastMetric):
    optimized_by_median = True

    def _compute_point_metric(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return self._safemean((y_true - y_pred).abs())


class WAPE(PointForecastMetric):
    optimized_by_median = True

    def _compute_point_metric(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return (y_true - y_pred).abs().sum() / y_true.abs().sum()


class sMAPE(PointForecastMetric):
    optimized_by_median = True

    def _compute_point_metric(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return self._safemean(2 * ((y_true - y_pred).abs() / (y_true.abs() + y_pred.abs())))


class MAPE(PointForecastMetric):
    optimized_by_median = True

    def _compute_point_metric(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return self._safemean((y_true - y_pred).abs() / y_true.abs())


class MASE(PointForecastMetric):
    optimized_by_median = True

    def __init__(self):
        self._past_abs_seasonal_error: Optional[pd.Series] = None

    def save_past_metrics(
        self,
        data_past: TimeSeriesDataFrame,
        target: str = "target",
        seasonal_period: int = 1,
        **kwargs,
    ) -> None:
        self._past_abs_seasonal_error = _in_sample_abs_seasonal_error(
            y_past=data_past[target], seasonal_period=seasonal_period
        )

    def clear_past_metrics(self) -> None:
        self._past_abs_seasonal_error = None

    def _compute_point_metric(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        if self._past_abs_seasonal_error is None:
            raise AssertionError("Call `save_past_metrics` before `compute_metric`")

        mae_per_item = (y_true - y_pred).abs().groupby(level=ITEMID, sort=False).mean()
        return self._safemean(mae_per_item / self._past_abs_seasonal_error)


class RMSSE(PointForecastMetric):
    def __init__(self):
        self._past_squared_seasonal_error: Optional[pd.Series] = None

    def save_past_metrics(
        self,
        data_past: TimeSeriesDataFrame,
        target: str = "target",
        seasonal_period: int = 1,
        **kwargs,
    ) -> None:
        self._past_squared_seasonal_error = _in_sample_squared_seasonal_error(
            y_past=data_past[target], seasonal_period=seasonal_period
        )

    def clear_past_metrics(self) -> None:
        self._past_squared_seasonal_error = None

    def _compute_point_metric(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        if self._past_squared_seasonal_error is None:
            raise AssertionError("Call `save_past_metrics` before `compute_metric`")

        rmse_per_item = (y_true - y_pred).pow(2.0).groupby(level=ITEMID, sort=False).mean()
        return self._safemean(rmse_per_item / self._past_squared_seasonal_error)

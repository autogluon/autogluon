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
    r"""Root mean squared error.

    Defined as
    
    .. math::

        \sqrt{\sum_{i,t}  (y_{i,t} - f_{i,t})^2}


    Properties:
    
    - estimates the mean (expected value) of the time series
    - scale-dependent

    
    References
    ----------
    - `Wikipedia <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_
    - `Forecasting: Principles and Practice <https://otexts.com/fpp3/accuracy.html#scale-dependent-errors>`_
    """

    equivalent_tabular_regression_metric = "root_mean_squared_error"

    def compute_metric(
        self, data_future: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame, target: str = "target", **kwargs
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        return np.sqrt(self._safemean((y_true - y_pred) ** 2))


class MSE(TimeSeriesScorer):
    r"""Mean squared error.

    Using this metric will lead to forecast of the mean.

    .. math::

        \sum_{i,t}  (y_{i,t} - f_{i,t})^2
    
    References
    ----------
    - `Wikipedia <https://en.wikipedia.org/wiki/Mean_squared_error>`_
    
    """

    equivalent_tabular_regression_metric = "mean_squared_error"

    def compute_metric(
        self, data_future: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame, target: str = "target", **kwargs
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        return self._safemean((y_true - y_pred) ** 2)


class MAE(TimeSeriesScorer):
    r"""Mean absolute error.


    .. math::

        \sum_{i,t}  |y_{i,t} - f_{i,t}|

    References
    ----------
    - `Wikipedia <https://en.wikipedia.org/wiki/Mean_absolute_percentage_error#WMAPE>`_
    - `Forecasting: Principles and Practice <https://otexts.com/fpp3/accuracy.html#scale-dependent-errors>`_
    """

    optimized_by_median = True
    equivalent_tabular_regression_metric = "mean_absolute_error"

    def compute_metric(
        self, data_future: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame, target: str = "target", **kwargs
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        return self._safemean((y_true - y_pred).abs())


class WAPE(TimeSeriesScorer):
    r"""Weighted absolute percentage error.

    Defined as sum of absolute errors divided by the sum of absolute time series values in the forecast horizon.

    .. math::

        \operatorname{WAPE} = \frac{1}{\sum_{i, t} |y_{i, t}|} \sum_{i,t}  |y_{i,t} - f_{i,t}|

    
    References
    ----------
    - `Wikipedia <https://en.wikipedia.org/wiki/Mean_absolute_percentage_error#WMAPE>`_
    """

    optimized_by_median = True
    equivalent_tabular_regression_metric = "mean_absolute_error"

    def compute_metric(
        self, data_future: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame, target: str = "target", **kwargs
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        return (y_true - y_pred).abs().sum() / y_true.abs().sum()


class SMAPE(TimeSeriesScorer):
    r"""Symmetric mean absolute percentage error.

    Properties:
    
    - Poorly suited 
    - Penalizes overpredictions stronger than underpredictions

    .. math::
        
        2 \cdot \sum_{i,t} \frac{ |y_{i,t} - f_{i,t}|}{|y_{i,t}| + |f_{i,t}|}
    
    References
    ----------
    - `Wikipedia <https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error>`_
    - `Forecasting: Principles and Practice <https://otexts.com/fpp3/accuracy.html#percentage-errors>`_
    """

    optimized_by_median = True
    equivalent_tabular_regression_metric = "symmetric_mean_absolute_percentage_error"

    def compute_metric(
        self, data_future: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame, target: str = "target", **kwargs
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        return self._safemean(2 * ((y_true - y_pred).abs() / (y_true.abs() + y_pred.abs())))


class MAPE(TimeSeriesScorer):
    r"""Mean absolute percentage error.

    .. math::
        
        \sum_{i,t} \frac{ |y_{i,t} - f_{i,t}|}{|y_{i,t}|}
    
    References
    ----------
    - `Wikipedia <https://en.wikipedia.org/wiki/Mean_absolute_percentage_error>`_
    - `Forecasting: Principles and Practice <https://otexts.com/fpp3/accuracy.html#percentage-errors>`_
    """

    optimized_by_median = True
    equivalent_tabular_regression_metric = "mean_absolute_percentage_error"

    def compute_metric(
        self, data_future: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame, target: str = "target", **kwargs
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        return self._safemean((y_true - y_pred).abs() / y_true.abs())


class MASE(TimeSeriesScorer):
    r"""Mean absolute scaled error.
 
    References
    ----------
    - `Wikipedia <https://en.wikipedia.org/wiki/Mean_absolute_scaled_error>`_
    - `Forecasting: Principles and Practice <https://otexts.com/fpp3/accuracy.html#scaled-errors>`_
    """

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
    r"""Root mean squared scaled error.
    
    References
    ----------
    - `Forecasting: Principles and Practice <https://otexts.com/fpp3/accuracy.html#scaled-errors>`_
    """

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

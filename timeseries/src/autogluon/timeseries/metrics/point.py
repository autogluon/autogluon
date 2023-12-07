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

    .. math::

        \operatorname{RMSE} = \sqrt{\frac{1}{N} \frac{1}{H} \sum_{i=1}^{N}\sum_{t=T+1}^{T+H}  (y_{i,t} - f_{i,t})^2}


    Properties:

    - scale-dependent (time series with large absolute value contribute more to the loss)
    - heavily penalizes models that cannot quickly adapt to abrupt changes in the time series
    - sensitive to outliers
    - prefers models that accurately estimate the mean (expected value)


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

        \operatorname{MSE} = \frac{1}{N} \frac{1}{H} \sum_{i=1}^{N}\sum_{t=T+1}^{T+H}  (y_{i,t} - f_{i,t})^2

    Properties:

    - scale-dependent (time series with large absolute value contribute more to the loss)
    - heavily penalizes models that cannot quickly adapt to abrupt changes in the time series
    - sensitive to outliers
    - prefers models that accurately estimate the mean (expected value)

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

        \operatorname{MAE} = \frac{1}{N} \frac{1}{H} \sum_{i=1}^{N}\sum_{t=T+1}^{T+H}  |y_{i,t} - f_{i,t}|

    Properties:

    - scale-dependent (time series with large absolute value contribute more to the loss)
    - not sensitive to outliers
    - prefers models that accurately estimate the median

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

        \operatorname{WAPE} = \frac{1}{\sum_{i=1}^{N} \sum_{t=T+1}^{T+H} |y_{i, t}|} \sum_{i=1}^{N} \sum_{t=T+1}^{T+H}  |y_{i,t} - f_{i,t}|

    Properties:

    - scale-dependent (time series with large absolute value contribute more to the loss)
    - not sensitive to outliers
    - prefers models that accurately estimate the median


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

    .. math::

        \operatorname{SMAPE} = 2 \frac{1}{N} \frac{1}{H} \sum_{i=1}^{N} \sum_{t=T+1}^{T+H} \frac{ |y_{i,t} - f_{i,t}|}{|y_{i,t}| + |f_{i,t}|}

    Properties:

    - should only be used if all time series have positive values
    - poorly suited for sparse & intermittent time series that contain zero values
    - penalizes overprediction more heavily than underprediction

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

        \operatorname{MAPE} = \frac{1}{N} \frac{1}{H} \sum_{i=1}^{N} \sum_{t=T+1}^{T+H} \frac{ |y_{i,t} - f_{i,t}|}{|y_{i,t}|}

    Properties:

    - should only be used if all time series have positive values
    - undefined for time series that contain zero values
    - penalizes overprediction more heavily than underprediction

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

    Normalizes the absolute error for each time series by the historic seasonal error of this time series.

    .. math::

        \operatorname{MASE} = \frac{1}{N} \frac{1}{H} \sum_{i=1}^{N} \frac{1}{a_i} \sum_{t=T+1}^{T+H} |y_{i,t} - f_{i,t}|

    where :math:`a_i` is the historic absolute seasonal error defined as

    .. math::

        a_i = \frac{1}{T-m} \sum_{t=m+1}^T |y_{i,t} - y_{i,t-m}|

    and :math:`m` is the seasonal period of the time series (``eval_metric_seasonal_period``).

    Properties:

    - scaled metric (normalizes the error for each time series by the scale of that time series)
    - undefined for constant time series
    - not sensitive to outliers
    - prefers models that accurately estimate the median

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

    Normalizes the absolute error for each time series by the historic seasonal error of this time series.

    .. math::

        \operatorname{RMSSE} = \sqrt{\frac{1}{N} \frac{1}{H} \sum_{i=1}^{N} \frac{1}{s_i} \sum_{t=T+1}^{T+H} (y_{i,t} - f_{i,t})^2}

    where :math:`s_i` is the historic squared seasonal error defined as

    .. math::

        s_i = \frac{1}{T-m} \sum_{t=m+1}^T (y_{i,t} - y_{i,t-m})^2

    and :math:`m` is the seasonal period of the time series (``eval_metric_seasonal_period``).


    Properties:

    - scaled metric (normalizes the error for each time series by the scale of that time series)
    - undefined for constant time series
    - heavily penalizes models that cannot quickly adapt to abrupt changes in the time series
    - sensitive to outliers
    - prefers models that accurately estimate the mean (expected value)


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

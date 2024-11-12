import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from autogluon.timeseries import TimeSeriesDataFrame

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

        num_items = len(self._past_abs_seasonal_error)
        # Reshape abs errors into [num_items, prediction_length] to normalize per item without groupby
        abs_errors = np.abs(y_true.values - y_pred.values).reshape([num_items, -1])
        return self._safemean(abs_errors / self._past_abs_seasonal_error.values[:, None])


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

        num_items = len(self._past_squared_seasonal_error)
        # Reshape squared errors into [num_items, prediction_length] to normalize per item without groupby
        squared_errors = ((y_true.values - y_pred.values) ** 2.0).reshape([num_items, -1])
        return np.sqrt(self._safemean(squared_errors / self._past_squared_seasonal_error.values[:, None]))


class RMSLE(TimeSeriesScorer):
    r"""Root mean squared logarithmic error.

    Applies a logarithmic transformation to the predictions before computing the root mean squared error. Assumes
    both the ground truth and predictions are positive. If negative predictions are given, they will be clipped to zero.

    .. math::

        \operatorname{RMSLE} = \sqrt{\frac{1}{N} \frac{1}{H} \sum_{i=1}^{N} \sum_{t=T+1}^{T+H} (\ln(1 + y_{i,t}) - \ln(1 + f_{i,t}))^2}


    Properties:

    - undefined for time series with negative values
    - penalizes models that underpredict more than models that overpredict
    - insensitive to effects of outliers and scale, best when targets can vary or trend exponentially


    References
    ----------
    - `Scikit-learn: <https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-log-error>`_
    """

    def compute_metric(self, data_future, predictions, target, **kwargs):
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        y_pred = np.clip(y_pred, a_min=0.0, a_max=None)

        return np.sqrt(np.power(np.log1p(y_pred) - np.log1p(y_true), 2).mean())

    def __call__(
        self,
        data: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        prediction_length: int = 1,
        target: str = "target",
        seasonal_period: Optional[int] = None,
        **kwargs,
    ) -> float:
        if (data[target] < 0).any():
            raise ValueError(f"{self.name} cannot be used if target time series contains negative values!")
        return super().__call__(
            data=data,
            predictions=predictions,
            prediction_length=prediction_length,
            target=target,
            seasonal_period=seasonal_period,
            **kwargs,
        )


class WCD(TimeSeriesScorer):
    r"""Weighted cumulative discrepancy.

    Measures the discrepancy between the cumulative sum of the forecast and the cumulative sum of the actual values.

    .. math::

        \operatorname{WCD} = 2 \cdot \frac{1}{N} \frac{1}{H} \sum_{i=1}^{N} \sum_{t=T+1}^{T+H} \alpha \cdot \max(0, -d_{i, t}) + (1 - \alpha) \cdot \max(0, d_{i, t})

    where :math:`d_{i, t}` is the difference between the cumulative predicted value and the cumulative actual value

    .. math::

        d_{i, t} = \left(\sum_{s=T+1}^t f_{i, s}) - \left(\sum_{s=T+1}^t y_{i, s})

    Parameters
    ----------
    alpha : float, default = 0.5
        Values > 0.5 correspond put a stronger penalty on underpredictions (when cumulative forecast is below the
        cumulative actual value). Values < 0.5 put a stronger penalty on overpredictions.
    """

    def __init__(self, alpha: float = 0.5):
        assert 0 < alpha < 1, "alpha must be in (0, 1)"
        self.alpha = alpha
        self.num_items: Optional[int] = None
        warnings.warn(
            f"{self.name} is an experimental metric. Its behavior may change in the future version of AutoGluon."
        )

    def save_past_metrics(self, data_past: TimeSeriesDataFrame, **kwargs) -> None:
        self.num_items = data_past.num_items

    def _fast_cumsum(self, y: np.ndarray) -> np.ndarray:
        """Compute the cumulative sum for each consecutive `prediction_length` items in the array."""
        y = y.reshape(self.num_items, -1)
        return np.nancumsum(y, axis=1).ravel()

    def compute_metric(
        self, data_future: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame, target: str = "target", **kwargs
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        cumsum_true = self._fast_cumsum(y_true.to_numpy())
        cumsum_pred = self._fast_cumsum(y_pred.to_numpy())
        diffs = cumsum_pred - cumsum_true
        error = diffs * np.where(diffs < 0, -self.alpha, (1 - self.alpha))
        return 2 * self._safemean(error)

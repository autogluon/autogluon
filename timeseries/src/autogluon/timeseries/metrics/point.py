import logging
import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from autogluon.timeseries import TimeSeriesDataFrame

from .abstract import TimeSeriesScorer
from .utils import in_sample_abs_seasonal_error, in_sample_squared_seasonal_error

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
        self,
        data_future: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        **kwargs,
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        y_true, y_pred = y_true.to_numpy(), y_pred.to_numpy()
        errors = ((y_true - y_pred) ** 2).reshape([-1, self.prediction_length])
        if self.horizon_weight is not None:
            errors *= self.horizon_weight
        return np.sqrt(self._safemean(errors))


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
        self,
        data_future: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        **kwargs,
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        y_true, y_pred = y_true.to_numpy(), y_pred.to_numpy()
        errors = ((y_true - y_pred) ** 2).reshape([-1, self.prediction_length])
        if self.horizon_weight is not None:
            errors *= self.horizon_weight
        return self._safemean(errors)


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
        self,
        data_future: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        **kwargs,
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        y_true, y_pred = y_true.to_numpy(), y_pred.to_numpy()
        errors = np.abs(y_true - y_pred).reshape([-1, self.prediction_length])
        if self.horizon_weight is not None:
            errors *= self.horizon_weight
        return self._safemean(errors)


class MAEB(TimeSeriesScorer):
    r"""Mean absolute error with a bias penalty.

    Adds a penalty for systematically biased (over- or under-) forecasts to :class:`MAE`. Defined as the mean absolute
    error plus the absolute mean forecast bias.

    .. math::

        \operatorname{MAEB} = \frac{1}{N} \frac{1}{H} \sum_{i=1}^{N} \sum_{t=T+1}^{T+H} |y_{i,t} - f_{i,t}| + \left| \frac{1}{N} \frac{1}{H} \sum_{i=1}^{N} \sum_{t=T+1}^{T+H} (f_{i,t} - y_{i,t}) \right|

    The first term measures forecast accuracy (as in MAE), while the second term penalizes forecast bias (the mean net
    over- or under-forecast). This discourages models that achieve low error by systematically under-forecasting
    demand, which is a common failure mode when optimizing MAE on intermittent (sparse) time series.

    Properties:

    - scale-dependent (time series with large absolute value contribute more to the loss)
    - equivalent to :class:`MAE` when the forecast is unbiased (mean net error is zero)

    See :class:`WAPEB` for a scale-free version of this metric that normalizes the error across time series.
    """

    # Target mean (RMSE + no median paste) to avoid underforecasting on intermittent series
    optimized_by_median = False
    equivalent_tabular_regression_metric = "root_mean_squared_error"

    def compute_metric(
        self,
        data_future: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        **kwargs,
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        y_true, y_pred = y_true.to_numpy(), y_pred.to_numpy()
        abs_errors = np.abs(y_true - y_pred).reshape([-1, self.prediction_length])
        biases = (y_pred - y_true).reshape([-1, self.prediction_length])
        if self.horizon_weight is not None:
            abs_errors = abs_errors * self.horizon_weight
            biases = biases * self.horizon_weight
        return self._safemean(abs_errors) + np.abs(self._safemean(biases))


class BIAS(TimeSeriesScorer):
    r"""Mean forecast bias (signed).

    Measures the average net over- or under-forecast. Positive values indicate that the forecast systematically
    *over-predicts* the target, negative values indicate systematic *under-prediction*.

    .. math::

        \operatorname{BIAS} = \frac{1}{N} \frac{1}{H} \sum_{i=1}^{N} \sum_{t=T+1}^{T+H} (f_{i,t} - y_{i,t})

    Forecast bias is a standard diagnostic in demand and retail forecasting, where the *direction* of the error
    matters: over-forecasting ties up inventory and capital, while under-forecasting causes stockouts.

    .. warning::
        Unlike other metrics, BIAS is **not** a loss function: its optimum is 0, and *both* positive and negative
        values indicate a worse forecast. For this reason it can only be used to evaluate a trained predictor via
        :meth:`~autogluon.timeseries.TimeSeriesPredictor.evaluate` and cannot be passed as ``eval_metric`` for model
        selection (``evaluate_only=True``). To penalize forecast bias *during* model selection, use :class:`MAEB` or
        :class:`WAPEB` instead, which combine forecast accuracy with an absolute bias penalty.

    Properties:

    - scale-dependent (time series with large absolute value contribute more to the metric)
    """

    evaluate_only = True
    # Reported without a sign flip so that predictor.evaluate() returns the raw signed bias (positive = over-forecast).
    greater_is_better_internal = True

    def compute_metric(
        self,
        data_future: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        **kwargs,
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        y_true, y_pred = y_true.to_numpy(), y_pred.to_numpy()
        biases = (y_pred - y_true).reshape([-1, self.prediction_length])
        if self.horizon_weight is not None:
            biases = biases * self.horizon_weight
        return self._safemean(biases)


class WAPE(TimeSeriesScorer):
    r"""Weighted absolute percentage error.

    Defined as sum of absolute errors divided by the sum of absolute time series values in the forecast horizon.

    .. math::

        \operatorname{WAPE} = \frac{1}{\sum_{i=1}^{N} \sum_{t=T+1}^{T+H} |y_{i, t}|} \sum_{i=1}^{N} \sum_{t=T+1}^{T+H}  |y_{i,t} - f_{i,t}|

    Properties:

    - scale-dependent (time series with large absolute value contribute more to the loss)
    - not sensitive to outliers
    - prefers models that accurately estimate the median

    If ``self.horizon_weight`` is provided, both the errors and the target time series in the denominator will be re-weighted.

    References
    ----------
    - `Wikipedia <https://en.wikipedia.org/wiki/Mean_absolute_percentage_error#WMAPE>`_
    """

    optimized_by_median = True
    equivalent_tabular_regression_metric = "mean_absolute_error"

    def compute_metric(
        self,
        data_future: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        **kwargs,
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        y_true, y_pred = y_true.to_numpy(), y_pred.to_numpy()
        errors = np.abs(y_true - y_pred).reshape([-1, self.prediction_length])
        if self.horizon_weight is not None:
            errors *= self.horizon_weight
            y_true = y_true.reshape([-1, self.prediction_length]) * self.horizon_weight
        return np.nansum(errors) / np.nansum(np.abs(y_true))


class WAPEB(TimeSeriesScorer):
    r"""Weighted absolute percentage error with a bias penalty.

    Adds a penalty for systematically biased (over- or under-) forecasts to :class:`WAPE`. Defined as the sum of
    absolute errors plus the absolute total forecast bias, divided by the sum of absolute time series values in the
    forecast horizon.

    .. math::

        \operatorname{WAPEB} = \frac{\sum_{i=1}^{N} \sum_{t=T+1}^{T+H}  |y_{i,t} - f_{i,t}| + \left| \sum_{i=1}^{N} \sum_{t=T+1}^{T+H} (f_{i,t} - y_{i,t}) \right|}{\sum_{i=1}^{N} \sum_{t=T+1}^{T+H} |y_{i, t}|}

    The numerator combines forecast accuracy (absolute error, as in WAPE) with forecast bias (the absolute net
    over- or under-forecast). This discourages models that achieve low error by systematically under-forecasting
    demand, which is a common failure mode for intermittent (sparse) time series. WAPEB is the scale-free counterpart
    of :class:`MAEB`, analogous to how WAPE relates to MAE.

    This metric was used to rank submissions in the VN1 Forecasting Accuracy Challenge (for non-negative target values,
    :math:`\sum |y_{i,t}|` equals the total actual sales used to normalize the score in the competition).

    Properties:

    - scale-dependent (time series with large absolute value contribute more to the loss)
    - equivalent to :class:`WAPE` when the forecast is unbiased (net error is zero)
    - well-suited for sparse (intermittent) time series that contain many zeros

    If ``self.horizon_weight`` is provided, the errors, the biases, and the target time series in the denominator will
    all be re-weighted.

    References
    ----------
    - `VN1 Forecasting Accuracy Challenge <https://www.datasource.ai/en/home/data-science-competitions-for-startups/vn1-forecasting-accuracy-challenge-phase-1/description>`_
    """

    # Target mean (RMSE + no median paste) to avoid underforecasting on intermittent series
    optimized_by_median = False
    equivalent_tabular_regression_metric = "root_mean_squared_error"

    def compute_metric(
        self,
        data_future: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        **kwargs,
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        y_true, y_pred = y_true.to_numpy(), y_pred.to_numpy()
        abs_errors = np.abs(y_true - y_pred).reshape([-1, self.prediction_length])
        biases = (y_pred - y_true).reshape([-1, self.prediction_length])
        if self.horizon_weight is not None:
            abs_errors = abs_errors * self.horizon_weight
            biases = biases * self.horizon_weight
            y_true = y_true.reshape([-1, self.prediction_length]) * self.horizon_weight
        return (np.nansum(abs_errors) + np.abs(np.nansum(biases))) / np.nansum(np.abs(y_true))


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
        self,
        data_future: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        **kwargs,
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        y_true, y_pred = y_true.to_numpy(), y_pred.to_numpy()
        errors = (np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))).reshape([-1, self.prediction_length])
        if self.horizon_weight is not None:
            errors *= self.horizon_weight
        return 2 * self._safemean(errors)


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
        self,
        data_future: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        **kwargs,
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        y_true, y_pred = y_true.to_numpy(), y_pred.to_numpy()
        errors = (np.abs(y_true - y_pred) / np.abs(y_true)).reshape([-1, self.prediction_length])
        if self.horizon_weight is not None:
            errors *= self.horizon_weight
        return self._safemean(errors)


class MASE(TimeSeriesScorer):
    r"""Mean absolute scaled error.

    Normalizes the absolute error for each time series by the historical seasonal error of this time series.

    .. math::

        \operatorname{MASE} = \frac{1}{N} \frac{1}{H} \sum_{i=1}^{N} \frac{1}{a_i} \sum_{t=T+1}^{T+H} |y_{i,t} - f_{i,t}|

    where :math:`a_i` is the historical absolute seasonal error defined as

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

    def __init__(
        self,
        prediction_length: int = 1,
        seasonal_period: int | None = None,
        horizon_weight: Sequence[float] | None = None,
    ):
        super().__init__(
            prediction_length=prediction_length, seasonal_period=seasonal_period, horizon_weight=horizon_weight
        )
        self._past_abs_seasonal_error: pd.Series | None = None

    def save_past_metrics(
        self, data_past: TimeSeriesDataFrame, target: str = "target", seasonal_period: int = 1, **kwargs
    ) -> None:
        self._past_abs_seasonal_error = in_sample_abs_seasonal_error(
            y_past=data_past[target], seasonal_period=seasonal_period
        )

    def clear_past_metrics(self) -> None:
        self._past_abs_seasonal_error = None

    def compute_metric(
        self,
        data_future: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        **kwargs,
    ) -> float:
        if self._past_abs_seasonal_error is None:
            raise AssertionError("Call `save_past_metrics` before `compute_metric`")

        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        y_true, y_pred = y_true.to_numpy(), y_pred.to_numpy()

        errors = np.abs(y_true - y_pred).reshape([-1, self.prediction_length])
        if self.horizon_weight is not None:
            errors *= self.horizon_weight
        return self._safemean(errors / self._past_abs_seasonal_error.to_numpy()[:, None])


class RMSSE(TimeSeriesScorer):
    r"""Root mean squared scaled error.

    Normalizes the absolute error for each time series by the historical seasonal error of this time series.

    .. math::

        \operatorname{RMSSE} = \sqrt{\frac{1}{N} \frac{1}{H} \sum_{i=1}^{N} \frac{1}{s_i} \sum_{t=T+1}^{T+H} (y_{i,t} - f_{i,t})^2}

    where :math:`s_i` is the historical squared seasonal error defined as

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

    def __init__(
        self,
        prediction_length: int = 1,
        seasonal_period: int | None = None,
        horizon_weight: Sequence[float] | None = None,
    ):
        super().__init__(
            prediction_length=prediction_length, seasonal_period=seasonal_period, horizon_weight=horizon_weight
        )
        self._past_squared_seasonal_error: pd.Series | None = None

    def save_past_metrics(
        self, data_past: TimeSeriesDataFrame, target: str = "target", seasonal_period: int = 1, **kwargs
    ) -> None:
        self._past_squared_seasonal_error = in_sample_squared_seasonal_error(
            y_past=data_past[target], seasonal_period=seasonal_period
        )

    def clear_past_metrics(self) -> None:
        self._past_squared_seasonal_error = None

    def compute_metric(
        self,
        data_future: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        **kwargs,
    ) -> float:
        if self._past_squared_seasonal_error is None:
            raise AssertionError("Call `save_past_metrics` before `compute_metric`")

        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        y_true, y_pred = y_true.to_numpy(), y_pred.to_numpy()
        errors = ((y_true - y_pred) ** 2).reshape([-1, self.prediction_length])
        if self.horizon_weight is not None:
            errors *= self.horizon_weight
        return np.sqrt(self._safemean(errors / self._past_squared_seasonal_error.to_numpy()[:, None]))


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

    def compute_metric(
        self,
        data_future: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        **kwargs,
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        y_true, y_pred = y_true.to_numpy(), y_pred.to_numpy()
        y_pred = np.clip(y_pred, a_min=0.0, a_max=None)

        errors = np.power(np.log1p(y_pred) - np.log1p(y_true), 2).reshape([-1, self.prediction_length])
        if self.horizon_weight is not None:
            errors *= self.horizon_weight
        return np.sqrt(self._safemean(errors))

    def __call__(
        self,
        data: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        **kwargs,
    ) -> float:
        if (data[target] < 0).any():
            raise ValueError(f"{self.name} cannot be used if target time series contains negative values!")
        return super().__call__(
            data=data,
            predictions=predictions,
            target=target,
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
        Values > 0.5 put a stronger penalty on underpredictions (when cumulative forecast is below the
        cumulative actual value). Values < 0.5 put a stronger penalty on overpredictions.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        prediction_length: int = 1,
        seasonal_period: int | None = None,
        horizon_weight: Sequence[float] | None = None,
    ):
        super().__init__(
            prediction_length=prediction_length, seasonal_period=seasonal_period, horizon_weight=horizon_weight
        )
        assert 0 < alpha < 1, "alpha must be in (0, 1)"
        self.alpha = alpha
        warnings.warn(
            f"{self.name} is an experimental metric. Its behavior may change in the future version of AutoGluon."
        )

    def _fast_cumsum(self, y: np.ndarray) -> np.ndarray:
        """Compute the cumulative sum for each consecutive `self.prediction_length` items in the array."""
        y = y.reshape(-1, self.prediction_length)
        return np.nancumsum(y, axis=1).ravel()

    def compute_metric(
        self,
        data_future: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        **kwargs,
    ) -> float:
        y_true, y_pred = self._get_point_forecast_score_inputs(data_future, predictions, target=target)
        cumsum_true = self._fast_cumsum(y_true.to_numpy())
        cumsum_pred = self._fast_cumsum(y_pred.to_numpy())
        diffs = cumsum_pred - cumsum_true
        errors = (diffs * np.where(diffs < 0, -self.alpha, (1 - self.alpha))).reshape([-1, self.prediction_length])
        if self.horizon_weight is not None:
            errors *= self.horizon_weight
        return 2 * self._safemean(errors)

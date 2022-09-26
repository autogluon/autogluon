"""Functions and objects for evaluating forecasts. Adapted from gluonts.evaluation.
See also, https://ts.gluon.ai/api/gluonts/gluonts.evaluation.html
"""
import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.dataset.ts_dataframe import ITEMID
from autogluon.timeseries.utils.warning_filters import evaluator_warning_filter

logger = logging.getLogger(__name__)


def in_sample_naive_1_error(*, y_history: pd.Series) -> pd.Series:
    """Compute the error of naive forecast (predict previous value) for each time series."""
    diff = y_history.diff()
    # We ignore the differences between the last value of prev item and the first value of the next item
    length_per_item = y_history.groupby(ITEMID, sort=False).size()
    first_index_for_each_item = length_per_item.cumsum().values[:-1]
    diff.iloc[first_index_for_each_item] = np.nan
    return diff.abs().groupby(ITEMID, sort=False).mean()


def mse_per_item(*, y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """Compute Mean Squared Error for each item (time series)."""
    return (y_true - y_pred).pow(2.0).groupby(ITEMID, sort=False).mean()


def mae_per_item(*, y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """Compute Mean Absolute Error for each item (time series)."""
    return (y_true - y_pred).abs().groupby(ITEMID, sort=False).mean()


def mape_per_item(*, y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """Compute Mean Absolute Percentage Error for each item (time series)."""
    return ((y_true - y_pred) / y_true).abs().groupby(ITEMID, sort=False).mean()


def symmetric_mape_per_item(*, y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """Compute symmetric Mean Absolute Percentage Error for each item (time series)."""
    return (2 * (y_true - y_pred).abs() / (y_true.abs() + y_pred.abs())).groupby(ITEMID, sort=False).mean()


def quantile_loss(*, y_true: pd.Series, y_pred: pd.Series, q: float) -> float:
    """Compute total quantile loss across all timesteps of all time series."""
    return 2 * ((y_true - y_pred) * ((y_true <= y_pred) - q)).abs().sum()


class TimeSeriesEvaluator:
    """Contains functions for computing forecast accuracy metrics.

    Forecast accuracy metrics measure discrepancy between forecasts and ground-truth time
    series. After being initialized, AutoGluon ``TimeSeriesEvaluator`` expects two time
    series data sets (``TimeSeriesDataFrame``) as input: the first of which contains the
    ground truth time series including both the "history" and the forecast horizon. The
    second input is the data frame of predictions corresponding only to the forecast
    horizon.

    .. warning::
        ``TimeSeriesEvaluator`` always computes metrics by their original definition, while
        AutoGluon-TimeSeries predictor and model objects always report their scores in
        higher-is-better fashion. The coefficients used to "flip" the signs of metrics to
        obey this convention are given in ``TimeSeriesEvaluator.METRIC_COEFFICIENTS``.

    .. warning::
        Definitions of forecast accuracy metrics may differ from package to package.
        For example, the definition of MASE is different between GluonTS and autogluon.

    Parameters
    ----------
    eval_metric: str
        Name of the metric to be computed. Available metrics are

        * ``MASE``: mean absolute scaled error. See https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
        * ``MAPE``: mean absolute percentage error. See https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
        * ``sMAPE``: "symmetric" mean absolute percentage error. See https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
        * ``mean_wQuantileLoss``: mean weighted quantile loss, i.e., average quantile loss scaled
         by the total absolute values of the time series. See https://docs.aws.amazon.com/forecast/latest/dg/metrics.html#metrics-wQL
        * ``MSE``: mean squared error
        * ``RMSE``: root mean squared error

    prediction_length: int
        Length of the forecast horizon
    target_column: str
        Name of the target column to be forecasting.

    Class Attributes
    ----------------
    AVAILABLE_METRICS
        list of names of available metrics
    METRIC_COEFFICIENTS
        coefficients by which each metric should be multiplied with to obey the higher-is-better
        convention
    DEFAULT_METRIC
        name of default metric returned by
        :meth:``~autogluon.timeseries.TimeSeriesEvaluator.check_get_evaluation_metric``.
    """

    AVAILABLE_METRICS = ["MASE", "MAPE", "sMAPE", "mean_wQuantileLoss", "MSE", "RMSE"]
    METRIC_COEFFICIENTS = {"MASE": -1, "MAPE": -1, "sMAPE": -1, "mean_wQuantileLoss": -1, "MSE": -1, "RMSE": -1}
    DEFAULT_METRIC = "mean_wQuantileLoss"

    def __init__(self, eval_metric: str, prediction_length: int, target_column: str = "target"):
        assert eval_metric in self.AVAILABLE_METRICS, f"Metric {eval_metric} not available"

        self.prediction_length = prediction_length
        self.eval_metric = eval_metric
        self.target_column = target_column

        self.metric_method = self.__getattribute__("_" + self.eval_metric.lower())

    @property
    def coefficient(self) -> int:
        return self.METRIC_COEFFICIENTS[self.eval_metric]

    @property
    def higher_is_better(self) -> bool:
        return self.coefficient > 0

    def _safemean(self, data: pd.Series) -> float:
        return data.replace([np.inf, -np.inf], np.nan).dropna().mean()

    def _mse(self, y_true: pd.Series, predictions: TimeSeriesDataFrame, **kwargs) -> float:
        y_pred = predictions["mean"]
        return self._safemean(mse_per_item(y_true=y_true, y_pred=y_pred))

    def _rmse(self, y_true: pd.Series, predictions: TimeSeriesDataFrame, **kwargs) -> float:
        return np.sqrt(self._mse(y_true=y_true, predictions=predictions))

    def _mase(self, y_true: pd.Series, predictions: TimeSeriesDataFrame, y_history: pd.Series) -> float:
        y_pred = self._get_median_forecast(predictions)
        mae = mae_per_item(y_true=y_true, y_pred=y_pred)
        naive_1_error = in_sample_naive_1_error(y_history=y_history)
        return self._safemean(mae / naive_1_error)

    def _mape(self, y_true: pd.Series, predictions: TimeSeriesDataFrame, **kwargs) -> float:
        y_pred = self._get_median_forecast(predictions)
        return self._safemean(mape_per_item(y_true=y_true, y_pred=y_pred))

    def _smape(self, y_true: pd.Series, predictions: TimeSeriesDataFrame, **kwargs) -> float:
        y_pred = self._get_median_forecast(predictions)
        return self._safemean(symmetric_mape_per_item(y_true=y_true, y_pred=y_pred))

    def _mean_wquantileloss(self, y_true: pd.Series, predictions: TimeSeriesDataFrame, **kwargs) -> float:
        loss_values = []
        abs_target_sum = y_true.abs().sum()
        for col in predictions.columns:
            if col != "mean":
                q = float(col)
                assert 0 <= q <= 1
                y_pred = predictions[col]
                loss_values.append(quantile_loss(y_true=y_true, y_pred=y_pred, q=q) / abs_target_sum)
        return np.mean(loss_values)

    def _get_median_forecast(self, predictions: TimeSeriesDataFrame) -> pd.Series:
        # TODO: Median forecast doesn't actually minimize the MAPE / sMAPE losses
        if "0.5" in predictions.columns:
            return predictions["0.5"]
        else:
            logger.warning("Median forecast not found. Defaulting to mean forecasts.")
            return predictions["mean"]

    @staticmethod
    def check_get_evaluation_metric(
        metric_name: Optional[str] = None,
        raise_if_not_available: bool = True,
    ):
        """A utility function that checks if a given evaluation metric
        name is available in autogluon.timeseries, and optionally raises
        a ValueError otherwise.

        Parameters
        ----------
        metric_name: str
            The requested metric name, currently one of the evaluation metrics available
            in GluonTS.
        raise_if_not_available: bool
            if True, a ValueError will be raised if the requested metric is not yet
            available in autogluon.timeseries. Otherwise, the default metric name will be
            returned instead of the requested metric.

        Returns
        -------
        checked_metric_name: str
            The requested metric name if it is available in autogluon.timeseries.
        """
        metric = metric_name or TimeSeriesEvaluator.DEFAULT_METRIC
        if metric not in TimeSeriesEvaluator.AVAILABLE_METRICS:
            if raise_if_not_available:
                raise ValueError(f"metric {metric} is not available yet.")
            return TimeSeriesEvaluator.DEFAULT_METRIC
        return metric

    def __call__(self, data: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame) -> float:
        assert (predictions.num_timesteps_per_item() == self.prediction_length).all()
        # Select entries in `data` that correspond to the forecast horizon
        data_history = data.slice_by_timestep(None, -self.prediction_length)
        data_future = data.slice_by_timestep(-self.prediction_length, None)
        assert data_future.index.equals(predictions.index), "Prediction and data indices do not match."

        with evaluator_warning_filter(), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            return self.metric_method(
                y_true=data_future[self.target_column],
                predictions=predictions,
                y_history=data_history[self.target_column],
            )

"""Functions and objects for evaluating forecasts. Adapted from gluonts.evaluation.
See also, https://ts.gluon.ai/api/gluonts/gluonts.evaluation.html
"""
import logging
import warnings
from typing import Callable, List, Optional, Any

import numpy as np
import pandas as pd

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.utils.warning_filters import evaluator_warning_filter

logger = logging.getLogger(__name__)


# unit metric callables -- compute an error or summary statistic of a single
# time series against a forecast


def mean_square_error(
    *, target: np.ndarray, forecast: np.ndarray, **kwargs  # noqa: F841
) -> float:
    return np.mean(np.square(target - forecast))  # noqa


def abs_error(
    *, target: np.ndarray, forecast: np.ndarray, **kwargs  # noqa: F841
) -> float:
    return np.sum(np.abs(target - forecast))  # noqa


def mean_abs_error(
    *, target: np.ndarray, forecast: np.ndarray, **kwargs  # noqa: F841
) -> float:
    return np.mean(np.abs(target - forecast))  # noqa


def quantile_loss(
    *, target: np.ndarray, forecast: np.ndarray, q: float, **kwargs  # noqa: F841
) -> float:
    return 2 * np.sum(np.abs((forecast - target) * ((target <= forecast) - q)))


def coverage(
    *, target: np.ndarray, forecast: np.ndarray, **kwargs  # noqa: F841
) -> float:
    return np.mean(target < forecast)  # noqa


def mape(*, target: np.ndarray, forecast: np.ndarray, **kwargs) -> float:  # noqa: F841
    return np.mean(np.abs(target - forecast) / np.abs(target))  # noqa


def symmetric_mape(
    *, target: np.ndarray, forecast: np.ndarray, **kwargs  # noqa: F841
) -> float:
    return 2 * np.mean(np.abs(target - forecast) / (np.abs(target) + np.abs(forecast)))


def abs_target_sum(*, target: np.ndarray, **kwargs):  # noqa: F841
    return np.sum(np.abs(target))


def abs_target_mean(*, target: np.ndarray, **kwargs):  # noqa: F841
    return np.mean(np.abs(target))


def in_sample_naive_1_error(*, target_history: np.ndarray, **kwargs):  # noqa: F841
    return np.nanmean(np.abs(np.diff(target_history)))


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

        * ``MASE``: mean absolute scaled error. See also, https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
        * ``MAPE``: mean absolute percentage error
        * ``sMAPE``: "symmetric" mean absolute percentage error
        * ``mean_wQuantileLoss``: mean weighted quantile loss, i.e., average quantile loss scaled
         by the total absolute values of the time series.
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
    METRIC_COEFFICIENTS = {
        "MASE": -1, "MAPE": -1, "sMAPE": -1, "mean_wQuantileLoss": -1, "MSE": -1, "RMSE": -1
    }
    DEFAULT_METRIC = "mean_wQuantileLoss"

    def __init__(
        self, eval_metric: str, prediction_length: int, target_column: str = "target"
    ):
        assert (
            eval_metric in self.AVAILABLE_METRICS
        ), f"Metric {eval_metric} not available"

        self.prediction_length = prediction_length
        self.eval_metric = eval_metric
        self.target_column = target_column

        self.metric_method = self.__getattribute__("_" + self.eval_metric.lower())

    def _safemean(self, data: Any):
        data_filled = np.nan_to_num(
            data, neginf=np.nan, posinf=np.nan, nan=np.nan
        )
        return np.nanmean(data_filled)

    def _mase(
        self, data: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame
    ) -> float:
        metric_callables = [mean_abs_error, in_sample_naive_1_error]
        df = self.get_metrics_per_ts(
            data, predictions, metric_callables=metric_callables
        )
        return float(self._safemean(df["mean_abs_error"] / df["in_sample_naive_1_error"]))

    def _mape(
        self, data: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame
    ) -> float:
        df = self.get_metrics_per_ts(data, predictions, metric_callables=[mape])
        return float(self._safemean(df["mape"]))

    def _smape(
        self, data: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame
    ) -> float:
        df = self.get_metrics_per_ts(
            data, predictions, metric_callables=[symmetric_mape]
        )
        return float(self._safemean(df["symmetric_mape"]))

    def _mse(self, data: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame):
        df = self.get_metrics_per_ts(
            data, predictions, metric_callables=[mean_square_error]
        )
        return float(np.mean(df["mean_square_error"]))

    def _rmse(self, data: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame):
        return np.sqrt(self._mse(data, predictions))

    def _mean_wquantileloss(
        self,
        data: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        quantiles: List[float] = None,
    ):
        if not quantiles:
            quantiles = [float(col) for col in predictions.columns if col != "mean"]
            assert all(0 <= q <= 1 for q in quantiles)

        df = self.get_metrics_per_ts(
            data=data,
            predictions=predictions,
            metric_callables=[quantile_loss, abs_target_sum],
            quantiles=quantiles,
        )

        w_quantile_losses = []
        total_abs_target = df["abs_target_sum"].sum()
        for q in quantiles:
            w_quantile_losses.append(
                df[f"quantile_loss[{str(q)}]"].sum() / total_abs_target
            )

        return float(np.mean(w_quantile_losses))

    def _get_minimizing_forecast(
        self, predictions: TimeSeriesDataFrame, metric_callable: Callable
    ) -> np.ndarray:
        """get field from among predictions that minimizes the given metric"""
        if "0.5" in predictions.columns and metric_callable is not mean_square_error:
            return np.array(predictions["0.5"])
        elif metric_callable is not mean_square_error:
            logger.warning("Median forecast not found. Defaulting to mean forecasts.")

        if "mean" not in predictions.columns:
            ValueError(
                f"Mean forecast not found. Cannot evaluate metric {metric_callable.__name__}"
            )
        return np.array(predictions["mean"])

    def get_metrics_per_ts(
        self,
        data: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        metric_callables: List[Callable],
        quantiles: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        metrics = []
        for item_id in data.iter_items():
            y_true_w_hist = data.loc[item_id][self.target_column]

            target = np.array(y_true_w_hist[-self.prediction_length:])
            target_history = np.array(y_true_w_hist[: -self.prediction_length])

            item_metrics = {}
            for metric_callable in metric_callables:
                if metric_callable is quantile_loss:
                    assert all(0 <= q <= 1 for q in quantiles)
                    for q in quantiles:
                        assert (
                            str(q) in predictions.columns
                        ), f"Quantile {q} not found in predictions"
                        item_metrics[f"quantile_loss[{str(q)}]"] = quantile_loss(
                            target=target,
                            forecast=np.array(predictions.loc[item_id][str(q)]),
                            q=q,
                        )
                else:
                    forecast = self._get_minimizing_forecast(
                        predictions.loc[item_id], metric_callable=metric_callable
                    )
                    item_metrics[metric_callable.__name__] = metric_callable(
                        target=target,
                        forecast=forecast,
                        target_history=target_history,
                    )

            metrics.append(item_metrics)

        return pd.DataFrame(metrics)

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

    def __call__(
        self, data: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame
    ) -> float:
        assert all(
            len(predictions.loc[i]) == self.prediction_length
            for i in predictions.iter_items()
        )
        assert set(predictions.iter_items()) == set(
            data.iter_items()
        ), "Prediction and data indices do not match."

        with evaluator_warning_filter(), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            return self.metric_method(data, predictions)

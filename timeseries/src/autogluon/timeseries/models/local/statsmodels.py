import logging
import warnings
from typing import Any, Callable, Dict, List

import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ModelWarning, ValueWarning

from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame
from autogluon.timeseries.utils.forecast import get_forecast_horizon_index_single_time_series
from autogluon.timeseries.utils.warning_filters import statsmodels_warning_filter

from .abstract_local_model import AbstractLocalModel

logger = logging.getLogger(__name__)

warnings.simplefilter("ignore", ModelWarning)
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", ValueWarning)


def get_quantiles_from_statsmodels(coverage_fn: Callable, quantile_levels: List[float]) -> List[pd.Series]:
    """Obtain quantile forecasts using a fitted Statsmodels model.

    The method for computing quantiles is different for all models in Statsmodels, this method unifies the interface.

    Parameters
    ----------
    coverage_fn : Callable
        Function that takes as input a coverage level between 0 and 100 and returns a pandas Dataframe with lower /
        upper values of the confidence interval.
    quantile_levels : List[float]
        List of quantiles between 0.0 and 1.0 to predict

    Returns
    -------
    results : List[pd.Series]
        List, where each element is a pandas Series containing the predictions for the corresponding quantile level.
    """
    results = []
    for q in quantile_levels:
        if q < 0.5:
            coverage = 2 * q
            column_index = 0
        else:
            coverage = 2 * (1 - q)
            column_index = 1
        quantile_pred = coverage_fn(coverage)
        # Select lower bound of the confidence interval if q < 0.5, upper bound otherwise
        results.append(quantile_pred.iloc[:, column_index].rename(str(q)))
    return results


class ETSModel(AbstractLocalModel):
    """Exponential smoothing with trend and seasonality.

    Based on `statsmodels.tsa.exponential_smoothing.ets.ETSModel <https://www.statsmodels.org/stable/generated/statsmodels.tsa.exponential_smoothing.ets.ETSModel.html>`_.

    Our implementation contains several improvements over the Statsmodels version, such
    as multi-CPU training and reducing the disk usage when saving models.


    Other Parameters
    ----------------
    error : {"add", "mul"}, default = "add"
        Error model. Allowed values are "add" (additive) and "mul" (multiplicative).
        Note that "mul" is only applicable to time series with positive values.
    trend : {"add", "mul", None}, default = "add"
        Trend component model. Allowed values are "add" (additive), "mul" (multiplicative) and None (disabled).
        Note that "mul" is only applicable to time series with positive values.
    damped_trend : bool, default = False
        Whether or not the included trend component is damped.
    seasonal : {"add", "mul", None}, default = "add"
        Seasonal component model. Allowed values are "add" (additive), "mul" (multiplicative) and None (disabled).
        Note that "mul" is only applicable to time series with positive values.
    seasonal_period : int or None, default = None
        Number of time steps in a complete seasonal cycle for seasonal models. For example, 7 for daily data with a
        weekly cycle or 12 for monthly data with an annual cycle.
        When set to None, seasonal_period will be inferred from the frequency of the training data. Can also be
        specified manually by providing an integer > 1.
        If seasonal_period (inferred or provided) is equal to 1, seasonality will be disabled.
        Seasonality will also be disabled, if the length of the time series is < 2 * seasonal_period.
    maxiter : int, default = 1000
        Number of iterations during optimization.
    n_jobs : int or float, default = 0.5
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    """

    allowed_local_model_args = [
        "error",
        "trend",
        "damped_trend",
        "seasonal",
        "seasonal_period",
        "maxiter",
    ]

    def _update_local_model_args(
        self, local_model_args: Dict[str, Any], data: TimeSeriesDataFrame, **kwargs
    ) -> Dict[str, Any]:
        local_model_args.setdefault("trend", "add")
        local_model_args.setdefault("maxiter", 1000)

        seasonal_period = local_model_args.pop("seasonal_period")
        seasonal = local_model_args.setdefault("seasonal", "add")
        if seasonal is not None and seasonal_period <= 1:
            logger.warning(
                f"{self.name} with seasonal = {seasonal} requires seasonal_period > 1 "
                f"(received seasonal_period = {seasonal_period}). Disabling seasonality."
            )
            local_model_args["seasonal"] = None
            local_model_args["seasonal_periods"] = 1
        else:
            local_model_args["seasonal_periods"] = seasonal_period

        return local_model_args

    @staticmethod
    def _predict_with_local_model(
        time_series: pd.Series,
        freq: str,
        prediction_length: int,
        quantile_levels: List[float],
        local_model_args: dict,
        **kwargs,
    ) -> pd.DataFrame:
        forecast_timestamps = get_forecast_horizon_index_single_time_series(
            past_timestamps=time_series.index, freq=freq, prediction_length=prediction_length
        )
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        maxiter = local_model_args.pop("maxiter")

        # Disable seasonality if timeseries is too short for given seasonal_period
        if local_model_args["seasonal"] is not None and len(time_series) < 2 * local_model_args["seasonal_periods"]:
            local_model_args["seasonal"] = None

        with statsmodels_warning_filter():
            model = ETSModel(
                endog=time_series,
                freq=freq,
                **local_model_args,
            )
            fit_result = model.fit(disp=False, maxiter=maxiter)
            predictions = fit_result.get_prediction(start=forecast_timestamps[0], end=forecast_timestamps[-1])

        results = [predictions.predicted_mean.rename("mean")]
        coverage_fn = lambda alpha: predictions.pred_int(alpha=alpha)
        results += get_quantiles_from_statsmodels(coverage_fn=coverage_fn, quantile_levels=quantile_levels)
        return pd.concat(results, axis=1)


class ARIMAModel(AbstractLocalModel):
    """Autoregressive Integrated Moving Average (ARIMA) model.

    Based on `statsmodels.tsa.statespace.sarimax.SARIMAX <https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html>`_.

    Our implementation contains several improvements over the Statsmodels version, such
    as multi-CPU training and reducing the disk usage when saving models.


    Other Parameters
    ----------------
    order: Tuple[int, int, int], default = (1, 1, 1)
        The (p, d, q) order of the model for the number of AR parameters, differences, and MA parameters to use.
    seasonal_order: Tuple[int, int, int], default = (0, 0, 0)
        The (P, D, Q) parameters of the seasonal ARIMA model. Setting to (0, 0, 0) disables seasonality.
    seasonal_period : int or None, default = None
        Number of time steps in a complete seasonal cycle for seasonal models. For example, 7 for daily data with a
        weekly cycle or 12 for monthly data with an annual cycle.
        When set to None, seasonal_period will be inferred from the frequency of the training data. Can also be
        specified manually by providing an integer > 1.
        If seasonal_period (inferred or provided) is equal to 1, seasonality will be disabled.
    trend : {"n", "c", "t", "ct"}, default = "c"
        Parameter controlling the trend polynomial. Allowed values are "n" (no trend), "c" (constant), "t" (linear) and
        "ct" (constant plus linear).
    enforce_stationarity : bool, default = True
        Whether to transform the AR parameters to enforce stationarity in the autoregressive component of the model.
        If ARIMA crashes during fitting with an LU decomposition error, you can either set enforce_stationarity to
        False or increase the differencing parameter ``d`` in ``order``.
    enforce_invertibility : bool, default = True
        Whether to transform the MA parameters to enforce invertibility in the moving average component of the model.
    maxiter : int, default = 50
        Number of iterations during optimization.
    n_jobs : int or float, default = 0.5
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    """

    allowed_local_model_args = [
        "order",
        "seasonal_order",
        "seasonal_period",
        "trend",
        "enforce_stationarity",
        "enforce_invertibility",
        "maxiter",
    ]

    def _update_local_model_args(
        self, local_model_args: Dict[str, Any], data: TimeSeriesDataFrame, **kwargs
    ) -> Dict[str, Any]:
        local_model_args.setdefault("trend", "c")
        local_model_args.setdefault("order", (1, 1, 1))
        local_model_args.setdefault("maxiter", 50)

        seasonal_period = local_model_args.pop("seasonal_period")
        seasonal_order = local_model_args.pop("seasonal_order", (0, 0, 0))

        seasonal_order_is_valid = len(seasonal_order) == 3 and all(isinstance(p, int) for p in seasonal_order)
        if not seasonal_order_is_valid:
            raise ValueError(
                f"{self.name} can't interpret received seasonal_order {seasonal_order} as a "
                "tuple with 3 nonnegative integers (P, D, Q)."
            )

        # Disable seasonality if seasonal_period is too short
        if seasonal_period <= 1:
            local_model_args["seasonal_order"] = (0, 0, 0, 0)
        else:
            local_model_args["seasonal_order"] = tuple(seasonal_order) + (seasonal_period,)

        return local_model_args

    @staticmethod
    def _predict_with_local_model(
        time_series: pd.Series,
        freq: str,
        prediction_length: int,
        quantile_levels: List[float],
        local_model_args: dict,
        **kwargs,
    ) -> pd.DataFrame:
        forecast_timestamps = get_forecast_horizon_index_single_time_series(
            past_timestamps=time_series.index, freq=freq, prediction_length=prediction_length
        )
        from statsmodels.tsa.statespace.sarimax import SARIMAX as StatsmodelSARIMAX

        maxiter = local_model_args.pop("maxiter")

        with statsmodels_warning_filter():
            model = StatsmodelSARIMAX(
                endog=time_series,
                freq=freq,
                **local_model_args,
            )
            fit_result = model.fit(disp=False, maxiter=maxiter)
            predictions = fit_result.get_prediction(start=forecast_timestamps[0], end=forecast_timestamps[-1])

        results = [predictions.predicted_mean.rename("mean")]
        coverage_fn = lambda alpha: predictions.conf_int(alpha=alpha)
        results += get_quantiles_from_statsmodels(coverage_fn=coverage_fn, quantile_levels=quantile_levels)
        return pd.concat(results, axis=1)


class ThetaModel(AbstractLocalModel):
    """The Theta forecasting model of Assimakopoulos and Nikolopoulos (2000).

    Based on `statsmodels.tsa.forecasting.theta.ThetaModel <https://www.statsmodels.org/stable/generated/statsmodels.tsa.forecasting.theta.ThetaModel.html>`_.

    Our implementation contains several improvements over the Statsmodels version, such
    as multi-CPU training and reducing the disk usage when saving models.


    References
    ----------
    Assimakopoulos, Vassilis, and Konstantinos Nikolopoulos.
    "The theta model: a decomposition approach to forecasting."
    International journal of forecasting 16.4 (2000): 521-530.


    Other Parameters
    ----------------
    deseasonalize : bool, default = True
        Whether to deseasonalize the data. If True and use_test is True, the data is only deseasonalized if the null
        hypothesis of no seasonality is rejected.
    seasonal_period : int or None, default = None
        Number of time steps in a complete seasonal cycle for seasonal models. For example, 7 for daily data with a
        weekly cycle or 12 for monthly data with an annual cycle.
        When set to None, seasonal_period will be inferred from the frequency of the training data. Can also be
        specified manually by providing an integer > 1.
        If seasonal_period (inferred or provided) is equal to 1, seasonality will be disabled.
        Seasonality will also be disabled, if the length of the time series is < 2 * seasonal_period.
    use_test : bool, default = True
        Whether to use a statistical test for determining if the seasonality is present.
    method : {"auto", "additive", "multiplicative"}, default = "auto"
        The model used for the seasonal decomposition. "auto" uses multiplicative if the time series is non-negative
        and all estimated seasonal components are positive. If either of these conditions is False, then it uses an
        additive decomposition.
    difference : bool, default = False
        Whether to difference the data before testing for seasonality.
    n_jobs : int or float, default = 0.5
        Number of CPU cores used to fit the models in parallel.
        When set to a float between 0.0 and 1.0, that fraction of available CPU cores is used.
        When set to a positive integer, that many cores are used.
        When set to -1, all CPU cores are used.
    """

    allowed_local_model_args = [
        "deseasonalize",
        "seasonal_period",
        "use_test",
        "method",
        "difference",
    ]

    def _update_local_model_args(
        self, local_model_args: Dict[str, Any], data: TimeSeriesDataFrame, **kwargs
    ) -> Dict[str, Any]:
        local_model_args.setdefault("deseasonalize", True)

        seasonal_period = local_model_args.pop("seasonal_period")
        local_model_args["period"] = seasonal_period

        return local_model_args

    @staticmethod
    def _predict_with_local_model(
        time_series: pd.Series,
        freq: str,
        prediction_length: int,
        quantile_levels: List[float],
        local_model_args: dict,
        **kwargs,
    ) -> pd.DataFrame:
        from statsmodels.tsa.forecasting.theta import ThetaModel as StatsmodelsTheta

        # Disable seasonality if timeseries is too short for given seasonal_period
        if local_model_args["deseasonalize"] and len(time_series) < 2 * local_model_args["period"]:
            local_model_args["deseasonalize"] = False

        time_series.index.freq = freq

        with statsmodels_warning_filter():
            model = StatsmodelsTheta(
                endog=time_series,
                **local_model_args,
            )
            fit_result = model.fit(disp=False)

        results = [fit_result.forecast(prediction_length).rename("mean")]
        coverage_fn = lambda alpha: fit_result.prediction_intervals(steps=prediction_length, alpha=alpha)
        results += get_quantiles_from_statsmodels(coverage_fn=coverage_fn, quantile_levels=quantile_levels)
        return pd.concat(results, axis=1)

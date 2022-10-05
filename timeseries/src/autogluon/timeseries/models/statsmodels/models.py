import logging
from typing import List

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel as StatsmodelsETS
from statsmodels.tsa.forecasting.theta import ThetaModel as StatsmodelsTheta
from statsmodels.tsa.statespace.sarimax import SARIMAX as StatsmodelsSARIMAX

from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame
from autogluon.timeseries.utils.seasonality import get_seasonality
from autogluon.timeseries.utils.warning_filters import statsmodels_warning_filter

from .abstract_statsmodels import AbstractStatsmodelsModel, FittedLocalModel

logger = logging.getLogger(__name__)


class ETSModel(AbstractStatsmodelsModel):
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

    quantile_method_name = "pred_int"
    statsmodels_allowed_init_args = [
        "error",
        "trend",
        "damped_trend",
        "seasonal",
        "seasonal_period",
    ]
    statsmodels_allowed_fit_args = [
        "maxiter",
    ]

    def _update_sm_model_init_args(self, sm_model_init_args: dict, data: TimeSeriesDataFrame) -> dict:
        sm_model_init_args = sm_model_init_args.copy()
        sm_model_init_args["freq"] = data.freq
        sm_model_init_args.setdefault("trend", "add")

        # Infer seasonal_period if seasonal_period is not given / is set to None
        seasonal_period = sm_model_init_args.pop("seasonal_period", None)
        if seasonal_period is None:
            seasonal_period = get_seasonality(data.freq)
        sm_model_init_args["seasonal_periods"] = seasonal_period

        seasonal = sm_model_init_args.setdefault("seasonal", "add")
        # Disable seasonality if seasonal_period is too short
        if seasonal is not None and seasonal_period <= 1:
            logger.warning(
                f"{self.name} with seasonal = {seasonal} requires seasonal_period > 1 "
                f"(received seasonal_period = {seasonal_period}). Disabling seasonality."
            )
            sm_model_init_args["seasonal"] = None
            sm_model_init_args["seasonal_periods"] = 1

        return sm_model_init_args

    def _fit_local_model(
        self, timeseries: pd.Series, sm_model_init_args: dict, sm_model_fit_args: dict
    ) -> FittedLocalModel:
        # Disable seasonality if timeseries is too short for given seasonal_period
        if sm_model_init_args["seasonal"] is not None and len(timeseries) < 2 * sm_model_init_args["seasonal_periods"]:
            sm_model_init_args = sm_model_init_args.copy()
            sm_model_init_args["seasonal"] = None

        with statsmodels_warning_filter():
            model = StatsmodelsETS(endog=timeseries, **sm_model_init_args)
            fit_result = model.fit(full_output=False, disp=False, **sm_model_fit_args)
        # Only save the parameters of the trained model, not the model itself
        parameters = dict(zip(fit_result.param_names, fit_result.params))
        return FittedLocalModel(model_name=self.name, sm_model_init_args=sm_model_init_args, parameters=parameters)

    def _predict_with_local_model(
        self, timeseries: pd.Series, fitted_model: FittedLocalModel, quantile_levels: List[float]
    ) -> pd.DataFrame:
        assert fitted_model.model_name == self.name
        with statsmodels_warning_filter():
            base_model = StatsmodelsETS(endog=timeseries, **fitted_model.sm_model_init_args)
            parameters = np.array(list(fitted_model.parameters.values()))
            # This is a hack that allows us to set the parameters to their estimated values & initialize the model
            sm_model = base_model.fit(start_params=parameters, maxiter=0, disp=False)
        return self._get_predictions_from_statsmodels_model(
            sm_model=sm_model,
            cutoff=timeseries.index.max(),
            quantile_levels=quantile_levels,
            freq=fitted_model.sm_model_init_args["freq"],
        )


class ARIMAModel(AbstractStatsmodelsModel):
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

    quantile_method_name = "conf_int"
    statsmodels_allowed_init_args = [
        "order",
        "seasonal_order",
        "seasonal_period",
        "enforce_stationarity",
        "enforce_invertibility",
    ]
    statsmodels_allowed_fit_args = [
        "maxiter",
    ]

    def _update_sm_model_init_args(self, sm_model_init_args: dict, data: TimeSeriesDataFrame) -> dict:
        sm_model_init_args = sm_model_init_args.copy()
        sm_model_init_args["freq"] = data.freq
        sm_model_init_args["trend"] = "c"
        sm_model_init_args.setdefault("enforce_stationarity", True)

        # Infer seasonal_period if seasonal_period is not given / is set to None
        seasonal_period = sm_model_init_args.pop("seasonal_period", None)
        if seasonal_period is None:
            seasonal_period = get_seasonality(data.freq)

        seasonal_order = sm_model_init_args.pop("seasonal_order", (0, 0, 0))
        seasonal_order_is_valid = len(seasonal_order) == 3 and all(isinstance(p, int) for p in seasonal_order)
        if not seasonal_order_is_valid:
            raise ValueError(
                f"{self.name} can't interpret received seasonal_order {seasonal_order} as a "
                "tuple with 3 nonnegative integers (P, D, Q)."
            )

        # Disable seasonality if seasonal_period is too short
        if seasonal_period <= 1:
            sm_model_init_args["seasonal_order"] = (0, 0, 0, 0)
        else:
            sm_model_init_args["seasonal_order"] = tuple(seasonal_order) + (seasonal_period,)

        return sm_model_init_args

    def _fit_local_model(
        self, timeseries: pd.Series, sm_model_init_args: dict, sm_model_fit_args: dict
    ) -> FittedLocalModel:
        with statsmodels_warning_filter():
            model = StatsmodelsSARIMAX(endog=timeseries, **sm_model_init_args)
            fit_result = model.fit(disp=False, **sm_model_fit_args)
        # Only save the parameters of the trained model, not the model itself
        parameters = dict(fit_result.params.iteritems())
        return FittedLocalModel(model_name=self.name, sm_model_init_args=sm_model_init_args, parameters=parameters)

    def _predict_with_local_model(
        self, timeseries: pd.Series, fitted_model: FittedLocalModel, quantile_levels: List[float]
    ) -> pd.DataFrame:
        assert fitted_model.model_name == self.name
        parameters = np.array(list(fitted_model.parameters.values()))
        with statsmodels_warning_filter():
            base_model = StatsmodelsSARIMAX(endog=timeseries, **fitted_model.sm_model_init_args)
            # This is a hack that allows us to set the parameters to their estimated values & initialize the model
            sm_model = base_model.fit(start_params=parameters, maxiter=0, disp=False)
        return self._get_predictions_from_statsmodels_model(
            sm_model=sm_model,
            cutoff=timeseries.index.max(),
            quantile_levels=quantile_levels,
            freq=fitted_model.sm_model_init_args["freq"],
        )


class ThetaModel(AbstractStatsmodelsModel):
    """The Theta forecasting model of Assimakopoulos and Nikolopoulos (2000).

    Based on `statsmodels.tsa.forecasting.theta.ThetaModel <https://www.statsmodels.org/stable/generated/statsmodels.tsa.forecasting.theta.ThetaModel.html>`_.

    Our implementation contains several improvements over the Statsmodels version, such
    as multi-CPU training and reducing the disk usage when saving models.


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

    statsmodels_allowed_init_args = [
        "deseasonalize",
        "seasonal_period",
        "use_test",
        "method",
        "difference",
    ]
    statsmodels_allowed_fit_args = []

    def _update_sm_model_init_args(self, sm_model_init_args: dict, data: TimeSeriesDataFrame) -> dict:
        sm_model_init_args = sm_model_init_args.copy()
        sm_model_init_args.setdefault("deseasonalize", True)

        # Infer seasonal_period if seasonal_period is not given / is set to None
        seasonal_period = sm_model_init_args.pop("seasonal_period", None)
        if seasonal_period is None:
            seasonal_period = get_seasonality(data.freq)
        sm_model_init_args["period"] = seasonal_period

        return sm_model_init_args

    def _fit_local_model(
        self, timeseries: pd.Series, sm_model_init_args: dict, sm_model_fit_args: dict
    ) -> FittedLocalModel:
        # ThetaModel in statsmodels doesn't provide a way to initialize the model with trained parameters,
        # so we delegate training to `_predict_with_local_model`
        if sm_model_init_args["deseasonalize"] and len(timeseries) < 2 * sm_model_init_args["period"]:
            sm_model_init_args = sm_model_init_args.copy()
            sm_model_init_args["deseasonalize"] = False
        return FittedLocalModel(model_name=self.name, sm_model_init_args=sm_model_init_args, parameters=None)

    def _predict_with_local_model(
        self, timeseries: pd.Series, fitted_model: FittedLocalModel, quantile_levels: List[float]
    ) -> pd.DataFrame:
        assert fitted_model.model_name == self.name
        timeseries.index.freq = self.freq
        with statsmodels_warning_filter():
            base_model = StatsmodelsTheta(endog=timeseries, **fitted_model.sm_model_init_args)
            sm_model = base_model.fit(disp=False)
        # The API is inconsistent with ETS/ARIMA, so we access the predictions differently here
        results = [sm_model.forecast(self.prediction_length).rename("mean")]
        for q in quantile_levels:
            if q < 0.5:
                coverage = 2 * q
                column_index = 0
            else:
                coverage = 2 * (1 - q)
                column_index = 1
            quantile_pred = sm_model.prediction_intervals(steps=self.prediction_length, alpha=coverage)
            # Select lower bound of the confidence interval if q < 0.5, upper bound otherwise
            results.append(quantile_pred.iloc[:, column_index].rename(str(q)))
        return pd.concat(results, axis=1)

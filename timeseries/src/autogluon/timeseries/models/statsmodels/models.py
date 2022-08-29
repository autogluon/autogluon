from typing import List

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame
from autogluon.timeseries.utils.seasonality import get_seasonality
from autogluon.timeseries.utils.warning_filters import statsmodels_warning_filter

from .abstract_statsmodels import AbstractStatsmodelsModel, FittedLocalModel


class StatsmodelsETSModel(AbstractStatsmodelsModel):
    """Exponential smoothing with trend and seasonality model.

    Based on `statsmodels.tsa.exponential_smoothing.ets.ETSModel`.

    See `AbstractStatsmodelsModel` for common parameters.


    Other Parameters
    ----------------
    error : str, default = "add"
        Error model. Allowed values are "add" and "mul".
        Note that "mul" is only applicable for time series with positive values.
    trend : str or None, default = "add"
        Trend component model. Allowed values are "add", "mul" and None.
        Note that "mul" is only applicable for time series with positive values.
    damped_trend : bool, default = False
        Whether or not the included trend component is damped.
    seasonal : bool, default = True
        Whether to enable additive seasonality.
    seasonal_periods : int or None, default = None
        Number of time steps in a complete seasonal cycle for seasonal models. For example, 7 for daily data with a
        weekly cycle or 12 for monthly data with an annual cycle.
        When set to None, seasonal_periods will be inferred from the frequency of the training data. Can also be
        specified manually by providing an integer > 1.
        If seasonal_periods (inferred or provided) is equal to 1, seasonality will be disabled.
    maxiter : int, default = 1000
        Number of iterations during optimization.
    n_jobs : int, default = -1
        Number of CPU cores used to fit the models in parallel. When set to -1, all CPU cores are used.
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
        "n_jobs",
    ]

    def _fit_local_model(
        self, timeseries: pd.Series, default_model_init_args: dict, default_model_fit_args: dict
    ) -> FittedLocalModel:
        model_init_args = default_model_init_args.copy()
        # Infer seasonal_periods if seasonal_periods is not given / is set to None
        seasonal_periods = model_init_args.pop("seasonal_periods", None)
        if seasonal_periods is None:
            seasonal_periods = get_seasonality(self.freq)
        # Disable seasonality if the model cannot be fit with seasonality
        if len(timeseries) < 2 * seasonal_periods or seasonal_periods == 1:
            model_init_args["seasonal"] = None
            model_init_args["seasonal_periods"] = 1
        else:
            model_init_args["seasonal_periods"] = seasonal_periods

        with statsmodels_warning_filter():
            model = ETSModel(endog=timeseries, freq=self.freq, **model_init_args)
            fit_result = model.fit(full_output=False, disp=False, **default_model_fit_args)
        # Only save the parameters of the trained model, not the model itself
        parameters = dict(zip(fit_result.param_names, fit_result.params))
        return FittedLocalModel(model_name=self.name, model_init_args=model_init_args, parameters=parameters)

    def _predict_with_local_model(
        self, timeseries: pd.Series, fitted_model: FittedLocalModel, quantile_levels: List[float]
    ) -> pd.DataFrame:
        assert fitted_model.model_name == self.name
        with statsmodels_warning_filter():
            base_model = ETSModel(endog=timeseries, freq=self.freq, **fitted_model.model_init_args)
            parameters = np.array(list(fitted_model.parameters.values()))
            # This is a hack that allows us to set the parameters to their estimated values & initialize the model
            initialized_model = base_model.fit(start_params=parameters, maxiter=0, disp=False)
        return self._get_predictions_from_initialized_model(
            initialized_model=initialized_model, cutoff=timeseries.index.max(), quantile_levels=quantile_levels
        )


class StatsmodelsARIMAModel(AbstractStatsmodelsModel):
    """Autoregressive Integrated Moving Average (ARIMA) model.

    Based on `statsmodels.tsa.statespace.sarimax.SARIMAX`

    See `AbstractStatsmodelsModel` for common parameters.

    Other Parameters
    ----------------
    order: Tuple[int, int, int], default = (1, 1, 1)
        The (p, d, q) order of the model for the number of AR parameters, differences, and MA parameters to use.
    seasonal_order: Tuple[int, int, int], default = (0, 0, 0)
        The (P, D, Q) parameters of the seasonal ARIMA model. Setting to (0, 0, 0) disables seasonality.
    seasonal_periods : int or None, default = None
        Number of time steps in a complete seasonal cycle for seasonal models. For example, 7 for daily data with a
        weekly cycle or 12 for monthly data with an annual cycle.
        When set to None, seasonal_periods will be inferred from the frequency of the training data. Can also be
        specified manually by providing an integer > 1.
        If seasonal_periods (inferred or provided) is equal to 1, seasonality will be disabled.
    enforce_stationarity : bool, default = True
        Whether to transform the AR parameters to enforce stationarity in the autoregressive component of the model.
        If ARIMA crashes during fitting with an LU decomposition error, you can either set enforce_stationarity to
        False or increase the differencing parameter `d` in `order`.
    maxiter : int, default = 1000
        Number of iterations during optimization.
    n_jobs : int, default = -1
        Number of CPU cores used to fit the models in parallel. When set to -1, all CPU cores are used.
    """

    quantile_method_name = "conf_int"
    statsmodels_allowed_init_args = [
        "order",
        "seasonal_order",
        "seasonal_period",
        "enforce_stationarity",
    ]
    statsmodels_allowed_fit_args = [
        "maxiter",
        "n_jobs",
    ]

    def _fit_local_model(
        self, timeseries: TimeSeriesDataFrame, default_model_init_args: dict, default_model_fit_args: dict
    ) -> FittedLocalModel:
        model_init_args = default_model_init_args.copy()
        # Set trend to constant if trend = True
        model_init_args.setdefault("enforce_stationarity", True)
        model_init_args.setdefault("with_intercept", True)
        # Infer seasonal_periods if seasonal_periods is not given / is set to None
        seasonal_period = model_init_args.pop("seasonal_period", None)
        if seasonal_period is None:
            seasonal_period = get_seasonality(self.freq)
        seasonal_order = model_init_args.pop("seasonal_order", (0, 0, 0))

        # Disable seasonality if seasonal_period is too short
        if seasonal_period <= 1:
            model_init_args["seasonal_order"] = (0, 0, 0, 0)
        else:
            model_init_args["seasonal_order"] = tuple(seasonal_order) + (seasonal_period,)

        with statsmodels_warning_filter():
            model = SARIMAX(endog=timeseries, freq=self.freq, **model_init_args)
            fit_result = model.fit(disp=False, **default_model_fit_args)
        # Only save the parameters of the trained model, not the model itself
        parameters = dict(fit_result.params.iteritems())
        return FittedLocalModel(model_name=self.name, model_init_args=model_init_args, parameters=parameters)

    def _predict_with_local_model(
        self, timeseries: pd.Series, fitted_model: FittedLocalModel, quantile_levels: List[float]
    ) -> pd.DataFrame:
        assert fitted_model.model_name == self.name
        parameters = np.array(list(fitted_model.parameters.values()))
        with statsmodels_warning_filter():
            base_model = SARIMAX(endog=timeseries, freq=self.freq, **fitted_model.model_init_args)
            # This is a hack that allows us to set the parameters to their estimated values & initialize the model
            initialized_model = base_model.fit(start_params=parameters, maxiter=0, disp=False)
        return self._get_predictions_from_initialized_model(
            initialized_model=initialized_model, cutoff=timeseries.index.max(), quantile_levels=quantile_levels
        )

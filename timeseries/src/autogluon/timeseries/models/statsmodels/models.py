from typing import List

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame
from autogluon.timeseries.utils.seasonality import get_seasonality

from .abstract_statsmodels import AbstractStatsmodelsModel, ModelFitSummary


class StatsmodelsETSModel(AbstractStatsmodelsModel):
    # TODO: Expose more hyperparameters & make them consistent across models
    # TODO: Check default values of hyperparameters
    # TODO: Add docstrings
    quantile_method_name = "pred_int"
    statsmodels_allowed_init_args = [
        "error",
        "trend",
        "seasonal",
        "seasonal_period",
    ]
    statsmodels_allowed_fit_args = [
        "maxiter",
    ]

    def _fit_local_model(
        self, timeseries: TimeSeriesDataFrame, default_model_init_args: dict, default_model_fit_args: dict
    ) -> ModelFitSummary:
        model_init_args = default_model_init_args.copy()
        # Infer seasonal_periods if seasonal_periods is not given / is set to None
        seasonal_period = model_init_args.pop("seasonal_period", None)
        if seasonal_period is None:
            seasonal_period = get_seasonality(self.freq)
        # Disable seasonality if the model cannot be fit with seasonality
        if len(timeseries) < 2 * seasonal_period or seasonal_period == 1:
            model_init_args["seasonal"] = None
            model_init_args["seasonal_periods"] = 1
        else:
            model_init_args["seasonal_periods"] = seasonal_period

        model = ETSModel(endog=timeseries.squeeze(1), freq=self.freq, **model_init_args)
        fit_result = model.fit(full_output=False, disp=False, **default_model_fit_args)
        # Only save the parameters of the trained model, not the model itself
        parameters = dict(zip(fit_result.param_names, fit_result.params))
        return ModelFitSummary(model_name=self.name, model_init_args=model_init_args, parameters=parameters)

    def _predict_using_fit_summary(
        self, fit_summary: ModelFitSummary, timeseries: TimeSeriesDataFrame, quantile_levels: List[float]
    ) -> pd.DataFrame:
        assert fit_summary.model_name == self.name
        # Initialize the model with fitted parameters
        model = ETSModel(endog=timeseries.squeeze(1), freq=self.freq, **fit_summary.model_init_args)
        parameters = np.array(list(fit_summary.parameters.values()))
        fitted_model = model.fit(start_params=parameters, maxiter=0, disp=False)
        return self._get_predictions_from_fitted_model(
            fitted_model=fitted_model, cutoff=timeseries.index.max(), quantile_levels=quantile_levels
        )


class StatsmodelsARIMAModel(AbstractStatsmodelsModel):
    # TODO: Expose more hyperparameters & make them consistent across models
    # TODO: Check default values of hyperparameters
    # TODO: Add docstrings
    quantile_method_name = "conf_int"
    statsmodels_allowed_init_args = [
        "order",
        "seasonal_order",
        "seasonal_period",
        "trend",
        "enforce_stationarity",
    ]
    statsmodels_allowed_fit_args = [
        "maxiter",
    ]

    def _fit_local_model(
        self, timeseries: TimeSeriesDataFrame, default_model_init_args: dict, default_model_fit_args: dict
    ) -> ModelFitSummary:
        model_init_args = default_model_init_args.copy()
        # Set trend to constant if trend = True
        trend = model_init_args.pop("trend", True)
        if trend:
            model_init_args["trend"] = "c"
        model_init_args.setdefault("enforce_stationarity", False)
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

        model = SARIMAX(endog=timeseries.squeeze(1), freq=self.freq, **model_init_args)
        fit_result = model.fit(disp=False, **default_model_fit_args)
        # Only save the parameters of the trained model, not the model itself
        parameters = dict(fit_result.params.iteritems())
        return ModelFitSummary(model_name=self.name, model_init_args=model_init_args, parameters=parameters)

    def _predict_using_fit_summary(
        self, fit_summary: ModelFitSummary, timeseries: TimeSeriesDataFrame, quantile_levels: List[float]
    ) -> pd.DataFrame:
        assert fit_summary.model_name == self.name
        model = SARIMAX(endog=timeseries.squeeze(1), freq=self.freq, **fit_summary.model_init_args)
        parameters = np.array(list(fit_summary.parameters.values()))
        fitted_model = model.fit(start_params=parameters, maxiter=0, disp=False)
        return self._get_predictions_from_fitted_model(
            fitted_model=fitted_model, cutoff=timeseries.index.max(), quantile_levels=quantile_levels
        )

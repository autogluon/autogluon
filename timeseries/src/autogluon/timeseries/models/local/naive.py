import numpy as np
import pandas as pd

from autogluon.timeseries.models.local.abstract_local_model import AbstractLocalModel, seasonal_naive_forecast


class NaiveModel(AbstractLocalModel):
    """Baseline model that sets the forecast equal to the last observed value.

    Quantiles are obtained by assuming that the residuals follow zero-mean normal distribution, scale of which is
    estimated from the empirical distribution of the residuals.
    As described in https://otexts.com/fpp3/prediction-intervals.html

    """

    allowed_local_model_args = ["seasonal_period"]

    def _predict_with_local_model(
        self,
        time_series: pd.Series,
        local_model_args: dict,
    ) -> pd.DataFrame:
        return seasonal_naive_forecast(
            target=time_series.values.ravel(),
            prediction_length=self.prediction_length,
            quantile_levels=self.quantile_levels,
            seasonal_period=1,
        )


class SeasonalNaiveModel(AbstractLocalModel):
    """Baseline model that sets the forecast equal to the last observed value from the same season.

    Quantiles are obtained by assuming that the residuals follow zero-mean normal distribution, scale of which is
    estimated from the empirical distribution of the residuals.
    As described in https://otexts.com/fpp3/prediction-intervals.html


    Other Parameters
    ----------------
    seasonal_period : int or None, default = None
        Number of time steps in a complete seasonal cycle for seasonal models. For example, 7 for daily data with a
        weekly cycle or 12 for monthly data with an annual cycle.
        When set to None, seasonal_period will be inferred from the frequency of the training data. Can also be
        specified manually by providing an integer > 1.
        If seasonal_period (inferred or provided) is equal to 1, will fall back to Naive forecast.
        Seasonality will also be disabled, if the length of the time series is < seasonal_period.
    """

    allowed_local_model_args = ["seasonal_period"]

    def _predict_with_local_model(
        self,
        time_series: np.ndarray,
        local_model_args: dict,
    ) -> pd.DataFrame:
        return seasonal_naive_forecast(
            target=time_series.values.ravel(),
            prediction_length=self.prediction_length,
            quantile_levels=self.quantile_levels,
            seasonal_period=local_model_args["seasonal_period"],
        )

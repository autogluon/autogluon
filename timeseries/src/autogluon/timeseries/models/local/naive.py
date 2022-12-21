from typing import List

import numpy as np
import pandas as pd
from scipy.stats import norm

from autogluon.timeseries.utils.forecast import get_forecast_horizon_index_single_time_series

from .abstract_local_model import AbstractLocalModel


def seasonal_naive_forecast(
    time_series: pd.Series, freq: str, prediction_length: int, quantile_levels: List[float], seasonal_period: int
):
    forecast_timestamps = get_forecast_horizon_index_single_time_series(
        past_timestamps=time_series.index, freq=freq, prediction_length=prediction_length
    )

    target = time_series.values.ravel()
    forecast = {}

    if len(target) > seasonal_period and seasonal_period > 1:
        indices = [len(target) - seasonal_period + k % seasonal_period for k in range(prediction_length)]
        forecast["mean"] = target[indices]
        residuals = target[seasonal_period:] - target[:-seasonal_period]

        sigma = np.sqrt(np.mean(np.square(residuals)))
        num_full_seasons = np.arange(1, prediction_length + 1) // seasonal_period
        sigma_per_timestep = sigma * np.sqrt(num_full_seasons + 1)
    else:
        # Fall back to naive forecast
        forecast["mean"] = np.full(shape=[prediction_length], fill_value=target[-1])
        residuals = target[1:] - target[:-1]

        sigma = np.sqrt(np.mean(np.square(residuals)))
        sigma_per_timestep = sigma * np.sqrt(np.arange(1, prediction_length + 1))

    for q in quantile_levels:
        forecast[str(q)] = forecast["mean"] + norm.ppf(q) * sigma_per_timestep

    return pd.DataFrame(forecast, index=forecast_timestamps)


class NaiveModel(AbstractLocalModel):
    """Baseline model that sets the forecast equal to the last observed value.

    Quantiles are obtained by assuming that the residuals follow zero-mean normal distribution, scale of which is
    estimated from the empirical distribution of the residuals.
    As described in https://otexts.com/fpp3/prediction-intervals.html

    """

    allowed_local_model_args = ["seasonal_period"]

    @staticmethod
    def _predict_with_local_model(
        time_series: pd.Series,
        freq: str,
        prediction_length: int,
        quantile_levels: List[float],
        local_model_args: dict,
        **kwargs,
    ) -> pd.DataFrame:
        return seasonal_naive_forecast(
            time_series=time_series,
            freq=freq,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
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

    @staticmethod
    def _predict_with_local_model(
        time_series: pd.Series,
        freq: str,
        prediction_length: int,
        quantile_levels: List[float],
        local_model_args: dict,
        **kwargs,
    ) -> pd.DataFrame:
        return seasonal_naive_forecast(
            time_series=time_series,
            freq=freq,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            seasonal_period=local_model_args["seasonal_period"],
        )

from typing import List
import numpy as np
import pandas as pd
from joblib import delayed, Parallel

from autogluon.timeseries.utils.forecast import get_forecast_horizon_timestamps
from .abstract_local_model import AbstractLocalModel


class SeasonalNaive(AbstractLocalModel):
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
        seasonal_period = local_model_args["seasonal_period"]
        forecast_timestamps = get_forecast_horizon_timestamps(
            past_timestamps=time_series.index, freq=freq, prediction_length=prediction_length
        )

        target = time_series.values
        forecast = {}
        if len(target) >= seasonal_period:
            indices = [len(target) - seasonal_period + k % seasonal_period for k in range(prediction_length)]
            forecast["mean"] = target[indices].reshape(-1)
        else:
            forecast["mean"] = np.full(shape=[prediction_length], fill_value=target.mean())

        for q in quantile_levels:
            forecast[str(q)] = forecast["mean"]

        return pd.DataFrame(forecast, index=forecast_timestamps)

import logging
import time
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Union
from autogluon.timeseries.dataset import TimeSeriesDataFrame

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from multiprocessing import TimeoutError
from scipy.stats import norm

from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.forecast import get_forecast_horizon_index_ts_dataframe
from autogluon.timeseries.utils.seasonality import get_seasonality
from autogluon.timeseries.utils.warning_filters import statsmodels_joblib_warning_filter, statsmodels_warning_filter

logger = logging.getLogger(__name__)


class AbstractLocalModel(AbstractTimeSeriesModel):
    allowed_local_model_args: List[str] = []
    MAX_TS_LENGTH: Optional[int] = None
    DEFAULT_N_JOBS: int = -1

    def __init__(
        self,
        freq: Optional[str] = None,
        prediction_length: int = 1,
        path: Optional[str] = None,
        name: Optional[str] = None,
        eval_metric: str = None,
        hyperparameters: Dict[str, Any] = None,
        **kwargs,  # noqa
    ):
        super().__init__(
            path=path,
            freq=freq,
            prediction_length=prediction_length,
            name=name,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            **kwargs,
        )
        if hyperparameters is None:
            hyperparameters = {}
        # TODO: Replace with 'num_cpus' argument passed to fit (after predictor API is changed)
        n_jobs = hyperparameters.get("n_jobs", self.DEFAULT_N_JOBS)
        if isinstance(n_jobs, float) and 0 < n_jobs <= 1:
            self.n_jobs = max(int(cpu_count() * n_jobs), 1)
        elif isinstance(n_jobs, int):
            self.n_jobs = n_jobs
        else:
            raise ValueError(f"n_jobs must be a float between 0 and 1 or an integer (received n_jobs = {n_jobs})")
        self._local_model_args: Dict[str, Any] = None
        self._seasonal_period: Optional[int] = None
        self.time_limit: Optional[float] = None

    def _fit(self, train_data: TimeSeriesDataFrame, time_limit: int = None, **kwargs):
        self._check_fit_params()
        # Initialize parameters passed to each local model
        raw_local_model_args = self._get_model_params().copy()
        raw_local_model_args.pop("n_jobs", None)

        unused_local_model_args = []
        local_model_args = {}
        for key, value in raw_local_model_args.items():
            if key in self.allowed_local_model_args:
                local_model_args[key] = value
            else:
                unused_local_model_args.append(key)

        if len(unused_local_model_args):
            logger.warning(
                f"{self.name} ignores following hyperparameters: {unused_local_model_args}. "
                f"See the docstring of {self.name} for the list of supported hyperparameters."
            )

        if "seasonal_period" not in local_model_args or local_model_args["seasonal_period"] is None:
            local_model_args["seasonal_period"] = get_seasonality(train_data.freq)
        self._seasonal_period = local_model_args["seasonal_period"]

        self._local_model_args = self._update_local_model_args(local_model_args=local_model_args)
        self.time_limit = time_limit

        logger.debug(f"{self.name} is a local model, so the model will be fit at prediction time.")
        return self

    def _update_local_model_args(self, local_model_args: Dict[str, Any]) -> Dict[str, Any]:
        return local_model_args

    def predict(self, data: TimeSeriesDataFrame, **kwargs) -> TimeSeriesDataFrame:
        if self.MAX_TS_LENGTH is not None:
            logger.debug(f"Shortening all time series to at most {self.MAX_TS_LENGTH}")
            data = data.groupby(level=ITEMID, sort=False).tail(self.MAX_TS_LENGTH)

        # Option 1: GroupedArray
        # from statsforecast.core import _grouped_array_from_df
        # df = pd.DataFrame(data).reset_index().rename(columns={ITEMID: "unique_id", TIMESTAMP: "ds", self.target: "y"})
        # df = df[["unique_id", "ds", "y"]].set_index("unique_id")
        # all_series, self.uids, self.last_dates, self.ds = _grouped_array_from_df(df, sort_df=False)

        # Option 2: Pandas groupby
        df = pd.DataFrame(data).reset_index(level=ITEMID)
        all_series = (ts for _, ts in df.groupby(by=ITEMID, as_index=False, sort=False)[self.target])

        # timeout ensures that no individual job takes longer than time_limit
        # TODO: a job started late may still exceed time_limit - how to prevent that?
        timeout = None if self.n_jobs == 1 else self.time_limit
        # end_time ensures that no new jobs are started after time_limit is exceeded
        end_time = None if self.time_limit is None else time.time() + self.time_limit

        executor = Parallel(self.n_jobs, timeout=timeout)
        # TODO: Make sure that the parallel workers are available at prediction time
        # executor._backend_args["idle_worker_timeout"] = 10000

        try:
            with statsmodels_joblib_warning_filter(), statsmodels_warning_filter():
                predictions = executor(delayed(self._predict_wrapper)(ts, end_time=end_time) for ts in all_series)
        except TimeoutError:
            raise TimeLimitExceeded

        predictions_df = pd.concat(predictions)
        predictions_df.index = get_forecast_horizon_index_ts_dataframe(data, self.prediction_length)
        return TimeSeriesDataFrame(predictions_df)

    def score_and_cache_oof(
        self, val_data: TimeSeriesDataFrame, store_val_score: bool = False, store_predict_time: bool = False
    ) -> None:
        super().score_and_cache_oof(val_data, store_val_score, store_predict_time)
        # Remove time_limit for future predictions
        self.time_limit = None

    def _predict_wrapper(self, time_series: pd.Series, end_time: Optional[float] = None) -> pd.DataFrame:
        if end_time is not None and time.time() >= end_time:
            raise TimeLimitExceeded
        try:
            result = self._predict_with_local_model(
                time_series=time_series,
                freq=self.freq,
                prediction_length=self.prediction_length,
                quantile_levels=self.quantile_levels,
                local_model_args=self._local_model_args.copy(),
            )
        except Exception as e:
            logger.info("Falling back to SeasonalNaive")
            result = seasonal_naive_forecast(
                target=time_series.values.ravel(),
                prediction_length=self.prediction_length,
                quantile_levels=self.quantile_levels,
                seasonal_period=self._seasonal_period,
            )
        return result

    def _predict_with_local_model(
        self,
        time_series: pd.Series,
        freq: str,
        prediction_length: int,
        quantile_levels: List[float],
        local_model_args: dict,
        **kwargs,
    ) -> pd.DataFrame:
        raise NotImplementedError


def seasonal_naive_forecast(
    target: np.ndarray, prediction_length: int, quantile_levels: List[float], seasonal_period: int
) -> pd.DataFrame:
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

    return pd.DataFrame(forecast)

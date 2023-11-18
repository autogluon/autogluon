import logging
import time
from multiprocessing import TimeoutError, cpu_count
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm

from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.datetime import get_seasonality
from autogluon.timeseries.utils.forecast import get_forecast_horizon_index_ts_dataframe
from autogluon.timeseries.utils.warning_filters import warning_filter

logger = logging.getLogger(__name__)


# We use the same default n_jobs across AG-TS to ensure that Joblib reuses the process pool
AG_DEFAULT_N_JOBS = max(int(cpu_count() * 0.5), 1)


class AbstractLocalModel(AbstractTimeSeriesModel):
    """Abstract class for local forecasting models that are trained separately for each time series.

    Prediction is parallelized across CPU cores using joblib.Parallel.

    Attributes
    ----------
    allowed_local_model_args : List[str]
        Argument that can be passed to the underlying local model.
    default_n_jobs : Union[int, float]
        Default number of CPU cores used to train models. If float, this fraction of CPU cores will be used.
    default_max_ts_length : Optional[int]
        If not None, only the last ``max_ts_length`` time steps of each time series will be used to train the model.
        This significantly speeds up fitting and usually leads to no change in accuracy.
    init_time_in_seconds : int
        Time that it takes to initialize the model in seconds (e.g., because of JIT compilation by Numba).
        If time_limit is below this number, model won't be trained.
    """

    allowed_local_model_args: List[str] = []
    default_n_jobs: Union[int, float] = AG_DEFAULT_N_JOBS
    default_max_ts_length: Optional[int] = 2500
    init_time_in_seconds: int = 0

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
        if hyperparameters is None:
            hyperparameters = {}
        else:
            hyperparameters = hyperparameters.copy()
        # TODO: Replace with 'num_cpus' argument passed to fit (after predictor API is changed)
        n_jobs = hyperparameters.pop("n_jobs", self.default_n_jobs)
        if isinstance(n_jobs, float) and 0 < n_jobs <= 1:
            self.n_jobs = max(int(cpu_count() * n_jobs), 1)
        elif isinstance(n_jobs, int):
            self.n_jobs = n_jobs
        else:
            raise ValueError(f"n_jobs must be a float between 0 and 1 or an integer (received n_jobs = {n_jobs})")
        # Default values, potentially overridden inside _fit()
        self.use_fallback_model = hyperparameters.pop("use_fallback_model", True)
        self.max_ts_length = hyperparameters.pop("max_ts_length", self.default_max_ts_length)

        super().__init__(
            path=path,
            freq=freq,
            prediction_length=prediction_length,
            name=name,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            **kwargs,
        )

        self._local_model_args: Dict[str, Any] = None
        self._seasonal_period: Optional[int] = None
        self.time_limit: Optional[float] = None

    def _fit(self, train_data: TimeSeriesDataFrame, time_limit: Optional[int] = None, verbosity: int = 2, **kwargs):
        self._check_fit_params()
        set_logger_verbosity(verbosity, logger=logger)

        if time_limit is not None and time_limit < self.init_time_in_seconds:
            raise TimeLimitExceeded

        # Initialize parameters passed to each local model
        raw_local_model_args = self._get_model_params().copy()

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
        return self

    def _update_local_model_args(self, local_model_args: Dict[str, Any]) -> Dict[str, Any]:
        return local_model_args

    def _predict(self, data: TimeSeriesDataFrame, **kwargs) -> TimeSeriesDataFrame:
        if self.max_ts_length is not None:
            logger.debug(f"Shortening all time series to at most {self.max_ts_length}")
            data = data.groupby(level=ITEMID, sort=False).tail(self.max_ts_length)

        df = pd.DataFrame(data).reset_index(level=ITEMID)
        all_series = (ts for _, ts in df.groupby(by=ITEMID, as_index=False, sort=False)[self.target])

        # timeout ensures that no individual job takes longer than time_limit
        # TODO: a job started late may still exceed time_limit - how to prevent that?
        timeout = None if self.n_jobs == 1 else self.time_limit
        # end_time ensures that no new jobs are started after time_limit is exceeded
        end_time = None if self.time_limit is None else time.time() + self.time_limit
        executor = Parallel(self.n_jobs, timeout=timeout)

        try:
            with warning_filter():
                predictions_with_flags = executor(
                    delayed(self._predict_wrapper)(ts, end_time=end_time) for ts in all_series
                )
        except TimeoutError:
            raise TimeLimitExceeded

        number_failed_models = sum(failed_flag for _, failed_flag in predictions_with_flags)
        if number_failed_models > 0:
            fraction_failed_models = number_failed_models / len(predictions_with_flags)
            logger.warning(
                f"\tWarning: {self.name} failed for {number_failed_models} time series "
                f"({fraction_failed_models:.1%}). Fallback model SeasonalNaive was used for these time series."
            )
        predictions_df = pd.concat([pred for pred, _ in predictions_with_flags])
        predictions_df.index = get_forecast_horizon_index_ts_dataframe(data, self.prediction_length)
        return TimeSeriesDataFrame(predictions_df)

    def score_and_cache_oof(
        self, val_data: TimeSeriesDataFrame, store_val_score: bool = False, store_predict_time: bool = False
    ) -> None:
        super().score_and_cache_oof(val_data, store_val_score, store_predict_time)
        # Remove time_limit for future predictions
        self.time_limit = None

    def _predict_wrapper(self, time_series: pd.Series, end_time: Optional[float] = None) -> Tuple[pd.DataFrame, bool]:
        if end_time is not None and time.time() >= end_time:
            raise TimeLimitExceeded
        try:
            result = self._predict_with_local_model(
                time_series=time_series,
                local_model_args=self._local_model_args.copy(),
            )
            if not np.isfinite(result.values).all():
                raise RuntimeError("Forecast contains NaN or Inf values.")
            model_failed = False
        except Exception:
            if self.use_fallback_model:
                result = seasonal_naive_forecast(
                    target=time_series.values.ravel(),
                    prediction_length=self.prediction_length,
                    quantile_levels=self.quantile_levels,
                    seasonal_period=self._seasonal_period,
                )
                model_failed = True
            else:
                raise
        return result, model_failed

    def _predict_with_local_model(
        self,
        time_series: pd.Series,
        local_model_args: dict,
    ) -> pd.DataFrame:
        raise NotImplementedError


def seasonal_naive_forecast(
    target: np.ndarray, prediction_length: int, quantile_levels: List[float], seasonal_period: int
) -> pd.DataFrame:
    """Generate seasonal naive forecast, predicting the last observed value from the same period."""
    forecast = {}
    # At least seasonal_period + 2 values are required to compute sigma for seasonal naive
    if len(target) > seasonal_period + 1 and seasonal_period > 1:
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

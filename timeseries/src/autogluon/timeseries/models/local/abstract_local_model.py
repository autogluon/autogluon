import logging
import time
from multiprocessing import TimeoutError, cpu_count
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm

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
        self._dummy_forecast: Optional[pd.DataFrame] = None

    @property
    def allowed_hyperparameters(self) -> List[str]:
        return (
            super().allowed_hyperparameters
            + ["use_fallback_model", "max_ts_length", "n_jobs"]
            + self.allowed_local_model_args
        )

    def preprocess(self, data: TimeSeriesDataFrame, is_train: bool = False, **kwargs) -> Any:
        if not self._get_tags()["allow_nan"]:
            data = data.fill_missing_values()
        return data

    def _fit(self, train_data: TimeSeriesDataFrame, time_limit: Optional[int] = None, **kwargs):
        self._check_fit_params()

        if time_limit is not None and time_limit < self.init_time_in_seconds:
            raise TimeLimitExceeded

        # Initialize parameters passed to each local model
        raw_local_model_args = self._get_model_params().copy()

        unused_local_model_args = []
        local_model_args = {}
        # TODO: Move filtering logic to AbstractTimeSeriesModel
        for key, value in raw_local_model_args.items():
            if key in self.allowed_hyperparameters:
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

        self._dummy_forecast = self._get_dummy_forecast(train_data)
        return self

    def _get_dummy_forecast(self, train_data: TimeSeriesDataFrame) -> pd.DataFrame:
        agg_functions = ["mean"] + [get_quantile_function(q) for q in self.quantile_levels]
        stats_marginal = train_data[self.target].agg(agg_functions)
        stats_repeated = np.tile(stats_marginal.values, [self.prediction_length, 1])
        return pd.DataFrame(stats_repeated, columns=stats_marginal.index)

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
        time_limit = kwargs.get("time_limit")
        timeout = None if self.n_jobs == 1 else time_limit
        # end_time ensures that no new jobs are started after time_limit is exceeded
        end_time = None if time_limit is None else time.time() + time_limit
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
        predictions_df.index = get_forecast_horizon_index_ts_dataframe(data, self.prediction_length, freq=self.freq)
        return TimeSeriesDataFrame(predictions_df)

    def score_and_cache_oof(
        self,
        val_data: TimeSeriesDataFrame,
        store_val_score: bool = False,
        store_predict_time: bool = False,
        **predict_kwargs,
    ) -> None:
        # All computation happens during inference, so we provide the time_limit at prediction time
        super().score_and_cache_oof(
            val_data, store_val_score, store_predict_time, time_limit=self.time_limit, **predict_kwargs
        )

    def _predict_wrapper(self, time_series: pd.Series, end_time: Optional[float] = None) -> Tuple[pd.DataFrame, bool]:
        if end_time is not None and time.time() >= end_time:
            raise TimeLimitExceeded

        model_failed = False
        if time_series.isna().all():
            result = self._dummy_forecast.copy()
        else:
            try:
                result = self._predict_with_local_model(
                    time_series=time_series,
                    local_model_args=self._local_model_args.copy(),
                )
                if not np.isfinite(result.values).all():
                    raise RuntimeError("Forecast contains NaN or Inf values.")
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

    def numpy_fillna(arr: np.ndarray) -> np.ndarray:
        """Fast implementation of forward fill + avg fill in numpy."""
        # First apply forward fill
        idx = np.arange(len(arr))
        mask = np.isnan(arr)
        idx[mask] = 0
        arr_filled = arr[np.maximum.accumulate(idx)]
        # Leading NaNs are filled with the mean
        arr_filled[np.isnan(arr_filled)] = np.nanmean(arr_filled)
        return arr_filled

    forecast = {}
    # At least seasonal_period + 2 values are required to compute sigma for seasonal naive
    if len(target) > seasonal_period + 1 and seasonal_period > 1:
        if np.isnan(target[-(seasonal_period + 2) :]).any():
            target = numpy_fillna(target)

        indices = [len(target) - seasonal_period + k % seasonal_period for k in range(prediction_length)]
        forecast["mean"] = target[indices]
        residuals = target[seasonal_period:] - target[:-seasonal_period]

        sigma = np.sqrt(np.nanmean(np.square(residuals)))
        num_full_seasons = np.arange(1, prediction_length + 1) // seasonal_period
        sigma_per_timestep = sigma * np.sqrt(num_full_seasons + 1)
    else:
        # Fall back to naive forecast
        last_observed_value = target[np.isfinite(target)][-1]
        forecast["mean"] = np.full(shape=[prediction_length], fill_value=last_observed_value)
        residuals = target[1:] - target[:-1]

        sigma = np.sqrt(np.nanmean(np.square(residuals)))
        if np.isnan(sigma):  # happens if there are no two consecutive non-nan observations
            sigma = 0.0
        sigma_per_timestep = sigma * np.sqrt(np.arange(1, prediction_length + 1))

    for q in quantile_levels:
        forecast[str(q)] = forecast["mean"] + norm.ppf(q) * sigma_per_timestep

    return pd.DataFrame(forecast)


def get_quantile_function(q: float) -> Callable:
    """Returns a function with name "q" that computes the q'th quantile of a pandas.Series."""

    def quantile_fn(x: pd.Series) -> pd.Series:
        return x.quantile(q)

    quantile_fn.__name__ = str(q)
    return quantile_fn

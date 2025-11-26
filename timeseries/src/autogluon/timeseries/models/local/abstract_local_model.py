import logging
import time
from multiprocessing import TimeoutError
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from scipy.stats import norm

from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.utils.constants import AG_DEFAULT_N_JOBS
from autogluon.timeseries.metrics import TimeSeriesScorer
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.datetime import get_seasonality
from autogluon.timeseries.utils.warning_filters import warning_filter

logger = logging.getLogger(__name__)


class AbstractLocalModel(AbstractTimeSeriesModel):
    """Abstract class for local forecasting models that are trained separately for each time series.

    Prediction is parallelized across CPU cores using joblib.Parallel.

    Attributes
    ----------
    allowed_local_model_args
        Argument that can be passed to the underlying local model.
    default_max_ts_length
        If not None, only the last ``max_ts_length`` time steps of each time series will be used to train the model.
        This significantly speeds up fitting and usually leads to no change in accuracy.
    init_time_in_seconds
        Time that it takes to initialize the model in seconds (e.g., because of JIT compilation by Numba).
        If time_limit is below this number, model won't be trained.
    """

    allowed_local_model_args: list[str] = []
    default_max_ts_length: Optional[int] = 2500
    default_max_time_limit_ratio = 1.0
    init_time_in_seconds: int = 0

    def __init__(
        self,
        freq: Optional[str] = None,
        prediction_length: int = 1,
        path: Optional[str] = None,
        name: Optional[str] = None,
        eval_metric: Union[str, TimeSeriesScorer, None] = None,
        hyperparameters: Optional[dict[str, Any]] = None,
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

        self._local_model_args: dict[str, Any]
        self._seasonal_period: int
        self._dummy_forecast: pd.DataFrame

    @property
    def allowed_hyperparameters(self) -> list[str]:
        return (
            super().allowed_hyperparameters
            + ["use_fallback_model", "max_ts_length", "n_jobs"]
            + self.allowed_local_model_args
        )

    def preprocess(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        is_train: bool = False,
        **kwargs,
    ) -> tuple[TimeSeriesDataFrame, Optional[TimeSeriesDataFrame]]:
        if not self._get_tags()["allow_nan"]:
            data = data.fill_missing_values()
        return data, known_covariates

    def _get_default_hyperparameters(self) -> dict:
        return {
            "n_jobs": AG_DEFAULT_N_JOBS,
            "use_fallback_model": True,
            "max_ts_length": self.default_max_ts_length,
        }

    @staticmethod
    def _compute_n_jobs(n_jobs: Union[int, float]) -> int:
        if isinstance(n_jobs, float) and 0 < n_jobs <= 1:
            return max(int(cpu_count() * n_jobs), 1)
        elif isinstance(n_jobs, int):
            return n_jobs
        else:
            raise ValueError(f"n_jobs must be a float between 0 and 1 or an integer (received n_jobs = {n_jobs})")

    def _fit(self, train_data: TimeSeriesDataFrame, time_limit: Optional[int] = None, **kwargs):
        self._check_fit_params()

        if time_limit is not None and time_limit < self.init_time_in_seconds:
            raise TimeLimitExceeded

        local_model_args = {}
        for key, value in self.get_hyperparameters().items():
            if key in self.allowed_local_model_args:
                local_model_args[key] = value

        self._log_unused_hyperparameters(extra_allowed_hyperparameters=self.allowed_local_model_args)

        if "seasonal_period" not in local_model_args or local_model_args["seasonal_period"] is None:
            local_model_args["seasonal_period"] = get_seasonality(self.freq)
        self._seasonal_period = local_model_args["seasonal_period"]

        self._local_model_args = self._update_local_model_args(local_model_args=local_model_args)

        self._dummy_forecast = self._get_dummy_forecast(train_data)
        return self

    def _get_dummy_forecast(self, train_data: TimeSeriesDataFrame, max_num_rows: int = 20_000) -> pd.DataFrame:
        agg_functions = ["mean"] + [get_quantile_function(q) for q in self.quantile_levels]
        target_series = train_data[self.target]
        if len(target_series) > max_num_rows:
            target_series = target_series.sample(max_num_rows, replace=True)
        stats_marginal = target_series.agg(agg_functions)
        stats_repeated = np.tile(stats_marginal.values, [self.prediction_length, 1])
        return pd.DataFrame(stats_repeated, columns=stats_marginal.index)

    def _update_local_model_args(self, local_model_args: dict[str, Any]) -> dict[str, Any]:
        return local_model_args

    def _predict(self, data: TimeSeriesDataFrame, **kwargs) -> TimeSeriesDataFrame:
        model_params = self.get_hyperparameters()
        max_ts_length = model_params["max_ts_length"]
        if max_ts_length is not None:
            logger.debug(f"Shortening all time series to at most {max_ts_length}")
            data = data.slice_by_timestep(-max_ts_length, None)

        indptr = data.get_indptr()
        target_series = data[self.target].droplevel(level=TimeSeriesDataFrame.ITEMID)
        all_series = (target_series[indptr[i] : indptr[i + 1]] for i in range(len(indptr) - 1))

        # timeout ensures that no individual job takes longer than time_limit
        # TODO: a job started late may still exceed time_limit - how to prevent that?
        time_limit = kwargs.get("time_limit")
        # TODO: Take into account num_cpus once the TimeSeriesPredictor API is updated
        n_jobs = self._compute_n_jobs(model_params["n_jobs"])
        timeout = None if n_jobs == 1 else time_limit
        # end_time ensures that no new jobs are started after time_limit is exceeded
        end_time = None if time_limit is None else time.time() + time_limit
        executor = Parallel(n_jobs=n_jobs, timeout=timeout)

        try:
            with warning_filter():
                predictions_with_flags = executor(
                    delayed(self._predict_wrapper)(
                        ts, use_fallback_model=model_params["use_fallback_model"], end_time=end_time
                    )
                    for ts in all_series
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
        predictions_df.index = self.get_forecast_horizon_index(data)
        return TimeSeriesDataFrame(predictions_df)

    def _predict_wrapper(
        self,
        time_series: pd.Series,
        use_fallback_model: bool,
        end_time: Optional[float] = None,
    ) -> tuple[pd.DataFrame, bool]:
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
                if use_fallback_model:
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
    target: np.ndarray, prediction_length: int, quantile_levels: list[float], seasonal_period: int
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

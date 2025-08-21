import warnings
from typing import Optional, Sequence, Union, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.utils.datetime import get_seasonality
from autogluon.timeseries.utils.warning_filters import warning_filter


class TimeSeriesScorer:
    """Base class for all evaluation metrics used in AutoGluon-TimeSeries.

    This object always returns the metric in greater-is-better format.

    Follows the design of ``autogluon.core.metrics.Scorer``.

    Parameters
    ----------
    prediction_length : int, default = 1
        The length of the forecast horizon. The predictions provided to the ``TimeSeriesScorer`` are expected to contain
        a forecast for this many time steps for each time series.
    seasonal_period : int or None, default = None
        Seasonal period used to compute some evaluation metrics such as mean absolute scaled error (MASE). Defaults to
        ``None``, in which case the seasonal period is computed based on the data frequency.
    horizon_weight : Sequence[float], np.ndarray or None, default = None
        Weight assigned to each time step in the forecast horizon when computing the metric. If provided, the
        ``horizon_weight`` will be stored as a numpy array of shape ``[1, prediction_length]``.

    Attributes
    ----------
    greater_is_better_internal : bool, default = False
        Whether internal method :meth:`~autogluon.timeseries.metrics.TimeSeriesScorer.compute_metric` is
        a loss function (default), meaning low is good, or a score function, meaning high is good.
    optimum : float, default = 0.0
        The best score achievable by the score function, i.e. maximum in case of scorer function and minimum in case of
        loss function.
    optimized_by_median : bool, default = False
        Whether given point forecast metric is optimized by the median (if True) or expected value (if False). If True,
        all models in AutoGluon-TimeSeries will attempt to paste median forecast into the "mean" column.
    needs_quantile : bool, default = False
        Whether the given metric uses the quantile predictions. Some models will modify the training procedure if they
        are trained to optimize a quantile metric.
    equivalent_tabular_regression_metric : str
        Name of an equivalent metric used by AutoGluon-Tabular with ``problem_type="regression"``. Used by forecasting
        models that train tabular regression models under the hood. This attribute should only be specified by point
        forecast metrics.
    """

    greater_is_better_internal: bool = False
    optimum: float = 0.0
    optimized_by_median: bool = False
    needs_quantile: bool = False
    equivalent_tabular_regression_metric: Optional[str] = None

    def __init__(
        self,
        prediction_length: int = 1,
        seasonal_period: Optional[int] = None,
        horizon_weight: Optional[Sequence[float]] = None,
    ):
        self.prediction_length = int(prediction_length)
        if self.prediction_length < 1:
            raise ValueError(f"prediction_length must be >= 1 (received {prediction_length})")
        self.seasonal_period = seasonal_period
        self.horizon_weight = self.check_get_horizon_weight(horizon_weight, prediction_length=prediction_length)

    @property
    def sign(self) -> int:
        return 1 if self.greater_is_better_internal else -1

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    @property
    def name_with_sign(self) -> str:
        if self.greater_is_better_internal:
            prefix = ""
        else:
            prefix = "-"
        return f"{prefix}{self.name}"

    def __call__(
        self,
        data: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        **kwargs,
    ) -> float:
        seasonal_period = get_seasonality(data.freq) if self.seasonal_period is None else self.seasonal_period

        if "prediction_length" in kwargs:
            warnings.warn(
                "Passing `prediction_length` to `TimeSeriesScorer.__call__` is deprecated and will be removed in v2.0. "
                "Please set the `eval_metric.prediction_length` attribute instead.",
                category=FutureWarning,
            )
            self.prediction_length = kwargs["prediction_length"]
            self.horizon_weight = self.check_get_horizon_weight(self.horizon_weight, self.prediction_length)

        data_past = data.slice_by_timestep(None, -self.prediction_length)
        data_future = data.slice_by_timestep(-self.prediction_length, None)

        assert not predictions.isna().any().any(), "Predictions contain NaN values."
        assert (predictions.num_timesteps_per_item() == self.prediction_length).all()
        assert data_future.index.equals(predictions.index), "Prediction and data indices do not match."

        try:
            with warning_filter():
                self.save_past_metrics(
                    data_past=data_past,
                    target=target,
                    seasonal_period=seasonal_period,
                    **kwargs,
                )
                metric_value = self.compute_metric(
                    data_future=data_future,
                    predictions=predictions,
                    target=target,
                    **kwargs,
                )
        finally:
            self.clear_past_metrics()
        return metric_value * self.sign

    score = __call__

    def compute_metric(
        self,
        data_future: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        **kwargs,
    ) -> float:
        """Internal method that computes the metric for given forecast & actual data.

        This method should be implemented by all custom metrics.

        Parameters
        ----------
        data_future : TimeSeriesDataFrame
            Actual values of the time series during the forecast horizon (``prediction_length`` values for each time
            series in the dataset). Must have the same index as ``predictions``.
        predictions : TimeSeriesDataFrame
            Data frame with predictions for the forecast horizon. Contain columns "mean" (point forecast) and the
            columns corresponding to each of the quantile levels. Must have the same index as ``data_future``.
        target : str, default = "target"
            Name of the column in ``data_future`` that contains the target time series.

        Returns
        -------
        score : float
            Value of the metric for given forecast and data. If self.greater_is_better_internal is True, returns score
            in greater-is-better format, otherwise in lower-is-better format.

        """
        raise NotImplementedError

    def save_past_metrics(
        self,
        data_past: TimeSeriesDataFrame,
        target: str = "target",
        seasonal_period: int = 1,
        **kwargs,
    ) -> None:
        """Compute auxiliary metrics on past data (before forecast horizon), if the chosen metric requires it.

        This method should only be implemented by metrics that rely on historical (in-sample) data, such as Mean Absolute
        Scaled Error (MASE) https://en.wikipedia.org/wiki/Mean_absolute_scaled_error.

        We keep this method separate from :meth:`compute_metric` to avoid redundant computations when fitting ensemble.
        """
        pass

    def clear_past_metrics(self) -> None:
        """Clear auxiliary metrics saved in :meth:`save_past_metrics`.

        This method should only be implemented if :meth:`save_past_metrics` has been implemented.
        """
        pass

    def error(self, *args, **kwargs):
        """Return error in lower-is-better format."""
        return self.optimum - self.score(*args, **kwargs)

    @staticmethod
    def _safemean(array: Union[np.ndarray, pd.Series]) -> float:
        """Compute mean of a numpy array-like object, ignoring inf, -inf and nan values."""
        return float(np.mean(array[np.isfinite(array)]))

    @staticmethod
    def _get_point_forecast_score_inputs(
        data_future: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame, target: str = "target"
    ) -> tuple[pd.Series, pd.Series]:
        """Get inputs necessary to compute point forecast metrics.

        Returns
        -------
        y_true
            Target time series values during the forecast horizon.
        y_pred
            Predicted time series values during the forecast horizon.
        """
        y_true = data_future[target]
        y_pred = predictions["mean"]
        return y_true, y_pred

    @staticmethod
    def _get_quantile_forecast_score_inputs(
        data_future: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame, target: str = "target"
    ) -> tuple[pd.Series, pd.DataFrame, np.ndarray]:
        """Get inputs necessary to compute quantile forecast metrics.

        Returns
        -------
        y_true
            Target time series values during the forecast horizon.
        q_pred
            Quantile forecast for each predicted quantile level. Column order corresponds to ``quantile_levels``.
        quantile_levels
            Quantile levels for which the forecasts are generated (as floats).
        """
        quantile_columns = [col for col in predictions.columns if col != "mean"]
        y_true = data_future[target]
        q_pred = pd.DataFrame(predictions[quantile_columns])
        quantile_levels = np.array(quantile_columns, dtype=float)
        return y_true, q_pred, quantile_levels

    @overload
    @staticmethod
    def check_get_horizon_weight(horizon_weight: None, prediction_length: int) -> None: ...
    @overload
    @staticmethod
    def check_get_horizon_weight(
        horizon_weight: Union[Sequence[float], np.ndarray], prediction_length: int
    ) -> npt.NDArray[np.float64]: ...

    @staticmethod
    def check_get_horizon_weight(
        horizon_weight: Union[Sequence[float], np.ndarray, None], prediction_length: int
    ) -> Optional[npt.NDArray[np.float64]]:
        """Convert horizon_weight to a non-negative numpy array that sums up to prediction_length.
        Raises an exception if horizon_weight has an invalid shape or contains invalid values.

        Returns
        -------
        horizon_weight
            None if the input is None, otherwise a numpy array of shape [1, prediction_length].
        """
        if horizon_weight is None:
            return None
        horizon_weight_np = np.ravel(horizon_weight).astype(np.float64)
        if horizon_weight_np.shape != (prediction_length,):
            raise ValueError(
                f"horizon_weight must have length equal to {prediction_length=} (got {len(horizon_weight)=})"
            )
        if not (horizon_weight_np >= 0).all():
            raise ValueError(f"All values in horizon_weight must be >= 0 (got {horizon_weight})")
        if not horizon_weight_np.sum() > 0:
            raise ValueError(f"At least some values in horizon_weight must be > 0 (got {horizon_weight})")
        if not np.isfinite(horizon_weight_np).all():
            raise ValueError(f"All horizon_weight values must be finite (got {horizon_weight})")
        horizon_weight_np = horizon_weight_np * prediction_length / horizon_weight_np.sum()
        return horizon_weight_np.reshape([1, prediction_length])

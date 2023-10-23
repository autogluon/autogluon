from typing import Optional

import numpy as np
import pandas as pd

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.utils.datetime import get_seasonality
from autogluon.timeseries.utils.warning_filters import warning_filter


class TimeSeriesScorer:
    """Base class for all evaluation metrics used in AutoGluon-TimeSeries.

    This object always returns the metric in greater-is-better format.

    Follows the design of ``autogluon.core.metrics.Scorer``.

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
    is_quantile_metric : bool, default = False
        Whether the given metric uses the quantile predictions. Some models will modify the training procedure if they
        are trained to optimize a quantile metric.
    equivalent_tabular_regression_metric : str
        Name of an equivalent metric used by AutoGluon-Tabular with ``problem_type="regression"``. Used by models that
        train a TabularPredictor under the hood. This attribute should only be specified by point forecast metrics.
    """

    greater_is_better_internal: bool = False
    optimum: float = 0.0
    optimized_by_median: bool = False
    is_quantile_metric: bool = False
    equivalent_tabular_regression_metric: str

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
        prediction_length: int = 1,
        target: str = "target",
        seasonal_period: Optional[int] = None,
        **kwargs,
    ) -> float:
        seasonal_period = get_seasonality(data.freq) if seasonal_period is None else seasonal_period

        data_past = data.slice_by_timestep(None, -prediction_length)
        data_future = data.slice_by_timestep(-prediction_length, None)

        assert (predictions.num_timesteps_per_item() == prediction_length).all()
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
            series in the dataset). This data frame is guaranteed to have the same index as ``predictions``.
        predictions : TimeSeriesDataFrame
            Data frame with predictions for the forecast horizon. Contain columns "mean" (point forecast) and the
            columns corresponding to each of the quantile levels.
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

        This method should only be implemented by metrics that rely on historic (in-sample) data, such as Mean Absolute
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
    def _safemean(data: pd.Series) -> float:
        return np.nanmean(data.replace([np.inf, -np.inf], np.nan).values)
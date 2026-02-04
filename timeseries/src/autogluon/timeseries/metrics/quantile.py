from typing import Sequence

import numpy as np
import pandas as pd

from autogluon.timeseries.dataset import TimeSeriesDataFrame

from .abstract import TimeSeriesScorer
from .utils import in_sample_abs_seasonal_error


class WQL(TimeSeriesScorer):
    r"""Weighted quantile loss.

    Also known as weighted pinball loss.

    Defined as total quantile loss divided by the sum of absolute time series values in the forecast horizon.

    .. math::

        \operatorname{WQL} = \frac{1}{\sum_{i=1}^{N} \sum_{t=T+1}^{T+H} |y_{i, t}|} \sum_{i=1}^{N} \sum_{t=T+1}^{T+H} \sum_{q}  \rho_q(y_{i,t}, f^q_{i,t})

    Properties:

    - scale-dependent (time series with large absolute value contribute more to the loss)
    - equivalent to WAPE if ``quantile_levels = [0.5]``

    If ``horizon_weight`` is provided, both the errors and the target time series in the denominator will be re-weighted.

    References
    ----------
    - `Forecasting: Principles and Practice <https://otexts.com/fpp3/distaccuracy.html#quantile-scores>`_
    """

    needs_quantile = True

    def compute_metric(
        self,
        data_future: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        **kwargs,
    ) -> float:
        y_true, q_pred, quantile_levels = self._get_quantile_forecast_score_inputs(data_future, predictions, target)
        y_true = y_true.to_numpy()[:, None]  # shape [N, 1]
        q_pred = q_pred.to_numpy()  # shape [N, len(quantile_levels)]

        errors = (
            np.abs((q_pred - y_true) * ((y_true <= q_pred) - quantile_levels))
            .mean(axis=1)
            .reshape([-1, self.prediction_length])
        )
        if self.horizon_weight is not None:
            errors *= self.horizon_weight
            y_true = y_true.reshape([-1, self.prediction_length]) * self.horizon_weight
        return 2 * np.nansum(errors) / np.nansum(np.abs(y_true))


class SQL(TimeSeriesScorer):
    r"""Scaled quantile loss.

    Also known as scaled pinball loss.

    Normalizes the quantile loss for each time series by the historical seasonal error of this time series.

    .. math::

        \operatorname{SQL} = \frac{1}{N} \frac{1}{H} \sum_{i=1}^{N} \frac{1}{a_i} \sum_{t=T+1}^{T+H} \sum_{q}  \rho_q(y_{i,t}, f^q_{i,t})

    where :math:`a_i` is the historical absolute seasonal error defined as

    .. math::

        a_i = \frac{1}{T-m} \sum_{t=m+1}^T |y_{i,t} - y_{i,t-m}|

    and :math:`m` is the seasonal period of the time series (``eval_metric_seasonal_period``).


    Properties:

    - scaled metric (normalizes the error for each time series by the scale of that time series)
    - undefined for constant time series
    - equivalent to MASE if ``quantile_levels = [0.5]``

    References
    ----------
    - `Forecasting: Principles and Practice <https://otexts.com/fpp3/distaccuracy.html#quantile-scores>`_
    """

    needs_quantile = True

    def __init__(
        self,
        prediction_length: int = 1,
        seasonal_period: int | None = None,
        horizon_weight: Sequence[float] | None = None,
    ):
        super().__init__(
            prediction_length=prediction_length, seasonal_period=seasonal_period, horizon_weight=horizon_weight
        )
        self._past_abs_seasonal_error: pd.Series | None = None

    def save_past_metrics(
        self, data_past: TimeSeriesDataFrame, target: str = "target", seasonal_period: int = 1, **kwargs
    ) -> None:
        self._past_abs_seasonal_error = in_sample_abs_seasonal_error(
            y_past=data_past[target], seasonal_period=seasonal_period
        )

    def clear_past_metrics(self) -> None:
        self._past_abs_seasonal_error = None

    def compute_metric(
        self,
        data_future: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        target: str = "target",
        **kwargs,
    ) -> float:
        if self._past_abs_seasonal_error is None:
            raise AssertionError("Call `save_past_metrics` before `compute_metric`")

        y_true, q_pred, quantile_levels = self._get_quantile_forecast_score_inputs(data_future, predictions, target)
        q_pred = q_pred.to_numpy()
        y_true = y_true.to_numpy()[:, None]  # shape [N, 1]

        errors = (
            np.abs((q_pred - y_true) * ((y_true <= q_pred) - quantile_levels))
            .mean(axis=1)
            .reshape([-1, self.prediction_length])
        )
        if self.horizon_weight is not None:
            errors *= self.horizon_weight
        return 2 * self._safemean(errors / self._past_abs_seasonal_error.to_numpy()[:, None])

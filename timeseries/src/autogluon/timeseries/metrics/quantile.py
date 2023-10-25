import numpy as np

from autogluon.timeseries import TimeSeriesDataFrame

from .abstract import TimeSeriesScorer


class WQL(TimeSeriesScorer):
    """Weighted quantile loss.

    Also known as weighted pinball loss.
    """

    needs_quantile = True

    def compute_metric(
        self, data_future: TimeSeriesDataFrame, predictions: TimeSeriesDataFrame, target: str = "target", **kwargs
    ) -> float:
        y_true, q_pred, quantile_levels = self._get_quantile_forecast_score_inputs(data_future, predictions, target)
        values_true = y_true.values[:, None]  # shape [N, 1]
        values_pred = q_pred.values  # shape [N, len(quantile_levels)]

        return 2 * np.mean(
            np.abs((values_true - values_pred) * ((values_true <= values_pred) - quantile_levels)).sum(axis=0)
            / np.abs(values_true).sum()
        )

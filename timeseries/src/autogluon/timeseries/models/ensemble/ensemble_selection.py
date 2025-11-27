import copy
from typing import Optional

import numpy as np

import autogluon.core as ag
from autogluon.core.models.greedy_ensemble.ensemble_selection import EnsembleSelection
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.metrics import TimeSeriesScorer
from autogluon.timeseries.utils.datetime import get_seasonality


class TimeSeriesEnsembleSelection(EnsembleSelection):
    def __init__(
        self,
        ensemble_size: int,
        metric: TimeSeriesScorer,
        problem_type: str = ag.constants.QUANTILE,
        sorted_initialization: bool = False,
        bagging: bool = False,
        tie_breaker: str = "random",
        random_state: Optional[np.random.RandomState] = None,
        prediction_length: int = 1,
        target: str = "target",
        **kwargs,
    ):
        super().__init__(
            ensemble_size=ensemble_size,
            metric=metric,  # type: ignore
            problem_type=problem_type,
            sorted_initialization=sorted_initialization,
            bagging=bagging,
            tie_breaker=tie_breaker,
            random_state=random_state,
            **kwargs,
        )
        self.prediction_length = prediction_length
        self.target = target
        self.metric: TimeSeriesScorer

        self.dummy_pred_per_window = []
        self.scorer_per_window = []

        self.dummy_pred_per_window: Optional[list[TimeSeriesDataFrame]]
        self.scorer_per_window: Optional[list[TimeSeriesScorer]]
        self.data_future_per_window: Optional[list[TimeSeriesDataFrame]]

    def fit(  # type: ignore
        self,
        predictions: list[list[TimeSeriesDataFrame]],
        labels: list[TimeSeriesDataFrame],
        time_limit: Optional[float] = None,
    ):
        return super().fit(
            predictions=predictions,  # type: ignore
            labels=labels,  # type: ignore
            time_limit=time_limit,
        )

    def _fit(  # type: ignore
        self,
        predictions: list[list[TimeSeriesDataFrame]],
        labels: list[TimeSeriesDataFrame],
        time_limit: Optional[float] = None,
        sample_weight: Optional[list[float]] = None,
    ):
        # Stack predictions for each model into a 3d tensor of shape [num_val_windows, num_rows, num_cols]
        stacked_predictions = [np.stack(preds) for preds in predictions]

        self.dummy_pred_per_window = []
        self.scorer_per_window = []
        self.data_future_per_window = []

        seasonal_period = self.metric.seasonal_period
        if seasonal_period is None:
            seasonal_period = get_seasonality(labels[0].freq)

        for window_idx, data in enumerate(labels):
            dummy_pred = copy.deepcopy(predictions[0][window_idx])
            # This should never happen; sanity check to make sure that all predictions have the same index
            assert all(dummy_pred.index.equals(pred[window_idx].index) for pred in predictions)
            assert all(dummy_pred.columns.equals(pred[window_idx].columns) for pred in predictions)

            self.dummy_pred_per_window.append(dummy_pred)

            scorer = copy.deepcopy(self.metric)
            # Split the observed time series once to avoid repeated computations inside the evaluator
            data_past = data.slice_by_timestep(None, -self.prediction_length)
            data_future = data.slice_by_timestep(-self.prediction_length, None)
            scorer.save_past_metrics(data_past, target=self.target, seasonal_period=seasonal_period)
            self.scorer_per_window.append(scorer)
            self.data_future_per_window.append(data_future)

        super()._fit(
            predictions=stacked_predictions,
            labels=data_future,  # type: ignore
            time_limit=time_limit,
        )
        self.dummy_pred_per_window = None
        self.evaluator_per_window = None
        self.data_future_per_window = None

    def _calculate_regret(  # type: ignore
        self,
        y_true,
        y_pred_proba,
        metric: TimeSeriesScorer,
        sample_weight=None,
    ):
        # Compute average score across all validation windows
        total_score = 0.0

        assert self.data_future_per_window is not None
        assert self.dummy_pred_per_window is not None
        assert self.scorer_per_window is not None

        for window_idx, data_future in enumerate(self.data_future_per_window):
            dummy_pred = self.dummy_pred_per_window[window_idx]
            dummy_pred[list(dummy_pred.columns)] = y_pred_proba[window_idx]
            # We use scorer.compute_metric instead of scorer.score to avoid repeated calls to scorer.save_past_metrics
            metric_value = self.scorer_per_window[window_idx].compute_metric(
                data_future,
                dummy_pred,
                target=self.target,
            )
            total_score += metric.sign * metric_value
        avg_score = total_score / len(self.data_future_per_window)
        # score: higher is better, regret: lower is better, so we flip the sign
        return -avg_score


def fit_time_series_ensemble_selection(
    data_per_window: list[TimeSeriesDataFrame],
    predictions_per_window: dict[str, list[TimeSeriesDataFrame]],
    ensemble_size: int,
    eval_metric: TimeSeriesScorer,
    prediction_length: int = 1,
    target: str = "target",
    time_limit: Optional[float] = None,
) -> dict[str, float]:
    """Fit ensemble selection for time series forecasting and return ensemble weights.

    Parameters
    ----------
    data_per_window:
        List of ground truth time series data for each validation window.
    predictions_per_window:
        Dictionary mapping model names to their predictions for each validation window.
    ensemble_size:
        Number of iterations of the ensemble selection algorithm.

    Returns
    -------
    weights:
        Dictionary mapping the model name to its weight in the ensemble.
    """
    ensemble_selection = TimeSeriesEnsembleSelection(
        ensemble_size=ensemble_size,
        metric=eval_metric,
        prediction_length=prediction_length,
        target=target,
    )
    ensemble_selection.fit(
        predictions=list(predictions_per_window.values()),
        labels=data_per_window,
        time_limit=time_limit,
    )
    return {model: float(weight) for model, weight in zip(predictions_per_window.keys(), ensemble_selection.weights_)}

import copy
import logging
import pprint
from typing import Dict, List, Optional

import numpy as np

import autogluon.core as ag
from autogluon.core.models.greedy_ensemble.ensemble_selection import EnsembleSelection
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.metrics import TimeSeriesScorer
from autogluon.timeseries.utils.datetime import get_seasonality

from .abstract import AbstractWeightedTimeSeriesEnsembleModel

logger = logging.getLogger(__name__)


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

        self.dummy_pred_per_window: Optional[List[TimeSeriesDataFrame]]
        self.scorer_per_window: Optional[List[TimeSeriesScorer]]
        self.data_future_per_window: Optional[List[TimeSeriesDataFrame]]

    def fit(  # type: ignore
        self,
        predictions: List[List[TimeSeriesDataFrame]],
        labels: List[TimeSeriesDataFrame],
        time_limit: Optional[float] = None,
    ):
        return super().fit(
            predictions=predictions,  # type: ignore
            labels=labels,  # type: ignore
            time_limit=time_limit,
        )

    def _fit(  # type: ignore
        self,
        predictions: List[List[TimeSeriesDataFrame]],
        labels: List[TimeSeriesDataFrame],
        time_limit: Optional[float] = None,
        sample_weight: Optional[List[float]] = None,
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


class GreedyEnsemble(AbstractWeightedTimeSeriesEnsembleModel):
    """Constructs a weighted ensemble using the greedy Ensemble Selection algorithm by
    Caruana et al. [Car2004]

    Other Parameters
    ----------------
    ensemble_size: int, default = 100
        Number of models (with replacement) to include in the ensemble.

    References
    ----------
    .. [Car2024] Caruana, Rich, et al. "Ensemble selection from libraries of models."
        Proceedings of the twenty-first international conference on Machine learning. 2004.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        if name is None:
            # FIXME: the name here is kept for backward compatibility. it will be called
            # GreedyEnsemble in v1.4 once ensemble choices are exposed
            name = "WeightedEnsemble"
        super().__init__(name=name, **kwargs)

    def _get_default_hyperparameters(self) -> Dict:
        return {"ensemble_size": 100}

    def _fit(
        self,
        predictions_per_window: Dict[str, List[TimeSeriesDataFrame]],
        data_per_window: List[TimeSeriesDataFrame],
        model_scores: Optional[Dict[str, float]] = None,
        time_limit: Optional[float] = None,
    ):
        ensemble_selection = TimeSeriesEnsembleSelection(
            ensemble_size=self.get_hyperparameters()["ensemble_size"],
            metric=self.eval_metric,
            prediction_length=self.prediction_length,
            target=self.target,
        )
        ensemble_selection.fit(
            predictions=list(predictions_per_window.values()),
            labels=data_per_window,
            time_limit=time_limit,
        )
        self.model_to_weight = {}
        for model_name, weight in zip(predictions_per_window.keys(), ensemble_selection.weights_):
            if weight != 0:
                self.model_to_weight[model_name] = weight

        weights_for_printing = {model: round(float(weight), 2) for model, weight in self.model_to_weight.items()}
        logger.info(f"\tEnsemble weights: {pprint.pformat(weights_for_printing, width=200)}")

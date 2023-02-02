import copy
import logging
from typing import Dict, List, Optional

import numpy as np

import autogluon.core as ag
from autogluon.core.models.greedy_ensemble.ensemble_selection import EnsembleSelection
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.evaluator import TimeSeriesEvaluator
from autogluon.timeseries.models.ensemble import AbstractTimeSeriesEnsembleModel

logger = logging.getLogger(__name__)


class TimeSeriesEnsembleSelection(EnsembleSelection):
    def __init__(
        self,
        ensemble_size: int,
        metric: TimeSeriesEvaluator,
        problem_type: str = ag.constants.QUANTILE,
        sorted_initialization: bool = False,
        bagging: bool = False,
        tie_breaker: str = "random",
        random_state: np.random.RandomState = None,
        **kwargs,
    ):
        super().__init__(
            ensemble_size=ensemble_size,
            metric=metric,
            problem_type=problem_type,
            sorted_initialization=sorted_initialization,
            bagging=bagging,
            tie_breaker=tie_breaker,
            random_state=random_state,
            **kwargs,
        )

    def _fit(
        self,
        predictions: List[TimeSeriesDataFrame],
        labels: TimeSeriesDataFrame,
        time_limit: Optional[int] = None,
        sample_weight=None,
    ):
        self.dummy_pred = copy.deepcopy(predictions[0])
        # This should never happen; sanity check to make sure that all predictions have the same index
        assert all(self.dummy_pred.index.equals(pred.index) for pred in predictions)
        # Split the observed time series once to avoid repeated computations inside the evaluator
        data_past = labels.slice_by_timestep(None, -self.metric.prediction_length)
        data_future = labels.slice_by_timestep(-self.metric.prediction_length, None)
        self.metric.save_past_metrics(data_past)
        super()._fit(
            predictions=[d.values for d in predictions],
            labels=data_future,
            time_limit=time_limit,
        )
        self.dummy_pred = None

    def _calculate_regret(self, y_true, y_pred_proba, metric, dummy_pred=None, sample_weight=None):  # noqa
        dummy_pred = copy.deepcopy(self.dummy_pred if dummy_pred is None else dummy_pred)
        dummy_pred[list(dummy_pred.columns)] = y_pred_proba
        score = metric.score_with_saved_past_metrics(y_true, dummy_pred) * metric.coefficient
        # score: higher is better, regret: lower is better, so we flip the sign
        return -score


class TimeSeriesGreedyEnsemble(AbstractTimeSeriesEnsembleModel):
    """Constructs a weighted ensemble using the greedy Ensemble Selection algorithm."""

    def __init__(self, name: str, ensemble_size: int = 100, **kwargs):
        super().__init__(name=name, **kwargs)
        self.ensemble_size = ensemble_size
        self.model_to_weight: Dict[str, float] = {}

    def _fit_ensemble(
        self,
        predictions: Dict[str, TimeSeriesDataFrame],
        data: TimeSeriesDataFrame,
        time_limit: Optional[int] = None,
        **kwargs,
    ):
        evaluator = TimeSeriesEvaluator(
            eval_metric=self.eval_metric,
            prediction_length=self.prediction_length,
            target_column=self.target,
        )
        ensemble_selection = TimeSeriesEnsembleSelection(ensemble_size=self.ensemble_size, metric=evaluator)
        ensemble_selection.fit(
            predictions=list(predictions.values()),
            labels=data,
            time_limit=time_limit,
        )
        self.model_to_weight = {}
        for model_name, weight in zip(predictions.keys(), ensemble_selection.weights_):
            if weight != 0:
                self.model_to_weight[model_name] = weight

    @property
    def model_names(self) -> List[str]:
        return list(self.model_to_weight.keys())

    @property
    def model_weights(self) -> np.ndarray:
        return np.array(list(self.model_to_weight.values()), dtype=np.float64)

    def predict(self, data: Dict[str, TimeSeriesDataFrame], **kwargs) -> TimeSeriesDataFrame:
        if set(data.keys()) != set(self.model_names):
            raise ValueError(
                f"Set of models given for prediction in {self.name} differ from those provided during initialization."
            )
        failed_models = [model_name for (model_name, model_pred) in data.items() if model_pred is None]
        if len(failed_models) == len(data):
            raise RuntimeError(f"All input models failed during prediction, {self.name} cannot predict.")
        if len(failed_models) > 0:
            logger.warning(
                f"Following models failed during prediction: {failed_models}. "
                f"{self.name} will set the weight of these models to zero and re-normalize the weights when predicting."
            )

        # Make sure that all predictions have same shape
        assert len(set(pred.shape for pred in data.values() if pred is not None)) == 1

        model_preds = [data[model_name] for model_name in self.model_names]
        weights = self.model_weights

        for idx, pred in enumerate(model_preds):
            if pred is None:
                weights[idx] = 0
        weights = weights / np.sum(weights)

        return sum(pred * w for pred, w in zip(model_preds, weights) if pred is not None)

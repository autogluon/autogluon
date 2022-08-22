import copy
import logging
from typing import Dict, List

import numpy as np

import autogluon.core as ag
from autogluon.core.models.greedy_ensemble.ensemble_selection import AbstractWeightedEnsemble, EnsembleSelection
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.evaluator import TimeSeriesEvaluator
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel

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

    def _fit(self, predictions, labels, time_limit=None, sample_weight=None):
        self.dummy_pred = copy.deepcopy(predictions[0])
        super()._fit(
            predictions=[d.values for d in predictions],
            labels=labels,
            time_limit=time_limit,
            sample_weight=sample_weight,
        )
        self.dummy_pred = None

    def _calculate_regret(self, y_true, y_pred_proba, metric, dummy_pred=None, sample_weight=None):  # noqa
        dummy_pred = copy.deepcopy(self.dummy_pred if dummy_pred is None else dummy_pred)
        dummy_pred[list(dummy_pred.columns)] = y_pred_proba
        return metric(y_true, dummy_pred) * metric.coefficient


class SimpleTimeSeriesWeightedEnsemble(AbstractWeightedEnsemble):
    """Predefined user-weights ensemble"""

    def __init__(self, weights: List[float], **kwargs):
        self.weights_ = weights
        self.problem_type = ag.constants.QUANTILE

    @property
    def ensemble_size(self):
        return len(self.weights_)

    def weight_pred_probas(
        self,
        preds: List[TimeSeriesDataFrame],
        weights: List[float],
    ) -> TimeSeriesDataFrame:
        assert len(set(v.shape for v in preds)) == 1

        # TODO: handle NaNs
        return sum(p * w for p, w in zip(preds, weights))


class TimeSeriesEnsembleWrapper(AbstractTimeSeriesModel):
    """AbstractTimeSeriesModel wrapper for simple weighted ensemble selection
    models.

    Parameters
    ----------
    weights : Dict[str, float]
        Dictionary mapping model names to weights in the ensemble
    name : str
        Name of the model. Usually like ``"WeightedEnsemble"``.
    """

    def __init__(self, weights: Dict[str, float], name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.model_names, model_weights = zip(*weights.items())
        self.weighted_ensemble = SimpleTimeSeriesWeightedEnsemble(weights=model_weights)

    def _fit(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @property
    def weights(self):
        return self.weighted_ensemble.weights_

    def predict(self, data: Dict[str, TimeSeriesDataFrame], **kwargs):
        if set(data.keys()) != set(self.model_names):
            raise ValueError(
                "Set of models given for prediction in the weighted ensemble differ from those "
                "provided during initialization."
            )

        return self.weighted_ensemble.predict([data[k] for k in self.model_names])

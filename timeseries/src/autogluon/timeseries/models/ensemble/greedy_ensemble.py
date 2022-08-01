import copy
import logging
from typing import List

import numpy as np

import autogluon.core as ag
from autogluon.core.models.greedy_ensemble.ensemble_selection import (
    AbstractWeightedEnsemble,
    EnsembleSelection,
)
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel

logger = logging.getLogger(__name__)


class TimeSeriesEnsembleSelection(EnsembleSelection):
    def __init__(
        self,
        ensemble_size: int,
        metric,
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

    def _calculate_regret(self, y_true, y_pred_proba, metric, dummy_pred=None, sample_weight=None):  # noqa
        dummy_pred = copy.deepcopy(self.dummy_pred if dummy_pred is None else dummy_pred)
        dummy_pred[list(dummy_pred.columns)] = y_pred_proba
        sign = 1 if metric.higher_is_better else -1
        return sign * metric(y_true, dummy_pred)


class SimpleTimeSeriesWeightedEnsemble(AbstractWeightedEnsemble):
    """Predefined user-weights ensemble"""

    def __init__(self, weights, **kwargs):
        self.weights_ = weights

    @property
    def ensemble_size(self):
        return len(self.weights_)

    def predict(self, X):
        return self.predict_proba(X)

    def predict_proba(self, X):
        return self.weight_pred_probas(X, weights=self.weights_)

    def weight_pred_probas(
        self,
        preds: List[TimeSeriesDataFrame],
        weights: List[float],
    ) -> TimeSeriesDataFrame:
        assert len(set(v.shape for v in preds)) == 1

        # TODO: handle NaNs
        return sum(p * w for p, w in zip(preds, weights))


class TimeSeriesEnsembleWrapper(AbstractTimeSeriesModel):
    """Predefined user-weights ensemble"""

    def __init__(self, weights, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.model = SimpleTimeSeriesWeightedEnsemble(weights=weights)
        self.name = name
        self.fit_time = None
        self.val_score = None

    def _fit(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @property
    def weights(self):
        return self.model.weights_

    def predict(self, data: List[TimeSeriesDataFrame], **kwargs):
        if isinstance(data, dict):  # FIXME: HACK, unclear what the input to predict should be for weighted ensemble
            data = [data[m] for m in data.keys()]
        return self.model.predict(data)

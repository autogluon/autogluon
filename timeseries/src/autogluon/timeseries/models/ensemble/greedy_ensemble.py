import copy
import logging
import time
from typing import List

import numpy as np

from autogluon.core.models.greedy_ensemble.ensemble_selection import (
    AbstractWeightedEnsemble,
    EnsembleSelection,
)

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel

logger = logging.getLogger(__name__)


class TimeSeriesEnsembleSelection(EnsembleSelection):
    def __init__(  # noqa
        self,
        ensemble_size: int,
        problem_type: str,
        metric,
        higher_is_better=False,
        sorted_initialization: bool = False,
        bagging: bool = False,
        tie_breaker: str = "random",
        random_state: np.random.RandomState = None,
        **kwargs,
    ):
        # TODO: call super init?
        self.ensemble_size = ensemble_size
        self.problem_type = problem_type
        self.metric = metric
        self.higher_is_better = higher_is_better
        self.sorted_initialization = sorted_initialization
        self.bagging = bagging
        self.use_best = True
        if tie_breaker not in ["random", "second_metric"]:
            raise ValueError(
                f"Unknown tie_breaker value: {tie_breaker}. Must be one of: ['random', 'second_metric']"
            )
        self.tie_breaker = tie_breaker
        if random_state is not None:
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(seed=0)
        self.quantile_levels = kwargs.get("quantile_levels", None)

    def fit(
        self, predictions, labels, time_limit=None, identifiers=None, sample_weight=None
    ):
        self.ensemble_size = int(self.ensemble_size)
        if self.ensemble_size < 1:
            raise ValueError("Ensemble size cannot be less than one!")
        self._fit(
            predictions=predictions,
            labels=labels,
            time_limit=time_limit,
            sample_weight=sample_weight,
        )
        self._calculate_weights()
        logger.log(10, "Ensemble weights: ")
        logger.log(10, self.weights_)
        return self

    # TODO: Consider having a removal stage, remove each model and see if
    #  score is affected, if improves or not effected, remove it.
    def _fit(self, predictions, labels, time_limit=None, sample_weight=None):
        ensemble_size = self.ensemble_size
        self.num_input_models_ = len(predictions)
        ensemble = []
        trajectory = []
        order = []
        used_models = set()

        time_start = time.time()
        round_scores = False
        epsilon = 1e-4
        round_decimals = 6
        dummy_pred = copy.deepcopy(predictions[0])
        for i in range(ensemble_size):
            scores = np.zeros((len(predictions)))
            s = len(ensemble)
            if s == 0:
                weighted_ensemble_prediction = np.zeros(predictions[0].shape)
            else:
                # Memory-efficient averaging!
                ensemble_prediction = np.zeros(ensemble[0].shape)
                for pred in ensemble:
                    ensemble_prediction += pred.values
                ensemble_prediction /= s

                weighted_ensemble_prediction = (s / float(s + 1)) * ensemble_prediction
            fant_ensemble_prediction = np.zeros(weighted_ensemble_prediction.shape)
            for j, pred in enumerate(predictions):
                fant_ensemble_prediction[:] = (
                    weighted_ensemble_prediction + (1.0 / float(s + 1)) * pred.values
                )
                scores[j] = self._calculate_regret(
                    y_true=labels,
                    y_pred_proba=fant_ensemble_prediction,
                    metric=self.metric,
                    sample_weight=sample_weight,
                    dummy_pred=dummy_pred,
                )
                if round_scores:
                    scores[j] = scores[j].round(round_decimals)

            all_best = np.argwhere(scores == np.nanmin(scores)).flatten()

            if (len(all_best) > 1) and used_models:
                # If tie, prioritize models already in ensemble to avoid unnecessarily large ensemble
                new_all_best = []
                for m in all_best:
                    if m in used_models:
                        new_all_best.append(m)
                if new_all_best:
                    all_best = new_all_best

            best = self.random_state.choice(all_best)
            best_score = scores[best]

            # If first iteration
            if i == 0:
                # If abs value of min score is large enough, round to 6 decimal places to avoid
                # floating point error deciding the best index.  This avoids 2 models with the same pred
                # proba both being used in the ensemble due to floating point error
                if np.abs(best_score) > epsilon:
                    round_scores = True
                    best_score = best_score.round(round_decimals)

            ensemble.append(predictions[best])
            trajectory.append(best_score)
            order.append(best)
            used_models.add(best)

            # Handle special case
            if len(predictions) == 1:
                break

            if time_limit is not None:
                time_elapsed = time.time() - time_start
                time_left = time_limit - time_elapsed
                if time_left <= 0:
                    logger.warning(
                        f"Warning: Ensemble Selection ran out of time, early stopping at iteration {i+1}. This "
                        f"may mean that the time_limit specified is very small for this problem."
                    )
                    break

        min_score = np.min(trajectory)
        first_index_of_best = trajectory.index(min_score)

        if self.use_best:
            self.indices_ = order[: first_index_of_best + 1]
            self.trajectory_ = trajectory[: first_index_of_best + 1]
            self.train_score_ = trajectory[first_index_of_best]
            self.ensemble_size = first_index_of_best + 1
            logger.log(15, "Ensemble size: %s" % self.ensemble_size)
        else:
            self.indices_ = order
            self.trajectory_ = trajectory
            self.train_score_ = trajectory[-1]

        logger.debug("Ensemble indices: " + str(self.indices_))

    def _calculate_regret(  # noqa
        self, y_true, y_pred_proba, metric, dummy_pred, sample_weight=None
    ):
        dummy_pred = copy.deepcopy(dummy_pred)
        dummy_pred[list(dummy_pred.columns)] = y_pred_proba
        score = metric(y_true, dummy_pred)
        if self.higher_is_better:
            score = -score
        return score


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
        # TODO: this is a hack. indices may not match, which should be checked or better,
        # TODO: weighted accordingly
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

    def _fit(
        self,
        train_data,
        val_data=None,
        time_limit=None,
        num_cpus=None,
        num_gpus=None,
        verbosity=2,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    @property
    def weights(self):
        return self.model.weights_

    def predict(self, data: List[TimeSeriesDataFrame], **kwargs):
        if isinstance(
            data, dict
        ):  # FIXME: HACK, unclear what the input to predict should be for weighted ensemble
            data = [data[m] for m in data.keys()]
        return self.model.predict(data)

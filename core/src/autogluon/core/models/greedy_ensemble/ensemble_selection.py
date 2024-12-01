from __future__ import annotations

import logging
import time
from collections import Counter
from typing import List

import numpy as np
import pandas as pd

from ...constants import PROBLEM_TYPES
from ...metrics import compute_metric, log_loss, Scorer
from ...utils import get_pred_from_proba

logger = logging.getLogger(__name__)


class AbstractWeightedEnsemble:
    def predict(self, X):
        y_pred_proba = self.predict_proba(X)
        return get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)

    def predict_proba(self, X):
        return self.weight_pred_probas(X, weights=self.weights_)

    @staticmethod
    def weight_pred_probas(pred_probas, weights):
        preds_norm = [pred * weight for pred, weight in zip(pred_probas, weights)]
        preds_ensemble = np.sum(preds_norm, axis=0)
        return preds_ensemble


class EnsembleSelection(AbstractWeightedEnsemble):
    def __init__(
        self,
        ensemble_size: int,
        problem_type: str,
        metric: Scorer,
        sorted_initialization: bool = False,
        bagging: bool = False,
        tie_breaker: str = "random",
        subsample_size: int | None = None,
        random_state: np.random.RandomState = None,
        **kwargs,
    ):
        self.ensemble_size = ensemble_size
        self.problem_type = problem_type
        self.metric = metric
        self.sorted_initialization = sorted_initialization
        self.bagging = bagging
        self.use_best = True
        if tie_breaker not in ["random", "second_metric"]:
            raise ValueError(f"Unknown tie_breaker value: {tie_breaker}. Must be one of: ['random', 'second_metric']")
        self.tie_breaker = tie_breaker
        self.subsample_size = subsample_size
        if random_state is not None:
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(seed=0)
        self.quantile_levels = kwargs.get("quantile_levels", None)

    def fit(self, predictions: List[np.ndarray], labels: np.ndarray, time_limit=None, identifiers=None, sample_weight=None):
        self.ensemble_size = int(self.ensemble_size)
        if self.ensemble_size < 1:
            raise ValueError("Ensemble size cannot be less than one!")
        if not self.problem_type in PROBLEM_TYPES:
            raise ValueError("Unknown problem type %s." % self.problem_type)
        # if not isinstance(self.metric, Scorer):
        #     raise ValueError('Metric must be of type scorer')

        self._fit(predictions=predictions, labels=labels, time_limit=time_limit, sample_weight=sample_weight)
        self._calculate_weights()
        logger.log(15, "Ensemble weights: ")
        logger.log(15, self.weights_)
        return self

    # TODO: Consider having a removal stage, remove each model and see if score is affected, if improves or not effected, remove it.
    def _fit(self, predictions: List[np.ndarray], labels: np.ndarray, time_limit=None, sample_weight=None):
        ensemble_size = self.ensemble_size
        if isinstance(labels, pd.Series):
            labels = labels.values
        self.num_input_models_ = len(predictions)
        ensemble = []
        trajectory = []
        order = []
        used_models = set()
        num_samples_total = len(labels)

        if self.subsample_size is not None and self.subsample_size < num_samples_total:
            logger.log(15, f"Subsampling to {self.subsample_size} samples to speedup ensemble selection...")
            subsample_indices = self.random_state.choice(num_samples_total, self.subsample_size, replace=False)
            labels = labels[subsample_indices]
            for i in range(self.num_input_models_):
                predictions[i] = predictions[i][subsample_indices]

        # if self.sorted_initialization:
        #     n_best = 20
        #     indices = self._sorted_initialization(predictions, labels, n_best)
        #     for idx in indices:
        #         ensemble.append(predictions[idx])
        #         order.append(idx)
        #         ensemble_ = np.array(ensemble).mean(axis=0)
        #         ensemble_performance = calculate_score(
        #             labels, ensemble_, self.task_type, self.metric,
        #             ensemble_.shape[1])
        #         trajectory.append(ensemble_performance)
        #     ensemble_size -= n_best

        time_start = time.time()
        round_scores = False
        epsilon = 1e-4
        round_decimals = 6
        ensemble_prediction = np.zeros(predictions[0].shape)
        weighted_ensemble_prediction = np.zeros(predictions[0].shape)
        fant_ensemble_prediction = np.zeros(predictions[0].shape)
        for i in range(ensemble_size):
            scores = np.zeros((len(predictions)))
            s = len(ensemble)

            if s != 0:
                ensemble_prediction *= (s - 1) / s
                ensemble_prediction += ensemble[-1] / s
                weighted_ensemble_prediction[:] = (s / float(s + 1)) * ensemble_prediction
            for j, pred in enumerate(predictions):
                fant_ensemble_prediction[:] = weighted_ensemble_prediction + (1.0 / float(s + 1)) * pred
                if self.problem_type in ["multiclass", "softclass"]:
                    # Renormalize
                    fant_ensemble_prediction /= fant_ensemble_prediction.sum(axis=1)[:, np.newaxis]
                scores[j] = self._calculate_regret(y_true=labels, y_pred_proba=fant_ensemble_prediction, metric=self.metric, sample_weight=sample_weight)
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

            if len(all_best) > 1:
                if self.tie_breaker == "second_metric":
                    if self.problem_type in ["binary", "multiclass"]:
                        # Tiebreak with log_loss
                        scores_tiebreak = np.zeros((len(all_best)))
                        secondary_metric = log_loss
                        fant_ensemble_prediction = np.zeros(weighted_ensemble_prediction.shape)
                        index_map = {}
                        for k, j in enumerate(all_best):
                            index_map[k] = j
                            pred = predictions[j]
                            fant_ensemble_prediction[:] = weighted_ensemble_prediction + (1.0 / float(s + 1)) * pred
                            scores_tiebreak[k] = self._calculate_regret(y_true=labels, y_pred_proba=fant_ensemble_prediction, metric=secondary_metric)
                        all_best_tiebreak = np.argwhere(scores_tiebreak == np.nanmin(scores_tiebreak)).flatten()
                        all_best = [index_map[index] for index in all_best_tiebreak]

            best = self.random_state.choice(all_best)
            best_score = scores[best]

            # If first iteration
            if i == 0:
                # If abs value of min score is large enough, round to 6 decimal places to avoid floating point error deciding the best index.
                # This avoids 2 models with the same pred proba both being used in the ensemble due to floating point error
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
                        "Warning: Ensemble Selection ran out of time, early stopping at iteration %s. This may mean that the time_limit specified is very small for this problem."
                        % (i + 1)
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

    def _calculate_regret(self, y_true: np.ndarray, y_pred_proba: np.ndarray, metric: Scorer, sample_weight: np.ndarray = None) -> float:
        if metric.needs_pred or metric.needs_quantile:
            y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)
            y_pred_proba = None
        else:
            y_pred = None
        regret = compute_metric(
            y=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            metric=metric,
            weights=sample_weight,
            quantile_levels=self.quantile_levels,
            as_error=True,
        )
        return regret

    def _calculate_weights(self):
        ensemble_members = Counter(self.indices_).most_common()
        weights = np.zeros((self.num_input_models_,), dtype=float)
        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / self.ensemble_size
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        self.weights_ = weights


class SimpleWeightedEnsemble(AbstractWeightedEnsemble):
    """Predefined user-weights ensemble"""

    def __init__(self, weights, problem_type, **kwargs):
        self.weights_ = weights
        self.problem_type = problem_type

    @property
    def ensemble_size(self):
        return len(self.weights_)

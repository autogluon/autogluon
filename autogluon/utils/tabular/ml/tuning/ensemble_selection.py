import logging, time
import numpy as np
from collections import Counter

from ..constants import PROBLEM_TYPES
from ...metrics import calculate_score, _ProbaScorer, _ThresholdScorer
from ..utils import get_pred_from_proba

logger = logging.getLogger(__name__)


class EnsembleSelection:
    def __init__(
            self,
            ensemble_size: int,
            problem_type: str,
            metric,
            sorted_initialization: bool = False,
            bagging: bool = False,
            random_state: np.random.RandomState = None,
    ):
        self.ensemble_size = ensemble_size
        self.problem_type = problem_type
        self.metric = metric
        self.sorted_initialization = sorted_initialization
        self.bagging = bagging
        self.use_best = True
        if random_state is not None:
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(seed=0)
        if isinstance(metric, _ProbaScorer):
            self.eval_metric_expects_y_pred = False
        elif isinstance(metric, _ThresholdScorer):
            self.eval_metric_expects_y_pred = False
        else:
            self.eval_metric_expects_y_pred = True

    def fit(self, predictions, labels, time_limit=None, identifiers=None):
        self.ensemble_size = int(self.ensemble_size)
        if self.ensemble_size < 1:
            raise ValueError('Ensemble size cannot be less than one!')
        if not self.problem_type in PROBLEM_TYPES:
            raise ValueError('Unknown problem type %s.' % self.problem_type)
        # if not isinstance(self.metric, Scorer):
        #     raise ValueError('Metric must be of type scorer')

        self._fit(predictions=predictions, labels=labels, time_limit=time_limit)
        self._calculate_weights()
        logger.log(15, 'Ensemble weights: ')
        logger.log(15, self.weights_)
        return self

    # TODO: Consider having a removal stage, remove each model and see if score is affected, if improves or not effected, remove it.
    def _fit(self, predictions, labels, time_limit=None):
        ensemble_size = self.ensemble_size
        self.num_input_models_ = len(predictions)
        ensemble = []
        trajectory = []
        order = []

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
        for i in range(ensemble_size):
            scores = np.zeros((len(predictions)))
            s = len(ensemble)
            if s == 0:
                weighted_ensemble_prediction = np.zeros(predictions[0].shape)
            else:
                # Memory-efficient averaging!
                ensemble_prediction = np.zeros(ensemble[0].shape)
                for pred in ensemble:
                    ensemble_prediction += pred
                ensemble_prediction /= s

                weighted_ensemble_prediction = (s / float(s + 1)) * \
                                               ensemble_prediction
            fant_ensemble_prediction = np.zeros(weighted_ensemble_prediction.shape)
            for j, pred in enumerate(predictions):
                fant_ensemble_prediction[:] = weighted_ensemble_prediction + (1. / float(s + 1)) * pred
                if self.eval_metric_expects_y_pred:
                    preds = get_pred_from_proba(y_pred_proba=fant_ensemble_prediction, problem_type=self.problem_type)
                else:
                    preds = fant_ensemble_prediction

                scores[j] = self.metric._optimum - calculate_score(
                    solution=labels,
                    prediction=preds,
                    task_type=self.problem_type,
                    metric=self.metric,
                    all_scoring_functions=False)

                # scores[j] = -self.metric(y_true=labels, y_pred=fant_ensemble_prediction)
            # print('scores:', scores)
            all_best = np.argwhere(scores == np.nanmin(scores)).flatten()
            best = self.random_state.choice(all_best)

            # TODO: Instead of selecting random, compute additional metric which can be a tie-breaker!

            ensemble.append(predictions[best])
            trajectory.append(scores[best])
            order.append(best)

            # Handle special case
            if len(predictions) == 1:
                break

            if time_limit is not None:
                time_elapsed = time.time() - time_start
                time_left = time_limit - time_elapsed
                if time_left <= 0:
                    logger.warning('Warning: Ensemble Selection ran out of time, early stopping at iteration %s. This may mean that the time_limit specified is very small for this problem.' % (i+1))
                    break

        min_score = np.min(trajectory)
        first_index_of_best = trajectory.index(min_score)

        self.indices_ = order
        self.trajectory_ = trajectory
        self.train_score_ = trajectory[-1]  # TODO: Select best iteration or select final iteration? Earlier iteration could have a better score!

        if self.use_best:
            self.indices_ = order[:first_index_of_best+1]
            self.trajectory_ = trajectory[:first_index_of_best+1]
            self.train_score_ = trajectory[first_index_of_best]  # TODO: Select best iteration or select final iteration? Earlier iteration could have a better score!
            self.ensemble_size = first_index_of_best + 1
            logger.log(15, 'Ensemble size: %s' % self.ensemble_size)

        logger.debug("Ensemble indices: "+str(self.indices_))

    def _calculate_weights(self):
        ensemble_members = Counter(self.indices_).most_common()
        weights = np.zeros((self.num_input_models_,), dtype=float)
        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / self.ensemble_size
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        self.weights_ = weights

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

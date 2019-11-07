
import numpy as np
from tabular.ml.constants import PROBLEM_TYPES

from collections import Counter

from tabular.metrics import calculate_score
import tabular.metrics

from tabular.ml.utils import get_pred_from_proba


class EnsembleSelection():
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
        if random_state is not None:
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(seed=0)
        if type(metric) == tabular.metrics._ProbaScorer:
            self.objective_func_expects_y_pred = False
        elif type(metric) == tabular.metrics._ThresholdScorer:
            self.objective_func_expects_y_pred = False
        else:
            self.objective_func_expects_y_pred = True

    def fit(self, predictions, labels, identifiers):
        self.ensemble_size = int(self.ensemble_size)
        if self.ensemble_size < 1:
            raise ValueError('Ensemble size cannot be less than one!')
        if not self.problem_type in PROBLEM_TYPES:
            raise ValueError('Unknown problem type %s.' % self.problem_type)
        # if not isinstance(self.metric, Scorer):
        #     raise ValueError('Metric must be of type scorer')

        self._fit(predictions=predictions, labels=labels)
        self._calculate_weights()
        print('weights:')
        print(self.weights_)

    # TODO: Consider having a removal stage, remove each model and see if score is affected, if improves or not effected, remove it.
    def _fit(self, predictions, labels):
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
                if self.objective_func_expects_y_pred:
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


        self.indices_ = order
        self.trajectory_ = trajectory
        self.train_score_ = trajectory[-1]  # TODO: Select best iteration or select final iteration? Earlier iteration could have a better score!
        print(order)

    def _calculate_weights(self):
        ensemble_members = Counter(self.indices_).most_common()
        weights = np.zeros((self.num_input_models_,), dtype=float)
        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / self.ensemble_size
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        self.weights_ = weights

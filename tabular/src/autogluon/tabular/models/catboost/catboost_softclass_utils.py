import math

import numpy as np
from catboost import MultiRegressionCustomMetric, MultiRegressionCustomObjective

from autogluon.core.metrics.softclass_metrics import EPS

from .catboost_utils import CustomMetric


# Ojectives for SOFTCLASS problem_type
class SoftclassCustomMetric(CustomMetric):
    def __init__(self, metric, is_higher_better, needs_pred_proba):  # metric is ignored
        super().__init__(metric, is_higher_better, needs_pred_proba)
        self.softlogloss = self.SoftLogLossMetric()  # the metric object to pass to CatBoostRegressor

    def evaluate(self, approxes, target, weight):
        return self.softlogloss.evaluate(approxes, target, weight)

    class SoftLogLossMetric(MultiRegressionCustomMetric):
        def get_final_error(self, error, weight):
            return error

        def is_max_optimal(self):
            return True

        def evaluate(self, approxes, target, weight):
            assert len(target) == len(approxes)
            assert len(target[0]) == len(approxes[0])
            weight_sum = len(target)
            # TODO: inefficient copy of approxes, targets to np.array from provided UniTuple (required for JIT to work)
            approxes2 = np.zeros((len(approxes[0]), len(approxes)))
            target2 = np.zeros((len(approxes[0]), len(approxes)))
            for i in range(len(approxes)):
                approxes2[:, i] = approxes[i]
                target2[:, i] = target[i]
            approxes = approxes2
            target = target2
            approxes = np.exp(approxes)
            approxes = (approxes.T / approxes.sum(axis=1)).T  # softmax
            # Numpy implementation of soft logloss:
            approxes = np.clip(approxes, a_min=EPS, a_max=None)  # clip 0s to avoid NaN
            approxes = (approxes.T / approxes.sum(axis=1)).T  # renormalize
            losses = np.multiply(np.log(approxes), target).sum(axis=1)
            error_sum = np.mean(losses)
            return error_sum, weight_sum
            """ The above numpy evaluate() function is necessary for JIT to work and not print warnings, here is the original function (that works without JIT):
            def evaluate(self, approxes, target, weight):
                assert len(target) == len(approxes)
                assert len(target[0]) == len(approxes[0])
                weight_sum = len(target)
                approxes = np.array(approxes)
                approxes = np.exp(approxes)
                approxes = np.multiply(approxes, 1/np.sum(approxes, axis=1)[:, np.newaxis])
                error_sum = soft_log_loss(np.array(target), np.array(approxes))
                return error_sum, weight_sum
            """


class SoftclassObjective(object):
    def __init__(self):
        self.softlogloss = self.SoftLogLossObjective()  # the objective object to pass to CatBoostRegressor

    class SoftLogLossObjective(MultiRegressionCustomObjective):
        # TODO: Consider replacing with C++ implementation (but requires building catboost from source).
        # This pure Python is 3x faster than optimized Numpy implementation. Tested C++ implementation was 3x faster than this one.
        def calc_ders_multi(self, approxes, targets, weight):
            exp_approx = [math.exp(val) for val in approxes]
            # exp_sum = sum(exp_approx)  # not yet supported in numba jit: https://stackoverflow.com/questions/64936311/numba-cannot-determine-numba-type-of-class-builtin-function-or-method, using for loop below instead:
            exp_sum = 0.0
            for x in exp_approx:
                exp_sum += x
            exp_approx = [val / exp_sum for val in exp_approx]
            grad = [(targets[j] - exp_approx[j]) * weight for j in range(len(targets))]
            hess = [[(exp_approx[j] * exp_approx[j2] - (j == j2) * exp_approx[j]) * weight for j in range(len(targets))] for j2 in range(len(targets))]
            return (grad, hess)

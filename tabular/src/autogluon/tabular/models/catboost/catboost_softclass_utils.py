import math
import numpy as np

from autogluon.core.metrics import soft_log_loss
from autogluon.core.utils import try_import_catboost, try_import_catboostdev

from .catboost_utils import CustomMetric


# Ojectives for SOFTCLASS problem_type
# TODO: these require catboost_dev for now. Swap catboost_dev for catboost>=0.24 once it is released. Lazy imports will no longer be needed at that time.
class SoftclassCustomMetric(CustomMetric):
    try_import_catboost()  # Need to first import catboost then catboost_dev not vice-versa.
    try_import_catboostdev()
    from catboost_dev import MultiRegressionCustomMetric
    def __init__(self, metric, is_higher_better, needs_pred_proba):  # metric is ignored
        super().__init__(metric, is_higher_better, needs_pred_proba)
        try_import_catboostdev()
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
            approxes = np.array(approxes)
            approxes = np.exp(approxes)
            approxes = np.multiply(approxes, 1/np.sum(approxes, axis=1)[:, np.newaxis])
            error_sum = soft_log_loss(np.array(target), np.array(approxes))
            return error_sum, weight_sum


class SoftclassObjective(object):
    try_import_catboost()  # Need to first import catboost then catboost_dev not vice-versa.
    try_import_catboostdev()
    from catboost_dev import MultiRegressionCustomObjective
    def __init__(self):
        try_import_catboostdev()
        self.softlogloss = self.SoftLogLossObjective()  # the objective object to pass to CatBoostRegressor

    class SoftLogLossObjective(MultiRegressionCustomObjective):
        # TODO: Consider replacing with C++ implementation (but requires building catboost from source).
        # This pure Python is 3x faster than optimized Numpy implementation. Tested C++ implementation was 3x faster than this one.
        def calc_ders_multi(self, approxes, targets, weight):
            exp_approx = [math.exp(val) for val in approxes]
            exp_sum = sum(exp_approx)
            exp_approx = [val / exp_sum for val in exp_approx]
            grad = [(targets[j] - exp_approx[j])*weight for j in range(len(targets))]
            hess = [[(exp_approx[j] * exp_approx[j2] - (j==j2)*exp_approx[j]) * weight
                    for j in range(len(targets))] for j2 in range(len(targets))]
            return (grad, hess)


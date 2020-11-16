import logging
import numpy as np

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS

logger = logging.getLogger(__name__)


# TODO: Add weight support?
# TODO: Can these be optimized? What computational cost do they have compared to the default catboost versions?
class CustomMetric:
    def __init__(self, metric, is_higher_better, needs_pred_proba):
        self.metric = metric
        self.is_higher_better = is_higher_better
        self.needs_pred_proba = needs_pred_proba

    @staticmethod
    def get_final_error(error, weight):
        return error

    def is_max_optimal(self):
        return self.is_higher_better

    def evaluate(self, approxes, target, weight):
        raise NotImplementedError


class BinaryCustomMetric(CustomMetric):
    @staticmethod
    def _get_y_pred_proba(approxes):
        return np.array(approxes[0])

    @staticmethod
    def _get_y_pred(y_pred_proba):
        return np.round(y_pred_proba)

    def evaluate(self, approxes, target, weight):
        y_pred_proba = self._get_y_pred_proba(approxes=approxes)

        # TODO: Binary log_loss doesn't work for some reason
        if self.needs_pred_proba:
            score = self.metric(np.array(target), y_pred_proba)
        else:
            raise NotImplementedError('Custom Catboost Binary prob metrics are not supported by AutoGluon.')
            # y_pred = self._get_y_pred(y_pred_proba=y_pred_proba)  # This doesn't work at the moment because catboost returns some strange valeus in approxes which are not the probabilities
            # score = self.metric(np.array(target), y_pred)

        return score, 1


class MulticlassCustomMetric(CustomMetric):
    @staticmethod
    def _get_y_pred_proba(approxes):
        return np.array(approxes)

    @staticmethod
    def _get_y_pred(y_pred_proba):
        return y_pred_proba.argmax(axis=0)

    def evaluate(self, approxes, target, weight):
        y_pred_proba = self._get_y_pred_proba(approxes=approxes)
        if self.needs_pred_proba:
            raise NotImplementedError('Custom Catboost Multiclass proba metrics are not supported by AutoGluon.')
            # y_pred_proba = y_pred_proba.reshape(len(np.unique(np.array(target))), -1).T
            # score = self.metric(np.array(target), y_pred_proba)  # This doesn't work at the moment because catboost returns some strange valeus in approxes which are not the probabilities
        else:
            y_pred = self._get_y_pred(y_pred_proba=y_pred_proba)
            score = self.metric(np.array(target), y_pred)

        return score, 1


class RegressionCustomMetric(CustomMetric):
    @staticmethod
    def _get_y_pred(approxes):
        return np.array(approxes[0])

    def evaluate(self, approxes, target, weight):
        y_pred = self._get_y_pred(approxes=approxes)
        score = self.metric(np.array(target), y_pred)

        return score, 1


metric_classes_dict = {
    BINARY: BinaryCustomMetric,
    MULTICLASS: MulticlassCustomMetric,
    REGRESSION: RegressionCustomMetric,
}


# TODO: Refactor as a dictionary mapping as done in LGBM
def construct_custom_catboost_metric(metric, is_higher_better, needs_pred_proba, problem_type):
    if problem_type == SOFTCLASS:
        from .catboost_softclass_utils import SoftclassCustomMetric
        if metric.name != 'soft_log_loss':
            logger.warning("Setting metric=soft_log_loss, the only metric supported for softclass problem_type")
        # SoftclassCustomMetric = make_softclass_metric()  # TODO: remove after catboost 0.24
        return SoftclassCustomMetric(metric=None, is_higher_better=True, needs_pred_proba=True)
    if (metric.name == 'log_loss') and (problem_type == MULTICLASS) and needs_pred_proba:
        return 'MultiClass'
    if metric.name == 'accuracy':
        return 'Accuracy'
    if (metric.name == 'log_loss') and (problem_type == BINARY) and needs_pred_proba:
        return 'Logloss'
    if (metric.name == 'roc_auc') and (problem_type == BINARY) and needs_pred_proba:
        return 'AUC'
    if (metric.name == 'f1') and (problem_type == BINARY) and not needs_pred_proba:
        return 'F1'
    if (metric.name == 'balanced_accuracy') and (problem_type == BINARY) and not needs_pred_proba:
        return 'BalancedAccuracy'
    if (metric.name == 'recall') and (problem_type == BINARY) and not needs_pred_proba:
        return 'Recall'
    if (metric.name == 'precision') and (problem_type == BINARY) and not needs_pred_proba:
        return 'Precision'
    if (metric.name == 'mean_absolute_error') and (problem_type == REGRESSION):
        return 'MAE'
    if (metric.name == 'mean_squared_error') and (problem_type == REGRESSION):
        return 'RMSE'
    if (metric.name == 'root_mean_squared_error') and (problem_type == REGRESSION):
        return 'RMSE'
    if (metric.name == 'median_absolute_error') and (problem_type == REGRESSION):
        return 'MedianAbsoluteError'
    if (metric.name == 'r2') and (problem_type == REGRESSION):
        return 'R2'
    metric_class = metric_classes_dict[problem_type]
    return metric_class(metric=metric, is_higher_better=is_higher_better, needs_pred_proba=needs_pred_proba)

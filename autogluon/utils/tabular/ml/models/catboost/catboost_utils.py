import numpy as np

from ...constants import BINARY, MULTICLASS, REGRESSION

# TODO: Add weight support?
# TODO: Can these be optimized? What computational cost do they have compared to the default catboost versions?
class CustomMetric:
    def __init__(self, metric, is_higher_better, needs_pred_proba):
        self.metric = metric
        self.is_higher_better = is_higher_better
        self.needs_pred_proba = needs_pred_proba

    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return self.is_higher_better

    def evaluate(self, approxes, target, weight):
        raise NotImplementedError


class BinaryCustomMetric(CustomMetric):
    def _get_y_pred_proba(self, approxes):
        return np.array(approxes[0])

    def _get_y_pred(self, y_pred_proba):
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
    def _get_y_pred_proba(self, approxes):
        return np.array(approxes)

    def _get_y_pred(self, y_pred_proba):
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
    def _get_y_pred(self, approxes):
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


def construct_custom_catboost_metric(metric, is_higher_better, needs_pred_proba, problem_type):
    if (metric.name == 'log_loss') and (problem_type == MULTICLASS) and needs_pred_proba:
        return 'MultiClass'
    if metric.name == 'accuracy':
        return 'Accuracy'
    if (metric.name == 'log_loss') and (problem_type == BINARY) and needs_pred_proba:
        return 'Logloss'
    metric_class = metric_classes_dict[problem_type]
    return metric_class(metric=metric, is_higher_better=is_higher_better, needs_pred_proba=needs_pred_proba)

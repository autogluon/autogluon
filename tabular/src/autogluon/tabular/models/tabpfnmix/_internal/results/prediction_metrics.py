from dataclasses import dataclass

import numpy as np
import scipy

from autogluon.core.metrics import Scorer

from ..core.enums import Task


@dataclass
class PredictionMetrics:
    task: Task
    loss: float
    score: float
    metrics: dict[str, float]

    @classmethod
    def from_prediction(cls, y_pred: np.ndarray, y_true: np.ndarray, task: Task, metric: Scorer):
        loss, score, metrics = compute_metrics(y_pred, y_true, task, metric=metric)

        return PredictionMetrics(task=task, loss=loss, score=score, metrics=metrics)


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray, task: Task, metric: Scorer) -> tuple[float, float, dict]:
    if task == Task.CLASSIFICATION:
        return compute_classification_metrics(y_pred, y_true, metric=metric)
    else:
        return compute_regression_metrics(y_pred, y_true, metric=metric)


def compute_classification_metrics(
    y_pred: np.ndarray, y_true: np.ndarray, metric: Scorer
) -> tuple[float, float, dict]:
    # predictions are assumed to be log-probabilities

    if metric.needs_pred or metric.needs_class:
        y_pred_class = np.argmax(y_pred, axis=1)
        metric_score = metric(y_true, y_pred_class)
    else:
        y_pred_proba = scipy.special.softmax(y_pred, axis=1)
        metric_score = metric(y_true, y_pred_proba)

    metric_error = metric.convert_score_to_error(metric_score)

    return metric_error, metric_score, {metric.name: metric_score}


def compute_regression_metrics(y_pred: np.ndarray, y_true: np.ndarray, metric: Scorer) -> tuple[float, float, dict]:
    metric_score = metric(y_true, y_pred)
    metric_error = metric.convert_score_to_error(metric_score)

    return metric_error, metric_score, {metric.name: metric_score}

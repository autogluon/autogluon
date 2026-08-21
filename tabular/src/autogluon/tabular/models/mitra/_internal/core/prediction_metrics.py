from dataclasses import dataclass

import numpy as np
import scipy.special
import torch
from loguru import logger
from sklearn.metrics import f1_score, mean_squared_error, r2_score, roc_auc_score, root_mean_squared_error

from ..._internal.config.enums import MetricName, Task
from ..._internal.data.preprocessor import Preprocessor


@dataclass
class PredictionMetrics:
    task: Task
    loss: float
    score: float
    metrics: dict[MetricName, float]

    @classmethod
    def from_prediction(cls, y_pred: np.ndarray, y_true: np.ndarray, task: Task) -> "PredictionMetrics":
        loss, score, metrics = compute_metrics(y_pred, y_true, task)

        return cls(task=task, loss=loss, score=score, metrics=metrics)


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray, task: Task) -> tuple[float, float, dict]:
    if task == Task.CLASSIFICATION:
        return compute_classification_metrics(y_pred, y_true)
    elif task == Task.REGRESSION:
        return compute_regression_metrics(y_pred, y_true)


def compute_classification_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, float, dict]:
    # predictions are assumed to be log-probabilities

    y_pred_class = np.argmax(y_pred, axis=1)
    y_pred_proba = scipy.special.softmax(y_pred, axis=1)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(
        axis=1, keepdims=True
    )  # softmax not completely numerically stable, so a small correction is needed
    labels = np.arange(y_pred_proba.shape[1])

    metrics = {
        MetricName.ACCURACY: (y_true == y_pred_class).mean(),
        MetricName.F1: f1_score(y_true, y_pred_class, average="weighted"),
        MetricName.AUC: roc_auc_score_multiclass(
            y_true, y_pred_proba, multi_class="ovo", average="macro", labels=labels
        ),
        MetricName.LOG_LOSS: torch.nn.functional.cross_entropy(
            torch.from_numpy(y_pred), torch.from_numpy(y_true)
        ).item(),
    }

    loss = metrics[MetricName.LOG_LOSS]
    score = metrics[MetricName.ACCURACY]

    return loss, score, metrics


def roc_auc_score_multiclass(y_true, y_pred_proba, multi_class="ovo", average="macro", labels=None) -> float:
    """
    The roc_auc_score multi_class is not supported for binary classification
    """

    if np.unique(y_true).shape[0] == 1:
        # AUC is not defined if there is only one class
        return float("nan")

    try:
        if y_pred_proba.shape[1] == 2:
            return roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            return roc_auc_score(y_true, y_pred_proba, multi_class=multi_class, average=average, labels=labels)
    except ValueError as e:
        logger.error(f"Error computing roc_auc_score: {e}")
        logger.error(f"Returning {-1}")
        return -1


def compute_regression_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, float, dict]:
    metrics = {
        MetricName.RMSE: root_mean_squared_error(y_true, y_pred),
        MetricName.MSE: mean_squared_error(y_true, y_pred),
        MetricName.MAE: np.abs(y_true - y_pred).mean(),
        MetricName.R2: r2_score(y_true, y_pred),
    }

    loss = metrics[MetricName.MSE]
    score = metrics[MetricName.R2]

    return loss, score, metrics


class PredictionMetricsTracker:
    """
    Prediction metrics tracker that accumulates predictions and true values to compute metrics at the end.
    Uses torch.Tensor for predictions and true values.
    """

    def __init__(self, task: Task, preprocessor: Preprocessor) -> None:
        self.task = task
        self.preprocessor = preprocessor
        self.reset()

    def reset(self) -> None:
        self.ys_pred: list[np.ndarray] = []
        self.ys_true: list[np.ndarray] = []

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor, train: bool) -> None:
        y_pred_np = y_pred.detach().cpu().numpy()[0]
        y_pred_ori = self.preprocessor.inverse_transform_y(y_pred_np)

        y_true_np = y_true.detach().cpu().numpy()[0]
        if train:
            y_true_ori = self.preprocessor.inverse_transform_y(y_true_np)
        else:
            y_true_ori = y_true_np

        self.ys_pred.append(y_pred_ori)
        self.ys_true.append(y_true_ori)

    def get_metrics(self) -> PredictionMetrics:
        y_pred = np.concatenate(self.ys_pred, axis=0)
        y_true = np.concatenate(self.ys_true, axis=0)

        return PredictionMetrics.from_prediction(y_pred, y_true, self.task)

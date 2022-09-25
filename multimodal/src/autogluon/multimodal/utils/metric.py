import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

from sklearn.metrics import f1_score

from autogluon.core.metrics import get_metric

from ..constants import (
    ACCURACY,
    AUTOMM,
    AVERAGE_PRECISION,
    BINARY,
    F1,
    METRIC_MODE_MAP,
    MULTICLASS,
    REGRESSION,
    RMSE,
    ROC_AUC,
    VALID_METRICS,
    Y_PRED,
    Y_PRED_PROB,
    Y_TRUE,
)

logger = logging.getLogger(AUTOMM)


def infer_metrics(
    problem_type: Optional[str] = None,
    eval_metric_name: Optional[str] = None,
):
    """
    Infer the validation metric and the evaluation metric if not provided.
    Validation metric is for early-stopping and selecting the best model checkpoints.
    Evaluation metric is to report performance to users.

    Parameters
    ----------
    problem_type
        Type of problem.
    eval_metric_name
        Name of evaluation metric provided by users.

    Returns
    -------
    validation_metric_name
        Name of validation metric.
    eval_metric_name
        Name of evaluation metric.
    """

    if eval_metric_name is not None:
        if problem_type != BINARY and eval_metric_name.lower() in [
            ROC_AUC,
            AVERAGE_PRECISION,
            F1,
        ]:
            raise ValueError(f"Metric {eval_metric_name} is only supported for binary classification.")

        if eval_metric_name in VALID_METRICS:
            validation_metric_name = eval_metric_name
            return validation_metric_name, eval_metric_name

        warnings.warn(
            f"Currently, we cannot convert the metric: {eval_metric_name} to a metric supported in torchmetrics. "
            f"Thus, we will fall-back to use accuracy for multi-class classification problems "
            f", ROC-AUC for binary classification problem, and RMSE for regression problems.",
            UserWarning,
        )

    if problem_type == MULTICLASS:
        eval_metric_name = ACCURACY
    elif problem_type == BINARY:
        eval_metric_name = ROC_AUC
    elif problem_type == REGRESSION:
        eval_metric_name = RMSE
    else:
        raise NotImplementedError(f"Problem type: {problem_type} is not supported yet!")

    validation_metric_name = eval_metric_name

    return validation_metric_name, eval_metric_name


def get_minmax_mode(metric_name: str):
    """
    Get minmax mode based on metric name

    Parameters
    ----------
    metric_name
        A string representing metric

    Returns
    -------
    mode
        The min/max mode used in selecting model checkpoints.
        - min
             Its means that smaller metric is better.
        - max
            It means that larger metric is better.
    """
    assert metric_name in METRIC_MODE_MAP, f"{metric_name} is not a supported metric. Options are: {VALID_METRICS}"
    return METRIC_MODE_MAP.get(metric_name)


def compute_score(
    metric_data: dict,
    metric_name: str,
    pos_label: Optional[int] = 1,
) -> float:
    """
    Use sklearn to compute the score of one metric.

    Parameters
    ----------
    metric_data
        A dictionary with the groundtruth (Y_TRUE) and predicted values (Y_PRED, Y_PRED_PROB).
        The predicted class probabilities are required to compute the roc_auc score.
    metric_name
        The name of metric to compute.
    pos_label
        The encoded label (0 or 1) of binary classification's positive class.

    Returns
    -------
    Computed score.
    """
    metric = get_metric(metric_name)
    if metric.name in [ROC_AUC, AVERAGE_PRECISION]:
        return metric._sign * metric(metric_data[Y_TRUE], metric_data[Y_PRED_PROB][:, pos_label])
    elif metric.name in [F1]:  # only for binary classification
        return f1_score(metric_data[Y_TRUE], metric_data[Y_PRED], pos_label=pos_label)
    else:
        return metric._sign * metric(metric_data[Y_TRUE], metric_data[Y_PRED])

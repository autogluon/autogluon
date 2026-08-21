import functools
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import evaluate
import torchmetrics
from torch.nn import functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from autogluon.core.metrics import Scorer, compute_metric, get_metric

from ...constants import (
    ACC,
    ACCURACY,
    AVERAGE_PRECISION,
    BER,
    COSINE_EMBEDDING_LOSS,
    COVERAGE,
    CROSS_ENTROPY,
    DETECTION_METRICS,
    DIRECT_LOSS,
    EM,
    F1,
    F1_MACRO,
    F1_MICRO,
    F1_WEIGHTED,
    FM,
    HIT_RATE,
    IOU,
    LOG_LOSS,
    MAE,
    MATCHING_METRICS,
    MATCHING_METRICS_WITHOUT_PROBLEM_TYPE,
    MAX,
    METRIC_MODE_MAP,
    MIN,
    MULTICLASS,
    NER_TOKEN_F1,
    OVERALL_ACCURACY,
    OVERALL_F1,
    PEARSONR,
    QUADRATIC_KAPPA,
    R2,
    RECALL,
    RETRIEVAL_METRICS,
    RMSE,
    ROC_AUC,
    ROOT_MEAN_SQUARED_ERROR,
    SM,
    SPEARMANR,
    Y_PRED,
    Y_PRED_PROB,
    Y_TRUE,
)
from .coverage_metrics import Coverage
from .hit_rate_metrics import CustomHitRate
from .semantic_seg_metrics import COD_METRICS_NAMES, Balanced_Error_Rate, Binary_IoU, Multiclass_IoU

logger = logging.getLogger(__name__)


def compute_score(
    metric_data: dict,
    metric: Union[str, Scorer],
    pos_label: Optional[int] = 1,
) -> float:
    """
    Use sklearn to compute the score of one metric.

    Parameters
    ----------
    metric_data
        A dictionary with the groundtruth (Y_TRUE) and predicted values (Y_PRED, Y_PRED_PROB).
        The predicted class probabilities are required to compute the roc_auc score.
    metric
        The name of metric or the function of metric to compute.
    pos_label
        The encoded label (0 or 1) of binary classification's positive class.

    Returns
    -------
    Computed score.
    """
    if isinstance(metric, str) and metric in [OVERALL_ACCURACY, OVERALL_F1]:
        metric = evaluate.load("seqeval")
        warnings.filterwarnings("ignore")
        for p in metric_data[Y_TRUE]:
            if "_" in p:
                print(p)
        for p in metric_data[Y_PRED]:
            if "_" in p:
                print(p)
        return metric.compute(references=metric_data[Y_TRUE], predictions=metric_data[Y_PRED])

    metric = get_metric(metric)

    y = metric_data[Y_TRUE]
    if metric.needs_proba or metric.needs_threshold:
        y_pred_proba = metric_data[Y_PRED_PROB]
        y_pred_proba = (
            y_pred_proba if y_pred_proba.shape[1] > 2 else y_pred_proba[:, pos_label]
        )  # only use pos_label for binary classification
        return metric.convert_score_to_original(
            compute_metric(y=y, y_pred_proba=y_pred_proba, metric=metric, weights=None)
        )
    else:
        y_pred = metric_data[Y_PRED]

        # TODO: This is a hack. Doesn't support `f1_macro`, `f1_micro`, `f1_weighted`, or custom `f1` metrics with different names.
        # TODO: Longterm the solution should be to have the input data to this function use the internal representation without the original class names. This way `pos_label` would not need to be specified.
        if metric.name == F1:  # only for binary classification
            y = (y == pos_label).astype(int)
            y_pred = (y_pred == pos_label).astype(int)

        return metric.convert_score_to_original(compute_metric(y=y, y_pred=y_pred, metric=metric, weights=None))


def infer_metrics(
    problem_type: Optional[str] = None,
    eval_metric: Optional[Union[str, Scorer]] = None,
    validation_metric_name: Optional[str] = None,
    is_matching: Optional[bool] = False,
):
    """
    Infer the validation metric and the evaluation metric if not provided.
    Validation metric is for early-stopping and selecting the best model checkpoints.
    Evaluation metric is to report performance to users.

    Parameters
    ----------
    problem_type
        Type of problem.
    eval_metric
        Name of evaluation metric provided by users.
    validation_metric_name
        The provided validation metric name
    is_matching
        Whether is matching.

    Returns
    -------
    validation_metric_name
        Name of validation metric.
    eval_metric_name
        Name of evaluation metric.
    """
    is_customized = False
    if eval_metric is None:
        eval_metric_name = None
    elif isinstance(eval_metric, str):
        eval_metric_name = eval_metric
    elif isinstance(eval_metric, Scorer):
        eval_metric_name = eval_metric.name
        is_customized = True
    else:
        raise TypeError(f"eval_metric can be a str, a Scorer, or None, but is type: {type(eval_metric)}")

    if problem_type is not None:
        from ...utils.problem_types import PROBLEM_TYPES_REG

        problem_property = PROBLEM_TYPES_REG.get(problem_type)

    if is_matching:
        if eval_metric_name is not None:
            # if eval_metric_name is a valid metric
            if eval_metric_name.lower() in METRIC_MODE_MAP.keys():
                validation_metric_name = eval_metric_name
                return validation_metric_name, eval_metric_name
            elif eval_metric_name.lower() in RETRIEVAL_METRICS:
                # Currently only support recall as validation metric in retrieval tasks.
                validation_metric_name = RECALL
                return validation_metric_name, eval_metric_name

        # When eval_metric_name is either None or not supported:
        # Fallback based on problem type unless it's a customized metric
        if problem_type is None:
            validation_metric_name, fallback_evaluation_metric = MATCHING_METRICS_WITHOUT_PROBLEM_TYPE
        elif problem_type in MATCHING_METRICS:
            validation_metric_name, fallback_evaluation_metric = MATCHING_METRICS[problem_type]
        else:
            raise NotImplementedError(f"Problem type: {problem_type} is not yet supported for matching!")
        if not is_customized:
            if eval_metric_name is not None:
                warnings.warn(
                    f"Metric {eval_metric_name} is not supported as the evaluation metric for {problem_type} in matching tasks."
                    f"The evaluation metric is changed to {fallback_evaluation_metric} by default."
                )
            eval_metric_name = fallback_evaluation_metric
        return validation_metric_name, eval_metric_name

    if eval_metric_name is not None:
        # Infer evaluation metric
        if eval_metric_name.lower() not in problem_property.supported_evaluation_metrics and not is_customized:
            warnings.warn(
                f"Metric {eval_metric_name} is not supported as the evaluation metric for {problem_type}. "
                f"The evaluation metric is changed to {problem_property.fallback_evaluation_metric} by default."
            )
            if problem_property.fallback_evaluation_metric is not None:
                eval_metric_name = problem_property.fallback_evaluation_metric
            else:
                # Problem types like extract_embedding does not need a eval/val metric
                return None, None

        # Infer validation metric
        if eval_metric_name.lower() in problem_property.supported_validation_metrics:
            validation_metric_name = eval_metric_name
        else:
            if problem_property.fallback_validation_metric is not None:
                validation_metric_name = problem_property.fallback_validation_metric
    else:
        eval_metric_name = problem_property.fallback_evaluation_metric
        validation_metric_name = problem_property.fallback_validation_metric

    return validation_metric_name, eval_metric_name


def get_minmax_mode(
    metric_name: Union[str, Scorer],
):
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
    if isinstance(metric_name, str):
        assert metric_name in METRIC_MODE_MAP, (
            f"{metric_name} is not a supported metric. Options are: {METRIC_MODE_MAP.keys()}"
        )
        return METRIC_MODE_MAP.get(metric_name)
    else:
        return MAX if metric_name.greater_is_better else MIN


def get_stopping_threshold(metric_name: str):
    """
    Get the metric threshold for early stopping.

    Parameters
    ----------
    metric_name
        Name of validation metric.

    Returns
    -------
    The stopping threshold.
    """
    try:
        metric = get_metric(metric_name)
        stopping_threshold = metric.optimum - metric._sign * 1e-7
    except:
        stopping_threshold = None

    return stopping_threshold


def get_torchmetric(
    metric_name: str,
    num_classes: Optional[int] = None,
    is_matching: Optional[bool] = False,
    problem_type: Optional[str] = None,
):
    """
    Obtain a torchmerics.Metric from its name.
    Define a customized metric function in case that torchmetrics doesn't support some metric.

    Parameters
    ----------
    metric_name
        Name of metric.
    num_classes
        Number of classes.
    is_matching
        Whether is matching.
    problem_type
        Type of problem, e.g., binary and multiclass.

    Returns
    -------
    torchmetrics.Metric
        A torchmetrics.Metric object.
    custom_metric_func
        A customized metric function.
    """
    metric_name = metric_name.lower()
    if metric_name in [ACC, ACCURACY, OVERALL_ACCURACY]:
        # use MULTICLASS since the head output dim is 2 for the binary problem type.
        return torchmetrics.Accuracy(task=MULTICLASS, num_classes=num_classes), None
    elif metric_name == NER_TOKEN_F1:
        return torchmetrics.F1Score(task=MULTICLASS, num_classes=num_classes, ignore_index=1), None
    elif metric_name in [RMSE, ROOT_MEAN_SQUARED_ERROR]:
        return torchmetrics.MeanSquaredError(squared=False), None
    elif metric_name == R2:
        return torchmetrics.R2Score(), None
    elif metric_name == QUADRATIC_KAPPA:
        return (
            torchmetrics.CohenKappa(task=problem_type, num_classes=num_classes, weights="quadratic"),
            None,
        )
    elif metric_name == ROC_AUC:
        return torchmetrics.AUROC(task=problem_type, num_classes=num_classes), None
    elif metric_name == AVERAGE_PRECISION:
        return torchmetrics.AveragePrecision(task=problem_type, num_classes=num_classes)
    elif metric_name in [LOG_LOSS, CROSS_ENTROPY]:
        return torchmetrics.MeanMetric(), functools.partial(F.cross_entropy, reduction="none")
    elif metric_name == COSINE_EMBEDDING_LOSS:
        return torchmetrics.MeanMetric(), functools.partial(F.cosine_embedding_loss, reduction="none")
    elif metric_name == PEARSONR:
        return torchmetrics.PearsonCorrCoef(), None
    elif metric_name == SPEARMANR:
        if is_matching:  # TODO: add support for matching.
            raise ValueError("spearman relation is not supported for matching yet.")
        else:
            return torchmetrics.SpearmanCorrCoef(), None
    elif metric_name == F1:
        return torchmetrics.F1Score(task=problem_type, num_classes=num_classes), None
    elif metric_name in [F1_MACRO, F1_MICRO, F1_WEIGHTED]:
        average = metric_name.split("_")[1]
        return torchmetrics.F1Score(task=problem_type, num_classes=num_classes, average=average), None
    elif metric_name in DETECTION_METRICS:
        return (
            MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=False),
            None,
        )  # TODO: remove parameter hardcodings here, and add class_metrics
    elif metric_name == DIRECT_LOSS:
        return (
            torchmetrics.MeanMetric(nan_strategy="warn"),
            None,
        )  # This only works for detection where custom_metric is not required for BaseAggregator
    elif metric_name in [RECALL, HIT_RATE]:
        if is_matching:
            return CustomHitRate(), None
        else:  # TODO: support recall for general classification tasks.
            raise ValueError("Recall is not supported yet.")
    elif metric_name == BER:
        return Balanced_Error_Rate(), None
    elif metric_name in [SM, EM, FM, MAE]:
        return COD_METRICS_NAMES[metric_name], None
    elif metric_name == IOU:
        if num_classes == 1:
            return Binary_IoU(), None
        else:
            return Multiclass_IoU(num_classes=num_classes), None
    elif metric_name == COVERAGE:
        return Coverage(), None
    else:
        raise ValueError(f"Unknown metric {metric_name}")

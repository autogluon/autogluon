import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import evaluate
import pytrec_eval
from sklearn.metrics import f1_score

from autogluon.core.metrics import get_metric

from ..constants import (
    ACCURACY,
    AUTOMM,
    AVERAGE_PRECISION,
    BINARY,
    DIRECT_LOSS,
    F1,
    MAP,
    METRIC_MODE_MAP,
    MULTICLASS,
    NER,
    OBJECT_DETECTION,
    OVERALL_ACCURACY,
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
    pipeline: Optional[str] = None,
    eval_metric_name: Optional[str] = None,
    validation_metric_name: Optional[str] = None,
):
    """
    Infer the validation metric and the evaluation metric if not provided.
    Validation metric is for early-stopping and selecting the best model checkpoints.
    Evaluation metric is to report performance to users.

    Parameters
    ----------
    problem_type
        Type of problem.
    pipeline
        Predictor pipeline, used when problem_type is None.
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
    elif problem_type == NER:
        eval_metric_name = OVERALL_ACCURACY
    elif problem_type == BINARY:
        eval_metric_name = ROC_AUC
    elif problem_type == REGRESSION:
        eval_metric_name = RMSE
    elif problem_type is None:
        if pipeline == OBJECT_DETECTION:
            if (not validation_metric_name) or validation_metric_name.lower() == DIRECT_LOSS:
                return DIRECT_LOSS, MAP
            elif validation_metric_name == MAP:
                return MAP, MAP
            else:
                raise ValueError(
                    f"Problem type: {problem_type}, pipeline: {pipeline}, validation_metric_name: {validation_metric_name} is not supported!"
                )
        else:
            raise NotImplementedError(f"Problem type: {problem_type}, pipeline: {pipeline} is not supported yet!")
    else:
        raise NotImplementedError(f"Problem type: {problem_type} is not supported yet!")

    validation_metric_name = eval_metric_name

    return validation_metric_name, eval_metric_name


def get_minmax_mode(
    metric_name: str,
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
    if metric_name == OVERALL_ACCURACY:
        metric = evaluate.load("seqeval")
        return metric.compute(references=metric_data[Y_TRUE], predictions=metric_data[Y_PRED])

    metric = get_metric(metric_name)
    if metric.name in [ROC_AUC, AVERAGE_PRECISION]:
        return metric._sign * metric(metric_data[Y_TRUE], metric_data[Y_PRED_PROB][:, pos_label])
    elif metric.name in [F1]:  # only for binary classification
        return f1_score(metric_data[Y_TRUE], metric_data[Y_PRED], pos_label=pos_label)
    else:
        return metric._sign * metric(metric_data[Y_TRUE], metric_data[Y_PRED])


def compute_ranking_score(
    results: Dict[str, Dict], qrel_dict: Dict[str, Dict], cutoff: Optional[List[int]] = [5, 10, 20]
):
    """
    Compute NDCG, MAP, Recall, and Precision.
    TODO: Consider MRR.

    Parameters
    ----------
    qrel_dict:
        the groundtruth query and document relavance
    results:
        the query/document ranking list by the model
    cutoff:
        the cutoff value for NDCG, MAP, Recall, and Precision
    """
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    for k in cutoff:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in cutoff])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in cutoff])
    recall_string = "recall." + ",".join([str(k) for k in cutoff])
    precision_string = "P." + ",".join([str(k) for k in cutoff])

    evaluator = pytrec_eval.RelevanceEvaluator(qrel_dict, {map_string, ndcg_string, recall_string, precision_string})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in cutoff:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

    for k in cutoff:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

    return ndcg, _map, recall, precision

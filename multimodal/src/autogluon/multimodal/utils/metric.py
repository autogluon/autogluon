import logging
import math
import operator
import warnings
from typing import Dict, List, Optional, Tuple, Union

import evaluate
import numpy as np
from sklearn.metrics import f1_score

from autogluon.core.metrics import Scorer, get_metric

from ..constants import (
    ACCURACY,
    AVERAGE_PRECISION,
    BINARY,
    DIRECT_LOSS,
    F1,
    FEW_SHOT_CLASSIFICATION,
    IOU,
    MAP,
    MATCHING_METRICS,
    MATCHING_METRICS_WITHOUT_PROBLEM_TYPE,
    MAX,
    METRIC_MODE_MAP,
    MIN,
    MULTICLASS,
    NDCG,
    NER,
    NER_TOKEN_F1,
    NUMERICAL,
    OBJECT_DETECTION,
    OPEN_VOCABULARY_OBJECT_DETECTION,
    OVERALL_ACCURACY,
    OVERALL_F1,
    PRECISION,
    RECALL,
    REGRESSION,
    RETRIEVAL_METRICS,
    RMSE,
    ROC_AUC,
    SEMANTIC_SEGMENTATION,
    SPEARMANR,
    Y_PRED,
    Y_PRED_PROB,
    Y_TRUE,
)
from ..problem_types import PROBLEM_TYPES_REG

logger = logging.getLogger(__name__)


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
    eval_metric_name
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
        assert (
            metric_name in METRIC_MODE_MAP
        ), f"{metric_name} is not a supported metric. Options are: {METRIC_MODE_MAP.keys()}"
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
    if metric.name in [ROC_AUC, AVERAGE_PRECISION]:
        return metric._sign * metric(metric_data[Y_TRUE], metric_data[Y_PRED_PROB][:, pos_label])
    elif metric.name in [F1]:  # only for binary classification
        return f1_score(metric_data[Y_TRUE], metric_data[Y_PRED], pos_label=pos_label)
    else:
        try:
            return metric._sign * metric(metric_data[Y_TRUE], metric_data[Y_PRED], y_prob=metric_data[Y_PRED_PROB])
        except:
            return metric._sign * metric(metric_data[Y_TRUE], metric_data[Y_PRED])


class RankingMetrics:
    def __init__(
        self,
        pred: Dict[str, Dict],
        target: Dict[str, Dict],
        is_higher_better=True,
    ):
        """
        Evaluation Metrics for information retrieval tasks such as document retrieval, image retrieval, etc.
        Reference: https://www.cs.cornell.edu/courses/cs4300/2013fa/lectures/metrics-2-4pp.pdf

        Parameters
        ----------
        pred:
            the prediction of the ranking model. It has the following form.
            pred = {
                'q1': {
                    'd1': 1,
                    'd3': 0,
                },
                'q2': {
                    'd2': 1,
                    'd3': 1,
                },
            }
            where q refers to queries, and d refers to documents, each query has a few relevant documents.
            0s and 1s are model predicted scores (does not need to be binary).
        target:
            the ground truth query and response relevance which has the same form as pred.
        is_higher_better:
            if higher relevance score means higher ranking.
            if the relevance score is cosine similarity / dot product, it should be set to True;
            if it is Eulidean distance, it should be False.
        """
        self.pred = pred
        self.target = target
        self.is_higher_better = is_higher_better
        # the supported metrics in this script
        self.supported_metrics = {
            "precision": 0,
            "recall": 1,
            "mrr": 2,
            "map": 3,
            "ndcg": 4,
        }

        assert len(pred) == len(
            target
        ), f"The prediction and groudtruth target should have the same number of queries, \
        while there are {len(pred)} queries in prediction and {len(target)} in the target."

        self.results = {}
        for key in target.keys():
            self.results.update({key: [target[key], pred[key]]})

    def compute(self, metrics: Union[str, list] = None, k: Optional[int] = 10):
        """
        compute and return ranking scores.

        Parameters
        ----------
        metrics:
            user provided metrics
        k:
            the cutoff value for NDCG, MAP, Recall, MRR, and Precision

        Returns
        -------
        Computed score.

        """
        if isinstance(metrics, str):
            metrics = [metrics]
        if not metrics:  # no metric is provided
            metrics = self.supported_metrics.keys()

        return_res = {}

        eval_res = np.mean(
            [list(self._compute_one(idx, k)) for idx in self.results.keys()],
            axis=0,
        )

        for metric in metrics:
            metric = metric.lower()
            if metric in self.supported_metrics:
                return_res.update({f"{metric}@{k}": eval_res[self.supported_metrics[metric]]})

        return return_res

    def _compute_one(self, idx, k):
        """
        compute and return the ranking scores for one query.
        for definition of these metrics, please refer to
        https://www.cs.cornell.edu/courses/cs4300/2013fa/lectures/metrics-2-4pp.pdf

        Parameters
        ----------
        idx:
            the index of the query
        k:
            the cutoff value for NDCG, MAP, Recall, MRR, and Precision

        Returns
        -------
        Computed score.
        """
        precision, recall, mrr, mAP, ndcg = 0, 0, 0, 0, 0
        target, pred = self.results[idx][0], self.results[idx][1]

        # sort the ground truth and predictions in descending order
        sorted_target = dict(
            sorted(
                target.items(),
                key=operator.itemgetter(1),
                reverse=self.is_higher_better,
            )
        )
        sorted_pred = dict(
            sorted(
                pred.items(),
                key=operator.itemgetter(1),
                reverse=self.is_higher_better,
            )
        )
        sorted_target_values = list(sorted_target.values())
        sorted_pred_values = list(sorted_pred.values())

        # number of positive relevance in target
        # negative numbers and zero are considered as negative response
        num_pos_target = len([val for val in sorted_target_values if val > 0])

        at_k = k if num_pos_target > k else num_pos_target

        first_k_items_list = list(sorted_pred.items())[0:k]

        rank = 0
        hit_rank = []  # correctly retrieved items
        for key, value in first_k_items_list:
            if key in sorted_target and sorted_target[key] > 0:
                hit_rank.append(rank)
            rank += 1
        count = len(hit_rank)
        # compute the precision and recall
        precision = count / k
        recall = count / num_pos_target

        dcg = 0
        if hit_rank:  # not empty
            # compute the mean reciprocal rank
            mrr = 1 / (hit_rank[0] + 1)
            # compute the mean average precision
            mAP = np.sum([sorted_pred_values[rank] * (i + 1) / (rank + 1) for i, rank in enumerate(hit_rank)])
            # compute the discounted cumulative gain
            dcg = np.sum([1 / math.log(rank + 2, 2) for rank in hit_rank])

        # compute the ideal discounted cumulative gain
        idcg = np.sum([1 / math.log(i + 2, 2) for i in range(at_k)])
        # compute the normalized discounted cumulative gain
        ndcg = dcg / idcg
        mAP /= at_k

        return precision, recall, mrr, mAP, ndcg


def compute_ranking_score(
    results: Dict[str, Dict],
    qrel_dict: Dict[str, Dict],
    metrics: List[str],
    cutoffs: Optional[List[int]] = [5, 10, 20],
):
    """
    Compute the ranking metrics, e.g., NDCG, MAP, Recall, and Precision.
    TODO: Consider MRR.

    Parameters
    ----------
    results:
        The query/document ranking list by the model.
    qrel_dict:
        The groundtruth query and document relevance.
    metrics
        A list of metrics to compute.
    cutoffs:
        The cutoff values for NDCG, MAP, Recall, and Precision.

    Returns
    -------
    A dict of metric scores.
    """
    scores = {}
    evaluator = RankingMetrics(pred=results, target=qrel_dict)
    for k in cutoffs:
        scores.update(evaluator.compute(k=k))

    metric_results = dict()
    for k in cutoffs:
        for per_metric in metrics:
            if per_metric.lower() == NDCG:
                metric_results[f"{NDCG}@{k}"] = 0.0
            elif per_metric.lower() == MAP:
                metric_results[f"{MAP}@{k}"] = 0.0
            elif per_metric.lower() == RECALL:
                metric_results[f"{RECALL}@{k}"] = 0.0
            elif per_metric.lower() == PRECISION:
                metric_results[f"{PRECISION}@{k}"] = 0.0

    for k in cutoffs:
        for per_metric in metrics:
            if per_metric.lower() == NDCG:
                metric_results[f"{NDCG}@{k}"] = round(scores[f"{NDCG}@{k}"], 5)
            elif per_metric.lower() == MAP:
                metric_results[f"{MAP}@{k}"] = round(scores[f"{MAP}@{k}"], 5)
            elif per_metric.lower() == RECALL:
                metric_results[f"{RECALL}@{k}"] = round(scores[f"{RECALL}@{k}"], 5)
            elif per_metric.lower() == PRECISION:
                metric_results[f"{PRECISION}@{k}"] = round(scores[f"{PRECISION}@{k}"], 5)

    return metric_results


def infer_problem_type_by_eval_metric(eval_metric_name: str, problem_type: str):
    if eval_metric_name is not None and eval_metric_name.lower() in [
        "rmse",
        "r2",
        "pearsonr",
        "spearmanr",
    ]:
        if problem_type is None:
            logger.debug(
                f"Infer problem type to be a regression problem "
                f"since the evaluation metric is set as {eval_metric_name}."
            )
            problem_type = REGRESSION
        else:
            problem_prop = PROBLEM_TYPES_REG.get(problem_type)
            if NUMERICAL not in problem_prop.supported_label_type:
                raise ValueError(
                    f"The provided evaluation metric will require the problem "
                    f"to support label type = {NUMERICAL}. However, "
                    f"the provided problem type = {problem_type} only "
                    f"supports label type = {problem_prop.supported_label_type}."
                )

    return problem_type

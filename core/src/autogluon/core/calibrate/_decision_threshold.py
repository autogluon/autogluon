from __future__ import annotations

import logging
from typing import Callable, List, Union

import numpy as np

from ..constants import BINARY
from ..metrics import Scorer
from ..utils import get_pred_from_proba

logger = logging.getLogger(__name__)


# TODO: docstring
# TODO: Can use a smarter search strategy than brute force for faster speed, such as bayes opt.
def calibrate_decision_threshold(
    y: np.array,
    y_pred_proba: np.array,
    metric: Union[Callable, Scorer],
    metric_kwargs: dict | None = None,
    decision_thresholds: Union[int, List[float]] = 50,
    metric_name: str | None = None,
    verbose: bool = True,
) -> float:
    problem_type = BINARY
    assert len(y_pred_proba.shape) == 1
    assert len(y.shape) == 1
    assert len(y) == len(y_pred_proba)

    if metric_kwargs is None:
        metric_kwargs = dict()

    if isinstance(metric, Scorer):
        if metric_name is None:
            metric_name = metric.name
        if not metric.needs_pred:
            logger.warning(
                f'WARNING: The provided metric "{metric_name}" does not use class predictions for scoring, '
                f"and thus is invalid for decision threshold calibration. "
                f"Falling back to `decision_threshold=0.5`."
            )
            return 0.5
    metric_name_log = f" {metric_name}" if metric_name is not None else ""

    if isinstance(decision_thresholds, int):
        # Order thresholds by their proximity to 0.5
        num_checks_half = decision_thresholds
        num_checks = num_checks_half * 2
        decision_thresholds = [[0.5]] + [[0.5 - (i / num_checks), 0.5 + (i / num_checks)] for i in range(1, num_checks_half + 1)]
        decision_thresholds = [item for sublist in decision_thresholds for item in sublist]
    else:
        for decision_threshold in decision_thresholds:
            if decision_threshold > 1 or decision_threshold < 0:
                raise ValueError(f"Invalid decision_threshold specified: {decision_threshold} |" f" Decision thresholds must be between 0 and 1.")
    best_score_val = None
    best_threshold = None

    y_pred_val = get_pred_from_proba(
        y_pred_proba=y_pred_proba,
        problem_type=problem_type,
        decision_threshold=0.5,
    )
    # TODO: Avoid calling like this, reuse logic that works with weights + extra args
    score_val_baseline = metric(y, y_pred_val, **metric_kwargs)

    if verbose:
        logger.log(20, f"Calibrating decision threshold to optimize metric{metric_name_log} " f"| Checking {len(decision_thresholds)} thresholds...")
    for decision_threshold in decision_thresholds:
        extra_log = ""
        y_pred_val = get_pred_from_proba(
            y_pred_proba=y_pred_proba,
            problem_type=problem_type,
            decision_threshold=decision_threshold,
        )
        # TODO: Avoid calling like this, reuse logic that works with weights + extra args
        score_val = metric(y, y_pred_val, **metric_kwargs)

        if best_score_val is None or score_val > best_score_val:
            best_threshold = decision_threshold
            best_score_val = score_val
            extra_log = "\t| NEW BEST"
        elif best_score_val == score_val:
            # If the new threshold is closer to 0.5 than the previous threshold, prioritize it.
            if abs(decision_threshold - 0.5) < abs(best_threshold - 0.5):
                best_threshold = decision_threshold
                best_score_val = score_val
                extra_log = "\t| NEW BEST (Tie, using threshold that is closer to 0.5)"

        if verbose:
            logger.log(15, f"\tthreshold: {decision_threshold:.3f}\t| val: {score_val:.4f}{extra_log}")
    if verbose:
        logger.log(20, f"\tBase Threshold: {0.5:.3f}\t| val: {score_val_baseline:.4f}")
        logger.log(20, f"\tBest Threshold: {best_threshold:.3f}\t| val: {best_score_val:.4f}")
    return best_threshold

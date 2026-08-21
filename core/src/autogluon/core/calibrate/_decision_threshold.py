from __future__ import annotations

import logging
from typing import Callable, List, Union

import numpy as np
import pandas as pd

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
    decision_thresholds: int | List[float] = 25,
    secondary_decision_thresholds: int | None = 19,
    subsample_size: int | None = None,
    seed: int = 0,
    metric_name: str | None = None,
    verbose: bool = True,
) -> float:
    assert isinstance(decision_thresholds, (int, list)), (
        f"decision_thresholds must be int or List[float] (decision_thresholds={decision_thresholds})"
    )
    assert secondary_decision_thresholds is None or isinstance(secondary_decision_thresholds, int), (
        f"secondary_decision_thresholds must be int or None (secondary_decision_thresholds={secondary_decision_thresholds})"
    )

    problem_type = BINARY

    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(y_pred_proba, pd.Series):
        y_pred_proba = y_pred_proba.values
    assert len(y_pred_proba.shape) == 1
    assert len(y.shape) == 1

    num_samples_total = len(y)
    assert num_samples_total == len(y_pred_proba)

    if subsample_size is not None and subsample_size < num_samples_total:
        logger.log(20, f"Subsampling y to {subsample_size} samples to speedup threshold calibration...")
        rng = np.random.default_rng(seed=seed)
        subsample_indices = rng.choice(num_samples_total, subsample_size, replace=False)
        y = y[subsample_indices]
        y_pred_proba = y_pred_proba[subsample_indices]

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
        decision_thresholds = [[0.5]] + [
            [0.5 - (i / num_checks), 0.5 + (i / num_checks)] for i in range(1, num_checks_half + 1)
        ]
        decision_thresholds = [item for sublist in decision_thresholds for item in sublist]
    else:
        for decision_threshold in decision_thresholds:
            if decision_threshold > 1 or decision_threshold < 0:
                raise ValueError(
                    f"Invalid decision_threshold specified: {decision_threshold} |"
                    f" Decision thresholds must be between 0 and 1."
                )
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
        logger.log(
            20,
            f"Calibrating decision threshold to optimize metric{metric_name_log} "
            f"| Checking {len(decision_thresholds)} thresholds...",
        )
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

    chosen_threshold = best_threshold
    if secondary_decision_thresholds is not None:
        sorted_decision_thresholds = sorted(decision_thresholds)
        idx_chosen = sorted_decision_thresholds.index(chosen_threshold)
        idx_left = idx_chosen - 1
        idx_right = idx_chosen + 1
        secondary_thresholds = []
        if idx_left >= 0:
            delta_left = sorted_decision_thresholds[idx_chosen] - sorted_decision_thresholds[idx_left]
            secondary_thresholds += [
                chosen_threshold + delta_left * ((i + 1) / (secondary_decision_thresholds + 1))
                for i in range(secondary_decision_thresholds)
            ]
        if idx_right < len(decision_thresholds):
            delta_right = sorted_decision_thresholds[idx_chosen] - sorted_decision_thresholds[idx_right]
            secondary_thresholds += [
                chosen_threshold + delta_right * ((i + 1) / (secondary_decision_thresholds + 1))
                for i in range(secondary_decision_thresholds)
            ]
        if verbose and secondary_thresholds:
            logger.log(
                20,
                f"Calibrating decision threshold via fine-grained search "
                f"| Checking {len(secondary_thresholds)} thresholds...",
            )

        for decision_threshold in secondary_thresholds:
            extra_log = ""
            y_pred_val = get_pred_from_proba(
                y_pred_proba=y_pred_proba,
                problem_type=problem_type,
                decision_threshold=decision_threshold,
            )
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

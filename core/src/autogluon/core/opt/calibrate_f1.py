from __future__ import annotations

import numpy as np


def calibrate_decision_threshold_f1(
    y_true: np.array,
    y_pred: np.array,
    **kwargs,
) -> float:
    """
    Fast function to calculate the optimal decision threshold for the f1 metric.
    """
    args = np.argsort(y_pred)
    tp = y_true.sum()
    fs = (tp - np.cumsum(y_true[args[:-1]])) / np.arange(y_true.shape[0] + tp - 1, tp, -1)
    res_idx = np.argmax(fs)
    return (y_pred[args[res_idx]] + y_pred[args[res_idx + 1]]) / 2

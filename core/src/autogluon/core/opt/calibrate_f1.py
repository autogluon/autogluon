import numpy as np
from sklearn.metrics import precision_recall_curve


def calibrate_decision_threshold_f1(
    y_true: np.array,
    y_pred: np.array,
    **kwargs,
) -> float:
    """
    Fast algorithm to calculate the optimal decision threshold for the f1 metric in binary classification.

    Note: This implementation goes beyond naive implementations through various edge-case handling
    related to duplicate y_pred values and extreme values like y_pred = 0 and y_pred = 1.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * recall * precision / (recall + precision)
    if thresholds[0] == 0:
        # Handles edge case where threshold=0 is the first threshold and is the best f1 score, since we technically cannot predict class 0 with threshold=0
        threshold_opt_idx = np.nanargmax(f1_scores[1:]) + 1
    else:
        threshold_opt_idx = np.nanargmax(f1_scores)
    threshold_opt = thresholds[threshold_opt_idx]

    if threshold_opt_idx == 0:
        # Handles edge case where the lowest threshold is the best, can set threshold to 0.
        threshold_opt = 0
    else:
        # Set the threshold to a value inbetween the edge of the optimal threshold value and the previous threshold
        # for slightly better expected generalization performance on future y_pred values between the two observed thresholds.
        # For example, if we only observe thresholds 0.5 and 0.9, and 0.9 is the best threshold, we set the threshold to 0.7,
        # anticipating that it is more likely that 0.7 threshold will get a better test score than 0.9 threshold.
        threshold_opt += (thresholds[threshold_opt_idx-1] - threshold_opt) / 2
    return threshold_opt

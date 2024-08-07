import numpy as np


# TODO: If 0.5 gets matching score, use 0.5?
def calibrate_decision_threshold_accuracy(
    y_true: np.array,
    y_pred: np.array,
    **kwargs,
) -> float:
    """
    Fast algorithm to calculate the optimal decision threshold for the accuracy metric in binary classification.

    Note: This implementation goes beyond naive implementations through various edge-case handling
    related to duplicate y_pred values and extreme values like y_pred = 0 and y_pred = 1.
    """
    args = np.argsort(y_pred)[::-1]
    y_pred_sorted = y_pred[args]
    y_true_sorted = y_true[args]
    y_true_sorted[y_true_sorted == 0] = -1

    return _calibrate_weighted_accuracy(y_true_sorted=y_true_sorted, y_pred_sorted=y_pred_sorted)


def calibrate_decision_threshold_balanced_accuracy(
    y_true: np.array,
    y_pred: np.array,
    **kwargs,
) -> float:
    """
    Fast algorithm to calculate the optimal decision threshold for the balanced accuracy metric in binary classification.

    Note: This implementation goes beyond naive implementations through various edge-case handling
    related to duplicate y_pred values and extreme values like y_pred = 0 and y_pred = 1.
    """
    proportion_class_1 = np.sum(y_true) / y_true.shape[0]
    if proportion_class_1 == 1:
        return 0
    elif proportion_class_1 == 0:
        return 1

    args = np.argsort(y_pred)[::-1]
    y_pred_sorted = y_pred[args]
    y_true_sorted = y_true[args]
    y_true_sorted = y_true_sorted.astype(float)

    relative_frequency = proportion_class_1 / (1 - proportion_class_1)
    y_true_sorted[y_true_sorted == 0] = -relative_frequency

    return _calibrate_weighted_accuracy(y_true_sorted=y_true_sorted, y_pred_sorted=y_pred_sorted)


def _calibrate_weighted_accuracy(y_true_sorted: np.ndarray, y_pred_sorted: np.ndarray) -> float:
    y_pred_sorted_delta = np.ediff1d(y_pred_sorted)
    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

    # If multiple rows contain the same y_pred_proba, only keep the last one's cumsum to ensure accurate counts.
    y_true_sorted_cumsum[np.argwhere(y_pred_sorted_delta == 0)] = np.min(y_true_sorted_cumsum)

    optimal_idx = np.argmax(y_true_sorted_cumsum)
    if optimal_idx == 0 and y_true_sorted_cumsum[optimal_idx] < 0:
        # Edge case where the best score is achieved by always predicting class 0.
        threshold_opt = 1
    elif optimal_idx == y_pred_sorted.shape[0] - 1:
        # Edge case where the best score is achieved by always predicting class 1.
        threshold_opt = 0
    else:
        # Set the threshold to a value between the edge of the optimal threshold value and the previous threshold
        # for slightly better expected generalization performance on future y_pred values between the two observed thresholds.
        # For example, if we only observe thresholds 0.5 and 0.9, and 0.9 is the best threshold, we set the threshold to 0.7,
        # anticipating that it is more likely that 0.7 threshold will get a better test score than 0.9 threshold.
        threshold_opt = y_pred_sorted[optimal_idx] + (y_pred_sorted_delta[optimal_idx] / 2)

    return threshold_opt

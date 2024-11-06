import numpy as np
import pandas as pd


def compute_conformity_score(y_val_pred: np.ndarray, y_val: np.ndarray, quantile_levels: list):
    """
    Compute conformity scores based on validation set. This is only applicable to quantile regression problems.
    The scores are used to conformalize new quantile predictions.
    This is based on the paper 'Conformalized Quantile Regression (https://arxiv.org/abs/1905.03222)',
    and its implementation at 'https://github.com/yromano/cqr'.

    Parameters:
    -----------
    y_val_preds: numpy ndarray
        [num_samples x num_quantiles]
        Quantile estimates by model on validation set
    y_val: numpy ndarray
        [num_samples x 1]
        The target values to the validation set
    quantile_levels: list
        List of quantile levels
    Return:
    numpy ndarray: values to conformalize the new quantile predictions
    """

    num_samples = y_val.shape[0]
    y_val = y_val.reshape(-1)
    assert y_val_pred.shape[0] == num_samples
    assert y_val_pred.shape[1] == len(quantile_levels)

    if isinstance(y_val_pred, pd.DataFrame):
        y_val_pred = y_val_pred.to_numpy()

    conformalize_list = []
    for i, q in enumerate(quantile_levels):
        if q > 0.5:
            error_high = y_val - y_val_pred[:, i]
            error_high = np.sort(error_high, 0)
            index_high = int(np.ceil(q * (num_samples + 1))) - 1
            index_high = min(max(index_high, 0), num_samples - 1)
            conformalize = error_high[index_high]
        elif q < 0.5:
            error_low = y_val_pred[:, i] - y_val
            error_low = np.sort(error_low, 0)
            index_low = int(np.ceil((1 - q) * (num_samples + 1))) - 1
            index_low = min(max(index_low, 0), num_samples - 1)
            conformalize = -error_low[index_low]
        else:
            conformalize = 0.0
        conformalize_list.append(conformalize)
    return np.array(conformalize_list)

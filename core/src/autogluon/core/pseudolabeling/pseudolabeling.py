import numpy as np
import pandas as pd
from autogluon.core.constants import PROBLEM_TYPES_CLASSIFICATION


def filter_pseudo(y_pred_proba_og, problem_type,
                  min_proportion_prob: float = 0.05, max_proportion_prob: float = 0.6,
                  threshold: float = 0.95, proportion_sample: float = 0.3):
    """
    Takes in the predicted probabilities of the model and chooses the indices that meet
    a criteria to incorporate into training data. Criteria is determined by problem_type.
    If multiclass or binary will choose all rows with max prob over threshold. For regression
    chooses 30% of the labeled data randomly. This filter is used pseudo labeled data.

    Parameters:
    -----------
    y_pred_proba_og: The predicted probabilities from the current best model. If problem is
        'binary' or 'multiclass' then it's Panda series of predictive probs, if it's 'regression'
        then it's a scalar. Binary probs should be set to multiclass.
    min_proportion_prob: Minimum proportion of indices in y_pred_proba_og to select. The filter
        threshold will be automatically adjusted until at least min_proportion_prob of the predictions
        in y_pred_proba_og pass the filter. This ensures we return at least min_proportion_prob of the
        pseudolabeled data to augment the training set in pseudolabeling.
    max_proportion_prob: Maximum proportion of indices in y_pred_proba_og to select. The filter threshold
        will be automatically adjusted until at most max_proportion_prob of the predictions in y_pred_proba_og
        pass the filter. This ensures we return at most max_proportion_prob of the pseudolabeled data to augment
        the training set in pseudolabeling.
    threshold: This filter will only return those indices of y_pred_proba_og where the probability
        of the most likely class exceeds the given threshold value.
    proportion_sample: When problem_type is regression this is percent of pseudo data
        to incorporate into train. Rows selected randomly.

    Returns:
    --------
    pd.Series of indices that met pseudolabeling requirements
    """
    if problem_type in PROBLEM_TYPES_CLASSIFICATION:
        y_pred_proba_max = y_pred_proba_og.max(axis=1)
        curr_threshold = threshold
        curr_percentage = (y_pred_proba_max >= curr_threshold).mean()
        num_rows = len(y_pred_proba_max)

        if curr_percentage > max_proportion_prob or curr_percentage < min_proportion_prob:
            if curr_percentage > max_proportion_prob:
                num_rows_threshold = max(np.ceil(max_proportion_prob * num_rows), 1)
            else:
                num_rows_threshold = max(np.ceil(min_proportion_prob * num_rows), 1)
            curr_threshold = y_pred_proba_max.sort_values(ascending=False).iloc[int(num_rows_threshold) - 1]

        test_pseudo_indices = (y_pred_proba_max >= curr_threshold)
    else:
        test_pseudo_indices = pd.Series(data=False, index=y_pred_proba_og.index)
        test_pseudo_indices_true = test_pseudo_indices.sample(frac=proportion_sample, random_state=0)
        test_pseudo_indices[test_pseudo_indices_true.index] = True

    test_pseudo_indices = test_pseudo_indices[test_pseudo_indices]

    return test_pseudo_indices

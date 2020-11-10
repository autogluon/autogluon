import logging

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

from autogluon.core.utils import normalize_binary_probas, normalize_multi_probas
from autogluon.core.constants import BINARY, REGRESSION, MULTICLASS, SOFTCLASS
from autogluon.core.metrics import accuracy, root_mean_squared_error, Scorer

logger = logging.getLogger(__name__)


def get_pred_from_proba(y_pred_proba, problem_type=BINARY):
    if problem_type == BINARY:
        y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred_proba]
    elif problem_type == REGRESSION:
        y_pred = y_pred_proba
    else:
        y_pred = np.argmax(y_pred_proba, axis=1)
    return y_pred


def generate_train_test_split(X: DataFrame, y: Series, problem_type: str, test_size: float = 0.1, random_state=0) -> (DataFrame, DataFrame, Series, Series):
    if (test_size <= 0.0) or (test_size >= 1.0):
        raise ValueError("fraction of data to hold-out must be specified between 0 and 1")

    if problem_type in [REGRESSION, SOFTCLASS]:
        stratify = None
    else:
        stratify = y

    # TODO: Enable stratified split when y class would result in 0 samples in test.
    #  One approach: extract low frequency classes from X/y, add back (1-test_size)% to X_train, y_train, rest to X_test
    #  Essentially stratify the high frequency classes, random the low frequency (While ensuring at least 1 example stays for each low frequency in train!)
    #  Alternatively, don't test low frequency at all, trust it to work in train set. Risky, but highest quality for predictions.
    X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=test_size, shuffle=True, random_state=random_state, stratify=stratify)
    if problem_type != SOFTCLASS:
        y_train = pd.Series(y_train, index=X_train.index)
        y_test = pd.Series(y_test, index=X_test.index)
    else:
        y_train = pd.DataFrame(y_train, index=X_train.index)
        y_test = pd.DataFrame(y_test, index=X_test.index)
    return X_train, X_test, y_train, y_test


def normalize_pred_probas(y_predprob, problem_type, eps=1e-7):
    """ Remaps the predicted probabilities to ensure there are no zeros (needed for certain metrics like log-loss)
        and that no predicted probability exceeds [0,1] (eg. in distillation when classification is treated as regression).
        Args:
            y_predprob: 1D (for binary classification) or 2D (for multiclass) numpy array of predicted probabilities
            problem_type: We only consider normalization if the problem_type is one of: [BINARY, MULTICLASS, SOFTCLASS]
            eps: controls around how far from 0 remapped predicted probabilities should be (larger `eps` means predicted probabilities will lie further from 0).
    """
    if (problem_type == REGRESSION) and (len(y_predprob.shape) > 1) and (y_predprob.shape[1] > 1):
        problem_type = SOFTCLASS  # this was MULTICLASS problem converted to REGRESSION (as done in distillation)

    if problem_type in [BINARY, REGRESSION]:
        if len(y_predprob.shape) > 1 and min(y_predprob.shape) > 1:
            raise ValueError(f"cannot call normalize_pred_probas with problem_type={problem_type} and y_predprob.shape=={y_predprob.shape}")
        return normalize_binary_probas(y_predprob, eps)
    elif problem_type in [MULTICLASS, SOFTCLASS]:  # clip all probs below at eps and then renormalize
        if len(y_predprob.shape) == 1:
            return normalize_binary_probas(y_predprob, eps)
        else:
            return normalize_multi_probas(y_predprob, eps)
    else:
        raise ValueError(f"Invalid problem_type")


def infer_problem_type(y: Series, silent=False) -> str:
    """ Identifies which type of prediction problem we are interested in (if user has not specified).
        Ie. binary classification, multi-class classification, or regression.
    """
    if len(y) == 0:
        raise ValueError("provided labels cannot have length = 0")
    y = y.dropna()  # Remove missing values from y (there should not be any though as they were removed in Learner.general_data_processing())
    num_rows = len(y)

    unique_values = y.unique()

    MULTICLASS_LIMIT = 1000  # if numeric and class count would be above this amount, assume it is regression
    if num_rows > 1000:
        REGRESS_THRESHOLD = 0.05  # if the unique-ratio is less than this, we assume multiclass classification, even when labels are integers
    else:
        REGRESS_THRESHOLD = 0.1

    unique_count = len(unique_values)
    if unique_count == 2:
        problem_type = BINARY
        reason = "only two unique label-values observed"
    elif y.dtype.name in ['object', 'category']:
        problem_type = MULTICLASS
        reason = f"dtype of label-column == {y.dtype.name}"
    elif np.issubdtype(y.dtype, np.floating):
        unique_ratio = unique_count / float(num_rows)
        if (unique_ratio <= REGRESS_THRESHOLD) and (unique_count <= MULTICLASS_LIMIT):
            try:
                can_convert_to_int = np.array_equal(y, y.astype(int))
                if can_convert_to_int:
                    problem_type = MULTICLASS
                    reason = "dtype of label-column == float, but few unique label-values observed and label-values can be converted to int"
                else:
                    problem_type = REGRESSION
                    reason = "dtype of label-column == float and label-values can't be converted to int"
            except:
                problem_type = REGRESSION
                reason = "dtype of label-column == float and label-values can't be converted to int"
        else:
            problem_type = REGRESSION
            reason = "dtype of label-column == float and many unique label-values observed"
    elif np.issubdtype(y.dtype, np.integer):
        unique_ratio = unique_count / float(num_rows)
        if (unique_ratio <= REGRESS_THRESHOLD) and (unique_count <= MULTICLASS_LIMIT):
            problem_type = MULTICLASS  # TODO: Check if integers are from 0 to n-1 for n unique values, if they have a wide spread, it could still be regression
            reason = "dtype of label-column == int, but few unique label-values observed"
        else:
            problem_type = REGRESSION
            reason = "dtype of label-column == int and many unique label-values observed"
    else:
        raise NotImplementedError(f'label dtype {y.dtype} not supported!')
    if not silent:
        logger.log(25, f"AutoGluon infers your prediction problem is: '{problem_type}' (because {reason}).")

        # TODO: Move this outside of this function so it is visible even if problem type was not inferred.
        if problem_type in [BINARY, MULTICLASS]:
            if unique_count > 10:
                logger.log(20, f'\tFirst 10 (of {unique_count}) unique label values:  {list(unique_values[:10])}')
            else:
                logger.log(20, f'\t{unique_count} unique label values:  {list(unique_values)}')
        elif problem_type == REGRESSION:
            y_max = y.max()
            y_min = y.min()
            y_mean = y.mean()
            y_stddev = y.std()
            logger.log(20, f'\tLabel info (max, min, mean, stddev): ({y_max}, {y_min}, {round(y_mean, 5)}, {round(y_stddev, 5)})')

        logger.log(25, f"\tIf '{problem_type}' is not the correct problem_type, please manually specify the problem_type argument in fit() (You may specify problem_type as one of: {[BINARY, MULTICLASS, REGRESSION]})")
    return problem_type


def infer_eval_metric(problem_type: str) -> Scorer:
    """Infers appropriate default eval metric based on problem_type. Useful when no eval_metric was provided."""
    if problem_type == BINARY:
        return accuracy
    elif problem_type == MULTICLASS:
        return accuracy
    else:
        return root_mean_squared_error



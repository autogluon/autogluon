import logging
import math
import pickle
import time
import sys
from typing import Callable

import numpy as np
import pandas as pd
import psutil
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

from autogluon.core.utils import normalize_binary_probas, normalize_multi_probas, shuffle_df_rows
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


# Note: Do not send training data as input or the importances will be overfit.
# TODO: Improve time estimate (Currently pessimistic)
def compute_permutation_feature_importance(X: pd.DataFrame, y: pd.Series, predict_func: Callable[..., np.ndarray], eval_metric: Scorer, features: list = None, subsample_size=1000, num_shuffle_sets: int = None,
                                           predict_func_kwargs: dict = None, transform_func: Callable[..., pd.DataFrame] = None, transform_func_kwargs: dict = None,
                                           time_limit: float = None, silent=False) -> pd.DataFrame:
    """
    Computes a trained model's feature importance via permutation shuffling (https://explained.ai/rf-importance/).
    A feature's importance score represents the performance drop that results when the model makes predictions on a perturbed copy of the dataset where this feature's values have been randomly shuffled across rows.
    A feature score of 0.01 would indicate that the predictive performance dropped by 0.01 when the feature was randomly shuffled.
    The higher the score a feature has, the more important it is to the model's performance.
    If a feature has a negative score, this means that the feature is likely harmful to the final model, and a model trained with the feature removed would be expected to achieve a better predictive performance.
    Note that calculating feature importance can be a very computationally expensive process, particularly if the model uses hundreds or thousands of features. In many cases, this can take longer than the original model training.

    Note: For highly accurate stddev and z_score estimates, it is recommend to set `subsample_size` to at least 5,000 if possible and `num_shuffle_sets` to at least 10.

    Parameters
    ----------
    X : pd.DataFrame
        Validation data to permute when calculating feature importances.
        Do not use training data as it will result in overfit feature importances.
    y : pd.Series
        Label values of X. The index of X and y must align.
    predict_func : Callable[..., np.ndarray]
        Function that computes model predictions or prediction probabilities on input data.
        Output must be in the form of a numpy ndarray or pandas Series or DataFrame.
        Output `y_pred` must be in a form acceptable as input to `eval_metric(y, y_pred)`.
        If using a fit model object, this is typically `model.predict` or `model.predict_proba`, depending on the `eval_metric` being used.
        If `eval_metric.needs_pred==True`, use `model.predict`, otherwise use `model.predict_proba`.
    eval_metric : Scorer
        Object that computes a score given ground truth labels and predictions or prediction probabilities (depending on the type of metric).
        If using a fit model object, this is typically `model.eval_metric`.
        Feature importances will be based on the delta permutation shuffling has on the score produced by `eval_metric`.
    features : list, default None
        List of features to calculate importances for.
        If None, all features' importances will be calculated.
    subsample_size : int, default 1000
        The amount of data rows to sample when computing importances.
        Higher values will improve the quality of feature importance estimates, but linearly increase the runtime.
        If None, all provided data will be used.
    num_shuffle_sets : int, default None
        The number of different permutation shuffles of the data that are evaluated.
        Shuffle sets are generated with different random seeds and importances are averaged across all shuffle sets to get the final feature importance values.
        Higher values will improve the quality of feature importance estimates, but linearly increase the runtime.
        `subsample_size` should be increased before `num_shuffle_sets` if runtime is a concern.
        Defaults to 1 if `time_limit` is None or 10 if `time_limit` is specified.
        When `num_shuffle_sets` is greater than 1, feature importance standard deviation and z-score will additionally be computed by using the results of each shuffle set as samples.
    predict_func_kwargs : dict, default {}
        Keyword arguments to be appended to calls to `predict_func(X, **kwargs)`.
    transform_func : Callable[..., pd.DataFrame], default None
        Transformation function that takes the raw input and transforms it row-wise to the input expected by `predict_func`.
        Common examples include `model.preprocess` and `feature_generator.transform`.
        If None, then no transformation is done on the data prior to calling `predict_func`.
        This is necessary to compute importance of original data features in `X` prior to their transformation assuming `predict_func` does not perform the transformation already.
            Example: `transform_func` is necessary to compute the importance of a text feature prior to being transformed into ngrams by `transform_func` when `predict_func` expects ngram features as input.
    transform_func_kwargs : dict, default {}
        Keyword arguments to be appended to calls to `transform_func(X, **kwargs)`.
    time_limit : float, default None
        Time in seconds to limit the calculation of feature importance.
        If None, feature importance will calculate without early stopping.
        A minimum of 1 full shuffle set will always be evaluated. If a shuffle set evaluation takes longer than `time_limit`, the method will take the length of a shuffle set evaluation to return regardless of the `time_limit`.
        If `num_shuffle_sets==1`, `time_limit` will be ignored.
    silent : bool, default False
        Whether to suppress logging output

    Returns
    -------
    Pandas `pandas.DataFrame` of feature importance scores with 3 columns:
        index: The feature name.
        'importance': The feature importance score.
        'stddev': The standard deviation of the feature importance score. If None, then not enough num_shuffle_sets were used to calculate a variance.
        'z_score': The z-score of the feature importance score. Equivalent to 'importance' / 'stddev'.
            A z-score of +4 or higher indicates that the feature is almost certainly useful and should be kept.
            A z-score of +2 indicates that the feature has a 97.5% chance of improving model quality when present.
            A z-score that is 0 or negative indicates that the feature can likely be dropped without negative impact to model quality.
            A z-score of None indicates that the feature's stddev was None or that the model predictions were never impacted by the feature (importance 0 and stddev 0). This indicates that the feature can safely be dropped.
            A z-score of +inf or -inf indicates that `subsample_size` and/or `num_shuffle_sets` were too small to calculate variance for the feature (non-zero importance with zero stddev).
    """
    if num_shuffle_sets is None:
        if time_limit is not None:
            num_shuffle_sets = 10
        else:
            num_shuffle_sets = 1

    time_start = time.time()
    if predict_func_kwargs is None:
        predict_func_kwargs = dict()
    if transform_func_kwargs is None:
        transform_func_kwargs = dict()
    if features is None:
        features = list(X.columns)
    feature_count = len(features)

    if (subsample_size is not None) and (len(X) > subsample_size):
        # Reset index to avoid error if duplicated indices.
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        X = X.sample(subsample_size, random_state=0)
        y = y.loc[X.index]

    if not silent:
        logging_message = f'Computing raw permutation feature importance for {feature_count} features using {len(X)} rows with {num_shuffle_sets} shuffle sets...'
        if time_limit is not None:
            logging_message = f'{logging_message} Time limit: {time_limit}s...'
        logger.log(20, logging_message)

    time_start_score = time.time()

    X_transformed = X if transform_func is None else transform_func(X, **transform_func_kwargs)
    y_pred = predict_func(X_transformed, **predict_func_kwargs)
    score_baseline = eval_metric(y, y_pred)
    time_score = time.time() - time_start_score

    if not silent:
        time_estimated = ((feature_count + 1) * time_score) * num_shuffle_sets + time_start_score - time_start
        time_estimated_per_set = time_estimated / num_shuffle_sets
        logger.log(20, f'\t{round(time_estimated, 2)}s\t= Expected runtime ({round(time_estimated_per_set, 2)}s per shuffle set)')

    row_count = X.shape[0]

    # calculating maximum number of features, which is safe to process parallel
    X_memory_ratio_max = 0.2
    compute_count_max = 200

    X_size_bytes = sys.getsizeof(pickle.dumps(X, protocol=4))
    if transform_func is not None:
        X_size_bytes += sys.getsizeof(pickle.dumps(X_transformed, protocol=4))
    available_mem = psutil.virtual_memory().available
    X_memory_ratio = X_size_bytes / available_mem

    compute_count_safe = math.floor(X_memory_ratio_max / X_memory_ratio)
    compute_count = max(1, min(compute_count_max, compute_count_safe))
    compute_count = min(compute_count, feature_count)

    # creating copy of original data N=compute_count times for parallel processing
    X_raw = pd.concat([X.copy() for _ in range(compute_count)], ignore_index=True, sort=False).reset_index(drop=True)

    time_permutation_start = time.time()
    permutation_importance_dict_list = []
    # TODO: Can speedup shuffle_repeats by incorporating into X_raw (do multiple repeats in a single predict call)
    shuffle_repeats_completed = 0
    for shuffle_repeat in range(num_shuffle_sets):
        permutation_importance_dict = dict()
        X_shuffled = shuffle_df_rows(X=X, seed=shuffle_repeat)
        for i in range(0, feature_count, compute_count):
            parallel_computed_features = features[i:i + compute_count]

            # if final iteration, leaving only necessary part of X_raw
            num_features_processing = len(parallel_computed_features)
            final_iteration = i + num_features_processing == feature_count
            if (num_features_processing < compute_count) and final_iteration:
                X_raw = X_raw.loc[:row_count * num_features_processing - 1]

            row_index = 0
            for feature in parallel_computed_features:
                row_index_end = row_index + row_count
                X_raw.loc[row_index:row_index_end - 1, feature] = X_shuffled[feature].values
                row_index = row_index_end

            X_raw_transformed = X_raw if transform_func is None else transform_func(X_raw, **transform_func_kwargs)
            y_pred = predict_func(X_raw_transformed, **predict_func_kwargs)

            row_index = 0
            for feature in parallel_computed_features:
                # calculating importance score for given feature
                row_index_end = row_index + row_count
                y_pred_cur = y_pred[row_index:row_index_end]
                score = eval_metric(y, y_pred_cur)
                permutation_importance_dict[feature] = score_baseline - score

                if not final_iteration:
                    # resetting to original values for processed feature
                    X_raw.loc[row_index:row_index_end - 1, feature] = X[feature].values

                row_index = row_index_end
        permutation_importance_dict_list.append(permutation_importance_dict)
        shuffle_repeats_completed = shuffle_repeat + 1
        if time_limit is not None and shuffle_repeat != (num_shuffle_sets - 1):
            time_now = time.time()
            time_left = time_limit - (time_now - time_start)
            time_permutation_average = (time_now - time_permutation_start) / (shuffle_repeat + 1)
            if time_left < (time_permutation_average * 1.1):
                if not silent:
                    logger.log(20, f'\tEarly stopping feature importance calculation before all shuffle sets have completed due to lack of time...')
                break
    permutation_importance_dict = dict()
    permutation_importance_stddev_dict = dict()
    permutation_importance_z_score_dict = dict()
    for feature in features:
        feature_shuffle_scores = [permutation_importance_dict_repeat[feature] for permutation_importance_dict_repeat in permutation_importance_dict_list]
        permutation_importance_dict[feature] = np.mean(feature_shuffle_scores)
        if len(feature_shuffle_scores) > 1:
            permutation_importance_stddev_dict[feature] = np.std(feature_shuffle_scores, ddof=1)
        else:
            permutation_importance_stddev_dict[feature] = None
        if permutation_importance_stddev_dict[feature] is not None and permutation_importance_stddev_dict[feature] != 0:
            permutation_importance_z_score_dict[feature] = permutation_importance_dict[feature] / permutation_importance_stddev_dict[feature]
        elif permutation_importance_stddev_dict[feature] is not None:  # stddev = 0
            if permutation_importance_dict[feature] == 0:
                permutation_importance_z_score_dict[feature] = None
            elif permutation_importance_dict[feature] > 0:
                permutation_importance_z_score_dict[feature] = np.inf
            else:  # < 0
                permutation_importance_z_score_dict[feature] = -np.inf
        else:
            permutation_importance_z_score_dict[feature] = None

    feature_importances = pd.Series(permutation_importance_dict).sort_values(ascending=False)
    feature_importances_stddev = pd.Series(permutation_importance_stddev_dict).sort_values(ascending=False)
    feature_importances_z_score = pd.Series(permutation_importance_z_score_dict).sort_values(ascending=False)

    fi_df = feature_importances.to_frame(name='importance')
    fi_df['stddev'] = feature_importances_stddev
    fi_df['z_score'] = feature_importances_z_score

    if not silent:
        logger.log(20, f'\t{round(time.time() - time_start, 2)}s\t= Actual runtime (Completed {shuffle_repeats_completed} of {num_shuffle_sets} shuffle sets)')

    return fi_df

import logging
import multiprocessing
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold, train_test_split

from .constants import BINARY, REGRESSION, MULTICLASS, SOFTCLASS
from ..metrics import accuracy, root_mean_squared_error, Scorer

logger = logging.getLogger(__name__)


def get_pred_from_proba(y_pred_proba, problem_type=BINARY):
    if problem_type == BINARY:
        y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred_proba]
    elif problem_type == REGRESSION:
        y_pred = y_pred_proba
    else:
        y_pred = np.argmax(y_pred_proba, axis=1)
    return y_pred


def generate_kfold(X, y=None, n_splits=5, random_state=0, stratified=False, n_repeats=1):
    if stratified and (y is not None):
        if n_repeats > 1:
            kf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        else:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        kf.get_n_splits(X, y)
        return [[train_index, test_index] for train_index, test_index in kf.split(X, y)]
    else:
        if n_repeats > 1:
            kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        kf.get_n_splits(X)
        return [[train_index, test_index] for train_index, test_index in kf.split(X)]


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


def setup_outputdir(output_directory):
    if output_directory is None:
        utcnow = datetime.utcnow()
        timestamp = utcnow.strftime("%Y%m%d_%H%M%S")
        output_directory = f"AutogluonModels/ag-{timestamp}{os.path.sep}"
        for i in range(1, 1000):
            try:
                os.makedirs(output_directory, exist_ok=False)
                break
            except FileExistsError as e:
                output_directory = f"AutogluonModels/ag-{timestamp}-{i:03d}{os.path.sep}"
        else:
            raise RuntimeError("more than 1000 jobs launched in the same second")
        logger.log(25, f"No output_directory specified. Models will be saved in: {output_directory}")
    output_directory = os.path.expanduser(output_directory)  # replace ~ with absolute path if it exists
    if output_directory[-1] != os.path.sep:
        output_directory = output_directory + os.path.sep
    return output_directory


def setup_compute(nthreads_per_trial, ngpus_per_trial):
    if nthreads_per_trial is None:
        nthreads_per_trial = multiprocessing.cpu_count()  # Use all of processing power / trial by default. To use just half: # int(np.floor(multiprocessing.cpu_count()/2))

    if ngpus_per_trial is None:
        ngpus_per_trial = 0  # do not use GPU by default
    elif ngpus_per_trial > 1:
        ngpus_per_trial = 1
        logger.debug("tabular_prediction currently doesn't use >1 GPU per training run. ngpus_per_trial set = 1")
    return nthreads_per_trial, ngpus_per_trial


def setup_trial_limits(time_limits, num_trials, hyperparameters):
    """ Adjust default time limits / num_trials """
    if num_trials is None:
        if time_limits is None:
            time_limits = 10 * 60  # run for 10min by default
        time_limits /= float(len(hyperparameters))  # each model type gets half the available time
        num_trials = 1000  # run up to 1000 trials (or as you can within the given time_limits)
    elif time_limits is None:
        time_limits = int(1e6)  # user only specified num_trials, so run all of them regardless of time-limits
    else:
        time_limits /= float(len(hyperparameters))  # each model type gets half the available time

    if time_limits <= 10:  # threshold = 10sec, ie. too little time to run >1 trial.
        num_trials = 1
    time_limits *= 0.9  # reduce slightly to account for extra time overhead
    return time_limits, num_trials


def dd_list():
    return defaultdict(list)


def get_leaderboard_pareto_frontier(leaderboard: DataFrame, score_col='score_val', inference_time_col='pred_time_val_full') -> DataFrame:
    """
    Given a set of models, returns in ranked order from best score to worst score models which satisfy the criteria:
    1. No other model in the set has both a lower inference time and a better or equal score.

    :param leaderboard: Leaderboard DataFrame of model info containing score_col and inference_time_col
    :param score_col: Column name in leaderboard of model score values
    :param inference_time_col: Column name in leaderboard of model inference times
    :return: Subset of the original leaderboard DataFrame containing only models that are a valid optimal choice at different valuations of score and inference time.
    """
    leaderboard = leaderboard.sort_values(by=[score_col, inference_time_col], ascending=[False, True]).reset_index(drop=True)
    leaderboard_unique = leaderboard.drop_duplicates(subset=[score_col])

    pareto_frontier = []
    inference_time_min = None
    for index, row in leaderboard_unique.iterrows():
        if row[inference_time_col] is None or row[score_col] is None:
            pass
        elif (inference_time_min is None) or (row[inference_time_col] < inference_time_min):
            inference_time_min = row[inference_time_col]
            pareto_frontier.append(index)
    leaderboard_pareto_frontier = leaderboard_unique.loc[pareto_frontier].reset_index(drop=True)
    return leaderboard_pareto_frontier


def shuffle_df_rows(X: DataFrame, seed=0, reset_index=True):
    """Returns DataFrame with rows shuffled based on seed value."""
    row_count = X.shape[0]
    np.random.seed(seed)
    rand_shuffle = np.random.randint(0, row_count, size=row_count)
    X_shuffled = X.iloc[rand_shuffle]
    if reset_index:
        X_shuffled.reset_index(inplace=True, drop=True)
    return X_shuffled


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


def normalize_binary_probas(y_predprob, eps):
    """ Remaps the predicted probabilities to open interval (0,1) while maintaining rank order """
    (pmin,pmax) = (eps, 1-eps)  # predicted probs outside this range will be remapped into (0,1)
    which_toobig = y_predprob > pmax
    if np.sum(which_toobig) > 0:  # remap overly large probs
        y_predprob = np.logical_not(which_toobig)*y_predprob + which_toobig*(1-(eps*np.exp(-(y_predprob-pmax))))
    which_toosmall = y_predprob < pmin
    if np.sum(which_toosmall) > 0:  # remap overly small probs
        y_predprob = np.logical_not(which_toosmall)*y_predprob + which_toosmall*eps*np.exp(-(pmin-y_predprob))
    return y_predprob


def normalize_multi_probas(y_predprob, eps):
    """ Remaps the predicted probabilities to lie in (0,1) where eps controls how far from 0 smallest class-probability lies """
    min_predprob = np.min(y_predprob)
    if min_predprob < 0:  # ensure nonnegative rows
        most_negative_rowvals = np.clip(np.min(y_predprob, axis=1), a_min=None, a_max=0)
        y_predprob = y_predprob - most_negative_rowvals[:,None]
    if min_predprob < eps:
        y_predprob = np.clip(y_predprob, a_min=eps, a_max=None)  # ensure no entries < eps
        y_predprob = y_predprob / y_predprob.sum(axis=1, keepdims=1)  # renormalize
    return y_predprob


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


def default_holdout_frac(num_train_rows, hyperparameter_tune=False):
    """ Returns default holdout_frac used in fit().
        Between row count 5,000 and 25,000 keep 0.1 holdout_frac, as we want to grow validation set to a stable 2500 examples.
    """
    if num_train_rows < 5000:
        holdout_frac = max(0.1, min(0.2, 500.0 / num_train_rows))
    else:
        holdout_frac = max(0.01, min(0.1, 2500.0 / num_train_rows))

    if hyperparameter_tune:
        holdout_frac = min(0.2, holdout_frac * 2)  # We want to allocate more validation data for HPO to avoid overfitting

    return holdout_frac


def augment_rare_classes(X, label, threshold):
    """ Use this method when using certain eval_metrics like log_loss, for which no classes may be filtered out.
        This method will augment dataset with additional examples of rare classes.
    """
    class_counts = X[label].value_counts()
    class_counts_invalid = class_counts[class_counts < threshold]
    if len(class_counts_invalid) == 0:
        logger.debug("augment_rare_classes did not need to duplicate any data from rare classes")
        return X

    aug_df = None
    for clss, n_clss in class_counts_invalid.iteritems():
        n_toadd = threshold - n_clss
        clss_df = X.loc[X[label] == clss]
        if aug_df is None:
            aug_df = clss_df[:0].copy()
        duplicate_times = int(np.floor(n_toadd / n_clss))
        remainder = n_toadd % n_clss
        new_df = clss_df.copy()
        new_df = new_df[:remainder]
        while duplicate_times > 0:
            logger.debug(f"Duplicating data from rare class: {clss}")
            duplicate_times -= 1
            new_df = new_df.append(clss_df.copy())
        aug_df = aug_df.append(new_df.copy())

    X = X.append(aug_df)
    class_counts = X[label].value_counts()
    class_counts_invalid = class_counts[class_counts < threshold]
    if len(class_counts_invalid) > 0:
        raise RuntimeError("augment_rare_classes failed to produce enough data from rare classes")
    logger.log(15, "Replicated some data from rare classes in training set because eval_metric requires all classes")
    return X

import logging
import multiprocessing
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
from pandas import DataFrame
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold


logger = logging.getLogger(__name__)


def get_cpu_count():
    return multiprocessing.cpu_count()


def get_gpu_count():
    from .nvutil import cudaInit, cudaDeviceGetCount, cudaShutdown
    if not cudaInit(): return 0
    gpu_count = cudaDeviceGetCount()
    cudaShutdown()
    return gpu_count


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


def setup_outputdir(output_directory, warn_if_exist=True):
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
    elif warn_if_exist:
        try:
            os.makedirs(output_directory, exist_ok=False)
        except FileExistsError as e:
            logger.warning(f'Warning: output_directory already exists! This predictor may overwrite an existing predictor! output_directory="{output_directory}"')
    output_directory = os.path.expanduser(output_directory)  # replace ~ with absolute path if it exists
    if output_directory[-1] != os.path.sep:
        output_directory = output_directory + os.path.sep
    return output_directory


def setup_compute(nthreads_per_trial, ngpus_per_trial):
    if nthreads_per_trial is None or nthreads_per_trial == 'auto':  # FIXME: Use 'auto' downstream
        nthreads_per_trial = multiprocessing.cpu_count()  # Use all of processing power / trial by default. To use just half: # int(np.floor(multiprocessing.cpu_count()/2))

    if ngpus_per_trial is None or ngpus_per_trial == 'auto':  # FIXME: Use 'auto' downstream
        ngpus_per_trial = 0  # do not use GPU by default
    elif ngpus_per_trial > 1:
        ngpus_per_trial = 1
        logger.debug("tabular_prediction currently doesn't use >1 GPU per training run. ngpus_per_trial set = 1")
    return nthreads_per_trial, ngpus_per_trial


def setup_trial_limits(time_limit, num_trials, hyperparameters):
    """ Adjust default time limits / num_trials """
    if num_trials is None:
        if time_limit is None:
            time_limit = 10 * 60  # run for 10min by default
        time_limit /= float(len(hyperparameters))  # each model type gets half the available time
        num_trials = 1000  # run up to 1000 trials (or as you can within the given time_limit)
    elif time_limit is None:
        time_limit = int(1e6)  # user only specified num_trials, so run all of them regardless of time_limit
    else:
        time_limit /= float(len(hyperparameters))  # each model type gets half the available time

    if time_limit <= 10:  # threshold = 10sec, ie. too little time to run >1 trial.
        num_trials = 1
    time_limit *= 0.9  # reduce slightly to account for extra time overhead
    return time_limit, num_trials


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

    missing_classes = []
    for clss, n_clss in class_counts_invalid.iteritems():
        if n_clss == 0:
            missing_classes.append(clss)
    if missing_classes:
        logger.warning(f'WARNING: Classes were found that have 0 training examples, and may lead to downstream issues. '
                       f'Consider either providing data for these classes or removing them from the class categories. '
                       f'These classes will be ignored: {missing_classes}')
        class_counts_invalid = class_counts_invalid[~class_counts_invalid.index.isin(set(missing_classes))]

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
    class_counts_invalid = class_counts_invalid[~class_counts_invalid.index.isin(set(missing_classes))]
    if len(class_counts_invalid) > 0:
        raise RuntimeError("augment_rare_classes failed to produce enough data from rare classes")
    logger.log(15, "Replicated some data from rare classes in training set because eval_metric requires all classes")
    return X

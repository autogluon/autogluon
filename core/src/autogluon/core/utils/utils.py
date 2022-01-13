import logging
import multiprocessing
import subprocess
import math
import pickle
import time
import random
import sys
from typing import Callable, List

import numpy as np
import pandas as pd
import psutil
import scipy.stats
from pandas import DataFrame, Series
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, LeaveOneGroupOut
from sklearn.model_selection import train_test_split

from .miscs import warning_filter
from ..constants import BINARY, REGRESSION, MULTICLASS, SOFTCLASS, QUANTILE
from ..metrics import accuracy, root_mean_squared_error, pinball_loss, Scorer

logger = logging.getLogger(__name__)


def get_cpu_count():
    return multiprocessing.cpu_count()


def get_gpu_count_all():
    """
    Attempts to get number of GPUs available for use via multiple means.
    """
    # FIXME: update to use only torch for TIMM or find a better GPU detection strategy
    # FIXME: get_gpu_count by itself doesn't always work for Windows
    num_gpus = _get_gpu_count_cuda()
    if num_gpus == 0:
        num_gpus = get_gpu_count_mxnet()
        if num_gpus == 0:
            num_gpus = get_gpu_count_torch()
    return num_gpus


def _get_gpu_count_cuda():
    # FIXME: Sometimes doesn't detect GPU on Windows
    # FIXME: Doesn't ensure the GPUs are actually usable by the model (MXNet, PyTorch, etc.)
    from .nvutil import cudaInit, cudaDeviceGetCount, cudaShutdown
    if not cudaInit(): return 0
    gpu_count = cudaDeviceGetCount()
    cudaShutdown()
    return gpu_count


def get_gpu_count_mxnet():
    try:
        import mxnet
        num_gpus = mxnet.context.num_gpus()
    except Exception:
        num_gpus = 0
    return num_gpus


def get_gpu_count_torch():
    try:
        import torch
        num_gpus = torch.cuda.device_count()
    except Exception:
        num_gpus = 0
    return num_gpus


class CVSplitter:
    def __init__(self,
                 splitter_cls=None,
                 n_splits=5,
                 n_repeats=1,
                 random_state=0,
                 stratified=False,
                 groups=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.stratified = stratified
        self.groups = groups
        if splitter_cls is None:
            splitter_cls = self._get_splitter_cls()
        self._splitter = self._get_splitter(splitter_cls)

    def _get_splitter_cls(self):
        if self.groups is not None:
            num_groups = len(self.groups.unique())
            if self.n_repeats != 1:
                raise AssertionError(f'n_repeats must be 1 when split groups are specified. (n_repeats={self.n_repeats})')
            self.n_splits = num_groups
            splitter_cls = LeaveOneGroupOut
            # pass
        elif self.stratified:
            splitter_cls = RepeatedStratifiedKFold
        else:
            splitter_cls = RepeatedKFold
        return splitter_cls

    def _get_splitter(self, splitter_cls):
        if splitter_cls == LeaveOneGroupOut:
            return splitter_cls()
        elif splitter_cls in [RepeatedKFold, RepeatedStratifiedKFold]:
            return splitter_cls(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)
        else:
            raise AssertionError(f'{splitter_cls} is not supported as a valid `splitter_cls` input to CVSplitter.')

    def split(self, X, y):
        if isinstance(self._splitter, RepeatedStratifiedKFold):
            # FIXME: There is a bug in sklearn that causes an incorrect ValueError if performing stratification and all classes have fewer than n_splits samples.
            #  This is hacked by adding a dummy class with n_splits samples, performing the kfold split, then removing the dummy samples from all resulting indices.
            #  This is very inefficient and complicated and ideally should be fixed in sklearn.
            with warning_filter():
                try:
                    out = [[train_index, test_index] for train_index, test_index in self._splitter.split(X, y)]
                except:
                    y_dummy = pd.concat([y, pd.Series([-1] * self.n_splits)], ignore_index=True)
                    X_dummy = pd.concat([X, X.head(self.n_splits)], ignore_index=True)
                    invalid_index = set(list(y_dummy.tail(self.n_splits).index))
                    out = [[train_index, test_index] for train_index, test_index in self._splitter.split(X_dummy, y_dummy)]
                    len_out = len(out)
                    for i in range(len_out):
                        train_index, test_index = out[i]
                        out[i][0] = [index for index in train_index if index not in invalid_index]
                        out[i][1] = [index for index in test_index if index not in invalid_index]
            return out
        else:
            return [[train_index, test_index] for train_index, test_index in self._splitter.split(X, y, groups=self.groups)]


def setup_compute(nthreads_per_trial, ngpus_per_trial):
    if nthreads_per_trial is None or nthreads_per_trial == 'all':
        nthreads_per_trial = get_cpu_count()  # Use all of processing power / trial by default. To use just half: # int(np.floor(multiprocessing.cpu_count()/2))
    if ngpus_per_trial is None:
        ngpus_per_trial = 0  # do not use GPU by default
    elif ngpus_per_trial == 'all':
        ngpus_per_trial = get_gpu_count_all()
    if not isinstance(nthreads_per_trial, int) and nthreads_per_trial != 'auto':
        raise ValueError(f'nthreads_per_trial must be an integer or "auto": nthreads_per_trial = {nthreads_per_trial}')
    if not isinstance(ngpus_per_trial, int) and ngpus_per_trial != 'auto':
        raise ValueError(f'ngpus_per_trial must be an integer or "auto": ngpus_per_trial = {ngpus_per_trial}')
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

    # Ensure new samples generated via augmentation have unique indices
    aug_df = aug_df.reset_index(drop=True)
    aug_df_len = len(aug_df)
    X_index_aug_start = X.index.max() + 1
    aug_index = [X_index_aug_start + i for i in range(aug_df_len)]
    aug_df.index = aug_index

    logger.log(20, f"Duplicated {len(aug_df)} samples from {len(class_counts_invalid)} rare classes in training set because eval_metric requires all classes have at least {threshold} samples.")

    X = X.append(aug_df)
    class_counts = X[label].value_counts()
    class_counts_invalid = class_counts[class_counts < threshold]
    class_counts_invalid = class_counts_invalid[~class_counts_invalid.index.isin(set(missing_classes))]
    if len(class_counts_invalid) > 0:
        raise RuntimeError("augment_rare_classes failed to produce enough data from rare classes")
    return X


def get_pred_from_proba_df(y_pred_proba, problem_type=BINARY):
    """From input DataFrame of pred_proba, return Series of pred"""
    if problem_type == REGRESSION:
        y_pred = y_pred_proba
    elif problem_type == QUANTILE:
        y_pred = y_pred_proba
    else:
        y_pred = y_pred_proba.idxmax(axis=1)
    return y_pred


def get_pred_from_proba(y_pred_proba, problem_type=BINARY):
    if problem_type == BINARY:
        y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred_proba]
    elif problem_type == REGRESSION:
        y_pred = y_pred_proba
    elif problem_type == QUANTILE:
        y_pred = y_pred_proba
    else:
        y_pred = []
        if not len(y_pred_proba) == 0:
            y_pred = np.argmax(y_pred_proba, axis=1)
    return y_pred


def generate_train_test_split(X: DataFrame,
                              y: Series,
                              problem_type: str,
                              test_size: float = 0.1,
                              random_state=0,
                              min_cls_count_train=1) -> (DataFrame, DataFrame, Series, Series):
    if (test_size <= 0.0) or (test_size >= 1.0):
        raise ValueError("fraction of data to hold-out must be specified between 0 and 1")

    X_split = X
    y_split = y
    if problem_type in [BINARY, MULTICLASS]:
        stratify = y
    else:
        stratify = None

    # This code block is necessary to avoid crashing when performing a stratified split and only 1 sample exists for a class.
    # This code will ensure that the sample will be part of the train split, meaning the test split will have 0 samples of the rare class.
    rare_indices = None
    if stratify is not None:
        cls_counts = y.value_counts()
        cls_counts_invalid = cls_counts[cls_counts < min_cls_count_train]

        if len(cls_counts_invalid) > 0:
            logger.error(f'ERROR: Classes have too few samples to split the data! At least {min_cls_count_train} are required.')
            logger.error(cls_counts_invalid)
            raise AssertionError('Not enough data to split data into train and val without dropping classes!')
        elif min_cls_count_train < 2:
            cls_counts_rare = cls_counts[cls_counts < 2]
            if len(cls_counts_rare) > 0:
                cls_counts_rare_val = set(cls_counts_rare.index)
                y_rare = y[y.isin(cls_counts_rare_val)]
                rare_indices = list(y_rare.index)
                X_split = X.drop(index=rare_indices)
                y_split = y.drop(index=rare_indices)
                stratify = y_split

    X_train, X_test, y_train, y_test = train_test_split(X_split, y_split.values, test_size=test_size, shuffle=True, random_state=random_state, stratify=stratify)
    if problem_type != SOFTCLASS:
        y_train = pd.Series(y_train, index=X_train.index)
        y_test = pd.Series(y_test, index=X_test.index)
    else:
        y_train = pd.DataFrame(y_train, index=X_train.index)
        y_test = pd.DataFrame(y_test, index=X_test.index)

    if rare_indices:
        X_train = X_train.append(X.loc[rare_indices])
        y_train = y_train.append(y.loc[rare_indices])

    if problem_type in [BINARY, MULTICLASS]:
        class_counts_dict_orig = y.value_counts().to_dict()
        class_counts_dict = y_train.value_counts().to_dict()
        class_counts_dict_test = y_test.value_counts().to_dict()

        indices_to_move = []
        random_state_init = random.getstate()
        random.seed(random_state)
        for cls in class_counts_dict_orig.keys():
            count = class_counts_dict.get(cls, 0)
            if count >= min_cls_count_train:
                continue
            count_test = class_counts_dict_test.get(cls, 0)
            if count + count_test < min_cls_count_train:
                raise AssertionError('Not enough data to split data into train and val without dropping classes!')
            count_to_move = min_cls_count_train - count
            indices_of_cls_test = list(y_test[y_test == cls].index)
            indices_to_move_cls = random.sample(indices_of_cls_test, count_to_move)
            indices_to_move += indices_to_move_cls
        random.setstate(random_state_init)
        if indices_to_move:
            y_test_moved = y_test.loc[indices_to_move].copy()
            X_test_moved = X_test.loc[indices_to_move].copy()
            y_train = pd.concat([y_train, y_test_moved])
            X_train = pd.concat([X_train, X_test_moved])
            y_test = y_test.drop(index=indices_to_move)
            X_test = X_test.drop(index=indices_to_move)
        y_train.name = y_split.name
        y_test.name = y_split.name
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
    elif y.dtype.name in ['object', 'category', 'string']:
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

        logger.log(25, f"\tIf '{problem_type}' is not the correct problem_type, please manually specify the problem_type parameter during predictor init "
                       f"(You may specify problem_type as one of: {[BINARY, MULTICLASS, REGRESSION]})")
    return problem_type


def infer_eval_metric(problem_type: str) -> Scorer:
    """Infers appropriate default eval metric based on problem_type. Useful when no eval_metric was provided."""
    if problem_type == BINARY:
        return accuracy
    elif problem_type == MULTICLASS:
        return accuracy
    elif problem_type == QUANTILE:
        return pinball_loss
    else:
        return root_mean_squared_error


def extract_column(X, col_name):
    """Extract specified column from dataframe. """
    if col_name is None or col_name not in list(X.columns):
        return X, None
    w = X[col_name].copy()
    X = X.drop(col_name, axis=1)
    return X, w


def compute_weighted_metric(y, y_pred, metric, weights, weight_evaluation=None, **kwargs):
    """ Report weighted metric if: weights is not None, weight_evaluation=True, and the given metric supports sample weights.
        If weight_evaluation=None, it will be set to False if weights=None, True otherwise.
    """
    if not metric.needs_quantile:
        kwargs.pop('quantile_levels', None)
    if weight_evaluation is None:
        weight_evaluation = not (weights is None)
    if weight_evaluation and weights is None:
        raise ValueError("Sample weights cannot be None when weight_evaluation=True.")
    if not weight_evaluation:
        return metric(y, y_pred, **kwargs)
    try:
        weighted_metric = metric(y, y_pred, sample_weight=weights, **kwargs)
    except (ValueError, TypeError, KeyError):
        if hasattr(metric, 'name'):
            metric_name = metric.name
        else:
            metric_name = metric
        logger.log(30, f"WARNING: eval_metric='{metric_name}' does not support sample weights so they will be ignored in reported metric.")
        weighted_metric = metric(y, y_pred, **kwargs)
    return weighted_metric


# Note: Do not send training data as input or the importances will be overfit.
# TODO: Improve time estimate (Currently pessimistic)
def compute_permutation_feature_importance(X: pd.DataFrame,
                                           y: pd.Series,
                                           predict_func: Callable[..., np.ndarray],
                                           eval_metric: Scorer,
                                           features: list = None,
                                           subsample_size=None,
                                           num_shuffle_sets: int = None,
                                           predict_func_kwargs: dict = None,
                                           transform_func: Callable[..., pd.DataFrame] = None,
                                           transform_func_kwargs: dict = None,
                                           time_limit: float = None,
                                           silent=False,
                                           log_prefix='',
                                           importance_as_list=False,
                                           random_state=0) -> pd.DataFrame:
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
        Can contain tuples as elements of (feature_name, feature_list) form.
            feature_name can be any string so long as it is unique with all other feature names / features in the list.
            feature_list can be any list of valid features in the data.
            This will compute importance of the combination of features in feature_list, naming the set of features in the returned DataFrame feature_name.
            This importance will differ from adding the individual importances of each feature in feature_list, and will be more accurate to the overall group importance.
            Example: ['featA', 'featB', 'featC', ('featBC', ['featB', 'featC'])]
            In this example, the importance of 'featBC' will be calculated by jointly permuting 'featB' and 'featC' together as if they were a single two-dimensional feature.
    subsample_size : int, default None
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
        Whether to suppress logging output.
    log_prefix : str, default ''
        Prefix to add to logging statements.
    importance_as_list : bool, default False
        Whether to return the 'importance' column values as a list of the importance from each shuffle (True) or a single averaged value (False).
    random_state : int, default 0
        Acts as a seed for data subsampling and permuting feature values.

    Returns
    -------
    Pandas `pandas.DataFrame` of feature importance scores with 4 columns:
        index: The feature name.
        'importance': The estimated feature importance score.
        'stddev': The standard deviation of the feature importance score. If NaN, then not enough num_shuffle_sets were used to calculate a variance.
        'p_value': P-value for a statistical t-test of the null hypothesis: importance = 0, vs the (one-sided) alternative: importance > 0.
            Features with low p-value appear confidently useful to the predictor, while the other features may be useless to the predictor (or even harmful to include in its training data).
            A p-value of 0.01 indicates that there is a 1% chance that the feature is useless or harmful, and a 99% chance that the feature is useful.
            A p-value of 0.99 indicates that there is a 99% chance that the feature is useless or harmful, and a 1% chance that the feature is useful.
        'n': The number of shuffles performed to estimate importance score (corresponds to sample-size used to determine confidence interval for true score).
    """
    if num_shuffle_sets is None:
        num_shuffle_sets = 1 if time_limit is None else 10

    time_start = time.time()
    if predict_func_kwargs is None:
        predict_func_kwargs = dict()
    if transform_func_kwargs is None:
        transform_func_kwargs = dict()
    if features is None:
        features = list(X.columns)

    _validate_features(features=features, valid_features=list(X.columns))

    num_features = len(features)

    if subsample_size is not None:
        num_rows = min(len(X), subsample_size)
    else:
        num_rows = len(X)
    subsample = num_rows < len(X)

    if not silent:
        logging_message = f'{log_prefix}Computing feature importance via permutation shuffling for {num_features} features using {num_rows} rows with {num_shuffle_sets} shuffle sets...'
        if time_limit is not None:
            logging_message = f'{logging_message} Time limit: {time_limit}s...'
        logger.log(20, logging_message)

    time_permutation_start = time.time()
    fi_dict_list = []
    shuffle_repeats_completed = 0
    log_final_suffix = ''

    X_orig = X
    y_orig = y
    feature_batch_count = None
    X_raw = None
    score_baseline = None
    initial_random_state = random_state
    # TODO: Can speedup shuffle_repeats by incorporating into X_raw (do multiple repeats in a single predict call)
    for shuffle_repeat in range(num_shuffle_sets):
        fi = dict()
        random_state = initial_random_state + shuffle_repeat

        if subsample:
            # TODO: Stratify? We currently don't know in this function the problem_type (could pass as additional arg).
            X = X_orig.sample(subsample_size, random_state=random_state)
            y = y_orig.loc[X.index]

        if subsample or shuffle_repeat == 0:
            time_start_score = time.time()
            X_transformed = X if transform_func is None else transform_func(X, **transform_func_kwargs)
            y_pred = predict_func(X_transformed, **predict_func_kwargs)
            score_baseline = eval_metric(y, y_pred)
            if shuffle_repeat == 0:
                if not silent:
                    time_score = time.time() - time_start_score
                    time_estimated = ((num_features + 1) * time_score) * num_shuffle_sets + time_start_score - time_start
                    time_estimated_per_set = time_estimated / num_shuffle_sets
                    logger.log(20, f'{log_prefix}\t{round(time_estimated, 2)}s\t= Expected runtime ({round(time_estimated_per_set, 2)}s per shuffle set)')

                if transform_func is None:
                    feature_batch_count = _get_safe_fi_batch_count(X=X, num_features=num_features)
                else:
                    feature_batch_count = _get_safe_fi_batch_count(X=X, num_features=num_features, X_transformed=X_transformed)

            # creating copy of original data N=feature_batch_count times for parallel processing
            X_raw = pd.concat([X.copy() for _ in range(feature_batch_count)], ignore_index=True, sort=False).reset_index(drop=True)

        row_count = len(X)

        X_shuffled = shuffle_df_rows(X=X, seed=random_state)

        for i in range(0, num_features, feature_batch_count):
            parallel_computed_features = features[i:i + feature_batch_count]

            # if final iteration, leaving only necessary part of X_raw
            num_features_processing = len(parallel_computed_features)
            final_iteration = i + num_features_processing == num_features

            row_index = 0
            for feature in parallel_computed_features:
                if isinstance(feature, tuple):
                    feature = feature[1]
                row_index_end = row_index + row_count
                X_raw.loc[row_index:row_index_end - 1, feature] = X_shuffled[feature].values
                row_index = row_index_end

            if (num_features_processing < feature_batch_count) and final_iteration:
                X_raw_transformed = X_raw.loc[:row_count * num_features_processing - 1]
                X_raw_transformed = X_raw_transformed if transform_func is None else transform_func(X_raw_transformed, **transform_func_kwargs)
            else:
                X_raw_transformed = X_raw if transform_func is None else transform_func(X_raw, **transform_func_kwargs)
            y_pred = predict_func(X_raw_transformed, **predict_func_kwargs)

            row_index = 0
            for feature in parallel_computed_features:
                if isinstance(feature, tuple):
                    feature_name = feature[0]
                    feature_list = feature[1]
                else:
                    feature_name = feature
                    feature_list = feature
                # calculating importance score for given feature
                row_index_end = row_index + row_count
                y_pred_cur = y_pred[row_index:row_index_end]
                score = eval_metric(y, y_pred_cur)
                fi[feature_name] = score_baseline - score

                # resetting to original values for processed feature
                X_raw.loc[row_index:row_index_end - 1, feature_list] = X[feature_list].values

                row_index = row_index_end
        fi_dict_list.append(fi)
        shuffle_repeats_completed = shuffle_repeat + 1
        if time_limit is not None and shuffle_repeat != (num_shuffle_sets - 1):
            time_now = time.time()
            time_left = time_limit - (time_now - time_start)
            time_permutation_average = (time_now - time_permutation_start) / (shuffle_repeat + 1)
            if time_left < (time_permutation_average * 1.1):
                log_final_suffix = ' (Early stopping due to lack of time...)'
                break

    fi_list_dict = dict()
    for val in fi_dict_list:
        for key in val:
            if key not in fi_list_dict:
                fi_list_dict[key] = []
            fi_list_dict[key].append(val[key])
    fi_df = _compute_fi_with_stddev(fi_list_dict, importance_as_list=importance_as_list)

    if not silent:
        logger.log(20, f'{log_prefix}\t{round(time.time() - time_start, 2)}s\t= Actual runtime (Completed {shuffle_repeats_completed} of {num_shuffle_sets} shuffle sets){log_final_suffix}')

    return fi_df


def _validate_features(features: list, valid_features: list):
    """Raises exception if features list contains invalid features or duplicate features"""
    valid_features = set(valid_features)
    used_features = set()
    # validate features
    for feature in features:
        if isinstance(feature, tuple):
            feature_name = feature[0]
            feature_list = feature[1]
            feature_list_set = set(feature_list)
            if len(feature_list_set) != len(feature_list):
                raise ValueError(f'Feature list contains duplicate features:\n'
                                 f'{feature_list}')
            for feature_in_list in feature_list:
                if feature_in_list not in valid_features:
                    raise ValueError(f'Feature does not exist in data: {feature_in_list}\n'
                                     f'This feature came from the following feature set:\n'
                                     f'{feature}\n'
                                     f'Valid Features:\n'
                                     f'{valid_features}')
        else:
            feature_name = feature
            if feature_name not in valid_features:
                raise ValueError(f'Feature does not exist in data: {feature_name}\n'
                                 f'Valid Features:\n'
                                 f'{valid_features}')
        if feature_name in used_features:
            raise ValueError(f'Feature is present multiple times in feature list: {feature_name}')
        used_features.add(feature_name)


def _compute_fi_with_stddev(fi_list_dict: dict, importance_as_list=False) -> DataFrame:
    features = list(fi_list_dict.keys())
    fi = dict()
    fi_stddev = dict()
    fi_p_value = dict()
    fi_n = dict()
    for feature in features:
        fi[feature], fi_stddev[feature], fi_p_value[feature], fi_n[feature] = _compute_mean_stddev_and_p_value(fi_list_dict[feature])
        if importance_as_list:
            fi[feature] = fi_list_dict[feature]

    fi = pd.Series(fi).sort_values(ascending=False)
    fi_stddev = pd.Series(fi_stddev)
    fi_p_value = pd.Series(fi_p_value)
    fi_n = pd.Series(fi_n, dtype='int64')

    fi_df = fi.to_frame(name='importance')
    fi_df['stddev'] = fi_stddev
    fi_df['p_value'] = fi_p_value
    fi_df['n'] = fi_n
    return fi_df


def _compute_mean_stddev_and_p_value(values: list):
    mean = np.mean(values)
    n = len(values)
    p_value = np.nan
    stddev = np.std(values, ddof=1) if n > 1 else np.nan
    if stddev != np.nan and stddev != 0:
        t_stat = mean / (stddev / math.sqrt(n))
        p_value = scipy.stats.t.sf(t_stat, n-1)
    elif stddev == 0:
        p_value = 0.5

    return mean, stddev, p_value, n


def _get_safe_fi_batch_count(X, num_features, X_transformed=None, max_memory_ratio=0.2, max_feature_batch_count=200):
    # calculating maximum number of features that are safe to process in parallel
    X_size_bytes = sys.getsizeof(pickle.dumps(X, protocol=4))
    if X_transformed is not None:
        X_size_bytes += sys.getsizeof(pickle.dumps(X_transformed, protocol=4))
    available_mem = psutil.virtual_memory().available
    X_memory_ratio = X_size_bytes / available_mem

    feature_batch_count_safe = math.floor(max_memory_ratio / X_memory_ratio)
    feature_batch_count = max(1, min(max_feature_batch_count, feature_batch_count_safe))
    feature_batch_count = min(feature_batch_count, num_features)
    return feature_batch_count


def get_gpu_free_memory():
    """Grep gpu free memory from nvidia-smi tool.
    This function can fail due to many reasons(driver, nvidia-smi tool, envs, etc) so please simply use
    it as a suggestion, stay away with any rules bound to it.
    E.g. for a 4-gpu machine, the result can be list of int
    >>> print(get_gpu_free_memory)
    >>> [13861, 13859, 13859, 13863]
    """
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    try:
        COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = _output_to_list(subprocess.check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    except:
        memory_free_values = []
    return memory_free_values


def unevaluated_fi_df_template(features: List[str]) -> pd.DataFrame:
    importance_df = pd.DataFrame({
        'importance': None,
        'stddev': None,
        'p_value': None,
        'n': 0
    }, index=features)
    return importance_df

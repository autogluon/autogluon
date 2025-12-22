import logging
from typing import Union

import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import coo_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.utils import check_consistent_length
from sklearn.utils.multiclass import unique_labels

try:
    from sklearn.metrics._classification import _check_targets, type_of_target
except:
    from sklearn.metrics.classification import _check_targets, type_of_target

try:
    # numpy>=2
    from numpy import trapezoid as trapezoid
except:
    # numpy<2, deprecated in numpy>=2
    from numpy import trapz as trapezoid

logger = logging.getLogger(__name__)


def balanced_accuracy(solution, prediction):
    y_type, solution, prediction = _check_targets(solution, prediction)

    if y_type not in ["binary", "multiclass", "multilabel-indicator"]:
        raise ValueError(f"{y_type} is not supported")

    if y_type == "binary":
        # Do not transform into any multiclass representation
        unique_sol = np.unique(solution)
        unique_pred = np.unique(prediction)
        classes = np.unique(np.concatenate((unique_sol, unique_pred)))
        if set(classes) != {0, 1}:
            pos_class = classes[-1]
            solution = np.array([1 if i == pos_class else 0 for i in solution])
            prediction = np.array([1 if i == pos_class else 0 for i in prediction])

    elif y_type == "multiclass":
        n = len(solution)
        unique_sol, encoded_sol = np.unique(solution, return_inverse=True)
        unique_pred, encoded_pred = np.unique(prediction, return_inverse=True)
        classes = np.unique(np.concatenate((unique_sol, unique_pred)))
        map_sol = np.array([np.where(classes == c)[0][0] for c in unique_sol])
        map_pred = np.array([np.where(classes == c)[0][0] for c in unique_pred])
        # one hot encoding
        sol_ohe = np.zeros((n, len(classes)))
        pred_ohe = np.zeros((n, len(classes)))
        sol_ohe[np.arange(n), map_sol[encoded_sol]] = 1
        pred_ohe[np.arange(n), map_pred[encoded_pred]] = 1
        solution = sol_ohe
        prediction = pred_ohe

    elif y_type == "multilabel-indicator":
        solution = solution.toarray()
        prediction = prediction.toarray()
    else:
        raise NotImplementedError(f"bac_metric does not support task type {y_type}")

    fn = np.sum(np.multiply(solution, (1 - prediction)), axis=0, dtype=float)
    tp = np.sum(np.multiply(solution, prediction), axis=0, dtype=float)
    # Bounding to avoid division by 0
    eps = 1e-15
    tp = np.maximum(eps, tp)
    pos_num = np.maximum(eps, tp + fn)
    tpr = tp / pos_num  # true positive rate (sensitivity)

    if y_type in ("binary", "multilabel-indicator"):
        tn = np.sum(np.multiply((1 - solution), (1 - prediction)), axis=0, dtype=float)
        fp = np.sum(np.multiply((1 - solution), prediction), axis=0, dtype=float)
        tn = np.maximum(eps, tn)
        neg_num = np.maximum(eps, tn + fp)
        tnr = tn / neg_num  # true negative rate (specificity)
        bac = 0.5 * (tpr + tnr)
    elif y_type == "multiclass":
        bac = tpr
    else:
        raise ValueError(y_type)

    return np.mean(bac)  # average over all classes


def pac(solution, prediction):
    """
    Probabilistic Accuracy based on log_loss metric.
    We assume the solution is in {0, 1} and prediction in [0, 1].
    Otherwise, run normalize_array.
    :param solution:
    :param prediction:
    :param task:
    :return:
    """

    def normalize_array(solution, prediction):
        """
        Use min and max of solution as scaling factors to normalize prediction,
        then threshold it to [0, 1].
        Binarize solution to {0, 1}. This allows applying classification
        scores to all cases. In principle, this should not do anything to
        properly formatted classification inputs and outputs.
        :param solution:
        :param prediction:
        :return:
        """
        # Binarize solution
        sol = np.ravel(solution)  # convert to 1-d array
        maxi = np.nanmax(sol[np.isfinite(sol)])
        mini = np.nanmin(sol[np.isfinite(sol)])
        if maxi == mini:
            logger.debug("Warning: cannot normalize array")
            return [solution, prediction]
        diff = maxi - mini
        mid = (maxi + mini) / 2.0

        solution[solution >= mid] = 1
        solution[solution < mid] = 0
        # Normalize and threshold predictions (takes effect only if solution not
        # in {0, 1})

        prediction -= float(mini)
        prediction /= float(diff)

        # and if predictions exceed the bounds [0, 1]
        prediction[prediction > 1] = 1
        prediction[prediction < 0] = 0
        # Make probabilities smoother
        # new_prediction = np.power(new_prediction, (1./10))

        return [solution, prediction]

    def log_loss(solution, prediction, task):
        """Log loss for binary and multiclass."""
        [sample_num, label_num] = solution.shape
        # Lower gives problems with float32!
        eps = 0.00000003

        if (task == "multiclass") and (label_num > 1):
            # Make sure the lines add up to one for multi-class classification
            norma = np.sum(prediction, axis=1)
            for k in range(sample_num):
                prediction[k, :] /= np.maximum(norma[k], eps)

            sample_num = solution.shape[0]
            for i in range(sample_num):
                j = np.argmax(solution[i, :])
                solution[i, :] = 0
                solution[i, j] = 1

            solution = solution.astype(np.int32, copy=False)
            # For the base prediction, this solution is ridiculous in the
            # multi-label case

            # Bounding of predictions to avoid log(0),1/0,...
        prediction = np.minimum(1 - eps, np.maximum(eps, prediction))
        # Compute the log loss
        pos_class_log_loss = -np.mean(solution * np.log(prediction), axis=0)
        if (task != "multiclass") or (label_num == 1):
            # The multi-label case is a bunch of binary problems.
            # The second class is the negative class for each column.
            neg_class_log_loss = -np.mean((1 - solution) * np.log(1 - prediction), axis=0)
            log_loss = pos_class_log_loss + neg_class_log_loss
            # Each column is an independent problem, so we average.
            # The probabilities in one line do not add up to one.
            # log_loss = mvmean(log_loss)
            # print('binary {}'.format(log_loss))
            # In the multilabel case, the right thing i to AVERAGE not sum
            # We return all the scores so we can normalize correctly later on
        else:
            # For the multiclass case the probabilities in one line add up one.
            log_loss = pos_class_log_loss
            # We sum the contributions of the columns.
            log_loss = np.sum(log_loss)
            # print('multiclass {}'.format(log_loss))
        return log_loss

    def prior_log_loss(frac_pos, task):
        """Baseline log loss.
        For multiple classes or labels return the values for each column
        """
        eps = 1e-15
        frac_pos_ = np.maximum(eps, frac_pos)
        if task != "multiclass":  # binary case
            frac_neg = 1 - frac_pos
            frac_neg_ = np.maximum(eps, frac_neg)
            pos_class_log_loss_ = -frac_pos * np.log(frac_pos_)
            neg_class_log_loss_ = -frac_neg * np.log(frac_neg_)
            base_log_loss = pos_class_log_loss_ + neg_class_log_loss_
            # base_log_loss = mvmean(base_log_loss)
            # print('binary {}'.format(base_log_loss))
            # In the multilabel case, the right thing i to AVERAGE not sum
            # We return all the scores so we can normalize correctly later on
        else:  # multiclass case
            fp = frac_pos_ / sum(frac_pos_)  # Need to renormalize the lines in multiclass case
            # Only ONE label is 1 in the multiclass case active for each line
            pos_class_log_loss_ = -frac_pos * np.log(fp)
            base_log_loss = np.sum(pos_class_log_loss_)
        return base_log_loss

    y_type = type_of_target(solution)

    if isinstance(solution, pd.Series):
        solution = solution.values
    if isinstance(prediction, pd.Series):
        prediction = prediction.values

    if y_type == "binary":
        if len(solution.shape) == 1:
            solution = solution.reshape((-1, 1))
        if len(prediction.shape) == 1:
            prediction = prediction.reshape((-1, 1))
        if len(prediction.shape) == 2:
            if prediction.shape[1] > 2:
                raise ValueError(f"A prediction array with probability values " f"for {prediction.shape[1]} classes is not a binary " f"classification problem")
            elif prediction.shape[1] == 2:
                prediction = prediction[:, 1]
            else:
                # Prediction will be copied into a new binary array - no copy
                prediction = prediction.reshape((-1, 1))
        else:
            raise ValueError(f"Invalid prediction shape {prediction.shape}")

    elif y_type == "multiclass":
        if len(solution.shape) == 2:
            if solution.shape[1] > 1:
                raise ValueError(f"Solution array must only contain one class " f"label, but contains {solution.shape[1]}")
        elif len(solution.shape) == 1:
            pass
        else:
            raise ValueError("Solution.shape %s" % solution.shape)

        # Need to create a multiclass solution and a multiclass predictions
        max_class = prediction.shape[1] - 1
        solution_binary = np.zeros((len(solution), max_class + 1))
        for i in range(len(solution)):
            solution_binary[i, int(solution[i])] = 1
        solution = solution_binary

    elif y_type == "multilabel-indicator":
        solution = solution.copy()

    else:
        raise NotImplementedError(f"pac_score does not support task {y_type}")

    solution, prediction = normalize_array(solution, prediction.copy())

    sample_num, _ = solution.shape

    eps = 1e-7
    # Compute the base log loss (using the prior probabilities)
    pos_num = 1.0 * np.sum(solution, axis=0, dtype=float)  # float conversion!
    frac_pos = pos_num / sample_num  # prior proba of positive class
    the_base_log_loss = prior_log_loss(frac_pos, y_type)
    the_log_loss = log_loss(solution, prediction, y_type)

    # Exponentiate to turn into an accuracy-like score.
    # In the multi-label case, we need to average AFTER taking the exp
    # because it is an NL operation
    pac = np.mean(np.exp(-the_log_loss))
    base_pac = np.mean(np.exp(-the_base_log_loss))
    # Normalize: 0 for random, 1 for perfect
    score = (pac - base_pac) / np.maximum(eps, (1 - base_pac))

    return score


def confusion_matrix(solution, prediction, labels=None, weights=None, normalize=None, output_format="numpy_array"):
    """
    Computes confusion matrix for a given true and predicted targets
    Parameters:
        solution - true targets
        prediction - predicted targets
        labels - list of labels for which confusion matrix should be calculated
        weights - list of weights of each target
        normalize - should the output be normalized. Can take values {'true', 'pred', 'all'}
        output_format - output format of the matrix. Can take values {'python_list', 'numpy_array', 'pandas_dataframe'}
    TODO : Add dedicated confusion_matrix function to AbstractLearner
    """
    y_type, solution, prediction = _check_targets(solution, prediction)
    # Only binary and multiclass data is supported
    if y_type not in ("binary", "multiclass"):
        raise ValueError(f"{y_type} dataset is not currently supported")

    if labels is None:
        labels = unique_labels(solution, prediction)
    else:
        # Ensure that label contains only 1-D binary or multi-class array
        labels_type = type_of_target(labels)
        if labels_type not in ("binary", "multiclass"):
            raise ValueError(f"{labels_type} labels are not supported")
        labels = np.array(labels)

    if weights is None:
        weights = np.ones(solution.size, dtype=int)
    else:
        # Ensure that weights contains only 1-D integer or float array
        weights_type = type_of_target(weights)
        if weights_type not in ("binary", "multiclass", "continuous"):
            raise ValueError(f"{weights_type} weights are not supported")
        weights = np.array(weights)

    n_labels = labels.size
    if n_labels == 0:
        raise ValueError("Labels cannot be empty")
    elif (np.unique(labels)).size != n_labels:
        raise ValueError("Labels cannot have duplicates")

    if solution.size == 0 or prediction.size == 0:
        return np.zeros((n_labels, n_labels), dtype=int)

    label_to_index = {y: x for x, y in enumerate(labels)}

    check_consistent_length(solution, prediction, weights)

    # Invalidate indexes with target labels outside the accepted set of labels
    valid_indexes = np.logical_and(np.isin(solution, labels), np.isin(prediction, labels))
    solution = np.array([label_to_index.get(i) for i in solution[valid_indexes]])
    prediction = np.array([label_to_index.get(i) for i in prediction[valid_indexes]])
    weights = weights[valid_indexes]
    # For high precision
    matrix_dtype = np.int64 if weights.dtype.kind in {"i", "u", "b"} else np.float64
    cm = coo_matrix((weights, (solution, prediction)), shape=(n_labels, n_labels), dtype=matrix_dtype).toarray()
    with np.errstate(all="ignore"):
        if normalize == "true":
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif normalize == "all":
            cm = cm / cm.sum()
        cm = np.nan_to_num(cm)
    if output_format == "python_list":
        return cm.tolist()
    elif output_format == "numpy_array":
        return cm
    elif output_format == "pandas_dataframe":
        cm_df = pd.DataFrame(data=cm, index=labels, columns=labels)
        return cm_df
    else:
        return cm


# TODO Add the "labels" option to metrics that will require the label map.
#  We will need to update how we use those metrics accordingly.
def quadratic_kappa(y_true, y_pred):
    """Calculate the cohen kappa score with quadratic weighting scheme.

    This is also known as "quadratic kappa" in the Kaggle competitions
    such as petfinder: https://www.kaggle.com/c/petfinder-adoption-prediction/overview/evaluation

    We will also support probabilistic input to ensure that the function knows
    the number of possible classes.

    Parameters
    ----------
    y_true
        Shape (#samples,)
    y_pred
        Shape (#samples, #class) or (#samples,)

    Returns
    -------
    score
        scalar score
    """
    labels = None
    if y_pred.ndim > 1:
        if labels is not None:
            assert len(labels) == y_pred.shape[1]
        else:
            labels = np.arange(y_pred.shape[1])
        y_pred = np.argmax(y_pred, axis=-1)
    return cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")


# Refer to https://github.com/scikit-learn/scikit-learn/blame/f3f51f9b611bf873bd5836748647221480071a87/sklearn/metrics/_ranking.py#L985-L1000
#  for the original logic and full explanation of what this does. This number has no impact on the score calculated, and is purely for speed.
#  This value was chosen as having a lower value simply slows down the majority of function calls more than it speeds them up.
#  It was observed that function calls were sped up by 25% by increasing this from 2 to 100000.
#  Values greater than this were not tested but would be marginal difference as large samples get less speed up.
_OPTIMIZE_INDICES_THRESHOLD = 100000


def customized_binary_roc_auc_score(y_true: Union[np.array, pd.Series], y_score: Union[np.array, pd.Series], **kwargs) -> float:
    """
    Functionally identical to sklearn.metrics.roc_auc_score for binary classification.
    Streamlined for binary classification to be faster by ~5x by avoiding validation checks of the inputs.
    We can do this in AutoGluon because we guarantee the data is of proper form when entering this logic.

    Parameters
    ----------
    y_true : Union[np.array, pd.Series] of type int
        Ground truth (correct) labels for n_samples samples. shape = (n_samples,)
        Valid sample values are 1 and 0.
    y_score : Union[np.array, pd.Series] of type float
        The prediction probabilities. shape = (n_samples,)
    **kwargs :
        Any additional arguments. If not empty, will fall back to sklearn's implementation

    Returns
    -------
    roc_auc_score : float
        The roc_auc_score in higher_is_better format.
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_score, pd.Series):
        y_score = y_score.values
    if y_true.size == 0 or y_score.size == 0:
        raise ValueError("Found array with 0 sample(s) (shape=(0,)) while a minimum of 1 is required.")
    if kwargs:
        return sklearn.metrics.roc_auc_score(y_true, y_score, **kwargs)

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # keep only indices that have different values to speed up future computation
    distinct_value_indices = np.where(np.diff(y_score))[0]
    # np.r_ is an optimized way to merge two or more arrays and/or singular values into one array
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # keep track of how many true positives and false positives have occurred at each threshold
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    if tps.size > _OPTIMIZE_INDICES_THRESHOLD:
        # optimize indices only when there is enough size to justify
        # this has no impact on the final score
        optimal_idxs = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]

    # Add an extra threshold position
    # to make sure that the curve starts at (0, 0)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    if fps[-1] <= 0 or tps[-1] <= 0:
        raise ValueError("Only one class present in y_true. ROC AUC score is not defined in that case.")
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    return trapezoid(tpr, fpr)

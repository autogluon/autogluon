import copy
from abc import ABCMeta, abstractmethod
from functools import partial

import scipy
import scipy.stats
import sklearn.metrics

from . import classification_metrics
from .util import sanitize_array
from ..constants import PROBLEM_TYPES_REGRESSION, PROBLEM_TYPES_CLASSIFICATION, QUANTILE
from ..utils.miscs import warning_filter
from .classification_metrics import *
from . import quantile_metrics


class Scorer(object, metaclass=ABCMeta):
    def __init__(self, name, score_func, optimum, sign, kwargs):
        self.name = name
        self._kwargs = kwargs
        self._score_func = score_func
        self._optimum = optimum
        self._sign = sign
        self.alias = set()

    def add_alias(self, alias):
        if alias == self.name:
            raise ValueError(f'The alias "{alias}" is the same as the original name "{self.name}". '
                             f'This is not allowed.')
        self.alias.add(alias)

    @property
    def greater_is_better(self) -> bool:
        """Return whether the score is greater the better.

        We use the stored `sign` object to decide the property.

        Returns
        -------
        flag
            The "greater_is_better" flag.
        """
        return self._sign > 0

    def convert_score_to_sklearn_val(self, score):
        """Scores are always greater_is_better, this flips the sign of metrics who were originally lower_is_better."""
        return self._sign * score

    @abstractmethod
    def __call__(self, y_true, y_pred, sample_weight=None):
        pass

    def __repr__(self):
        return self.name

    def sklearn_scorer(self):
        with warning_filter():
            ret = sklearn.metrics.scorer.make_scorer(score_func=self, greater_is_better=True, needs_proba=self.needs_proba, needs_threshold=self.needs_threshold)
        return ret

    @property
    @abstractmethod
    def needs_pred(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def needs_proba(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def needs_threshold(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def needs_quantile(self) -> bool:
        raise NotImplementedError


class _PredictScorer(Scorer):
    def __call__(self, y_true, y_pred, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        y_true : array-like
            Gold standard target values for X.

        y_pred : array-like, [n_samples x n_classes]
            Model predictions

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        if isinstance(y_true, list):
            y_true = np.array(y_true)
        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)
        type_true = type_of_target(y_true)

        if len(y_pred.shape) == 1 or y_pred.shape[1] == 1 or type_true == 'continuous':
            pass  # must be regression, all other task types would return at least two probabilities
        elif type_true in ['binary', 'multiclass']:
            y_pred = np.argmax(y_pred, axis=1)
        elif type_true == 'multilabel-indicator':
            y_pred[y_pred > 0.5] = 1.0
            y_pred[y_pred <= 0.5] = 0.0
        else:
            raise ValueError(type_true)

        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred,
                                                 **self._kwargs)

    @property
    def needs_pred(self):
        return True

    @property
    def needs_proba(self):
        return False

    @property
    def needs_threshold(self):
        return False

    @property
    def needs_quantile(self):
        return False


class _ProbaScorer(Scorer):
    def __call__(self, y_true, y_pred, sample_weight=None):
        """Evaluate predicted probabilities for X relative to y_true.
        Parameters
        ----------
        y_true : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.

        y_pred : array-like, [n_samples x n_classes]
            Model predictions

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred, **self._kwargs)

    @property
    def needs_pred(self):
        return False

    @property
    def needs_proba(self):
        return True

    @property
    def needs_threshold(self):
        return False

    @property
    def needs_quantile(self):
        return False


class _ThresholdScorer(Scorer):
    def __call__(self, y_true, y_pred, sample_weight=None):
        """Evaluate decision function output for X relative to y_true.

        Parameters
        ----------
        y_true : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.

        y_pred : array-like, [n_samples x n_classes]
            Model predictions

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        if isinstance(y_true, list):
            y_true = np.array(y_true)
        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)
        y_type = type_of_target(y_true)
        if y_type not in ("binary", "multilabel-indicator"):
            raise ValueError("{0} format is not supported".format(y_type))

        if y_type == "binary":
            pass
            # y_pred = y_pred[:, 1]
        elif isinstance(y_pred, list):
            y_pred = np.vstack([p[:, -1] for p in y_pred]).T

        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred, **self._kwargs)

    @property
    def needs_pred(self):
        return False

    @property
    def needs_proba(self):
        return False

    @property
    def needs_threshold(self):
        return True

    @property
    def needs_quantile(self):
        return False


class _QuantileScorer(Scorer):
    def __call__(self, y_true, y_pred, quantile_levels, sample_weight=None):
        """Evaluate predicted quantile target values for X relative to y_true.

        Parameters
        ----------
        y_true : array-like
            Gold standard target values for X.

        y_pred : array-like, [n_samples x n_quantiles]
            Model quantile predictions

        quantile_levels : array-like
            List of quantile levels

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        if isinstance(y_true, list):
            y_true = np.array(y_true)
        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)
        if isinstance(quantile_levels, list):
            quantile_levels = np.array(quantile_levels)
        type_true = type_of_target(y_true)

        if len(y_pred.shape) == 2 or y_pred.shape[1] >= 1 or type_true == 'continuous':
            pass  # must be quantile regression, all other task types would return at least two probabilities
        else:
            raise ValueError(type_true)

        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred,
                                                 quantile_levels,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred,
                                                 quantile_levels,
                                                 **self._kwargs)

    @property
    def needs_pred(self):
        return False

    @property
    def needs_proba(self):
        return False

    @property
    def needs_threshold(self):
        return False

    @property
    def needs_quantile(self):
        return True


def make_scorer(name, score_func, optimum=1, greater_is_better=True,
                needs_proba=False, needs_threshold=False, needs_quantile=False, **kwargs) -> Scorer:
    """Make a scorer from a performance metric or loss function.

    Factory inspired by scikit-learn which wraps scikit-learn scoring functions
    to be used in auto-sklearn.

    Parameters
    ----------
    score_func : callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    optimum : int or float, default=1
        The best score achievable by the score function, i.e. maximum in case of
        scorer function and minimum in case of loss function.

    greater_is_better : boolean, default=True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.

    needs_proba : boolean, default=False
        Whether score_func requires predict_proba to get probability estimates
        out of a classifier.

    needs_threshold : boolean, default=False
        Whether score_func takes a continuous decision certainty.
        This only works for binary classification.

    needs_quantile : boolean, default=False
        Whether score_func is based on quantile predictions.
        This only works for quantile regression.

    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Returns
    -------
    scorer
        Callable object that returns a scalar score; greater is better.
    """
    sign = 1 if greater_is_better else -1
    if needs_proba:
        cls = _ProbaScorer
    elif needs_threshold:
        cls = _ThresholdScorer
    elif needs_quantile:
        cls = _QuantileScorer
    else:
        cls = _PredictScorer
    return cls(name, score_func, optimum, sign, kwargs)


# Standard regression scores
r2 = make_scorer('r2',
                 sklearn.metrics.r2_score)
mean_squared_error = make_scorer('mean_squared_error',
                                 sklearn.metrics.mean_squared_error,
                                 optimum=0,
                                 greater_is_better=False)
mean_squared_error.add_alias('mse')

mean_absolute_error = make_scorer('mean_absolute_error',
                                  sklearn.metrics.mean_absolute_error,
                                  optimum=0,
                                  greater_is_better=False)
mean_absolute_error.add_alias('mae')

median_absolute_error = make_scorer('median_absolute_error',
                                    sklearn.metrics.median_absolute_error,
                                    optimum=0,
                                    greater_is_better=False)


def local_spearmanr(y_true, y_pred):
    return float(scipy.stats.spearmanr(y_true, y_pred)[0])


spearmanr = make_scorer('spearmanr',
                        local_spearmanr,
                        optimum=1.0,
                        greater_is_better=True)


def local_pearsonr(y_true, y_pred):
    return float(scipy.stats.pearsonr(y_true, y_pred)[0])


pearsonr = make_scorer('pearsonr',
                       local_pearsonr,
                       optimum=1.0,
                       greater_is_better=True)


def rmse_func(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())


root_mean_squared_error = make_scorer('root_mean_squared_error',
                                      rmse_func,
                                      optimum=0,
                                      greater_is_better=False)
root_mean_squared_error.add_alias('rmse')

# Quantile pinball loss
pinball_loss = make_scorer('pinball_loss',
                           quantile_metrics.pinball_loss,
                           needs_quantile=True,
                           optimum=0.0,
                           greater_is_better=False)
pinball_loss.add_alias('pinball')


# Standard Classification Scores
accuracy = make_scorer('accuracy',
                       sklearn.metrics.accuracy_score)
accuracy.add_alias('acc')

balanced_accuracy = make_scorer('balanced_accuracy',
                                classification_metrics.balanced_accuracy)
f1 = make_scorer('f1',
                 sklearn.metrics.f1_score)
mcc = make_scorer('mcc', sklearn.metrics.matthews_corrcoef)


# Score functions that need decision values
roc_auc = make_scorer('roc_auc',
                      sklearn.metrics.roc_auc_score,
                      greater_is_better=True,
                      needs_threshold=True)

roc_auc_ovo_macro = make_scorer('roc_auc_ovo_macro',
                                sklearn.metrics.roc_auc_score,
                                multi_class='ovo',
                                average='macro',
                                greater_is_better=True,
                                needs_proba=True,
                                needs_threshold=False)

average_precision = make_scorer('average_precision',
                                sklearn.metrics.average_precision_score,
                                needs_threshold=True)
precision = make_scorer('precision',
                        sklearn.metrics.precision_score)
recall = make_scorer('recall',
                     sklearn.metrics.recall_score)

# Register other metrics
quadratic_kappa = make_scorer('quadratic_kappa', quadratic_kappa, needs_proba=False)


def customized_log_loss(y_true, y_pred, eps=1e-15):
    """

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels for n_samples samples.

    y_pred : array-like of float
        The predictions. shape = (n_samples, n_classes) or (n_samples,)

    eps : float
        The epsilon

    Returns
    -------
    loss
        The negative log-likelihood
    """
    assert y_true.ndim == 1
    if y_pred.ndim == 1:
        # First clip the y_pred which is also used in sklearn
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()
    else:
        assert y_pred.ndim == 2, 'Only ndim=2 is supported'
        labels = np.arange(y_pred.shape[1], dtype=np.int32)
        return sklearn.metrics.log_loss(y_true.astype(np.int32), y_pred,
                                        labels=labels,
                                        eps=eps)


# Score function for probabilistic classification
log_loss = make_scorer('log_loss',
                       customized_log_loss,
                       optimum=0,
                       greater_is_better=False,
                       needs_proba=True)
log_loss.add_alias('nll')

pac_score = make_scorer('pac_score',
                        classification_metrics.pac_score,
                        greater_is_better=True,
                        needs_proba=True)

REGRESSION_METRICS = dict()
for scorer in [r2, mean_squared_error, root_mean_squared_error, mean_absolute_error,
                   median_absolute_error, spearmanr, pearsonr]:
    if scorer.name in REGRESSION_METRICS:
        raise ValueError(f'Duplicated score name found! scorer={scorer}, name={scorer.name}. '
                         f'Consider to register with a different name.')
    REGRESSION_METRICS[scorer.name] = scorer
    for alias in scorer.alias:
        if alias in REGRESSION_METRICS:
            raise ValueError(f'Duplicated alias found! scorer={scorer}, alias={alias}. '
                             f'Consider to use a different alias.')
        REGRESSION_METRICS[alias] = scorer

QUANTILE_METRICS = dict()
for scorer in [pinball_loss]:
    if scorer.name in QUANTILE_METRICS:
        raise ValueError(f'Duplicated score name found! scorer={scorer}, name={scorer.name}. '
                         f'Consider to register with a different name.')
    QUANTILE_METRICS[scorer.name] = scorer
    for alias in scorer.alias:
        if alias in QUANTILE_METRICS:
            raise ValueError(f'Duplicated alias found! scorer={scorer}, alias={alias}. '
                             f'Consider to use a different alias.')
        QUANTILE_METRICS[alias] = scorer

CLASSIFICATION_METRICS = dict()
for scorer in [accuracy, balanced_accuracy, mcc, roc_auc, roc_auc_ovo_macro, average_precision,
               log_loss, pac_score, quadratic_kappa]:
    CLASSIFICATION_METRICS[scorer.name] = scorer
    for alias in scorer.alias:
        CLASSIFICATION_METRICS[alias] = scorer


for name, metric in [('precision', sklearn.metrics.precision_score),
                     ('recall', sklearn.metrics.recall_score),
                     ('f1', sklearn.metrics.f1_score)]:
    globals()[name] = make_scorer(name, metric)
    CLASSIFICATION_METRICS[name] = globals()[name]
    for average in ['macro', 'micro', 'samples', 'weighted']:
        qualified_name = '{0}_{1}'.format(name, average)
        globals()[qualified_name] = make_scorer(qualified_name,
                                                partial(metric, pos_label=None, average=average))
        CLASSIFICATION_METRICS[qualified_name] = globals()[qualified_name]


def get_metric(metric, problem_type=None, metric_type=None) -> Scorer:
    """Returns metric function by using its name if the metric is str.
    Performs basic check for metric compatibility with given problem type."""
    all_available_metric_names = list(CLASSIFICATION_METRICS.keys()) + list(REGRESSION_METRICS.keys()) + list(QUANTILE_METRICS.keys()) + ['soft_log_loss']

    if metric is not None and isinstance(metric, str):
        if metric in CLASSIFICATION_METRICS:
            if problem_type is not None and problem_type not in PROBLEM_TYPES_CLASSIFICATION:
                raise ValueError(f"{metric_type}={metric} can only be used for classification problems")
            return CLASSIFICATION_METRICS[metric]
        elif metric in REGRESSION_METRICS:
            if problem_type is not None and problem_type not in PROBLEM_TYPES_REGRESSION:
                raise ValueError(f"{metric_type}={metric} can only be used for regression problems")
            return REGRESSION_METRICS[metric]
        elif metric in QUANTILE_METRICS:
            if problem_type is not None and problem_type != QUANTILE:
                raise ValueError(f"{metric_type}={metric} can only be used for quantile problems")
            return QUANTILE_METRICS[metric]
        elif metric == 'soft_log_loss':
            if problem_type == QUANTILE:
                raise ValueError(f"{metric_type}={metric} can not be used for quantile problems")
            # Requires mxnet
            from .softclass_metrics import soft_log_loss
            return soft_log_loss
        else:
            raise ValueError(
                f"{metric} is an unknown metric, all available metrics are "
                f"'{all_available_metric_names}'. You can also refer to "
                f"autogluon.core.metrics to see how to define your own {metric_type} function"
            )
    else:
        return metric

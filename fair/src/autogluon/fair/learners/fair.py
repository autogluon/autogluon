"""The entry point to fair. Defines the FairPredictor object used to access fairness functionality."""
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from autogluon.core.metrics import Scorer
from autogluon.tabular import TabularPredictor
from ..utils import group_metrics
from ..utils.group_metric_classes import BaseGroupMetric
from . import efficient_compute, fair_frontier
logger = logging.getLogger(__name__)


class FairPredictor:
    """Assess and mitigate the fairness and effectiveness of a autogluon binary predictor post-fit
    by computing group specific metrics, and performing threshold adjustment.
    Parameters
    ----------
    predictor: an autogluon binary  predictor that will be modified
    validation_data: a dataframe that can be read by predictor.
    groups (optional, default None): is an indicator of protected attributes, i.e.  the discrete groups used to measure
    fairness
    it may be:
        1. The name of a pandas column containing discrete values
        2. a vector of the same size as the validation set containing discrete values
        3. The value None   (used when we don't require groups, for example,
               if we are optimizing F1 without per-group thresholds)
    inferred_groups: (Optional, default False) A binary or multiclass autogluon predictor that infers the protected
                                attributes.
        This can be used to enforce fairness when no information about protected attribtutes is
        avalible at test time. If this is not false, fairness will be measured using the variable
        'groups', but enforced using the predictor response.
    use_fast: (Optional) Bool
    If use_fast is True, the fair search is much more efficient, but the objectives must take the
    form of a GroupMetric
    If use_fast is False, autogluon scorers are also supported.
    """

    def __init__(self, predictor, validation_data, groups=None, *, inferred_groups=False,
                 use_fast=True) -> None:

        if predictor.problem_type != 'binary':
            logger.error('Fairpredictor only takes a binary predictor as input')
        self.predictor = predictor
        if groups is None:
            groups = False
        # Internal logic differentiates between groups are not provided on other data
        # i.e. groups = None
        # and there are no groups i.e. groups = False
        # However, as a user interface groups = None makes more sense for instantiation.
        self.groups = groups
        self.use_fast: bool = use_fast
        self.validation_data = validation_data
        validation_labels = self.validation_data[predictor.label]

        # We use _internal_groups as a standardized argument that is always safe to pass
        # to functions expecting a vector
        self._internal_groups = self.groups_to_numpy(groups, validation_data)

        if self._internal_groups.shape[0] != validation_labels.shape[0]:
            logger.error('The size of the groups does not match the dataset size')

        self.inferred_groups: bool = inferred_groups
        if inferred_groups:
            self._val_thresholds = np.asarray(inferred_groups.predict_proba(self.validation_data))
        else:
            # Use OneHot and store encoder so it will work on new data
            self.group_encoder: OneHotEncoder = OneHotEncoder(handle_unknown='ignore')
            self.group_encoder.fit(self._internal_groups.reshape(-1, 1))
            self._val_thresholds = self.group_encoder.transform(
                self._internal_groups.reshape(-1, 1)).toarray()

        self.proba = np.asarray(predictor.predict_proba(self.validation_data))
        self.y_true = np.asarray(validation_labels == self.predictor.class_labels[1])
        self.frontier = None
        if self.use_fast:
            self.offset = np.zeros((self._val_thresholds.shape[1],))
        else:
            self.offset = np.zeros((self._val_thresholds.shape[1], self.proba.shape[1]))
        self.objective1 = None
        self.objective2 = None
        self.round = False

    def groups_to_numpy(self, groups, data: pd.DataFrame) -> np.ndarray:
        """helper function for transforming groups into a numpy array of unique values
        parameters
        ----------
        groups: one of the standard represenations of groups (see class doc)
        data: a pandas dataframe
        returns
        -------
        numpy array
        """
        if groups is None:
            groups = self.groups
        if callable(groups):
            return groups(data).argmax(1)
        if isinstance(groups, str):
            return np.asarray(data[groups])
        if groups is False:
            return np.zeros(data.shape[0])
        return groups

    def fit(self, objective, constraint=group_metrics.accuracy, value=0.0, *,
            greater_is_better_obj=None, greater_is_better_const=None,
            recompute=True, tol=False, grid_width=False):
        """Fits the chosen predictor to optimize an objective while satisfing a constraint.
        parameters
        ----------
        objective: a BaseGroupMetric or Scorable to be optimised
        constraint (optional): a BaseGroupMetric or Scorable that must be above/below a certain value
        value (optional): float the value constraint must be above or below
        If neither constraint nor value are provided fit enforces the constraint that accuracy is greater or equal to zero.

        greater_is_better_obj: bool or None Governs if the objective is maximised (True) or
                             minimized (False).
                If None the value of objective.greater_is_better is used.
        greater_is_better_const: bool or None Governs if the constraint has to be greater (True) or
                                smaller (False) than value.
                If None the value of constraint.greater_is_better is used.
        recompute: governs if the the parato frontier should be recomputed. Use False to efficiently
                    adjusting the threshold while keeping objective and constraint fixed.
        tol: float or False. Can round the solutions found by predict_proba to within a particular
                            tolerance to prevent overfitting.
                               Generally not needed.
        grid_width: allows manual specification of the grid size. N.B. the overall computational
                    budget is O(grid_width**groups)
                 By default the grid_size is 30
        returns
        -------
        Nothing
        """
        if greater_is_better_obj is None:
            greater_is_better_obj = objective.greater_is_better
        if greater_is_better_const is None:
            greater_is_better_const = constraint.greater_is_better

        if recompute is True or self.frontier is None:
            self.compute_frontier(objective, constraint,
                                  greater_is_better_obj1=greater_is_better_obj,
                                  greater_is_better_obj2=greater_is_better_const, tol=tol,
                                  grid_width=grid_width)
        if greater_is_better_const:
            mask = self.frontier[0][1] >= value
        else:
            mask = self.frontier[0][1] <= value

        if mask.sum() == 0:
            logger.warning('No solutions satisfy the constraint found, selecting the closest solution')
            weights = self.frontier[1]
            vmax = [self.frontier[0][1].argmin(),
                    self.frontier[0][1].argmax()][int(greater_is_better_const)]
        else:
            values = self.frontier[0][0][mask]
            weights = self.frontier[1].T[mask].T

            vmax = [values.argmin(),
                    values.argmax()][int(greater_is_better_obj)]
        self.offset = weights.T[vmax].T

    def compute_frontier(self, objective1, objective2, *, greater_is_better_obj1,
                         greater_is_better_obj2, tol=False,
                         grid_width=False) -> None:
        """ Computes the parato frontier. Internal logic used by fit
        parameters
        ----------
        objective1: a BaseGroupMetric or Scorable to be optimised
        objective2: a BaseGroupMetric or Scorable to be optimised
        greater_is_better_obj1: bool or None Governs if the objective is maximised (True)
                                 or  minimized (False).
                If None the value of objective.greater_is_better is used.
        greater_is_better_obj2: bool or None Governs if the constraint has to be greater (True)
                                or  smaller (False) than value.
                If None the value of constraint.greater_is_better is used.
        tol: float or False. Can round the solutions found by predict_proba to within a given
                            tolerance to prevent overfitting
                            Generally not needed.
        grid_width: allows manual specification of the grid size. N.B. the overall computational
                    budget is O(grid_width**groups)
        returns
        -------
        Nothing
        """
        self.objective1 = objective1
        self.objective2 = objective2

        if self.use_fast is False:
            if _needs_groups(objective1):
                objective1 = fix_groups(objective1, self._internal_groups)
            if _needs_groups(objective2):
                objective2 = fix_groups(objective2, self._internal_groups)
        direction = np.ones(2)
        if greater_is_better_obj1 is False:
            direction[0] = -1
        if greater_is_better_obj2 is False:
            direction[1] = -1

        if grid_width is False:
            if self.use_fast:
                grid_width = min(30, (30**5)**(1 / self._val_thresholds.shape[1]))
            else:
                grid_width = 14
                if self._val_thresholds.shape[1] == 2:
                    grid_width = 18

        self.round = tol

        proba = self.proba
        if tol is not False:
            proba = np.around(self.proba / tol) * tol
        if self.use_fast:
            self.frontier = efficient_compute.grid_search(self.y_true, proba, objective1,
                                                          objective2,
                                                          self._val_thresholds.argmax(1),
                                                          self._internal_groups, steps=grid_width,
                                                          directions=direction)
        else:
            self.frontier = fair_frontier.build_coarse_to_fine_front(objective1, objective2,
                                                                     self.y_true, proba,
                                                                     np.asarray(self._val_thresholds,
                                                                                dtype=np.float16),
                                                                     directions=direction,
                                                                     nr_of_recursive_calls=3,
                                                                     initial_divisions=grid_width)

    def plot_frontier(self, data=None, groups=None, objective1=False, objective2=False) -> None:
        """ Plots an existing parato frontier with respect to objective1 and objective2.
            These do not need to be the same objectives as used when computing the frontier
            The original predictor, and the predictor selected by fit is shown in different colors.
            fit() must be called first.
            parameters
            ----------
            data: (optional) pandas dataset. If not specified, uses the data used to run fit.
            groups: (optional) groups data (see class definition). If not specified, uses the
                                definition provided at initialisation
            objective1: (optional) an objective to be plotted, if not specified use the
                                    objective provided to fit is used in its place.
            objective2: (optional) an objective to be plotted, if not specified use the
                                    constraint provided to fit is used in its place.
        """
        if self.frontier is None:
            logger.error('Call fit before plot_frontier')

        objective1 = objective1 or self.objective1
        objective2 = objective2 or self.objective2
        plt.figure()
        plt.title('Frontier found')
        plt.xlabel(objective2.name)
        plt.ylabel(objective1.name)

        if data is None:
            data = self.validation_data
            labels = self.y_true
            proba = self.proba
            groups = self.groups_to_numpy(groups, data)
            val_thresholds = self._val_thresholds
        else:
            labels = np.asarray(data[self.predictor.label])
            proba = np.asarray(self.predictor.predict_proba(data))
            labels = (labels == self.predictor.positive_class) * 1
            groups = self.groups_to_numpy(groups, data)

            if self.inferred_groups is False:
                if self.groups is False:
                    val_thresholds = np.ones((data.shape[0], 1))
                else:
                    val_thresholds = self.group_encoder.transform(groups.reshape(-1, 1)).toarray()
            else:
                val_thresholds = np.asarray(self.inferred_groups.predict_proba(data))

        if self.use_fast is False:
            if _needs_groups(objective1):
                objective1 = fix_groups(objective1, groups)
            if _needs_groups(objective2):
                objective2 = fix_groups(objective2, groups)

            front1 = fair_frontier.compute_metric(objective1, labels, proba,
                                                  val_thresholds, self.frontier[1])
            front2 = fair_frontier.compute_metric(objective2, labels, proba,
                                                  val_thresholds, self.frontier[1])

            zero = [dispatch_metric(objective1, labels, proba, groups),
                    dispatch_metric(objective2, labels, proba, groups)]

            front1_u = fair_frontier.compute_metric(objective1, labels, proba,
                                                    val_thresholds, self.offset[:, :, np.newaxis])
            front2_u = fair_frontier.compute_metric(objective2, labels, proba,
                                                    val_thresholds, self.offset[:, :, np.newaxis])

        else:
            front1 = efficient_compute.compute_metric(objective1, labels, proba,
                                                      groups,
                                                      val_thresholds.argmax(1), self.frontier[1])
            front2 = efficient_compute.compute_metric(objective2, labels, proba,
                                                      groups,
                                                      val_thresholds.argmax(1), self.frontier[1])

            zero = [objective1(labels, proba.argmax(1), groups),
                    objective2(labels, proba.argmax(1), groups)]

            front1_u = efficient_compute.compute_metric(objective1, labels, proba, groups,
                                                        val_thresholds.argmax(1),
                                                        self.offset[:, np.newaxis])
            front2_u = efficient_compute.compute_metric(objective2, labels, proba, groups,
                                                        val_thresholds.argmax(1),
                                                        self.offset[:, np.newaxis])

        plt.scatter(front2, front1, label='Frontier')
        plt.scatter(zero[1], zero[0], label='Original predictor')

        plt.scatter(front2_u, front1_u, label='Updated predictor')
        plt.legend(loc='best')

    def evaluate(self, data=None, metrics=None, verbose=False) -> pd.DataFrame:
        """Compute standard metrics of the original predictor and the updated predictor
         found by fit and return them in a dataframe.
          If fit has not been called only return the metrics of the original predictor.
        parameters
        ----------
        data: (optional) a pandas dataframe to evaluate over. If not provided evaluate over
            the dataset provided at initialisation.
        metrics: (optional) a dictionary where the keys are metric names and the elements are either
                    scoreables or group metrics. If not provided report the standard metrics
                    reported by autogluon on binary predictors
        returns
        -------
        a pandas dataset containing rows indexed by metric name, and columns by
        ['original', 'updated']
         """
        if metrics is None:
            metrics = group_metrics.ag_metrics

        groups = None
        if data is not None:
            groups = np.ones(data.shape[0])

        return self.evaluate_fairness(data, groups, metrics=metrics, verbose=verbose)

    def evaluate_fairness(self, data=None, groups=None, *, metrics=None, verbose=False) -> pd.DataFrame:
        """Compute standard fairness metrics of the orginal predictor and the new predictor
         found by fit. If fit has not been called return a dataframe containing
         only the metrics of the original predictor.
         parameters
        ----------
        data: (optional) a pandas dataframe to evaluate over. If not provided evaluate over
                the dataset provided at initialisation.
        groups (optional) a specification of the groups (see class defintion). If not provided use
                the defintion provided at init.
        metrics: (optional) a dictionary where the keys are metric names and the elements are either
                    scoreables or group metrics. If not provided report the standard metrics
                    reported by SageMaker Clarify
                    https://mkai.org/learn-how-amazon-sagemaker-clarify-helps-detect-bias
        returns
        -------
        a pandas dataset containing rows indexed by fairness measure name, and columns by
        ['original', 'updated']
         """
        if metrics is None:
            metrics = group_metrics.clarify_metrics

        if data is None:
            data = self.validation_data
            labels = self.y_true
            y_pred_proba = self.predictor.predict_proba(data)
        else:
            labels = np.asarray(data[self.predictor.label])
            y_pred_proba = self.predictor.predict_proba(data)
            labels = (labels == self.predictor.positive_class) * 1
        groups = self.groups_to_numpy(groups, data)

        collect = self.fairness_metrics(labels, y_pred_proba, groups, metrics, verbose=verbose)
        collect.columns = ['original']

        if np.any(self.offset):
            y_pred_proba = self.predict_proba(data)
            new_pd = self.fairness_metrics(labels, y_pred_proba, groups, metrics, verbose=verbose)
            new_pd.columns = ['updated']
            collect = pd.concat([collect, new_pd], axis='columns')
        return collect

    def fairness_metrics(self, y_true: np.ndarray, proba, groups: np.ndarray,
                         metrics: dict, *, verbose=False) -> pd.DataFrame:
        """Helper function for evaluate_fairness
        Report fairness metrics that do not require additional information.
        parameters
        ----------
        y_true: numpy array containing true binary labels of the dataset
        proba: numpy or pandas array containing the output of predict_proba
        groups: numpy array containing discrete group labelling
        metrics: a dictionary where keys are the names and values are either
        Scorable or a BaseGroupMetric.
        returns
        -------
        a pandas dataframe of fairness metrics
        """
        values = np.zeros(len(metrics))
        names = []
        for i, k in enumerate(metrics.keys()):
            if verbose is False:
                names.append(k)
            else:
                names.append(metrics[k].name)
            values[i] = dispatch_metric(metrics[k], y_true, proba, groups)

        return pd.DataFrame(values, index=names)

    def evaluate_groups(self, data=None, groups=None, metrics=None, *, return_original=False,
                        verbose=False):
        """Evaluate standard metrics per group and returns dataframe.
        parameters
        ----------
        data: (optional) a pandas dataframe to evaluate over. If not provided evaluate over
            the dataset provided at initialisation.
        groups (optional) a specification of the groups (see class defintion). If not provided
                use the defintion provided at init.
        metrics: (optional) a dictionary where the keys are metric names and the elements are either
                    scoreables or group metrics. If not provided report the standard autogluon
                    binary predictor evaluations plus measures of the size of each group and their
                    labels.
        return_original: (optional) bool.
                            If return_original is true, it returns a hierarchical dataframe
                            of the scores of the original classifier under key 'original'and the
                            scores of the updated classifier under key 'updated'.
                            If return_original is false it returns a dataframe of the scores of the
                            updated classifier only.
        returns
        -------
        either a dict of pandas dataframes or a single pandas dataframe, depending on the value of
        return original.
        """
        if metrics is None:
            metrics = group_metrics.default_group_metrics
        if data is None:
            data = self.validation_data
            y_true = self.y_true
            new_pred_proba = self.predict_proba(data)
            if return_original:
                orig_pred_proba = self.predictor.predict_proba(data)
        else:
            y_true = np.asarray(data[self.predictor.label])
            new_pred_proba = self.predict_proba(data)
            if return_original:
                orig_pred_proba = self.predictor.predict_proba(data)
            y_true = (y_true == self.predictor.positive_class) * 1

        groups = self.groups_to_numpy(groups, data)

        if return_original:
            original = self.evaluate_predictor_binary(y_true,
                                                      orig_pred_proba,
                                                      groups,
                                                      metrics=metrics,
                                                      verbose=verbose)

        updated = self.evaluate_predictor_binary(y_true,
                                                 new_pred_proba,
                                                 groups,
                                                 metrics=metrics,
                                                 verbose=verbose)

        out = updated
        if return_original:
            out = pd.concat([original, updated], keys=['original', 'updated'])
        return out

    def evaluate_predictor_binary(self, y_true, proba, groups, metrics, *, verbose=True) -> pd.DataFrame:
        """Helper function for evaluate_groups
        Compute standard per-group  metrics for binary classification
        parameters
        ----------
        y_true: numpy array containing true binary labels of the dataset
        proba: numpy or pandas array containing the output of predict_proba
        groups: numpy array containing discrete group labelling
        metrics: a dictionary where keys are the names and values are either Scorable
        or a BaseGroupMetrics.
        returns
        -------
        a pandas dataframe of fairness metrics.
        Rows are indexed by 'Overall', 'Maximum difference', or group labels.
        Columns are indexed by metric names.
        """
        proba = np.asarray(proba)
        group_names = np.unique(groups)

        names = metrics.keys()
        names = list(names)

        overall_scores = list(map(lambda n: dispatch_metric(metrics[n], y_true, proba, groups),
                                  names))
        scores = list(map(lambda n: dispatch_metric_per_group(metrics[n], y_true, proba, groups),
                          names))

        scores = np.stack(scores)
        overall_scores = np.stack(overall_scores)

        if verbose is False:
            pandas_names = names
        else:
            pandas_names = list(map(lambda n: metrics[n].name, names))
        gap = (scores.max(-1) - scores.min(-1)).reshape(-1, 1)
        collect = np.hstack((overall_scores.reshape(-1, 1), scores, gap))
        out = pd.DataFrame(collect.T, index=(['Overall'] + group_names.tolist()
                                             + ["Maximum difference"]),
                           columns=pandas_names)
        if verbose:
            out.index.name = 'Groups'
        else:
            out.index.name = 'groups'
        return out

    def predict_proba(self, data, *, transform_features=True) -> pd.DataFrame:
        """Duplicates the functionality of predictor.predict_proba with the updated predictor.
        parameters
        ----------
        data a pandas array to make predictions over.
        return
        ------
        a  pandas array of scores. Note, these scores are not probabilities, and not guarenteed to
        be non-negative or to sum to 1.
        """
        proba: pd.DataFrame = self.predictor.predict_proba(data,
                                                           transform_features=transform_features)

        if self.inferred_groups is False:
            if self.groups is False:
                onehot = np.ones((data.shape[0], 1))
            else:
                groups = self.groups_to_numpy(self.groups, data)
                onehot = self.group_encoder.transform(groups.reshape(-1, 1)).toarray()
        else:
            onehot = np.asarray(self.inferred_groups.predict_proba(data))
        if self.use_fast:
            tmp = np.zeros_like(proba)
            tmp[:, 1] = self.offset[onehot.argmax(1)]
        else:
            tmp = onehot.dot(self.offset)
        if self.round is not False:
            proba = np.around(proba / self.round) * self.round
        proba += tmp
        return proba

    def predict(self, data, *, transform_features=True) -> pd.Series:
        "duplicates the functionality of predictor.predict but with the fair predictor"
        return self.predict_proba(data, transform_features=transform_features).idxmax(1)


def _needs_groups(func) -> bool:
    """Internal helper function. Check if a metric is a scorer. If not assume it requires a group
    argument.
    parameters
    ----------
    func either a Scorable or GroupMetric
    """
    return not isinstance(func, Scorer)


def inferred_attribute_builder(train, target, protected, *args):
    """Helper function that trains tabular predictors suitible for use when the protected attribute
        is inferred when enforcing fairness.
        parameters
        ----------
        train: a pandas dataframe
        target: a string identifying the column of the dataframe the predictor should try to
        estimate.
        protected: a string identifying the column of the dataframe that represents the
        protected attribute.
        returns
        -------
        a pair of autogluon tabular predictors.
            1. a predictor predicting the target that doesn't use the protected attribute
            2. a predictor predicting the protected attribute that doesn't use the target.

        """
    target_train = train.drop(protected, axis=1, inplace=False)
    protected_train = train.drop(target, axis=1, inplace=False)
    target_predictor = TabularPredictor(label=target).fit(train_data=target_train, *args)
    protected_predictor = TabularPredictor(label=protected).fit(train_data=protected_train, *args)
    return target_predictor, protected_predictor


def fix_groups(metric: BaseGroupMetric, groups):
    """fixes the choice of groups so that BaseGroupMetrics can be passed as Scorable analogs to the
    slow pathway.

    Parameters
    ----------
    metric: a BaseGroupMetric
    groups: a 1D pandas dataframe or numpy array

    Returns
    -------
    a function that takes y_true and y_pred as an input.

        todo: return scorable"""
    groups = np.asarray(groups)

    def new_metric(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return metric(y_true, y_pred, groups)
    return new_metric


def dispatch_metric(metric, y_true, proba, groups) -> np.ndarray:
    """Helper function for making sure different types of Scorer and GroupMetrics get the right data

    Parameters
    ----------
    metric: a BaseGroupMetric or Scorable
    y_true: a binary numpy array indicating positive or negative labels
    proba: a 2xdatapoints numpy or pandas array
    groups: a numpy array indicating group membership.

    Returns
    -------
     a numpy array containing the score provided by metrics
    """
    proba = np.asarray(proba)
    try:
        if isinstance(metric, BaseGroupMetric):
            return metric(y_true, proba.argmax(1), groups)[0]

        if isinstance(metric, Scorer) and (metric.needs_pred is False):
            return metric(y_true, proba[:, 1] - proba[:, 0])

        return metric(y_true, proba.argmax(1))
    except ValueError:
        return '-'


def dispatch_metric_per_group(metric, y_true: np.ndarray, proba: np.ndarray,
                              groups: np.ndarray) -> np.ndarray:
    """Helper function for making sure different types of Scorer and GroupMetrics get the right data
    parameters
    ----------
    metric: a GroupMetric or Scorable
    y_true: a binary numpy array indicating positive or negative labels
    proba: a 2xdatapoints numpy or pandas array
    groups: a numpy array indicating group membership.

    returns
    -------
    a numpy array containing the per group score provided by metrics """
    if isinstance(metric, group_metrics.GroupMetric):
        return metric.per_group(y_true, proba.argmax(1), groups)[0]
    unique = np.unique(groups)
    out = np.empty_like(unique, dtype=float)
    if isinstance(metric, Scorer) and (metric.needs_pred is False):
        for i, grp in enumerate(unique):
            mask = (grp == groups)
            try:
                out[i] = metric(y_true[mask], proba[mask, 1] - proba[mask, 0])
            except ValueError:
                out[i] = np.nan
    else:
        out = metric(y_true, proba.argmax(1), groups)

    return out

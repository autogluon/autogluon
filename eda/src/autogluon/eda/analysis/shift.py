import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from autogluon.core.constants import BINARY
from autogluon.core.metrics import BINARY_METRICS, roc_auc
from autogluon.core.utils import generate_train_test_split
from autogluon.tabular import TabularPredictor

from .. import AnalysisState
from ..state import StateCheckMixin
from .base import AbstractAnalysis

__all__ = ["XShiftDetector"]


class XShiftDetector(AbstractAnalysis, StateCheckMixin):
    """Detect a change in covariate (X) distribution between training and test, which we call XShift.  It can tell you
    if your training set is not representative of your test set distribution.  This is done with a Classifier 2
    Sample Test.

    State attributes

    - `xshift_results.detection_status`:
        bool, True if detected
    - `xshift_results.test_statistic`: float
        Classifier Two-Sample Test (C2ST) statistic. It is a measure how well a classifier distinguishes between the samples from the training and test sets.
        If the classifier can accurately separate the samples, it suggests that the input distributions differ significantly, indicating the presence of
        covariate shift. A C2ST value close to 0.5 implies that the classifier struggles to differentiate between the sets, indicating minimal covariate shift.
        In contrast, a value significantly different from 0.5 suggests the presence of covariate shift, warranting further investigation and potential
        adjustments to the model or data preprocessing.
    - `xshift_results.pvalue`: float
        p-value using permutation test
    - `xshift_results.pvalue_threshold`: float,
        decision boundary of p-value threshold
    - `xshift_results.feature_importance`: DataFrame,
        the feature importance dataframe, if computed
    - `xshift_results.shift_features`
        list of features whose contribution is statistically significant; only present if `xshift_results.detection_status = True`

    Parameters
    ----------
    classifier_class : an AutoGluon predictor, such as autogluon.tabular.TabularPredictor (default)
        The predictor that will be fit on training set and predict the test set
    compute_fi : bool, default = True
        To compute the feature importances set to True, this can be computationally intensive
    pvalue_thresh : float, default = 0.01
        The threshold for the pvalue
    eval_metric : str, default = 'balanced_accuracy'
        The metric used for the C2ST, it must be one of the binary metrics from autogluon.core.metrics
    sample_label : str, default = '__label__'
        The label internally used for the classifier 2 sample test, the only reason to change it is in the off chance
        that the default value is a column in the data.
    classifier_kwargs : dict, default = {}
        The kwargs passed to the classifier, a member of classifier_class
    classifier_fit_kwargs : dict, default = {}
        The kwargs passed to the classifier's `fit` call, a member of classifier_class
    num_permutations: int, default = 1000
        The number of permutations used for any permutation based method
    test_size_2st: float, default = 0.3
        The size of the test set in the training test split in 2ST

    """

    def __init__(
        self,
        classifier_class: Any = TabularPredictor,
        compute_fi: bool = True,
        pvalue_thresh: float = 0.01,
        eval_metric: str = "roc_auc",
        sample_label: str = "__label__",
        classifier_kwargs: Optional[dict] = None,
        classifier_fit_kwargs: Optional[dict] = None,
        num_permutations: int = 1000,
        test_size_2st: float = 0.3,
        parent: Union[None, AbstractAnalysis] = None,
        children: Optional[List[AbstractAnalysis]] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, children, **kwargs)
        if classifier_kwargs is None:
            classifier_kwargs = {}
        if classifier_fit_kwargs is None:
            classifier_fit_kwargs = {}
        self.classifier_kwargs = classifier_kwargs
        self.classifier_fit_kwargs = classifier_fit_kwargs
        self.classifier_class = classifier_class
        self.compute_fi = compute_fi
        named_metrics = BINARY_METRICS
        assert eval_metric in named_metrics.keys(), (
            "eval_metric must be one of [" + ", ".join(named_metrics.keys()) + "]"
        )
        self.eval_metric = named_metrics[eval_metric]
        self.C2ST = Classifier2ST(
            classifier_class,
            sample_label=sample_label,
            eval_metric=self.eval_metric,
            compute_fi=compute_fi,
            classifier_kwargs=classifier_kwargs,
            test_size_2st=test_size_2st,
        )
        self.fi_scores = None
        self.compute_fi = compute_fi
        self.pvalue_thresh = pvalue_thresh
        self.num_permutations = num_permutations

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return self.all_keys_must_be_present(args, "train_data", "test_data")

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        """Fit method.  `args` can contain
        - 'train_data': pd.DataFrame, required
        - 'test_data': pd.DataFrame, required
        - 'label': str, optional
            The Y variable that is to be predicted (if it appears in the train/test data then it will be removed)
        """
        X = args["train_data"].copy()
        X_test = args["test_data"].copy()
        assert (
            self.C2ST.sample_label not in X.columns
        ), f"your data columns contain {self.C2ST.sample_label} which is used internally"
        if "label" in args:
            label = args["label"]
            if label in X.columns:
                X = X.drop(columns=[label])
            if label in X_test.columns:
                X_test = X_test.drop(columns=[label])
        self.C2ST.fit((X, X_test), **self.classifier_fit_kwargs, **fit_kwargs)
        # Feature importance
        if self.C2ST.has_fi and self.compute_fi:
            fi_scores = self.C2ST.feature_importance()
        else:
            fi_scores = None
        pvalue = self.C2ST.pvalue(num_permutations=self.num_permutations)

        detection_status = bool(pvalue <= self.pvalue_thresh)  # numpy.bool_ -> bool

        state.xshift_results = {
            "detection_status": detection_status,
            "test_statistic": self.C2ST.test_stat,
            "pvalue": pvalue,
            "pvalue_threshold": self.pvalue_thresh,
            "eval_metric": self.eval_metric.name,
            "feature_importance": fi_scores,
        }

        if detection_status:
            fi_scores = fi_scores[fi_scores.p_value <= self.pvalue_thresh]
            shift_features = fi_scores.index.tolist()
            state.xshift_results["shift_features"] = shift_features


def post_fit(func):
    """decorator for post-fit methods"""

    def pff_wrapper(self, *args, **kwargs):
        assert self._is_fit, f".fit needs to be called prior to .{func.__name__}"
        return func(self, *args, **kwargs)

    return pff_wrapper


class Classifier2ST:
    """A classifier 2 sample test, which tests for a difference between a source and target dataset.  It fits a
    classifier to predict if a sample is in the source and target dataset, then computes an evaluation metric on a
    holdout which becomes the test statistic.

    Parameters
    ----------
    classifier_class : an AutoGluon predictor, such as autogluon.tabular.TabularPredictor
        The predictor (classifier) class to classify the source from target dataset, predictor class needs to support
        binary classification
    sample_label : str, default = 'xshift_label'
        The label that will be used to indicate if the sample is from training or test
    eval_metric : callable, default = autogluon.core.metrics.balanced_accuracy
        Binary classification metric to use for the classifier 2 sample test, currently only metrics that accept binary
        predictions are supported, such as balanced_accuracy
    compute_fi : bool, default = True
        To compute the feature importances set to True, this can be computationally intensive
    split : float, default = 0.5
        Training/test split proportion for classifier 2 sample test
    classifier_kwargs : dict, default = {}
        The kwargs passed to the classifier, a member of classifier_class
    test_size_2st: float, default = 0.3
        The size of the test set in the training test split in 2ST
    """

    def __init__(
        self,
        classifier_class,
        sample_label="xshift_label",
        eval_metric=roc_auc,
        split=0.5,
        compute_fi=True,
        classifier_kwargs: Optional[Dict] = None,
        test_size_2st=0.3,
    ):
        if classifier_kwargs is None:
            classifier_kwargs = {}
        else:
            classifier_kwargs = copy.deepcopy(classifier_kwargs)
        classifier_kwargs.update({"label": sample_label, "eval_metric": eval_metric})
        self.classifier = classifier_class(**classifier_kwargs)
        self.classifier_class = classifier_class
        self.split = split
        self.sample_label = sample_label
        self.eval_metric = eval_metric
        self._is_fit = False
        self._test = None
        self.test_stat = None
        self.has_fi: Optional[bool] = None
        self.compute_fi = compute_fi
        self.test_size_2st = test_size_2st

    @staticmethod
    def _make_source_target_label(data, sample_label):
        """Turn a source, target pair into a single dataframe with label column"""
        source, target = data[0].copy(), data[1].copy()
        source.loc[:, sample_label] = 0
        target.loc[:, sample_label] = 1
        data = pd.concat((source, target), ignore_index=True)
        return data

    def fit(self, data, **kwargs):
        """Fit the classifier for predicting if source or target and compute the 2-sample test statistic.

        Parameters
        ----------
        data : pd.DataFrame, or tuple
            either
            - a dataframe with a label column where 1 = target and 0 = source
            - a tuple of source dataframe and target dataframe
        """
        if isinstance(data, pd.DataFrame):
            sample_label = self.sample_label
            assert sample_label in data.columns, "sample_label needs to be a column of data"
            assert self.split, "sample_label requires the split parameter"
            data = data.copy()  # makes a copy
        else:
            assert len(data) == 2, "Data needs to be tuple/list of (source, target) if sample_label is None"
            data = self._make_source_target_label(data, self.sample_label)  # makes a copy
        if data.index.has_duplicates:
            data = data.reset_index(drop=True)
        train, test, y_train, y_test = generate_train_test_split(
            data.drop(columns=[self.sample_label]), data[self.sample_label], BINARY, test_size=self.test_size_2st
        )
        train[self.sample_label] = y_train
        test[self.sample_label] = y_test
        self.classifier.fit(train, **kwargs)
        yhat = self.classifier.predict_proba(test)[1]
        self.test_stat = self.eval_metric(test[self.sample_label], yhat)
        self.has_fi = getattr(self.classifier, "feature_importance", None) is not None
        if self.has_fi and self.compute_fi:
            self._test = test  # for feature importance
        self._is_fit = True

    @post_fit
    def _pvalue_half_permutation(self, num_permutations=1000):
        """The half permutation method for computing p-values.
        See Section 9.2 of https://arxiv.org/pdf/1602.02210.pdf
        """
        perm_stats = [self.test_stat]
        yhat = self.classifier.predict_proba(self._test)[1]
        for _ in range(num_permutations):
            perm_yhat = np.random.permutation(yhat)
            perm_test_stat = self.eval_metric(self._test[self.sample_label], perm_yhat)  # type: ignore
            perm_stats.append(perm_test_stat)
        pval = (self.test_stat <= np.array(perm_stats)).mean()
        return pval

    @post_fit
    def pvalue(self, num_permutations: int = 1000):
        """Compute the p-value which measures the significance level for the test statistic

        Parameters
        ----------
        num_permutations: int, default = 1000
            The number of permutations used for any permutation based method

        Returns
        -------
        float of the p-value for the 2-sample test
        """
        pval = self._pvalue_half_permutation(num_permutations=num_permutations)
        return pval

    @post_fit
    def feature_importance(self):
        """Returns the feature importances for the trained classifier for source v. target

        Returns
        -------
        pd.DataFrame of feature importances
        """
        assert self.has_fi, "Classifier class does not have feature_importance method"
        assert self.compute_fi, "Set compute_fi to True to compute feature importances"
        fi_scores = self.classifier.feature_importance(self._test)
        return fi_scores

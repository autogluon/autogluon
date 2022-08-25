## This is the public API that should be exposed to the general user
import pandas as pd
from autogluon.core.metrics import BINARY_METRICS
import warnings
from ..utils import post_fit
from ..models.classifier2ST import Classifier2ST
from typing import Any, Union, Optional


class C2STShiftDetector:
    """Detect a change in covariate (X) distribution between training and test, which we call XShift.  It can tell you
    if your training set is not representative of your test set distribution.  This is done with a Classifier 2
    Sample Test.

    Parameters
    ----------
    classifier_class : an AutoGluon predictor, such as autogluon.tabular.TabularPredictor
        The predictor that will be fit on training set and predict the test set
    label : str, default = None
        The Y variable that is to be predicted (if it appears in the train/test data then it will be removed)
    eval_metric : str, default = 'balanced_accuracy'
        The metric used for the C2ST, it must be one of the binary metrics from autogluon.core.metrics
    sample_label : str, default = 'i2vkyc0p64'
        The label internally used for the classifier 2 sample test, the only reason to change it is in the off chance
        that the default value is a column in the data.
    classifier_kwargs : dict, default = {}
        The kwargs passed to the classifier, a member of classifier_class

    Methods
    -------
    fit : fits the detector on training and test covariate data
    results, summary : outputs the results of XShift detection
        - test statistic
        - de
        - detector feature importances
    anomaly_scores : computes anomaly scores for test samples
    pvalue: a p-value for the two sample test

    Usage
    -----
    >>> xshiftd = XShiftDetector(TabularPredictor, label='class')
    Fit the detector...
    >>> xshiftd.fit(X, X_test)
    Output the decision...
    >>> xshiftd.decision()
    Output the summary...
    >>> xshiftd.summary()
    Return the anomaly scores...
    >>> asdf = xshiftd.anomaly_scores()
    """

    def __init__(self,
                 classifier_class: Any,
                 label: Optional[str]=None,
                 eval_metric: str='balanced_accuracy',
                 sample_label: str='i2vkyc0p64',
                 classifier_kwargs: dict={}):
        named_metrics = BINARY_METRICS
        assert eval_metric in named_metrics.keys(), \
            'eval_metric must be one of [' + ', '.join(named_metrics.keys()) + ']'
        self.eval_metric = named_metrics[eval_metric]  #is this necessary?
        self.C2ST = Classifier2ST(classifier_class,
                                  sample_label=sample_label,
                                  eval_metric=self.eval_metric,
                                  classifier_kwargs=classifier_kwargs)
        if not label:
            warnings.warn('label is not specified, please ensure that X, X_test do not have the Y (label) '
                          'variable')
        self.label = label
        self._is_fit = False
        self.anomalies = None
        self.fi_scores = None
        self.teststat_thresh = None

    def fit(self,
            X: pd.DataFrame,
            X_test: pd.DataFrame,
            compute_fi: bool=True,
            **kwargs):
        """Fit the XShift detector.

        Parameters
        ----------
        X, X_test : pd.DataFrame
            Training dataframe and test dataframe
        compute_fi : bool, default = True
            True to compute the feature importances, this may be computationally intensive
        **kwargs (optional): keyword arguments to .fit() for the classifier_class
        """
        assert self.C2ST.sample_label not in X.columns, \
            f'your data columns contain {self.C2ST.sample_label} which is used internally'

        if self.label:
            if self.label in X.columns:
                X = X.drop(columns=[self.label])
            if self.label in X_test.columns:
                X_test = X_test.drop(columns=[self.label])

        self.C2ST.fit((X, X_test), **kwargs)

        # Feature importance
        if self.C2ST.has_fi and compute_fi:
            self.fi_scores = self.C2ST.feature_importance()

        self._is_fit = True
        self._X_test = X_test

    @post_fit
    def decision(self,
                 teststat_thresh: float=0.55) -> str:
        """Decision function for testing XShift.  Uncertainty quantification is currently not supported.

        Parameters
        ----------
        teststat_thresh : float
            the threshold for the test statistic

        Returns
        -------
        One of ['detected', 'not detected']
        """
        # default teststat_thresh by metric
        self.teststat_thresh = teststat_thresh
        if self.C2ST.test_stat > teststat_thresh:
            return 'detected'
        else:
            return 'not_detected'

    @post_fit
    def anomaly_scores(self,
                       join_test: bool=True,
                       how: str='all',
                       sample_size: int=100) -> pd.DataFrame:
        """Return anomaly scores for all points

        Parameters
        ----------
        join_test : bool, default = True
            If True, join the anomaly scores to the original test dataframe
        how: str, default = 'all'
            - 'all' = all test points
            - 'rand' = random selection of held out rows in test set
            - 'top' = most anomalous values in held out test set
        sample_size: int
            size of the subsample to compute anomaly scores, only relevant for how = 'rand', 'top'

        Returns
        -------
        pd.DataFrame of test data anomaly score (joined with test dataframe if join_test=True)
        """
        as_top = self.C2ST.sample_anomaly_scores(how=how, sample_size=sample_size)
        as_top = as_top[[1]].rename(columns={1: 'xshift_test_proba'})
        if join_test:
            return as_top.join(self._X_test)
        return as_top

    @post_fit
    def results(self) -> dict:
        """Output the results of the C2ST in dictionary

        Returns
        -------
        dict of
            - `detection_status`: One of ['detected', 'not detected']
            - `test_statistic`: the C2ST statistic
            - `feature_importance`: the feature importance dataframe, if computed
        """
        res_json = {
            'detection_status': self.decision(),
            'test_statistic': self.C2ST.test_stat,
        }
        if self.fi_scores is not None:
            res_json['feature_importance'] = self.fi_scores
        return res_json

    @post_fit
    def summary(self,
                format: str="markdown") -> str:
        """Output the results of C2ST in a human readable format

        Parameters
        ----------
        format : str, default = 'markdown'
            The format of the outputted string

        Returns
        -------
        str of summary
        """
        assert format == 'markdown', 'Only markdown format is supported'
        if self.decision() == 'not_detected':
            ret_md = (
                f"# Detecting distribution shift\n"
                f"We did not detect a substantial difference between the training and test X distributions."
            )
            return ret_md
        else:
            ret_md = (
                f"# Detecting distribution shift\n"
                f"We detected a substantial difference between the training and test X distributions,\n"
                f"a type of distribution shift.\n"
                f"\n"
                f"## Test results\n"
                f"We can predict whether a sample is in the test vs. training set with a {self.eval_metric.name} of\n"
                f"{self.C2ST.test_stat} (larger than the threshold of {self.teststat_thresh}).\n"
                f"\n"
            )
            if self.fi_scores is not None:
                fi_md = (
                    f"## Feature importances\n"
                    f"The variables that are the most responsible for this shift are those with high feature importance:\n"
                    f"{self.fi_scores.to_markdown()}"
                )
                return ret_md + fi_md
        return ret_md

    @post_fit
    def pvalue(self,
               method: str='half permutation',
               num_permutations: int=1000) -> float:
        """Compute the p-value which measures the significance level for the test statistic

        Parameters
        ----------
        method : str
            One of 'half permutation' (method 1 of https://arxiv.org/pdf/1602.02210.pdf), ...
        num_permutations: int, default = 1000
            The number of permutations used for any permutation based method

        Returns
        -------
        float of the p-value for the 2-sample test
        """
        return self.C2ST.pvalue(method=method, num_permutations=num_permutations)

from typing import Union, List, Any, Optional
import warnings
import pandas as pd
import numpy as np
from autogluon.core.metrics import balanced_accuracy, BINARY_METRICS
from autogluon.tabular import TabularPredictor
from .. import AnalysisState
from .base import AbstractAnalysis


def post_fit(func):
    """decorator for post-fit methods"""

    def pff_wrapper(self, *args, **kwargs):
        assert self._is_fit, f'.fit needs to be called prior to .{func.__name__}'
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
        Binary classification metric to use for the classifier 2 sample test
    split : float, default = 0.5
        Training/test split proportion for classifier 2 sample test
    classifier_kwargs : dict, default = {}
        The kwargs passed to the classifier, a member of classifier_class
    """
    def __init__(self,
                 classifier_class,
                 sample_label='xshift_label',
                 eval_metric=balanced_accuracy,
                 split=0.5,
                 compute_fi = True,
                 classifier_kwargs = {}
                 ):
        classifier_kwargs.update({'label': sample_label, 'eval_metric': eval_metric})
        self.classifier = classifier_class(**classifier_kwargs)
        self.classifier_class = classifier_class
        self.split = split
        self.sample_label = sample_label
        self.eval_metric = eval_metric
        self._is_fit = False
        self._test = None
        self.test_stat = None
        self.has_fi = None
        self.compute_fi = compute_fi

    @staticmethod
    def _make_source_target_label(data, sample_label):
        """Turn a source, target pair into a single dataframe with label column"""
        source, target = data[0].copy(), data[1].copy()
        source.loc[:,sample_label] = 0
        target.loc[:,sample_label] = 1
        data = pd.concat((source, target))
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
            data = data.copy() # makes a copy
        else:
            assert len(data) == 2, "Data needs to be tuple/list of (source, target) if sample_label is None"
            data = self._make_source_target_label(data, self.sample_label) # makes a copy
        if data.index.has_duplicates:
            data.index = pd.RangeIndex(data.shape[0])
        train = data.sample(frac=self.split)
        test = data.drop(train.index)
        self.classifier.fit(train, **kwargs)
        yhat = self.classifier.predict(test)
        self.test_stat = self.eval_metric(test[self.sample_label], yhat)
        self.has_fi = (getattr(self.classifier, "feature_importance", None) is not None)
        if self.has_fi and self.compute_fi:
            self._test = test  # for feature importance
        self._is_fit = True

    @post_fit
    def _pvalue_half_permutation(self,
                                 num_permutations=1000):
        """The half permutation method for computing p-values.
        See Section 9.1 of https://arxiv.org/pdf/1602.02210.pdf

        Parameters
        ----------
        num_permutations: int, default = 1000
            The number of permutations for the permutation test

        Returns
        -------
        float of the p-value for the 2-sample test
        """
        perm_stats = [self.test_stat]
        yhat = self.classifier.predict(self._test)
        for i in range(num_permutations):
            perm_yhat = np.random.permutation(yhat)
            perm_test_stat = self.eval_metric(
                self._test[self.sample_label],
                perm_yhat
            )
            perm_stats.append(perm_test_stat)
        pval = (self.test_stat <= np.array(perm_stats)).mean()
        return pval

    @post_fit
    def pvalue(self,
               num_permutations: int=1000):
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
        - detection status
        - p-value
        - detector feature importances
    anomaly_scores : computes anomaly scores for test samples
    pvalue: a p-value for the two sample test

    Usage
    -----
    >>> xshiftd = C2STShiftDetector(TabularPredictor, label='class')
    Fit the detector...
    >>> xshiftd.fit(X, X_test)
    Output the decision...
    >>> xshiftd.decision()
    Output the summary...
    >>> xshiftd.summary()
    """

    def __init__(self,
                 classifier_class: Any,
                 label: Optional[str]=None,
                 compute_fi: bool = True,
                 eval_metric: str='balanced_accuracy',
                 sample_label: str='i2vkyc0p64',
                 classifier_kwargs: dict={}):
        named_metrics = BINARY_METRICS
        assert eval_metric in named_metrics.keys(), \
            'eval_metric must be one of [' + ', '.join(named_metrics.keys()) + ']'
        self.eval_metric = named_metrics[eval_metric]
        self.C2ST = Classifier2ST(classifier_class,
                                  sample_label=sample_label,
                                  eval_metric=self.eval_metric,
                                  compute_fi = compute_fi,
                                  classifier_kwargs=classifier_kwargs)
        if not label:
            warnings.warn('label is not specified, please ensure that train_data, test_data do not have the Y (label) '
                          'variable')
        self.label = label
        self._is_fit = False
        self.fi_scores = None
        self.compute_fi = compute_fi

    def fit(self,
            X: pd.DataFrame,
            X_test: pd.DataFrame,
            **kwargs):
        """Fit the XShift detector.

        Parameters
        ----------
        X : pd.DataFrame
            Training dataframe
        X_test : pd.DataFrame
            Test dataframe
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
        if self.C2ST.has_fi and self.compute_fi:
            self.fi_scores = self.C2ST.feature_importance()

        self._is_fit = True
        self._X_test = X_test

    @post_fit
    def decision(self,
                 pvalue_thresh: float=0.01,
                 pvalue_kwargs: dict = {}) -> str:
        """Decision function for testing XShift.  Uncertainty quantification is currently not supported.

        Parameters
        ----------
        teststat_thresh : float, default = 0.55
            the threshold for the test statistic

        Returns
        -------
        One of ['detected', 'not detected']
        """
        # default teststat_thresh by metric
        p_value = self.pvalue(**pvalue_kwargs)
        if p_value < pvalue_thresh:
            return 'detected', p_value
        else:
            return 'not_detected', p_value

    @post_fit
    def results(self,
                pvalue_thresh: float=0.01,
                pvalue_kwargs: dict={}) -> dict:
        """Output the results of the C2ST in dictionary

        Returns
        -------
        dict of
            - `detection_status`: One of ['detected', 'not detected']
            - `test_statistic`: the C2ST statistic
            - 'pvalue'
            - 'pvalue_threshold'
            - `feature_importance`: the feature importance dataframe, if computed
        """
        det_status, pvalue = self.decision(pvalue_thresh=pvalue_thresh, pvalue_kwargs=pvalue_kwargs)
        res_json = {
            'detection_status': det_status,
            'test_statistic': self.C2ST.test_stat,
            'pvalue': pvalue,
            'pvalue_threshold': pvalue_thresh,
            'eval_metric': self.eval_metric.name,
        }
        if self.fi_scores is not None:
            res_json['feature_importance'] = self.fi_scores
        return res_json

    @post_fit
    def pvalue(self,
               num_permutations: int=1000) -> float:
        """Compute the p-value which measures the significance level for the test statistic

        Parameters
        ----------
        num_permutations: int, default = 1000
            The number of permutations used for any permutation based method

        Returns
        -------
        float of the p-value for the 2-sample test
        """
        return self.C2ST.pvalue(num_permutations=num_permutations)


class XShiftDetector(AbstractAnalysis):
    """Detect a change in covariate (X) distribution between training and test, which we call XShift.  It can tell you
    if your training set is not representative of your test set distribution.  This is done with a Classifier 2
    Sample Test.
    """

    def __init__(self,
                 classifier_class: Union[Any,None] = TabularPredictor,
                 compute_fi: bool = True,
                 classifier_kwargs: dict = {},
                 parent: Union[None,AbstractAnalysis] = None,
                 children: List[AbstractAnalysis] = [],
                 **kwargs) -> None:
        super().__init__(parent, children, **kwargs)
        self.classifier_kwargs = classifier_kwargs
        self.classifier_class = classifier_class
        self.compute_fi = compute_fi

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        # where to put path?
        # how to sample?
        if 'label' in args:
            label = args['label']
        else:
            label = None
        tst = C2STShiftDetector(classifier_class=self.classifier_class,
                                label=label,
                                compute_fi=self.compute_fi,
                                classifier_kwargs=self.classifier_kwargs)
        assert 'train_data' in args, 'train_data required as arg'
        assert 'test_data' in args, 'test_data required as arg'
        tst.fit(X=args['train_data'],
                X_test=args['test_data'],
                verbosity=0)
        state.xshift_results = tst.results()
        pass

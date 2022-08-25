## This API should be assumed to be private and only exposed to developers
## for the public API look at learners

from autogluon.core.metrics import balanced_accuracy
import numpy as np
import pandas as pd
from ..utils import post_fit

class Classifier2ST:
    """A classifier 2 sample test, which tests for a difference between a source and target dataset.  It fits a
    classifier to predict if a sample is in the source and target dataset, then computes a evaluation metric on a
    holdout test which becomes the test statistic.

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
    """
    def __init__(self,
                 classifier_class,
                 sample_label='xshift_label',
                 eval_metric=balanced_accuracy,
                 split=0.5,
                 classifier_kwargs = {}
                 ):
        classifier_kwargs.update({'label': sample_label, 'eval_metric': eval_metric})
        self.classifier_kwargs = classifier_kwargs
        self.classifier = classifier_class(**self.classifier_kwargs)
        self.classifier_class = classifier_class
        self.split = split
        self.sample_label = sample_label
        self.eval_metric = eval_metric
        self._is_fit = False
        self._test = None
        self._train = None
        self.test_stat = None
        self.original_index = None
        self.has_fi = None
        self.fit_kwargs = None

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
        else:
            assert len(data) == 2, "Data needs to be tuple/list of (source, target) if sample_label is None"
            data = self._make_source_target_label(data, self.sample_label)
        if data.index.has_duplicates:
            self.original_index = data.index
            data.index = pd.RangeIndex(data.shape[0])
        self.fit_kwargs = kwargs
        train = data.sample(frac=self.split)
        test = data.drop(train.index)
        self.classifier.fit(train, **kwargs)
        yhat = self.classifier.predict(test)
        self.test_stat = self.eval_metric(test[self.sample_label], yhat)
        self._test = test  # for feature importance and sample anomalies
        self._train = train # for sample anomalies with how = 'all'
        self.has_fi = (getattr(self.classifier, "feature_importance", None) is not None)
        self._is_fit = True

    @post_fit
    def _pvalue_half_permutation(self,
                                 num_permutations=1000):
        """The half permutation method for computing p-values.
        See Section 9.1 of https://arxiv.org/pdf/1602.02210.pdf"""
        perm_stats = [self.test_stat]
        yhat = self.classifier.predict(self._test)
        for i in range(num_permutations):
            perm_yhat = np.random.permutation(yhat)
            perm_test_stat = self.eval_metric(
                self._test[self.sample_label],
                perm_yhat
            )
            perm_stats.append(perm_test_stat)
        p_val = (self.test_stat <= np.array(perm_stats)).mean()
        return p_val

    @post_fit
    def pvalue(self,
               method='half permutation',
               num_permutations=1000):
        """Compute the p-value which measures the significance level for the test statistic

        Parameters
        ----------
        method : str
            one of 'half permutation' (method 1 of https://arxiv.org/pdf/1602.02210.pdf), ...
        num_permutations: int
            the number of permutations used for any permutation based method

        Returns
        -------
        pval: float
            The p-value for the 2-sample test
        """
        valid_methods = [
            'half permutation'
        ]
        assert method in valid_methods, 'method must be one of ' + ', '.join(valid_methods)
        if method == 'half permutation':
            pval = self._pvalue_half_permutation(
                num_permutations=num_permutations)
        return pval

    @post_fit
    def sample_anomaly_scores(self,
                              how='all',
                              sample_size=100):
        """Return anomaly ranks for a subset of test datapoint from target set.  Rank of 1 means most like the target
        set and unlike source set, rank of 0 means opposite.

        Parameters
        ----------
        how: str, default = 'all'
            - 'all' = all test points
            - 'rand' = random selection of held out rows in test set
            - 'top' = most anomalous values in held out test set
        sample_size: int
            size of the subsample to compute anomaly scores, only relevant for how = 'rand', 'top'

        Returns
        -------
        phat: pd.DataFrame
            prediction probability for subset with labels as columns and "rank" as column for the rank
        """
        how_valid = ['all', 'top', 'rand']
        assert how in how_valid, 'parameter how is not in valid set: ' + ' '.join(how_valid)
        if not sample_size:
            sample_size = self._test.shape[0]
        test = self._test.loc[self._test[self.sample_label] == 1].copy()
        if how == 'rand':
            test = test.sample(sample_size)
            phat = self.classifier.predict_proba(test)
            phat_samp = phat.sort_values(1, ascending=False)
        if how == 'top':
            phat = self.classifier.predict_proba(test)
            phat = phat.sort_values(1, ascending=False)
            phat_samp = phat.iloc[:sample_size,:]
        if how == 'all':
            train = self._train.loc[self._train[self.sample_label] == 1].copy()
            phat_test = self.classifier.predict_proba(test)
            phat_test = phat_test.sort_values(1, ascending=False)
            # train a classifier on the remainder on the holdout and predict on the training set
            # this means that all of the original test dataset gets an anomaly score
            classifier_test = self.classifier_class(**self.classifier_kwargs)
            classifier_test.fit(self._test, **self.fit_kwargs)
            phat_train = self.classifier.predict_proba(train)
            phat_train = phat_train.sort_values(1, ascending=False)
            phat_samp = pd.concat((phat_train, phat_test))
        return phat_samp

    @post_fit
    def feature_importance(self):
        """Returns the feature importances for the trained classifier for source v. target
        """
        assert self.has_fi, "Classifier class does not have feature_importance method"
        fi_scores = self.classifier.feature_importance(self._test)
        return fi_scores

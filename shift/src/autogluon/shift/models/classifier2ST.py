## This API should be assumed to be private and only exposed to developers
## for the public API look at learners

from sklearn.metrics import balanced_accuracy_score
import numpy as np
import pandas as pd


class Classifier2ST:
    """A classifier 2 sample test, which tests for a difference between a source and target dataset.  It fits a
    classifier to predict if a sample is in the source and target dataset, then evaluate a holdout test error which
    becomes the test statistic.

    Parameters
    ----------
    classifier : an autogluon predictor, i.e. ...
        the predictor (classifier) to classify the source from target dataset

    Attributes
    ----------
    test_stat: float
        the test statistic, which is the error metric on the test dataset
    """
    def __init__(self,
                 classifier):
        self._classifier = classifier
        self._is_fit = False

    def _post_fit(func):
        """decorator for post-fit methods"""
        def pff_wrapper(self, *args, **kwargs):
            assert self._is_fit, f'.fit needs to be called prior to .{func.__name__}'
            return func(self, *args, **kwargs)
        return pff_wrapper

    @staticmethod
    def _make_source_target_label(data, sample_label):
        """turn a source, target pain into a single dataframe with label column"""
        source, target = data[0].copy(), data[1].copy()
        source.loc[:,sample_label] = 0
        target.loc[:,sample_label] = 1
        data = pd.concat((source, target))
        return data

    def fit(self,
            data,
            sample_label = 'shift_label',
            accuracy_metric = balanced_accuracy_score,
            split=0.5):
        """Fit the classifier for predicting if source or target and compute the 2-sample test statistic.

        Parameters
        ----------
        data : pd.DataFrame, or tuple
            either
            - a dataframe with a label column where 1 = target and 0 = source
            - a tuple of source dataframe and target dataframe
        sample_label : str
            the name of the label that indicates whether the sample is in source or target, None if data is tuple
        """
        if isinstance(data, pd.DataFrame):
            assert sample_label in data.columns, "sample_label needs to be a column of data"
            assert split, "sample_label requires the split parameter"
        else:
            assert len(data) == 2, "Data needs to be tuple/list of (source, target) if sample_label is None"
            data = self._make_source_target_label(data, sample_label)
        if data.index.has_duplicates:
            self.original_index = data.index
            data.index = pd.RangeIndex(data.shape[0])
        else:
            self.original_index = None
        train = data.sample(frac=split)
        test = data.drop(train.index)
        self._classifier.fit(train)
        yhat = self._classifier.predict(test)
        self._accuracy_metric = accuracy_metric
        self._sample_label = sample_label
        self.test_stat = accuracy_metric(test[sample_label], yhat)
        self._test = test # for feature importance and sample anomalies
        self.has_fi = getattr(self._classifier, "feature_importance", None)
        self._is_fit = True

    @_post_fit
    def _pvalue_half_permutation(self,
                                 num_permutations=1000):
        """The half permutation method for computing p-values.
        See Section 9.1 of https://arxiv.org/pdf/1602.02210.pdf"""
        perm_stats = [self.test_stat]
        for i in range(num_permutations):
            perm_data = self._test.copy()
            perm_data[self._sample_label] = \
                np.random.permutation(perm_data[self._sample_label])
            yhat = self._classifier.predict(perm_data)
            perm_test_stat = self._accuracy_metric(
                perm_data[self._sample_label],
                yhat)
            perm_stats.append(perm_test_stat)
        p_val = (self.test_stat <= np.array(perm_stats)).mean()
        return p_val

    @_post_fit
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

    @_post_fit
    def sample_anomaly_scores(self,
                              sample_size=100,
                              how='rand'):
        """Return anomaly ranks for a subset of test datapoint from target set.  Rank of 1 means most like the target
        set and unlike source set, rank of 0 means opposite.

        Parameters
        ----------
        sample_size: int
            size of the subsample to compute anomaly scores
        how: str
            'rand' = random selection of rows in test set
            'top' = most anomalous values in test set

        Returns
        -------
        phat: pd.DataFrame
            prediction probability for subset with labels as columns and "rank" as column for the rank
        """
        how_valid = ['top', 'rand']
        assert how in how_valid, 'how is not in valid set: ' + ' '.join(how_valid)
        test = self._test.loc[self._test[self._sample_label]==1].copy()
        if how == 'rand':
            test = test.sample(sample_size)
            phat = self._classifier.predict_proba(test)
            phat_samp = phat.sort_values(1, ascending=False)
        if how == 'top':
            phat = self._classifier.predict_proba(test)
            phat = phat.sort_values(1, ascending=False)
            phat_samp = phat.iloc[:sample_size,:]
        return phat_samp

    @_post_fit
    def feature_importance(self):
        """Returns the feature importances for the trained classifier for source v. target
        """
        assert self.has_fi, "Classifier class does not have feature_importance method"
        fi_scores = self._classifier.feature_importance(self._test)
        return fi_scores
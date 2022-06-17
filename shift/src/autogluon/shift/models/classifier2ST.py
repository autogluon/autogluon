from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


class Classifier2ST:
    """A classifier 2 sample test, which tests for a difference between a source and target dataset.  It fits a
    classifier to predict if a sample is in the source and target dataset, then evaluate a holdout test error which
    becomes the test statistic.

    Parameters
    ----------
    classifier : an autogluon predictor, i.e. ...
        description
    """
    def __init__(self,
                 classifier):
        self._classifier = classifier

    @staticmethod
    def _make_source_target_label(data, sample_label='label'):
        """turn a source, target pain into a single dataframe with label column"""
        source, target = data
        source.loc[:,sample_label] = 0
        target.loc[:,sample_label] = 1
        data = pd.concat((source, target))
        return data

    def fit(self,
            data,
            sample_label = None,
            accuracy_metric = accuracy_score,
            split=0.5):
        """Fit the classifier for predicting if source or target and compute the test statistic.

        Parameters
        ----------
        data : pd.DataFrame
            description
        """
        if sample_label:
            assert sample_label in data.columns, "sample_label needs to be a column of data"
            assert split, "sample_label requires the split parameter"
            train, _, test = data.random_split(split)
        else:
            assert len(data) == 2, "Data needs to be tuple/list of (source, target) if sample_label is None"
            data = _make_source_target_label(data)
            train, _, test = data.random_split(split)
        self._classifier.fit(train)
        self._test = test
        yhat = self._classifier.predict(test)
        self._accuracy_metric = accuracy_metric
        self._sample_label = sample_label
        self.test_stat = accuracy_metric(test[sample_label], yhat)

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

    def pvalue(self,
               method='half permutation',
               num_permutations=1000):
        """Compute the p-value which measures the significance level for the test statistic

        Parameters
        ----------
        method : str
            one of 'half permutation' (method 1 of https://arxiv.org/pdf/1602.02210.pdf), ...
        """
        valid_methods = [
            'half permutation'
        ]
        assert method in valid_methods, 'method must be one of ' + ', '.join(valid_methods)
        if method == 'half permutation':
            pval = self._pvalue_half_permutation(
                num_permutations=num_permutations)
        return pval
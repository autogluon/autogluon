from sklearn.metrics import accuracy_score
import numpy as np

class Classifier2ST:
    def __init__(self,
                 classifier):
        self._classifier = classifier

    def fit(self,
            data,
            sample_label = None,
            accuracy_metric = accuracy_score,
            split=0.5):

        if sample_label:
            train, _, test = data.random_split(split)
        # need to catch all sorts of data cases
        # particularly add support for passing tuple of source and target
        self._classifier.fit(train)
        self._test = test
        yhat = self._classifier.predict(test)
        self._accuracy_metric = accuracy_metric
        self._sample_label = sample_label
        self.test_stat = accuracy_metric(test[sample_label], yhat)

    def _pvalue_half_permutation(self,
                                 num_permutations=1000):
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
        valid_methods = [
            'half permutation'
        ]
        assert method in valid_methods, \
            'method must be one of ' + ', '.join(valid_methods)
        if method == 'half permutation':
            pval = self._pvalue_half_permutation(
                num_permutations=num_permutations)
        return pval
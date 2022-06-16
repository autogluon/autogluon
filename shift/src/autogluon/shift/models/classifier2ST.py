from sklearn.metrics import accuracy_score

"""
def acc_test_stat(data, pred, split=0.5):
    train, _ ,test = data.random_split(split)
    pred.fit(train)
    test_acc = pred.evaluate(test)
    p = test_acc['top1']
    n = test.shape[0]
    se = (p * (1-p) / n)**0.5
    return p, se

def acc_test(data, pred, split=0.5, perm_num = 100):
    train, _ ,test = data.random_split(split)
    pred.fit(train)
    test_acc = pred.evaluate(test)
    p = test_acc['top1']
    perm_p = [p]
    for i in range(perm_num):
        perm_data = test.copy()
        perm_data['label'] = np.random.permutation(test['label'])
        perm_p.append(pred.evaluate(perm_data)['top1'])
    p_val = (p <= np.array(perm_p)).mean()
    return p, p_val

def sim_null_test(data, pred, split = 0.5, perm_num = 100):
    perm_data = data.copy()
    perm_data['label'] = np.random.permutation(data['label'])
    return acc_test(perm_data, pred, split=split, perm_num=100)
"""

class Classifier2ST:
    def __init__(self, classifier):
        self._classifier = classifier

    def fit(self, data, sample_label = None, accuracy_metric = accuracy_score):
        if sample_label:
            train, _, test = data.random_split(split)
        # need to catch all sorts of data cases
        # particularly add support for passing tuple of source and target
        self._classifier.fit(train)
        self._test = test
        yhat = self._classifier.predict(test)
        self.test_stat = accuracy_metric(test[label], yhat)

    def pvalue(self, method='half permutation'):
        pass

    def sim_null(self):
        pass
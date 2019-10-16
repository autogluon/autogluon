from sklearn.metrics import f1_score
import numpy as np


class DummyData:
    def get_label(self):
        return [0]


def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat)  # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True


# TODO: Add others, such as AUC. Try to generalize this logic to AbstractModel
_eval_metric_dict = {
    f1_score: lgb_f1_score,
}

supported_metrics = _eval_metric_dict.keys()


def get_eval_metric(metric):
    return _eval_metric_dict[metric]

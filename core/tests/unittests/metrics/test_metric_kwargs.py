import numpy as np
import sklearn.metrics

from autogluon.core.metrics import f1
from autogluon.core.metrics import make_scorer


def test_metric_kwargs():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 1])
    score = f1(y_true, y_pred)

    score_pos_label_1 = f1(y_true, y_pred, pos_label=1)
    score_pos_label_0 = f1(y_true, y_pred, pos_label=0)

    assert score == score_pos_label_1
    assert score > score_pos_label_0
    assert score_pos_label_0 == 0.4


def test_metric_kwargs_init():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 1])
    f1_pos_label_0 = make_scorer('f1', sklearn.metrics.f1_score, pos_label=0)
    f1_pos_label_0_v2 = make_scorer('f1', sklearn.metrics.f1_score, metric_kwargs=dict(pos_label=0))
    f1_pos_label_0_test_override = make_scorer('f1', sklearn.metrics.f1_score, metric_kwargs=dict(pos_label=0), pos_label=1)

    score_og = f1(y_true, y_pred)
    score_pos_label_0_og = f1(y_true, y_pred, pos_label=0)
    score_pos_label_0 = f1_pos_label_0(y_true, y_pred)
    score_pos_label_0_v2 = f1_pos_label_0_v2(y_true, y_pred)
    score_pos_label_0_test_override = f1_pos_label_0_test_override(y_true, y_pred)

    assert score_og > score_pos_label_0_og
    assert score_pos_label_0_og == score_pos_label_0
    assert score_pos_label_0_og == score_pos_label_0_v2
    assert score_pos_label_0_og == score_pos_label_0_test_override

    try:
        f1_pos_label_0(y_true, y_pred, pos_label=1)
    except TypeError:
        pass  # TypeError is expected
    else:
        raise AssertionError('metric should raise TypeError if keyword argument is specified in both init and call.')

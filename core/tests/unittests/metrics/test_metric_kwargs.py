import numpy as np
import sklearn.metrics

from autogluon.core.metrics import f1, make_scorer


def test_metric_kwargs():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 1])
    score = f1(y_true, y_pred)

    score_pos_label_1 = f1(y_true, y_pred, pos_label=1)
    score_pos_label_0 = f1(y_true, y_pred, pos_label=0)

    assert score == score_pos_label_1
    assert score > score_pos_label_0
    assert score_pos_label_0 == 0.4

    error = f1.error(y_true, y_pred)
    error_pos_label_1 = f1.error(y_true, y_pred, pos_label=1)
    error_pos_label_0 = f1.error(y_true, y_pred, pos_label=0)

    assert f1.optimum == 1
    assert error == (f1.optimum - score)
    assert error_pos_label_1 == (f1.optimum - score_pos_label_1)
    assert error_pos_label_0 == (f1.optimum - score_pos_label_0)
    assert error_pos_label_0 == 0.6


def test_metric_kwargs_init():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 1])
    f1_pos_label_0 = make_scorer("f1", sklearn.metrics.f1_score, pos_label=0, needs_class=True)
    f1_pos_label_0_v2 = make_scorer("f1", sklearn.metrics.f1_score, metric_kwargs=dict(pos_label=0), needs_class=True)
    f1_pos_label_0_test_override = make_scorer(
        "f1", sklearn.metrics.f1_score, metric_kwargs=dict(pos_label=0), pos_label=1, needs_class=True
    )

    score_og = f1(y_true, y_pred)
    score_pos_label_0_og = f1(y_true, y_pred, pos_label=0)
    score_pos_label_0 = f1_pos_label_0(y_true, y_pred)
    score_pos_label_0_v2 = f1_pos_label_0_v2(y_true, y_pred)
    score_pos_label_0_test_override = f1_pos_label_0_test_override(y_true, y_pred)

    assert score_og > score_pos_label_0_og
    assert score_pos_label_0_og == score_pos_label_0
    assert score_pos_label_0_og == score_pos_label_0_v2
    assert score_pos_label_0_og == score_pos_label_0_test_override

    # assert that kwargs passed during call override init kwargs
    assert score_og == f1_pos_label_0(y_true, y_pred, pos_label=1)

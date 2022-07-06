import numpy as np
from autogluon.core.metrics import f1


def test_metric_kwargs():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 1])
    score = f1(y_true, y_pred)

    score_pos_label_1 = f1(y_true, y_pred, pos_label=1)
    score_pos_label_0 = f1(y_true, y_pred, pos_label=0)

    assert score == score_pos_label_1
    assert score > score_pos_label_0
    assert score_pos_label_0 == 0.4

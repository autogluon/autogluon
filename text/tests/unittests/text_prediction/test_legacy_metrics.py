import pytest
from autogluon.text.text_prediction.metrics import calculate_metric_by_expr


def test_calculate_metric_by_expr():
    metrics = {'label1': {'acc': 0.9, 'f1': 0.8, 'auc': 0.9},
               'label2': {'mse': 0.1, 'mae': 0.2}}
    label_names = ['label1', 'label2']
    score = calculate_metric_by_expr(label_metric_scores=metrics,
                                     label_names=label_names,
                                     expr='label1.acc - label2.mse')
    assert score == 0.9 - 0.1
    score = calculate_metric_by_expr(label_metric_scores=metrics,
                                     label_names=label_names,
                                     expr='label1.acc * label2.mse')
    assert score == 0.9 * 0.1
    with pytest.raises(Exception):
        calculate_metric_by_expr(label_metric_scores=metrics,
                                 label_names=label_names,
                                 expr='label1.acc * mse')
    metrics2 = {'label1': {'acc': 0.9}, 'label2': {'acc': 0.8}}
    score = calculate_metric_by_expr(label_metric_scores=metrics2,
                                     label_names=label_names,
                                     expr='acc')
    assert score == (0.9 + 0.8) / 2
    metrics2 = {'label1': {'acc': 0.9}, 'label2': {'acc': 0.8}}
    score = calculate_metric_by_expr(label_metric_scores=metrics2,
                                     label_names=label_names,
                                     expr='acc')
    assert score == (0.9 + 0.8) / 2
    metrics2 = {'label1': {'acc': 0.9}, 'label2': {'acc': 0.8}}
    score = calculate_metric_by_expr(label_metric_scores=metrics2,
                                     label_names=label_names,
                                     expr='2.0 / (1 / label1.acc + 1 / label2.acc)')
    assert score == 2.0 / (1 / 0.9 + 1 / 0.8)

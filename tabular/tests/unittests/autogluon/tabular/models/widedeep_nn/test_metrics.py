import pytest
import torchmetrics as tm
from pytorch_widedeep.metrics import Accuracy

from autogluon.core.metrics import CLASSIFICATION_METRICS
from autogluon.tabular.models.widedeep_nn.metrics import *


def test_get_nn_metric():
    metric = get_nn_metric(BINARY, CLASSIFICATION_METRICS['accuracy'], 2)
    assert get_metrics_map(2)['accuracy'] == metric


def test_get_nn_metric_not_supported_binary():
    metric = get_nn_metric(BINARY, CLASSIFICATION_METRICS['quadratic_kappa'], 2)
    assert get_metrics_map(2)['log_loss'] == metric


def test_get_nn_metric_not_supported_regression():
    metric = get_nn_metric(REGRESSION, CLASSIFICATION_METRICS['quadratic_kappa'], 2)
    assert get_metrics_map(2)['mean_squared_error'] == metric


def test_get_objective_binary():
    assert 'binary' == get_objective(BINARY, None)


def test_get_objective_multiclass():
    assert 'multiclass' == get_objective(MULTICLASS, None)


def test_get_objective_multiclass_regression():
    mapping = {
        'root_mean_squared_error': 'root_mean_squared_error',
        'mean_squared_error': 'regression',
        'mean_absolute_error': 'mean_absolute_error',
    }

    for stopping_metric, exp_objective in mapping.items():
        assert exp_objective == get_objective(REGRESSION, stopping_metric)


def test_get_objective_multiclass_regression_unknown():
    assert 'regression' == get_objective(REGRESSION, 'unknown')


def test_get_objective_unsupported():
    with pytest.raises(ValueError):
        get_objective('unsupported', None)


def test_get_monitor_metric_wideep_metric_class():
    assert get_monitor_metric(Accuracy) == 'val_acc'

def test_get_monitor_metric_None():
    assert get_monitor_metric(None) == 'val_loss'

def test_get_monitor_metric_wideep_metric_instance():
    assert get_monitor_metric(Accuracy()) == 'val_acc'


def test_get_monitor_metric_torchmetrics():
    assert get_monitor_metric(tm.F1Score(average='macro', num_classes=2)) == 'val_F1Score'

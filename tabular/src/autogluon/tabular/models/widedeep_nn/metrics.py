import logging
from inspect import isclass

from autogluon.core.constants import BINARY, REGRESSION, MULTICLASS

logger = logging.getLogger(__name__)


def get_metrics_map(num_classes):
    from pytorch_widedeep.metrics import Accuracy, R2Score, F1Score
    import torchmetrics as tm
    num_classes = 2 if num_classes is None else num_classes
    metrics_map = {
        # Regression
        'root_mean_squared_error': tm.MeanSquaredError(squared=False),
        'mean_squared_error': tm.MeanSquaredError(),
        'mean_absolute_error': tm.MeanAbsoluteError(),
        'r2': R2Score,
        # Not supported: 'median_absolute_error': None,

        # Classification
        'accuracy': Accuracy,

        'f1': F1Score,
        'f1_macro': tm.F1Score(average='macro', num_classes=num_classes),
        'f1_micro': tm.F1Score(average='micro', num_classes=num_classes),
        'f1_weighted': tm.F1Score(average='weighted', num_classes=num_classes),

        'roc_auc': tm.AUROC(num_classes=num_classes),

        'precision': tm.Precision(num_classes=num_classes),
        'precision_macro': tm.Precision(average='macro', num_classes=num_classes),
        'precision_micro': tm.Precision(average='micro', num_classes=num_classes),
        'precision_weighted': tm.Precision(average='weighted', num_classes=num_classes),

        'recall': tm.Recall(num_classes=num_classes),
        'recall_macro': tm.Recall(average='macro', num_classes=num_classes),
        'recall_micro': tm.Recall(average='micro', num_classes=num_classes),
        'recall_weighted': tm.Recall(average='weighted', num_classes=num_classes),
        'log_loss': None,

        # Not supported: 'pinball_loss': None
        # Not supported: pac_score
    }
    return metrics_map


def get_nn_metric(problem_type, stopping_metric, num_classes):
    metrics_map = get_metrics_map(num_classes)

    # Unsupported metrics will be replaced by defaults for a given problem type
    objective_func_name = stopping_metric.name
    if objective_func_name not in metrics_map.keys():
        if problem_type == REGRESSION:
            objective_func_name = 'mean_squared_error'
        else:
            objective_func_name = 'log_loss'
        logger.warning(f'Metric {stopping_metric.name} is not supported by this model - using {objective_func_name} instead')

    nn_metric = metrics_map.get(objective_func_name, None)

    return nn_metric


def get_monitor_metric(nn_metric):
    if nn_metric is None:
        return 'val_loss'
    monitor_metric = nn_metric
    if isclass(monitor_metric):
        monitor_metric = monitor_metric()
    if hasattr(monitor_metric, '_name'):
        return f'val_{monitor_metric._name}'
    else:
        return f'val_{monitor_metric.__class__.__name__}'


def get_objective(problem_type, stopping_metric):
    if problem_type == BINARY:
        return 'binary'
    elif problem_type == MULTICLASS:
        return 'multiclass'
    elif problem_type == REGRESSION:
        # See supported objectives: https://pytorch-widedeep.readthedocs.io/en/latest/trainer.html#pytorch_widedeep.training.Trainer
        return {
            'root_mean_squared_error': 'root_mean_squared_error',
            'mean_squared_error': 'regression',
            'mean_absolute_error': 'mean_absolute_error',
        }.get(stopping_metric, 'regression')
    else:
        raise ValueError(f'Unsupported problem type {problem_type}')

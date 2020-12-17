import logging

from autogluon.core.constants import REGRESSION

logger = logging.getLogger(__name__)


def get_objective_func_name(problem_type, stopping_metric):
    from fastai.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, accuracy, FBeta, AUROC, Precision, Recall, r2_score

    metrics_map = {
        # Regression
        'root_mean_squared_error': root_mean_squared_error,
        'mean_squared_error': mean_squared_error,
        'mean_absolute_error': mean_absolute_error,
        'r2': r2_score,
        # Not supported: median_absolute_error

        # Classification
        'accuracy': accuracy,

        'f1': FBeta(beta=1),
        'f1_macro': FBeta(beta=1, average='macro'),
        'f1_micro': FBeta(beta=1, average='micro'),
        'f1_weighted': FBeta(beta=1, average='weighted'),  # this one has some issues

        'roc_auc': AUROC(),

        'precision': Precision(),
        'precision_macro': Precision(average='macro'),
        'precision_micro': Precision(average='micro'),
        'precision_weighted': Precision(average='weighted'),

        'recall': Recall(),
        'recall_macro': Recall(average='macro'),
        'recall_micro': Recall(average='micro'),
        'recall_weighted': Recall(average='weighted'),
        'log_loss': None,
        # Not supported: pac_score
    }

    # Unsupported metrics will be replaced by defaults for a given problem type
    objective_func_name = stopping_metric.name
    if objective_func_name not in metrics_map.keys():
        if problem_type == REGRESSION:
            objective_func_name = 'mean_squared_error'
        else:
            objective_func_name = 'log_loss'
        logger.warning(f'Metric {stopping_metric.name} is not supported by this model - using {objective_func_name} instead')

    if objective_func_name in metrics_map.keys():
        nn_metric = metrics_map[objective_func_name]
    else:
        nn_metric = None
    return nn_metric, objective_func_name


def get_objective_func_to_monitor(objective_func_name):
    monitor_obj_func = {
        'roc_auc': 'auroc',

        'f1': 'f_beta',
        'f1_macro': 'f_beta',
        'f1_micro': 'f_beta',
        'f1_weighted': 'f_beta',

        'precision_macro': 'precision',
        'precision_micro': 'precision',
        'precision_weighted': 'precision',

        'recall_macro': 'recall',
        'recall_micro': 'recall',
        'recall_weighted': 'recall',
        'log_loss': 'valid_loss',
    }
    objective_func_name_to_monitor = objective_func_name
    if objective_func_name in monitor_obj_func:
        objective_func_name_to_monitor = monitor_obj_func[objective_func_name]
    return objective_func_name_to_monitor

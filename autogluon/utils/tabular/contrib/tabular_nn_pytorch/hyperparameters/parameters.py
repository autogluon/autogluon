from autogluon.utils.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION


def get_param_baseline(problem_type, num_classes=None):
    if problem_type == BINARY:
        return get_param_binary_baseline()
    elif problem_type == MULTICLASS:
        return get_param_multiclass_baseline()
    elif problem_type == REGRESSION:
        return get_param_regression_baseline()
    else:
        return get_param_binary_baseline()


def get_param_multiclass_baseline():
    params = {
        'nn.tabular.dropout': 0.1,
        'nn.tabular.bs': 256,
        'nn.tabular.lr': 1e-2,
        'nn.tabular.epochs': 30,
        'nn.tabular.metric': 'accuracy',
        'nn.tabular.early.stopping.min_delta': 0.001,
        'nn.tabular.early.stopping.patience': 7,
    }
    return params


def get_param_binary_baseline():
    params = get_param_multiclass_baseline()
    return params


def get_param_regression_baseline():
    params = {
        'nn.tabular.dropout': 0.1,
        'nn.tabular.bs': 256,
        'nn.tabular.lr': 1e-2,
        'nn.tabular.epochs': 30,
        'nn.tabular.metric': 'root_mean_squared_error',
        'nn.tabular.early.stopping.min_delta': 0.001,
        'nn.tabular.early.stopping.patience': 7,
    }
    return params

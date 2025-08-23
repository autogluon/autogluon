from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS

def get_param_baseline(problem_type, num_classes=None):
    if problem_type == BINARY:
        return get_param_binary_baseline()
    elif problem_type == MULTICLASS:
        return get_param_multiclass_baseline(num_classes=num_classes)
    elif problem_type == SOFTCLASS:
        return get_param_multiclass_baseline(num_classes=num_classes)
    elif problem_type == REGRESSION:
        return get_param_regression_baseline()
    else:
        return get_param_binary_baseline()


def get_base_params():
    base_params = {}
    return base_params


def get_param_binary_baseline():
    params = get_base_params()
    baseline_params = {}
    params.update(baseline_params)
    return params


def get_param_multiclass_baseline(num_classes):
    params = get_base_params()
    baseline_params = {}
    params.update(baseline_params)
    return params


def get_param_regression_baseline():
    params = get_base_params()
    baseline_params = {}
    params.update(baseline_params)
    return params

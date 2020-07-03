from ....constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS

DEFAULT_ITERATIONS = 10000


def get_param_baseline(problem_type, num_classes=None):
    if problem_type == BINARY:
        return get_param_binary_baseline()
    elif problem_type in [MULTICLASS, SOFTCLASS]:
        return get_param_multiclass_baseline(num_classes=num_classes)
    elif problem_type == REGRESSION:
        return get_param_regression_baseline()
    else:
        return get_param_binary_baseline()


def get_param_binary_baseline():
    params = {
        'iterations': DEFAULT_ITERATIONS,
        'learning_rate': 0.1,
    }
    return params


def get_param_multiclass_baseline(num_classes):
    params = {
        'iterations': DEFAULT_ITERATIONS,
        'learning_rate': 0.1,
    }
    return params


def get_param_regression_baseline():
    params = {
        'iterations': DEFAULT_ITERATIONS,
        'learning_rate': 0.1,
    }
    return params

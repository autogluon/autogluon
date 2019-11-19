""" Lists the default (fixed) hyperparameter values we use in our Gradient Boosting model. """

from autogluon.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION

DEFAULT_NUM_BOOST_ROUND = 10000

def get_param_baseline(problem_type, num_classes=None):
    if problem_type == BINARY:
        return get_param_binary_baseline()
    elif problem_type == MULTICLASS:
        return get_param_multiclass_baseline(num_classes=num_classes)
    elif problem_type == REGRESSION:
        return get_param_regression_baseline()
    else:
        return get_param_binary_baseline()


def get_param_baseline_dummy(problem_type, num_classes=None):
    if problem_type == BINARY:
        return get_param_binary_baseline_dummy()
    elif problem_type == MULTICLASS:
        return get_param_multiclass_baseline_dummy(num_classes=num_classes)
    elif problem_type == REGRESSION:
        return get_param_regression_baseline_dummy()
    else:
        return get_param_binary_baseline_dummy()


def get_param_binary_baseline():
    params = {

    }
    return params


def get_param_binary_baseline_dummy():
    params = {
        'iterations': 10000,
        'learning_rate': 0.1,
    }
    return params


def get_param_multiclass_baseline(num_classes):
    params = {

    }
    return params



def get_param_multiclass_baseline_dummy(num_classes):
    params = {
        'iterations': 10000,
        'learning_rate': 0.1,
    }
    return params


def get_param_regression_baseline():
    params = {

    }
    return params


def get_param_regression_baseline_dummy():
    params = {
        'iterations': 10000,
        'learning_rate': 0.1,
    }
    return params

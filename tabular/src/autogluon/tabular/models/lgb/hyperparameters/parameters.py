"""Default (fixed) hyperparameter values used in Gradient Boosting model."""

from autogluon.core.constants import BINARY, MULTICLASS, QUANTILE, REGRESSION, SOFTCLASS

DEFAULT_NUM_BOOST_ROUND = 10000  # default for single training run


def get_lgb_objective(problem_type):
    return {
        BINARY: "binary",
        MULTICLASS: "multiclass",
        QUANTILE: "quantile",
        REGRESSION: "regression",
        SOFTCLASS: "multiclass",
    }[problem_type]


def get_param_baseline(problem_type):
    if problem_type == BINARY:
        return get_param_binary_baseline()
    elif problem_type == MULTICLASS:
        return get_param_multiclass_baseline()
    elif problem_type == REGRESSION:
        return get_param_regression_baseline()
    elif problem_type == SOFTCLASS:
        return get_param_softclass_baseline()
    else:
        return get_param_binary_baseline()


def get_param_binary_baseline():
    params = {
        "learning_rate": 0.05,
    }
    return params


def get_param_multiclass_baseline():
    params = {
        "learning_rate": 0.05,
    }
    return params


def get_param_regression_baseline():
    params = {
        "learning_rate": 0.05,
    }
    return params


def get_param_softclass_baseline():
    params = get_param_multiclass_baseline()
    params.pop("metric", None)
    return params

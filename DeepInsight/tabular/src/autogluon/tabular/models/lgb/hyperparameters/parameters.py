""" Default (fixed) hyperparameter values used in Gradient Boosting model. """

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS

DEFAULT_NUM_BOOST_ROUND = 10000  # default for single training run


def get_param_baseline_custom(problem_type):
    if problem_type == BINARY:
        return get_param_binary_baseline_custom()
    elif problem_type == MULTICLASS:
        return get_param_multiclass_baseline_custom()
    elif problem_type == REGRESSION:
        return get_param_regression_baseline_custom()
    elif problem_type == SOFTCLASS:
        return get_param_softclass_baseline_custom()
    else:
        return get_param_binary_baseline_custom()


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


def get_param_multiclass_baseline_custom():
    params = {
        'learning_rate': 0.03,
        'num_leaves': 128,
        'feature_fraction': 0.9,
        'min_data_in_leaf': 3,
        # TODO: Bin size max increase
    }
    return params


def get_param_binary_baseline():
    params = {
        'num_boost_round': DEFAULT_NUM_BOOST_ROUND,
        'num_threads': -1,
        'learning_rate': 0.05,
        'objective': 'binary',
        'verbose': -1,
        'boosting_type': 'gbdt',
        'two_round': True,
    }
    return params


def get_param_multiclass_baseline():
    params = {
        'num_boost_round': DEFAULT_NUM_BOOST_ROUND,
        'num_threads': -1,
        'learning_rate': 0.05,
        'objective': 'multiclass',
        'verbose': -1,
        'boosting_type': 'gbdt',
        'two_round': True,
    }
    return params


def get_param_regression_baseline():
    params = {
        'num_boost_round': DEFAULT_NUM_BOOST_ROUND,
        'num_threads': -1,
        'learning_rate': 0.05,
        'objective': 'regression',
        'verbose': -1,
        'boosting_type': 'gbdt',
        'two_round': True,
    }
    return params


def get_param_binary_baseline_custom():
    params = {
        'learning_rate': 0.03,
        'num_leaves': 128,
        'feature_fraction': 0.9,
        'min_data_in_leaf': 5,
    }
    return params


def get_param_regression_baseline_custom():
    params = {
        'learning_rate': 0.03,
        'num_leaves': 128,
        'feature_fraction': 0.9,
        'min_data_in_leaf': 5,
    }
    return params


def get_param_softclass_baseline():
    params = get_param_multiclass_baseline()
    params.pop('metric', None)
    return params


def get_param_softclass_baseline_custom():
    params = get_param_multiclass_baseline_custom()
    params.pop('metric', None)
    return params

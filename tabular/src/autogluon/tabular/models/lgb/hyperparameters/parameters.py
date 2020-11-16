""" Default (fixed) hyperparameter values used in Gradient Boosting model. """

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS

DEFAULT_NUM_BOOST_ROUND = 10000  # default for single training run


def get_param_baseline_custom(problem_type, num_classes=None):
    if problem_type == BINARY:
        return get_param_binary_baseline_custom()
    elif problem_type == MULTICLASS:
        return get_param_multiclass_baseline_custom(num_classes=num_classes)
    elif problem_type == REGRESSION:
        return get_param_regression_baseline_custom()
    elif problem_type == SOFTCLASS:
        return get_param_softclass_baseline_custom(num_classes=num_classes)
    else:
        return get_param_binary_baseline_custom()


def get_param_baseline(problem_type, num_classes=None):
    if problem_type == BINARY:
        return get_param_binary_baseline()
    elif problem_type == MULTICLASS:
        return get_param_multiclass_baseline(num_classes=num_classes)
    elif problem_type == REGRESSION:
        return get_param_regression_baseline()
    elif problem_type == SOFTCLASS:
        return get_param_softclass_baseline(num_classes=num_classes)
    else:
        return get_param_binary_baseline()


def get_param_multiclass_baseline_custom(num_classes):
    params = {
        'num_boost_round': DEFAULT_NUM_BOOST_ROUND,
        'num_threads': -1,
        'objective': 'multiclass',
        'num_classes': num_classes,
        'verbose': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 128,
        'feature_fraction': 0.9,
        'min_data_in_leaf': 3,
        'two_round': True,
        'seed_value': 0,
        # 'device': 'gpu'  # needs GPU-enabled lightGBM build
        # TODO: Bin size max increase
    }
    return params.copy()


def get_param_binary_baseline():
    params = {
        'num_boost_round': DEFAULT_NUM_BOOST_ROUND,
        'num_threads': -1,
        'objective': 'binary',
        'verbose': -1,
        'boosting_type': 'gbdt',
        'two_round': True,
    }
    return params


def get_param_multiclass_baseline(num_classes):
    params = {
        'num_boost_round': DEFAULT_NUM_BOOST_ROUND,
        'num_threads': -1,
        'objective': 'multiclass',
        'num_classes': num_classes,
        'verbose': -1,
        'boosting_type': 'gbdt',
        'two_round': True,
    }
    return params


def get_param_regression_baseline():
    params = {
        'num_boost_round': DEFAULT_NUM_BOOST_ROUND,
        'num_threads': -1,
        'objective': 'regression',
        'verbose': -1,
        'boosting_type': 'gbdt',
        'two_round': True,
    }
    return params


def get_param_binary_baseline_dummy_gpu():
    params = {
        'num_boost_round': DEFAULT_NUM_BOOST_ROUND,
        'num_threads': -1,
        'objective': 'binary',
        'verbose': -1,
        'boosting_type': 'gbdt',
        'two_round': True,
        'device_type': 'gpu',
    }
    return params


def get_param_binary_baseline_custom():
    params = {
        'num_boost_round': DEFAULT_NUM_BOOST_ROUND,
        'num_threads': -1,
        'objective': 'binary',
        'verbose': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 128,
        'feature_fraction': 0.9,
        'min_data_in_leaf': 5,
        # 'is_unbalance': True,  # TODO: Set is_unbalanced: True for F1-score, AUC!
        'two_round': True,
        'seed_value': 0,
    }
    return params.copy()


def get_param_regression_baseline_custom():
    params = {
        'num_boost_round': DEFAULT_NUM_BOOST_ROUND,
        'num_threads': -1,
        'objective': 'regression',
        'verbose': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 128,
        'feature_fraction': 0.9,
        'min_data_in_leaf': 5,
        'two_round': True,
        'seed_value': 0,
    }
    return params.copy()


def get_param_softclass_baseline(num_classes):
    params = get_param_multiclass_baseline(num_classes)
    params.pop('metric', None)
    return params.copy()


def get_param_softclass_baseline_custom(num_classes):
    params = get_param_multiclass_baseline_custom(num_classes)
    params.pop('metric', None)
    return params.copy()

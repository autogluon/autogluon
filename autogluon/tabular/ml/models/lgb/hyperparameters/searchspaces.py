""" Lists the default (fixed) hyperparameter search spaces we use in our Gradient Boosting model. """
from autogluon.core import *

# TODO: move these files
from autogluon.tabular.ml.constants import BINARY, MULTICLASS, REGRESSION

DEFAULT_NUM_BOOST_ROUND = 10000 # default for HPO

def get_default_searchspace(problem_type, num_classes=None):
    if problem_type == BINARY:
        return get_searchspace_binary_baseline()
    elif problem_type == MULTICLASS:
        return get_searchspace_multiclass_baseline(num_classes=num_classes)
    elif problem_type == REGRESSION:
        return get_searchspace_regression_baseline()
    else:
        return get_searchspace_binary_baseline()


def get_searchspace_multiclass_baseline(num_classes):
    params = {
        'objective': 'multiclass',
        'metric': 'multi_error,multi_logloss',
        'num_classes': num_classes,
        'learning_rate': Real(lower=1e-3, upper=0.2, default=0.1, log=True),
        'feature_fraction': Real(lower=0.25, upper=1.0, default=1.0),
        'min_data_in_leaf': Int(lower=2, upper=100, default=20),
        'num_leaves': Int(lower=16, upper=96, default=31),
        'num_boost_round': DEFAULT_NUM_BOOST_ROUND,
        'boosting_type': 'gbdt',
        'verbose': -1,
        'two_round': True,
        'seed_value': None,
        # 'device': 'gpu'  # needs GPU-enabled lightGBM build
        # TODO: Bin size max increase
    }
    return params


def get_searchspace_binary_baseline():
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss,binary_error',
        'learning_rate': Real(lower=1e-3, upper=0.2, default=0.1, log=True),
        'feature_fraction': Real(lower=0.25, upper=1.0, default=1.0),
        'min_data_in_leaf': Int(lower=2, upper=100, default=20),
        'num_leaves': Int(lower=16, upper=96, default=31),
        'num_boost_round': DEFAULT_NUM_BOOST_ROUND,
        'boosting_type': 'gbdt',
        'verbose': -1,
        'two_round': True,
        'seed_value': None,
    }
    return params


def get_searchspace_regression_baseline():
    params = {
        'objective': 'regression',
        'metric': 'l1',
        'learning_rate': Real(lower=1e-3, upper=0.2, default=0.1, log=True),
        'feature_fraction': Real(lower=0.25, upper=1.0, default=1.0),
        'min_data_in_leaf': Int(lower=2, upper=100, default=20),
        'num_leaves': Int(lower=16, upper=96, default=31),
        'num_boost_round': DEFAULT_NUM_BOOST_ROUND,
        'boosting_type': 'gbdt',
        'verbose': -1,
        'two_round': True,
        'seed_value': None,
    }
    return params



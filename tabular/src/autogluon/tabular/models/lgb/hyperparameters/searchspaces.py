""" Default hyperparameter search spaces used in Gradient Boosting model """
from autogluon.core import Real, Int
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION


def get_default_searchspace(problem_type):
    if problem_type == BINARY:
        return get_searchspace_binary_baseline()
    elif problem_type == MULTICLASS:
        return get_searchspace_multiclass_baseline()
    elif problem_type == REGRESSION:
        return get_searchspace_regression_baseline()
    else:
        return get_searchspace_binary_baseline()


def get_searchspace_multiclass_baseline():
    params = {
        'learning_rate': Real(lower=5e-3, upper=0.2, default=0.05, log=True),
        'feature_fraction': Real(lower=0.75, upper=1.0, default=1.0),
        'min_data_in_leaf': Int(lower=2, upper=60, default=20),  # TODO: Use size of dataset to set upper, if row count is small upper should be small
        'num_leaves': Int(lower=16, upper=96, default=31),  # TODO: Use row count and feature count to set this, the higher feature count the higher num_leaves upper
        # TODO: Bin size max increase
    }
    return params


def get_searchspace_binary_baseline():
    params = {
        'learning_rate': Real(lower=5e-3, upper=0.2, default=0.05, log=True),
        'feature_fraction': Real(lower=0.75, upper=1.0, default=1.0),
        'min_data_in_leaf': Int(lower=2, upper=60, default=20),
        'num_leaves': Int(lower=16, upper=96, default=31),
    }
    return params


def get_searchspace_regression_baseline():
    params = {
        'learning_rate': Real(lower=5e-3, upper=0.2, default=0.05, log=True),
        'feature_fraction': Real(lower=0.75, upper=1.0, default=1.0),
        'min_data_in_leaf': Int(lower=2, upper=60, default=20),
        'num_leaves': Int(lower=16, upper=96, default=31),
    }
    return params

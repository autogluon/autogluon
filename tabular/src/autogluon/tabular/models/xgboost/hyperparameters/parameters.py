import os
from autogluon.core.constants import BINARY, MULTICLASS, SOFTCLASS, REGRESSION

DEFAULT_NUM_BOOST_ROUND = 10000
MAX_CATEGORY_LEVELS = 100  # maximum number of allowed levels per categorical feature
# Options: [10, 100, 200, 300, 400, 500, 1000, 10000]


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


# TODO: `n_jobs` : the value -1, 0 doesn't set a model to use whole parallel threads in Scikit-learn API.
#  Use the number of CPUs in the training system for now, but it could be lower than the number of CPUs in the inferencing system.
#  xgboost plans to accept -1 for compability with other packages. After that, resolving this issue.
def get_base_params():
    base_params = {
        'n_estimators': DEFAULT_NUM_BOOST_ROUND,
        'learning_rate': 0.1,
        'n_jobs': os.cpu_count(),
        'proc.max_category_levels' : MAX_CATEGORY_LEVELS,
    }
    return base_params


def get_param_binary_baseline():
    params = get_base_params()
    baseline_params = {
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'use_label_encoder': False,
    }
    params.update(baseline_params)
    return params


def get_param_multiclass_baseline(num_classes):
    params = get_base_params()
    baseline_params = {
        'objective': 'multi:softmax',
        'booster': 'gbtree',
        'num_class': num_classes,
        'use_label_encoder': False,
    }
    params.update(baseline_params)
    return params


def get_param_regression_baseline():
    params = get_base_params()
    baseline_params = {
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
    }
    params.update(baseline_params)
    return params

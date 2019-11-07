from tabular.ml.constants import BINARY, MULTICLASS, REGRESSION


def get_param_baseline(problem_type, num_classes=None):
    if problem_type == BINARY:
        return get_param_binary_baseline()
    elif problem_type == MULTICLASS:
        return get_param_multiclass_baseline(num_classes=num_classes)
    elif problem_type == REGRESSION:
        return get_param_regression_baseline()
    else:
        return get_param_binary_baseline()


def get_param_multiclass_baseline(num_classes):
    params = {
        'num_threads': -1,
        'objective': 'multiclass',
        'metric': 'multi_error,multi_logloss',
        'num_classes': num_classes,
        'verbose': -1,

        'boosting_type': 'gbdt',
        'learning_rate': 0.005,
        'num_leaves': 256,
        'feature_fraction': 0.9,
        'min_data_in_leaf': 3,
        'two_round': True,
        # 'device': 'gpu'  # needs GPU-enabled lightGBM build
        # TODO: Bin size max increase
    }
    return params


def get_param_binary_baseline():
    params = {
        'num_threads': -1,
        'objective': 'binary',
        'metric': 'binary_logloss,binary_error',
        'verbose': -1,

        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'num_leaves': 256,
        'feature_fraction': 0.9,
        'min_data_in_leaf': 5,
        # 'is_unbalance': True,  # TODO: Set is_unbalanced: True for F1-score, AUC!
        'two_round': True,
    }
    return params


def get_param_regression_baseline():
    params = {
        'num_threads': -1,
        'objective': 'regression',
        'metric': 'l1',
        'verbose': -1,

        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'num_leaves': 128,
        'feature_fraction': 0.9,
        'min_data_in_leaf': 5,
        'two_round': True,
    }
    return params

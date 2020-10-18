""" Default hyperparameter search spaces used in XGBoost Boosting model """
from .......core import Real, Int
from ....constants import BINARY, MULTICLASS, REGRESSION

DEFAULT_NUM_BOOST_ROUND = 10000  # default for HPO


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
        'objective': 'multi:softmax',
        'num_class': num_classes,
        'n_estimators': DEFAULT_NUM_BOOST_ROUND,
        'booster': 'gbtree',
        'learning_rate': Real(lower=5e-3, upper=0.2, default=0.03, log=True),
        'max_depth': Int(lower=3, upper=10, default=3),
        'min_child_weight': Int(lower=1, upper=5, default=1),
        'gamma': Real(lower=0, upper=5, default=0.01),
        'subsample': Real(lower=0.5, upper=1.0, default=1.0),
        'colsample_bytree': Real(lower=0.5, upper=1.0, default=1.0),
        'reg_alpha': Real(lower=0.0, upper=10.0, default=0.0),
        'reg_lambda': Real(lower=0.0, upper=10.0, default=1.0),
    }
    return params


def get_searchspace_binary_baseline():
    params = {
        'objective': 'binary:logistic',
        'n_estimators': DEFAULT_NUM_BOOST_ROUND,
        'booster': 'gbtree',
        'learning_rate': Real(lower=5e-3, upper=0.2, default=0.03, log=True),
        'max_depth': Int(lower=3, upper=10, default=3),
        'min_child_weight': Int(lower=1, upper=5, default=1),
        'gamma': Real(lower=0, upper=5, default=0.01),
        'subsample': Real(lower=0.5, upper=1.0, default=1.0),
        'colsample_bytree': Real(lower=0.5, upper=1.0, default=1.0),
        'reg_alpha': Real(lower=0.0, upper=10.0, default=0.0),
        'reg_lambda': Real(lower=0.0, upper=10.0, default=1.0),
    }
    return params


def get_searchspace_regression_baseline():
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': DEFAULT_NUM_BOOST_ROUND,
        'booster': 'gbtree',
        'learning_rate': Real(lower=5e-3, upper=0.2, default=0.03, log=True),
        'max_depth': Int(lower=3, upper=10, default=3),
        'min_child_weight': Int(lower=1, upper=5, default=1),
        'gamma': Real(lower=0, upper=5, default=0.01),
        'subsample': Real(lower=0.5, upper=1.0, default=1.0),
        'colsample_bytree': Real(lower=0.5, upper=1.0, default=1.0),
        'reg_alpha': Real(lower=0.0, upper=10.0, default=0.0),
        'reg_lambda': Real(lower=0.0, upper=10.0, default=1.0),
    }
    return params

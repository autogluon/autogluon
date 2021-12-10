""" Default hyperparameter search spaces used in Neural network model """
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, QUANTILE
from autogluon.core import Categorical, Real


def get_default_searchspace(problem_type, num_classes=None):
    params = {
        'learning_rate': Real(1e-4, 3e-2, default=3e-4, log=True),
        'weight_decay': Real(1e-12, 0.1, default=1e-6, log=True),
        'activation': Categorical('relu', 'softrelu'),
        'dropout_prob': Categorical(0.1, 0.0, 0.5, 0.2, 0.3, 0.4),
        'layers': Categorical(None, [200, 100], [256], [100, 50], [200, 100, 50], [1024], [32], [300, 150]),
        'embedding_size_factor': Categorical(1.0, 0.5, 1.5, 0.7, 0.6, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4),
        'network_type': Categorical('widedeep', 'feedforward'),
        'use_batchnorm': Categorical(True, False),
        'batch_size': Categorical(512, 1024, 2056, 128),
        'proc.embed_min_categories': Categorical(4, 3, 10, 100, 1000),
        'proc.impute_strategy': Categorical('median', 'mean', 'most_frequent'),
        'proc.max_category_levels': Categorical(100, 10, 20, 200, 300, 400, 500, 1000, 10000),
        'proc.skew_threshold': Categorical(0.99, 0.2, 0.3, 0.5, 0.8, 0.9, 0.999, 1.0, 10.0, 100.0),
    }
    if problem_type == QUANTILE:
        return get_searchspace_quantile().copy()
    elif problem_type == BINARY:
        problem_params = get_searchspace_binary()
    elif problem_type == MULTICLASS:
        problem_params = get_searchspace_multiclass(num_classes=num_classes)
    elif problem_type == REGRESSION:
        problem_params = get_searchspace_regression()
    params.update(problem_params)
    return params.copy()

def get_searchspace_multiclass(num_classes):
    return {}

def get_searchspace_binary():
    return {}

def get_searchspace_regression():
    params = {
        'weight_decay': Real(1e-12, 1.0, default=1e-6, log=True),
        'activation': Categorical('relu', 'softrelu', 'tanh'),
    }
    return params

def get_searchspace_quantile():
    params = {
        'learning_rate': Real(1e-4, 3e-2, default=3e-4, log=True),
        'weight_decay': Real(1e-12, 0.1, default=1e-6, log=True),
        'dropout_prob': Real(0.0, 0.2, default=0.1),
        'gamma': Real(0.0, 5.0, default=5.0),
        'num_layers': Categorical(2, 3),
        'hidden_size': Categorical(64, 128, 256),
        'embedding_size_factor': Real(0.5, 1.5, default=1.0),
        'alpha': Categorical(0.0, 0.01),
    }
    return params

""" Default (fixed) hyperparameter search spaces used in Tabular Neural Network models.
    MXNet vs Torch backend frameworks use slightly different hyperparameters.
"""

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, QUANTILE
from autogluon.core import Categorical, Real

from .parameters import merge_framework_params


def get_default_searchspace(problem_type, framework, num_classes=None):
    params = {
        'learning_rate': Real(1e-4, 3e-2, default=3e-4, log=True),
        'weight_decay': Real(1e-12, 0.1, default=1e-6, log=True),
        'dropout_prob': Categorical(0.1, 0.0, 0.5, 0.2, 0.3, 0.4),
        'embedding_size_factor': Categorical(1.0, 0.5, 1.5, 0.7, 0.6, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4),
        'proc.embed_min_categories': Categorical(4, 3, 10, 100, 1000),
        'proc.impute_strategy': Categorical('median', 'mean', 'most_frequent'),
        'proc.max_category_levels': Categorical(100, 10, 20, 200, 300, 400, 500, 1000, 10000),
        'proc.skew_threshold': Categorical(0.99, 0.2, 0.3, 0.5, 0.8, 0.9, 0.999, 1.0, 10.0, 100.0),
    }
    mxnet_params = {
        'use_batchnorm': Categorical(True, False),
        'layers': Categorical(None, [200, 100], [256], [100, 50], [200, 100, 50], [1024], [32], [300, 150]),
        'network_type': Categorical('widedeep', 'feedforward'),
        'activation': Categorical('relu', 'softrelu'),
        'batch_size': Categorical(512, 1024, 2056, 128),
    }
    pytorch_params = {
        'use_batchnorm': Categorical(False, True),
        'num_layers': Categorical(2, 3, 4),
        'hidden_size': Categorical(128, 256, 512),
        'activation': Categorical('relu', 'elu'),
    }
    params = merge_framework_params(framework=framework, shared_params=params, mxnet_params=mxnet_params, pytorch_params=pytorch_params)
    if problem_type == QUANTILE:
        problem_params =  get_searchspace_quantile(framework)
    elif problem_type == BINARY:
        problem_params = get_searchspace_binary(framework)
    elif problem_type == MULTICLASS:
        problem_params = get_searchspace_multiclass(framework, num_classes=num_classes)
    elif problem_type == REGRESSION:
        problem_params = get_searchspace_regression(framework)
    params.update(problem_params)
    return params.copy()


def get_searchspace_multiclass(framework, num_classes):
    return {}


def get_searchspace_binary(framework):
    return {}


def get_searchspace_regression(framework):
    params = {
        'weight_decay': Real(1e-12, 1.0, default=1e-6, log=True),
    }
    mxnet_params = {
        'activation': Categorical('relu', 'softrelu', 'tanh'),
    }
    pytorch_params = {
        'activation': Categorical('relu', 'elu', 'tanh'),
    }
    return merge_framework_params(framework=framework, shared_params=params, mxnet_params=mxnet_params, pytorch_params=pytorch_params)


def get_searchspace_quantile(framework):
    if framework != 'pytorch':
        raise ValueError("Only pytorch tabular neural network is currently supported for quantile regression.")
    params = {
        'activation': Categorical('relu', 'elu', 'tanh'),
        'weight_decay': Real(1e-12, 1.0, default=1e-6, log=True),
        'gamma': Real(0.1, 10.0, default=5.0),
        'alpha': Categorical(0.001, 0.01, 0.1, 1.0),
    }
    return params

""" Default (fixed) hyperparameter search spaces used in Tabular Neural Network models.
"""
from autogluon.common import space
from autogluon.core.constants import BINARY, MULTICLASS, QUANTILE, REGRESSION

from .parameters import merge_framework_params


def get_default_searchspace(problem_type, framework, num_classes=None):
    params = {
        "learning_rate": space.Real(1e-4, 3e-2, default=3e-4, log=True),
        "weight_decay": space.Real(1e-12, 0.1, default=1e-6, log=True),
        "dropout_prob": space.Categorical(0.1, 0.0, 0.5, 0.2, 0.3, 0.4),
        "embedding_size_factor": space.Categorical(1.0, 0.5, 1.5, 0.7, 0.6, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4),
        "proc.embed_min_categories": space.Categorical(4, 3, 10, 100, 1000),
        "proc.impute_strategy": space.Categorical("median", "mean", "most_frequent"),
        "proc.max_category_levels": space.Categorical(100, 10, 20, 200, 300, 400, 500, 1000, 10000),
        "proc.skew_threshold": space.Categorical(0.99, 0.2, 0.3, 0.5, 0.8, 0.9, 0.999, 1.0, 10.0, 100.0),
    }
    pytorch_params = {
        "use_batchnorm": space.Categorical(False, True),
        "num_layers": space.Categorical(2, 3, 4),
        "hidden_size": space.Categorical(128, 256, 512),
        "activation": space.Categorical("relu", "elu"),
    }
    params = merge_framework_params(framework=framework, shared_params=params, pytorch_params=pytorch_params)
    if problem_type == QUANTILE:
        problem_params = get_searchspace_quantile(framework)
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
        "weight_decay": space.Real(1e-12, 1.0, default=1e-6, log=True),
    }
    pytorch_params = {
        "activation": space.Categorical("relu", "elu", "tanh"),
    }
    return merge_framework_params(framework=framework, shared_params=params, pytorch_params=pytorch_params)


def get_searchspace_quantile(framework):
    if framework != "pytorch":
        raise ValueError("Only pytorch tabular neural network is currently supported for quantile regression.")
    params = {
        "activation": space.Categorical("relu", "elu", "tanh"),
        "weight_decay": space.Real(1e-12, 1.0, default=1e-6, log=True),
        "gamma": space.Real(0.1, 10.0, default=5.0),
        "alpha": space.Categorical(0.001, 0.01, 0.1, 1.0),
    }
    return params

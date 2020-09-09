""" Default hyperparameter search spaces used in Neural network model """
from ....constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core import Categorical, Real


def get_default_searchspace(problem_type, num_classes=None):
    if problem_type == BINARY:
        return get_searchspace_binary().copy()
    elif problem_type == MULTICLASS:
        return get_searchspace_multiclass(num_classes=num_classes)
    elif problem_type == REGRESSION:
        return get_searchspace_regression().copy()
    else:
        return get_searchspace_binary().copy()


def get_searchspace_multiclass(num_classes):
    # Search space we use by default (only specify non-fixed hyperparameters here):  # TODO: move to separate file
    params = {
        'learning_rate': Real(1e-4, 3e-2, default=3e-4, log=True),
        'weight_decay': Real(1e-12, 0.1, default=1e-6, log=True),
        'dropout_prob': Real(0.0, 0.5, default=0.1),
        # 'layers': Categorical(None, [200, 100], [256], [2056], [1024, 512, 128], [1024, 1024, 1024]),
        'layers': Categorical(None, [200, 100], [256], [100, 50], [200, 100, 50], [50, 25], [300, 150]),
        'embedding_size_factor': Real(0.5, 1.5, default=1.0),
        'network_type': Categorical('widedeep', 'feedforward'),
        'use_batchnorm': Categorical(True, False),
        'activation': Categorical('relu', 'softrelu'),
        # 'batch_size': Categorical(512, 1024, 2056, 128), # this is used in preprocessing so cannot search atm
    }
    return params


def get_searchspace_binary():
    params = {
        'learning_rate': Real(1e-4, 3e-2, default=3e-4, log=True),
        'weight_decay': Real(1e-12, 0.1, default=1e-6, log=True),
        'dropout_prob': Real(0.0, 0.5, default=0.1),
        # 'layers': Categorical(None, [200, 100], [256], [2056], [1024, 512, 128], [1024, 1024, 1024]),
        'layers': Categorical(None, [200, 100], [256], [100, 50], [200, 100, 50], [50, 25], [300, 150]),
        'embedding_size_factor': Real(0.5, 1.5, default=1.0),
        'network_type': Categorical('widedeep', 'feedforward'),
        'use_batchnorm': Categorical(True, False),
        'activation': Categorical('relu', 'softrelu'),
        # 'batch_size': Categorical(512, 1024, 2056, 128), # this is used in preprocessing so cannot search atm
    }
    return params


def get_searchspace_regression():
    params = {
        'learning_rate': Real(1e-4, 3e-2, default=3e-4, log=True),
        'weight_decay': Real(1e-12, 0.1, default=1e-6, log=True),
        'dropout_prob': Real(0.0, 0.5, default=0.1),
        # 'layers': Categorical(None, [200, 100], [256], [2056], [1024, 512, 128], [1024, 1024, 1024]),
        'layers': Categorical(None, [200, 100], [256], [100, 50], [200, 100, 50], [50, 25], [300, 150]),
        'embedding_size_factor': Real(0.5, 1.5, default=1.0),
        'network_type': Categorical('widedeep', 'feedforward'),
        'use_batchnorm': Categorical(True, False),
        'activation': Categorical('relu', 'softrelu', 'tanh'),
        # 'batch_size': Categorical(512, 1024, 2056, 128), # this is used in preprocessing so cannot search atm
    }
    return params

import functools
import types

from ..space import *

__all__ = ['autogluon_optims', 'Optimizer']


def get_hyper_params(self):
    return self.hyper_params


def autogluon_optims(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        optim = func(*args, **kwargs)
        # TODO (cgraywang): parse user provided config to generate hyper_params
        if optim.name == 'sgd':
            setattr(optim, 'hyper_params',
                    [Log('lr', 10 ** -5, 10 ** -1).get_hyper_param(),
                     Linear('momentum', 0.85, 0.95).get_hyper_param()])
        elif optim.name == 'adam':
            setattr(optim, 'hyper_params',
                    [Log('lr', 10 ** -5, 10 ** -1).get_hyper_param()])

        elif optim.name == 'ftml':
            setattr(optim, 'hyper_params',
                    [Log('lr', 10 ** -5, 10 ** -1).get_hyper_param()])

        elif optim.name == 'bertadam':
            setattr(optim, 'hyper_params',
                    [Log('lr', 10 ** -5, 10 ** -1).get_hyper_param()])

        else:
            raise NotImplementedError
        return optim

    return wrapper_decorator


class Optimizer(object):
    def __init__(self, name):
        self.name = name
        self.hyper_params = None

    def get_hyper_params(self):
        return self.hyper_params

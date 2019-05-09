import functools
import types

from ..space import *

__all__ = ['autogluon_optims']


def get_hyper_params(self):
    return self.hyper_params


def autogluon_optims(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        optim = func(*args, **kwargs)
        #TODO: parse user provided config to generate hyper_params
        if optim == 'sgd':
            optim.hyper_params = [Log('learning_rate', 10**-4, 10**-1).get_hyper_param(),
                                  Linear('momentum', 0.85, 0.95).get_hyper_param()]
        elif optim == 'adam':
            optim.hyper_params = [Log('learning_rate', 10 ** -4, 10 ** -1).get_hyper_param()]
        else:
            raise NotImplementedError
        optim.get_hyper_params = types.MethodType(get_hyper_params, optim)
        return optim
    return wrapper_decorator
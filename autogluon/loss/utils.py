import functools
import types

from ..space import *

__all__ = ['autogluon_losses', 'Loss']


def autogluon_losses(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        loss = func(*args, **kwargs)
        #TODO (cgraywang): add more hparams
        return loss
    return wrapper_decorator


#TODO(cgraywang): consider organize as a class decorator?
class Loss(object):
    def __init__(self, name):
        self.name = name
        self.hyper_params = None

    def get_hyper_params(self):
        return self.hyper_params

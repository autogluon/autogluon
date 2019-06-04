import functools
import types

from ..space import *

__all__ = ['autogluon_metrics', 'Metric']


def get_hyper_params(self):
    return self.hyper_params


def autogluon_metrics(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        metric = func(*args, **kwargs)
        #TODO (cgraywang): add more hparams
        return metric
    return wrapper_decorator


#TODO(cgraywang): consider organize as a class decorator?
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.hyper_params = None

    def get_hyper_params(self):
        return self.hyper_params

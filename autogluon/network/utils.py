import functools
import types

from ..space import *

__all__ = ['autogluon_net_instances', 'autogluon_nets']


def get_hyper_params(self):
    return self.hyper_params


def autogluon_nets(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        net = func(*args, **kwargs)
        net.hyper_params = [List('pretrained', [True, False]).get_hyper_param()]
        net.get_hyper_params = types.MethodType(get_hyper_params, net)
        return net
    return wrapper_decorator


def autogluon_net_instances(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        net = func(*args, **kwargs)
        net.hyper_params = [List('pretrained', [True, False]).get_hyper_param()]
        net.get_hyper_params = types.MethodType(get_hyper_params, net)
        return net
    return wrapper_decorator
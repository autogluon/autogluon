import functools
import types

from mxnet import gluon

from ..space import *

__all__ = ['autogluon_net_instances', 'autogluon_nets', 'Net']


def get_hyper_params(self):
    return self.hyper_params


def autogluon_nets(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        net = func(*args, **kwargs)
        #TODO (cgraywang): add more hparams
        setattr(net, 'hyper_params', [List('pretrained', [False]).get_hyper_param(),
                                      List('pretrained_base', [False]).get_hyper_param(),
                                      List('norm_layer', ['BatchNorm']).get_hyper_param()])
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


#TODO(cgraywang): consider organize as a class decorator?
class Net(object):
    def __init__(self, name):
        self.name = name
        self.hyper_params = None

    def get_hyper_params(self):
        return self.hyper_params

import functools

from ..space import *

__all__ = ['autogluon_net_instances', 'autogluon_nets', 'Net']


def get_hyper_params(self):
    return self.hyper_params


def autogluon_nets(func):
    """The auto net decorator.

    Args:
        args: args for the network.
        kwargs: kwargs for the network.

    Example:
        >>> @autogluon_nets
        >>> def resnet18_v1(**kwargs):
        >>>     pass
    """
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        net = func(*args, **kwargs)
        setattr(net, 'hyper_params', [List('pretrained', [True]).get_hyper_param(),
                                      List('pretrained_base', [True]).get_hyper_param(),
                                      List('norm_layer', ['BatchNorm']).get_hyper_param(),
                                      Linear('dense_layers', lower=1, upper=3).get_hyper_param(),
                                      Linear('dropout', lower=0.0, upper=0.50).get_hyper_param()])
        return net
    return wrapper_decorator


def autogluon_net_instances(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        net = func(*args, **kwargs)
        # TODO (ghaipiyu) : This was causing an exception. Should this hyperparam be here?
        # net.hyper_params = [List('pretrained', [True, False]).get_hyper_param()]
        # net.get_hyper_params = types.MethodType(get_hyper_params, net)
        return net

    return wrapper_decorator


# TODO(cgraywang): consider organize as a class decorator?
class Net(object):
    """The net with the search space.

    Args:
        name: the network's name
        hyper_params: the hyper-parameters for the network

    Example:
        >>> net = Net('resnet18_v1')
    """
    def __init__(self, name):
        self.name = name
        self.hyper_params = None

    def get_hyper_params(self):
        return self.hyper_params

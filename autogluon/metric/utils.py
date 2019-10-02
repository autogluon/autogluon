import functools
import types

from ..space import *

__all__ = ['autogluon_metrics', 'Metric']


def autogluon_metrics(func):
    """The auto metric decorator.

    Args:
        args: args for the metric.
        kwargs: kwargs for the metric.

    Example:
        >>> @autogluon_metrics
        >>> def Accuracy(**kwargs):
        >>>     pass
    """
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        metric = func(*args, **kwargs)
        #TODO (cgraywang): add more hparams
        return metric
    return wrapper_decorator


#TODO(cgraywang): consider organize as a class decorator?
class Metric(object):
    """The metric with the search space.

    Args:
        name: the metric name
        hyper_params: the hyper-parameters for the metric

    Example:
        >>> metrics = {'Accuracy': mxnet.metric.Accuracy}
        >>> metric = metrics['Accuracy']():
    """
    def __init__(self, name):
        self.name = name
        self.hyper_params = None

    def get_hyper_params(self):
        return self.hyper_params

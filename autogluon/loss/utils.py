import functools
import types

from ..space import *

__all__ = ['autogluon_losses', 'Loss']


def autogluon_losses(func):
    """The auto loss decorator.

    Args:
        args: args for the loss.
        kwargs: kwargs for the loss.

    Example:
        >>> @autogluon_losses
        >>> def L2Loss(**kwargs):
        >>>     pass
    """
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        loss = func(*args, **kwargs)
        #TODO (cgraywang): add more hparams
        return loss
    return wrapper_decorator


#TODO(cgraywang): consider organize as a class decorator?
class Loss(object):
    """The Loss with the search space.

    Args:
        name: the loss name
        hyper_params: the hyper-parameters for the loss

    Example:
        >>> losses = {'SoftmaxCrossEntropyLoss': gluon.loss.SoftmaxCrossEntropyLoss}
        >>> loss = losses['SoftmaxCrossEntropyLoss']()
    """
    def __init__(self, name):
        self.name = name
        self.hyper_params = None

    def get_hyper_params(self):
        return self.hyper_params

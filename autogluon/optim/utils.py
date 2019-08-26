import functools
import types

from ..space import *

__all__ = ['autogluon_optims', 'Optimizer']


def autogluon_optims(func):
    """The auto optimizer decorator.

        Args:
            args: args for the optimizer.
            kwargs: kwargs for the optimizer.

        Example:
            >>> @autogluon_optims
            >>> def SGD(**kwargs):
            >>>     return Optimizer('sgd')
        """
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        optim = func(*args, **kwargs)
        # TODO (cgraywang): parse user provided config to generate hyper_params
        if not kwargs:
            if optim.name == 'sgd':
                setattr(optim, 'hyper_params',
                        [Log('lr', 10 ** -4, 10 ** -1).get_hyper_param(),
                         Linear('momentum', 0.85, 0.95).get_hyper_param(),
                         Log('wd', 10 ** -6, 10 ** -2).get_hyper_param()])
            elif optim.name == 'adam':
                setattr(optim, 'hyper_params',
                        [Log('lr', 10 ** -4, 10 ** -1).get_hyper_param(),
                         Log('wd', 10 ** -6, 10 ** -2).get_hyper_param()])
            elif optim.name == 'nag':
                setattr(optim, 'hyper_params',
                        [Log('lr', 10 ** -4, 10 ** -1).get_hyper_param(),
                         Linear('momentum', 0.85, 0.95).get_hyper_param(),
                         Log('wd', 10 ** -6, 10 ** -2).get_hyper_param()])
            else:
                raise NotImplementedError
        else:
            hyper_param_lst = []
            for k, v in kwargs.items():
                hyper_param_lst.append(v.get_hyper_param())
            setattr(optim, 'hyper_params', hyper_param_lst)
        return optim

    return wrapper_decorator


class Optimizer(object):
    """The optimizer with the search space.

    Args:
        name: the optimizer name
        hyper_params: the hyper-parameters for the optmizer

    Example:
        >>> sgd = Optimizer('sgd')
    """
    def __init__(self, name):
        self.name = name
        self.hyper_params = None

    def get_hyper_params(self):
        return self.hyper_params

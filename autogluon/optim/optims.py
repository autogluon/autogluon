import ConfigSpace as CS
from mxnet import optimizer as optim

from ..core import *
from ..basic.space import *
from ..basic.decorators import autogluon_object
#from .utils import autogluon_optims, Optimizer

__all__ = ['Adam', 'NAG', 'SGD']

@autogluon_object(
    learning_rate=LogLinearSpace(10 ** -4, 10 ** -1),
    wd=LogLinearSpace(10 ** -6, 10 ** -2),
    )
class Adam(optim.Adam):
    pass

@autogluon_object(
    learning_rate=LogLinearSpace(10 ** -4, 10 ** -1),
    wd=LogLinearSpace(10 ** -6, 10 ** -2),
    momentum=LinearSpace(0.85, 0.95),
    )
class NAG(optim.NAG):
    pass

@autogluon_object(
    learning_rate=LogLinearSpace(10 ** -4, 10 ** -1),
    wd=LogLinearSpace(10 ** -6, 10 ** -2),
    momentum=LinearSpace(0.85, 0.95),
    )
class SGD(optim.SGD):
    pass

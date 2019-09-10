import ConfigSpace as CS
from mxnet import optimizer as optim

from ..core import *
from ..basic.space import *
from ..basic.decorators import autogluon_object

__all__ = ['Adam', 'NAG', 'SGD']

@autogluon_object(
    learning_rate=LogLinearSpace(1e-4, 1e-1),
    wd=LogLinearSpace(1e-6, 1e-2),
    )
class Adam(optim.Adam):
    pass

@autogluon_object(
    learning_rate=LogLinearSpace(1e-4, 1e-1),
    wd=LogLinearSpace(1e-6, 1e-2),
    momentum=LinearSpace(0.85, 0.95),
    )
class NAG(optim.NAG):
    pass

@autogluon_object(
    learning_rate=LogLinearSpace(1e-4, 1e-1),
    wd=LogLinearSpace(1e-6, 1e-2),
    momentum=LinearSpace(0.85, 0.95),
    )
class SGD(optim.SGD):
    pass

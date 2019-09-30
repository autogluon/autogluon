import ConfigSpace as CS
from mxnet import optimizer as optim

from ..core import *

__all__ = ['Adam', 'NAG', 'SGD']

@autogluon_object()
class Adam(optim.Adam):
    pass

@autogluon_object()
class NAG(optim.NAG):
    pass

@autogluon_object()
class SGD(optim.SGD):
    pass

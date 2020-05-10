from mxnet import optimizer as optim
from ..core import obj

__all__ = ['Adam', 'NAG', 'SGD']

@obj()
class Adam(optim.Adam):
    pass

@obj()
class NAG(optim.NAG):
    pass

@obj()
class SGD(optim.SGD):
    pass

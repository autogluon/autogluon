from .scheduler import *
from .fifo import *
from .hyperband import *

from ..core import List

class DeprecatedObject(List):
    def __init__(self, objs):
        super().__init__(*objs)

class Optimizers(DeprecatedObject):
    pass

class Nets(DeprecatedObject):
    pass

class Losses(DeprecatedObject):
    pass

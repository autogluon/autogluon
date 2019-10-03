from .scheduler import *
from .fifo import *
from .hyperband import *
from .space import *

from ..core import List
from ..utils import DeprecationHelper

class DeprecatedObject(List):
    def __init__(self, objs):
        super().__init__(*objs)

    def __repr__(self):
        reprstr = 'List' + str(self.data)
        return reprstr

Optimizers = DeprecationHelper(DeprecatedObject, 'List')
Nets = DeprecationHelper(DeprecatedObject, 'List')
Losses = DeprecationHelper(DeprecatedObject, 'List')

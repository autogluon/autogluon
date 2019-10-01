from .scheduler import *
from .fifo import *
from .hyperband import *

from ..core import List
from ..utils import DeprecationHelper

class DeprecatedObject(List):
    def __init__(self, objs):
        super().__init__(*objs)

Optimizers = DeprecationHelper(DeprecatedObject, 'Optimizers')

Nets = DeprecationHelper(DeprecatedObject, 'Nets')

Losses = DeprecationHelper(DeprecatedObject, 'Losses')

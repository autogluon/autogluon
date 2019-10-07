from ..core import Choice
from ..utils import DeprecationHelper

__all__ = ['Optimizers', 'Nets', 'Losses']

class DeprecatedObject(Choice):
    def __init__(self, objs):
        super().__init__(*objs)

    def __repr__(self):
        reprstr = 'Choice' + str(self.data)
        return reprstr

Optimizers = DeprecationHelper(DeprecatedObject, 'Choice')
Nets = DeprecationHelper(DeprecatedObject, 'Choice')
Losses = DeprecationHelper(DeprecatedObject, 'Choice')

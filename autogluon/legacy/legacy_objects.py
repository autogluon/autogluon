from ..core import Choice, Space
from ..utils import DeprecationHelper
from .space import List

__all__ = ['Optimizers', 'Nets', 'Losses']

Optimizers = DeprecationHelper(List, 'Choice')
Nets = DeprecationHelper(List, 'Choice')
Losses = DeprecationHelper(List, 'Choice')

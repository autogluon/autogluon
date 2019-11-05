from ..core import Categorical, Space
from ..utils import DeprecationHelper
from .space import List

__all__ = ['Optimizers', 'Nets', 'Losses']

Optimizers = DeprecationHelper(List, 'Categorical')
Nets = DeprecationHelper(List, 'Categorical')
Losses = DeprecationHelper(List, 'Categorical')

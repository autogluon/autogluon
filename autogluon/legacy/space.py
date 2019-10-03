from ..utils import DeprecationHelper
from ..core.space import *

__all__ = ['ListSpace', 'IntSpace', 'LogLinearSpace']

ListSpace = DeprecationHelper(List, 'List')
IntSpace = DeprecationHelper(List, 'Int')
LogLinearSpace = DeprecationHelper(List, 'LogLinear')

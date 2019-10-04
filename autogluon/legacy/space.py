from ..utils import DeprecationHelper
from ..core.space import *

__all__ = ['ListSpace', 'IntSpace', 'LogLinearSpace']

ListSpace = DeprecationHelper(Choice, 'List')
IntSpace = DeprecationHelper(Int, 'Int')
LogLinearSpace = DeprecationHelper(LogLinear, 'LogLinear')

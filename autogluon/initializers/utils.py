import functools
from typing import AnyStr

from ..space import *

__all__ = ['autogluon_initializers']

def autogluon_initializers(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        # TODO (ghaipiyu): Parse user provided config to generate initializers
        pass
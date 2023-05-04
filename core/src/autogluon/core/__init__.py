from  typing import Any
from autogluon.common.utils.log_utils import _add_stream_handler

from .dataset import TabularDataset
from . import metrics
from . import constants
from .space import Space, Categorical, Real, Int, Bool, spaces
from .version import __version__


import warnings

# TODO: Remove deprecation warning for v1.0
class DeprecatedSpacesWrapper:
    def __getattr__(self, attr: str) -> Any:
        import autogluon.common as ag

        if attr in spaces:
            warnings.warn(
                "Accessing search spaces as `autogluon.core.space` is deprecated as of v0.8 and won't be supported "
                "in the next release. Please use `autogluon.common.space` instead."
            )
            return getattr(ag.space, attr)
        else:
            raise AttributeError


_add_stream_handler()
space = DeprecatedSpacesWrapper()

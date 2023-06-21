import warnings
from typing import Any

from autogluon.common.utils.log_utils import _add_stream_handler

from . import constants, metrics
from .dataset import TabularDataset
from .space import Bool, Categorical, Int, Real, Space, spaces
from .version import __version__


# TODO: Remove deprecation warning for v1.0
class DeprecatedSpacesWrapper:
    def __getattr__(self, attr: str) -> Any:
        from autogluon.common import space

        if attr in spaces:
            warnings.warn(
                "Accessing search spaces as `autogluon.core.space` is deprecated as of v0.8 and won't be supported "
                "in the next release. Please use `autogluon.common.space` instead.",
                category=DeprecationWarning,
            )
            return getattr(space, attr)
        else:
            raise AttributeError


_add_stream_handler()
space = DeprecatedSpacesWrapper()

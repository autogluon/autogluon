from  typing import Any
from autogluon.common.utils.log_utils import _add_stream_handler

from .dataset import TabularDataset
from . import metrics
from . import constants
from .version import __version__

import warnings


class DeprecatedSpacesWrapper:
    def __getattr__(self, attr: str) -> Any:
        import autogluon.common as ag

        if attr in ["Space", "Categorical", "Real", "Int", "Bool", "SimpleSpace", "DiscreteSpace"]:
            warnings.warn(
                "Search spaces have been moved to `autogluon.common.space` as of v0.8 and won't be supported"
                "in the next release. Please use `autogluon.common.space` instead."
            )
            return getattr(ag.space, attr)
        else:
            raise AttributeError

_add_stream_handler()
# TODO: Remove the deprecated warning in v1.0
space = DeprecatedSpacesWrapper()

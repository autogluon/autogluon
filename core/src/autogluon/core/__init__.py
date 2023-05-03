from  typing import Any
from autogluon.common.utils.log_utils import _add_stream_handler
# TODO: Remove space import for v1.0
from autogluon.common import space
from autogluon.common.space import Space, Categorical, Real, Int, Bool

from .dataset import TabularDataset
from . import metrics
from . import constants
from .version import __version__


_add_stream_handler()


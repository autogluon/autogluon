from autogluon.common.utils.log_utils import _add_stream_handler

from . import constants, metrics
from .dataset import TabularDataset
from .version import __version__

_add_stream_handler()

# noinspection PyUnresolvedReferences
from autogluon.common.dataset import TabularDataset
from autogluon.common.utils.log_utils import _add_stream_handler

from . import constants, metrics
from .version import __version__

_add_stream_handler()

from autogluon.common.utils.log_utils import _add_stream_handler

try:
    from .version import __version__
except ImportError:
    pass

from .predictor import MultiModalPredictor

_add_stream_handler()

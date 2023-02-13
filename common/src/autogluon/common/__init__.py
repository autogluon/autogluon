from .version import __version__

from .utils.log_utils import _add_stream_handler, fix_logging_if_kaggle as __fix_logging_if_kaggle

# Fixes logger in Kaggle to show logs in notebook.
__fix_logging_if_kaggle()

_add_stream_handler()

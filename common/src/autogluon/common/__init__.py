from .dataset import TabularDataset
from .features.feature_metadata import FeatureMetadata
from .utils.log_utils import _add_stream_handler
from .utils.log_utils import fix_logging_if_kaggle as __fix_logging_if_kaggle
from .version import __version__

# Fixes logger in Kaggle to show logs in notebook.
__fix_logging_if_kaggle()

_add_stream_handler()

from autogluon.common.dataset import TabularDataset
from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.common.utils.log_utils import _add_stream_handler

try:
    from .version import __version__
except ImportError:
    pass

from .predictor import TabularPredictor

_add_stream_handler()

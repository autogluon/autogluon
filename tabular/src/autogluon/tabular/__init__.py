import logging

from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.core.dataset import TabularDataset

try:
    from .version import __version__
except ImportError:
    pass

from .predictor import TabularPredictor

logging.basicConfig(format='%(message)s')  # just print message in logs

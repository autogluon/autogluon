import logging

from autogluon.core.dataset import TabularDataset
from autogluon.core.features.feature_metadata import FeatureMetadata
#from autogluon.tabular_to_image.utils_pro import Utils_pro
#from autogluon.tabular_to_image.models_zoo import ModelsZoo

try:
    from .version import __version__
except ImportError:
    pass

from autogluon.tabular_to_image.prediction import ImagePredictions #.predictor import TabularPredictor

logging.basicConfig(format='%(message)s')  # just print message in logs

from .predictor import ImagePredictor
from .detector import ObjectDetector

ImageDataset = ImagePredictor.Dataset
ImageDetectionDataset = ObjectDetector.Dataset

try:
    from .version import __version__
except ImportError:
    pass

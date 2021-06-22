from .predictor import ImagePredictor
from .detector import ObjectDetector

ImageDataset = ImagePredictor.Dataset
ImageDetectionDataset = ObjectDetector.Dataset

from .version import __version__

try:
    from .version import __version__
except ImportError:
    pass

from . import constants, data, models, optimization, predictor, utils
from .base import MultiModalPredictor
from .predictor import AutoMMPredictor
from .utils import download

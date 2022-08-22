try:
    from .version import __version__
except ImportError:
    pass

from . import constants, data, models, optimization, predictor, utils
from .predictor import AutoMMPredictor, MultiModalPredictor
from .utils import download

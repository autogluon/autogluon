try:
    from .version import __version__
except ImportError:
    pass

from . import constants, data, models, optimization, predictor, utils
from .matcher import MultiModalMatcher
from .predictor import AutoMMPredictor, MultiModalPredictor
from .utils import download

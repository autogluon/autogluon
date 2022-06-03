import logging

try:
    from .version import __version__
except ImportError:
    pass

from .predictor import TimeSeriesPredictor
from .dataset import TimeSeriesDataFrame

logging.basicConfig(format="%(message)s")  # just print message in logs

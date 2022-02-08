import logging
try:
    from .version import __version__
except ImportError:
    pass

from .predictor.predictor import ForecastingPredictor

logging.basicConfig(format='%(message)s')  # just print message in logs

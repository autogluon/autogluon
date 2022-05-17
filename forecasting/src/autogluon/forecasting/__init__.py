import logging

try:
    from .version import __version__
except ImportError:
    pass

from autogluon.core.dataset import TabularDataset  # noqa: F401

# TODO: make ForecastingPredictor available

logging.basicConfig(format="%(message)s")  # just print message in logs

import logging
from packaging.version import parse

try:
    from .version import __version__
except ImportError:
    pass

from .dataset import TimeSeriesDataFrame
from .evaluator import TimeSeriesEvaluator
from .predictor import TimeSeriesPredictor

try:
    from mxnet import __version__ as mxnet_version

    assert parse("2.0") > parse(mxnet_version) >= parse("1.9")
except (ImportError, AssertionError):
    raise ImportError(
        "autogluon.forecasting depends on Apache MxNet v1.9 or greater (below v2.0). "
        "Please install a suitable version of MxNet in order to use autogluon.forecasting using "
        "`pip install mxnet==1.9` or `pip install mxnet-cu112==1.9` for GPU support."
    )

logging.basicConfig(format="%(message)s")  # just print message in logs

import warnings

from packaging.version import parse

from autogluon.common.utils.log_utils import _add_stream_handler

try:
    from .version import __version__
except ImportError:
    pass

MXNET_INSTALLED = False
try:
    from mxnet import __version__ as mxnet_version

    if parse("2.0") > parse(mxnet_version) >= parse("1.9"):
        MXNET_INSTALLED = True
    else:
        warnings.warn(
            "autogluon.timeseries depends on Apache MXNet version >=1.9.0 and <2.0 for some "
            f"additional models, although {mxnet_version} was found. MXNet features will be disabled."
        )
except ImportError:
    pass


from .dataset import TimeSeriesDataFrame
from .evaluator import TimeSeriesEvaluator
from .predictor import TimeSeriesPredictor

_add_stream_handler()


__all__ = ["TimeSeriesDataFrame", "TimeSeriesEvaluator", "TimeSeriesPredictor"]

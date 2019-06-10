# coding: utf-8
# pylint: disable=wrong-import-position
"""AutoGluon: AutoML toolkit with Gluon."""
from __future__ import absolute_import

# mxnet version check
mx_version = '1.4.0'
try:
    import mxnet as mx
    from distutils.version import LooseVersion

    if LooseVersion(mx.__version__) < LooseVersion(mx_version):
        msg = (
            "Legacy mxnet-mkl=={} detected, some new modules may not work properly. "
            "mxnet-mkl>={} is required. You can use pip to upgrade mxnet "
            "`pip install mxnet-mkl --pre --upgrade` "
            "or `pip install mxnet-cu90mkl --pre --upgrade`").format(mx.__version__, mx_version)
        raise ImportError(msg)
except ImportError:
    raise ImportError(
        "Unable to import dependency mxnet. "
        "A quick tip is to install via `pip install mxnet-mkl/mxnet-cu90mkl --pre`. ")

try:
    import mxboard
except ImportError:
    raise ImportError(
        "Unable to import dependency mxboard. "
        "A quick tip is to install via `pip install mxboard`. ")

try:
    import tensorboard
except ImportError:
    raise ImportError(
        "Unable to import dependency tensorboard. "
        "A quick tip is to install via `pip install tensorflow tensorboard`. ")

try:
    import dask
except ImportError:
    raise ImportError(
        "Unable to import dependency dask. "
        "A quick tip is to install via `pip install dask[complete]`. ")


__version__ = '0.0.1'

from .core import *
from .dataset import *
from .loss import *
from .metric import *
from .network import *
from .optim import *
from .space import *
from .task import image_classification as image_classification
from .basic import *
from .utils import *
from . import scheduler, searcher, distributed, resource

__all__ = dataset.__all__ \
          + loss.__all__ \
          + metric.__all__ \
          + network.__all__ \
          + optim.__all__ \
          + space.__all__ \
          + task.__all__ \
          + core.__all__

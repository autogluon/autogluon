# coding: utf-8
# pylint: disable=wrong-import-position
"""AutoGluon: AutoML toolkit with Gluon."""
from __future__ import absolute_import
from .version import __version__

from .utils.try_import import *
try_import_mxnet()
try_import_dask()

from .core import *
from .dataset import *
from .loss import *
from .metric import *
from .network import *
from .optim import *
from .space import *
from .utils import *
from .task import image_classification as image_classification
from .task import object_detection as object_detection
from .basic import *
from . import scheduler, searcher, distributed, resource

__all__ = dataset.__all__ \
          + loss.__all__ \
          + metric.__all__ \
          + network.__all__ \
          + optim.__all__ \
          + space.__all__ \
          + task.__all__ \
          + core.__all__

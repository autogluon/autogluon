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
from .optim import optims as optims
from .space import *
from .basic import *
from .utils import *
from .task.image_classification import ImageClassification as image_classification
from .task.text_classification import TextClassification as text_classification
from .task.object_detection import ObjectDetection as object_detection
from .basic import *
from . import scheduler, searcher, distributed, resource

__all__ = dataset.__all__ \
          + loss.__all__ \
          + metric.__all__ \
          + network.__all__ \
          + optim.__all__ \
          + space.__all__ \
          + core.__all__

# coding: utf-8
# pylint: disable=wrong-import-position
"""AutoGluon: AutoML toolkit with Gluon."""
from __future__ import absolute_import
from .version import __version__
from .utils.try_import import *

try_import_mxnet()

from . import scheduler, searcher, nas, utils
from .scheduler import get_cpu_count, get_gpu_count

from .utils import *
from .core import *
from .task import *

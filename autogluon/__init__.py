# coding: utf-8
# pylint: disable=wrong-import-position
""" AutoGluon: AutoML Toolkit for Deep Learning """
from __future__ import absolute_import
from .version import __version__
from .utils.try_import import *

try_import_mxnet()

from autogluon_core import scheduler, searcher
from autogluon_core.scheduler import get_cpu_count, get_gpu_count
from autogluon_core.core import *

from .utils import *
from .task import *

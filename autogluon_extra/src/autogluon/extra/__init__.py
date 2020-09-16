# coding: utf-8
# pylint: disable=wrong-import-position
""" AutoGluon: AutoML Toolkit for Deep Learning """
from __future__ import absolute_import
from .version import __version__

import logging
logger = logging.getLogger("distributed.utils_perf")
logger.setLevel(logging.ERROR)

logger = logging.getLogger("distributed.logging.distributed")
logger.setLevel(logging.ERROR)

logger = logging.getLogger("distributed.worker")
logger.setLevel(logging.ERROR)

from .utils.try_import import *
try_import_mxnet()

from autogluon.core import scheduler, searcher, utils
from autogluon.core.scheduler import get_cpu_count, get_gpu_count

from .utils import *
from autogluon.core import *

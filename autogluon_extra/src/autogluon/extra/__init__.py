# coding: utf-8
# pylint: disable=wrong-import-position
""" AutoGluon: AutoML Toolkit for Deep Learning """
from __future__ import absolute_import

import logging

from .version import __version__

logger = logging.getLogger("distributed.utils_perf")
logger.setLevel(logging.ERROR)

logger = logging.getLogger("distributed.logging.distributed")
logger.setLevel(logging.ERROR)

logger = logging.getLogger("distributed.worker")
logger.setLevel(logging.ERROR)


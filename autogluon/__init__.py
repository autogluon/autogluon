# coding: utf-8
# pylint: disable=wrong-import-position
""" AutoGluon: AutoML Toolkit for Deep Learning """
from __future__ import absolute_import
from .version import __version__
from autogluon_core.utils.try_import import *

try_import_mxnet()

from autogluon_core import *
from .utils import *
from .task import *

from .space import *
from .task.task import *
from .task.task import Task
from .decorator import *
from .utils.files import *
from .scheduler.resource.resource import *
from .scheduler.scheduler import *

# TODO: v0.1 Identify why distributed logs are spammed if not suppressed via the below code
#  Refer to https://github.com/awslabs/autogluon/issues/493
#  Example 1 of spam: http://autogluon-staging.s3-website-us-west-2.amazonaws.com/PR-694/16/tutorials/tabular_prediction/tabular-indepth.html#keeping-models-in-memory
#  Example 2 of spam: http://autogluon-staging.s3-website-us-west-2.amazonaws.com/PR-764/1/tutorials/tabular_prediction/tabular-indepth.html#keeping-models-in-memory
import logging as __logging
__logging.getLogger("distributed.utils_perf").setLevel(__logging.ERROR)
__logging.getLogger("distributed.logging.distributed").setLevel(__logging.ERROR)
__logging.getLogger("distributed.worker").setLevel(__logging.ERROR)

import os
import time
import logging

from ......core import args
from ......scheduler.reporter import LocalStatusReporter
from .....tabular.utils.exceptions import TimeLimitExceeded

logger = logging.getLogger(__name__)


@args()
# @ag.args(context_length=ag.space.Int(1, 20))
def model_trial(args, reporter):
    util_args, params = prepare_inputs(args)
    model = util_args.model
    train_ds = util_args.train_ds
    test_ds = util_args.test_ds
    model.params.update(params)
    model.fit(train_ds)
    model.save(path=f"trial_{args['task_id']}")
    score = model.score(test_ds)
    reporter(epoch=1, validation_performance=-score)


def prepare_inputs(args):
    util_args = args.pop("util_args")
    return util_args, args

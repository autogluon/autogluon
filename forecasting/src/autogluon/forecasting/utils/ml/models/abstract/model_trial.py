import os
import time
import logging

from core import args

logger = logging.getLogger(__name__)


@args()
def model_trial(args, reporter):
    util_args, params = prepare_inputs(args)
    model = util_args.model
    train_ds = util_args.train_ds
    test_ds = util_args.test_ds
    metric = util_args.metric
    model.params.update(params)
    model.fit(train_ds)
    model.save(path=f"./model/{model.name}/trial_{args['task_id']}.pkl")
    score = model.score(test_ds, metric=metric)
    reporter(epoch=1, validation_performance=-score)


def prepare_inputs(args):
    util_args = args.pop("util_args")
    return util_args, args

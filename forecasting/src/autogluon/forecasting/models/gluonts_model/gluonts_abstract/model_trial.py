import os
import time
import logging
from core.utils.loaders import load_pkl
from core.utils.exceptions import TimeLimitExceeded
from core import args

logger = logging.getLogger(__name__)


@args()
def model_trial(args, reporter):
    try:
        model, util_args, params = prepare_inputs(args)

        train_data = load_pkl.load(util_args.directory + util_args.train_data_path)
        test_data = load_pkl.load(util_args.directory + util_args.test_data_path)
        eval_metric = model.eval_metric
        model = fit_and_save_model(model, params, train_data, test_data, eval_metric, util_args.time_start,
                                   time_limit=util_args.get('time_limit', None))

    except Exception as e:
        if not isinstance(e, TimeLimitExceeded):
            logger.exception(e, exc_info=True)
        reporter.terminate()
    else:
        # TODO: specify whether score is lower is better
        reporter(epoch=1, validation_performance=-model.test_score)


def prepare_inputs(args):
    task_id = args.pop('task_id')
    util_args = args.pop('util_args')
    file_prefix = f"trial_{task_id}"  # append to all file names created during this trial. Do NOT change!
    model = util_args.model  # the model object must be passed into model_trial() here
    model.name = model.name + os.path.sep + file_prefix
    model.set_contexts(path_context=model.path_root + model.name + os.path.sep)
    return model, util_args, args


def fit_and_save_model(model, params, train_data, test_data, eval_metric, time_start, time_limit=None):
    time_current = time.time()
    time_elapsed = time_current - time_start
    if time_limit is not None:
        time_left = time_limit - time_elapsed
        if time_left <= 0:
            raise TimeLimitExceeded
    else:
        time_left = None

    model.params.update(params)
    time_fit_start = time.time()
    model.fit(train_data, time_limit=time_left)
    time_fit_end = time.time()
    model.test_score = -model.score(test_data, eval_metric)
    model.fit_time = time_fit_end - time_fit_start
    model.save()
    return model

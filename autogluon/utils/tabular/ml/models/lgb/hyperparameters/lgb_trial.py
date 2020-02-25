import logging
import os
import time

from .....utils.exceptions import TimeLimitExceeded
from .......core import args
from ......try_import import try_import_lightgbm

logger = logging.getLogger(__name__)


@args()
def lgb_trial(args, reporter):
    """ Training script for hyperparameter evaluation of Gradient Boosting model """
    try:
        try_import_lightgbm()
        import lightgbm as lgb
        dataset_train = lgb.Dataset(args.directory + args.dataset_train_filename)
        dataset_val = lgb.Dataset(args.directory + args.dataset_val_filename)

        nonparam_args = {'directory', 'task_id', 'time_start', 'time_limit', 'model', 'dataset_train_filename', 'dataset_val_filename'}  # list of args which are not model hyperparameters
        # Note: args.task_id may not start at 0 if HPO has been run for other models with same scheduler
        file_prefix = "trial_" + str(args.task_id)  # append to all file names created during this trial. Do NOT change!
        model = args.model
        model.name = model.name + os.path.sep + file_prefix
        model.set_contexts(path_context=model.path_root + model.name + os.path.sep)
        for key in args:
            if key not in nonparam_args:
                model.params[key] = args[key]  # use these hyperparam values in this trial

        time_current = time.time()
        time_elapsed = time_current - args.time_start
        if 'time_limit' in args:
            time_left = args.time_limit - time_elapsed
            if time_left <= 0:
                raise TimeLimitExceeded
        else:
            time_left = None

        model.fit(dataset_train=dataset_train, dataset_val=dataset_val, time_limit=time_left, reporter=reporter)
        # TODO: Set fit_time variable in model here
        model.save()
        # TODO: add to reporter: time_of_trial without load/save time (isn't this just function of early-stopping point?), memory/inference ??
        # TODO: add to reporter: Name/path of model
    except Exception as e:
        if not isinstance(e, TimeLimitExceeded):
            logger.exception(e, exc_info=True)
        reporter.terminate()

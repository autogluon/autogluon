import os
import time
import logging

from ....utils.loaders import load_pkl
from ....utils.exceptions import TimeLimitExceeded
from ......core import args
from ......scheduler.reporter import LocalStatusReporter

logger = logging.getLogger(__name__)


@args()
def model_trial(args, reporter: LocalStatusReporter):
    """ Training script for hyperparameter evaluation of an arbitrary model that subclasses AbstractModel.
        
        Notes:
            - Model object itself must be passed as kwarg: model
            - All model hyperparameters must be stored in model.params dict that may contain special keys such as:
                'seed_value' to ensure reproducibility
                'num_threads', 'num_gpus' to set specific resources in model.fit()
            - model.save() must have return_filename, file_prefix, directory options
    """
    try:
        task_id = args.pop('task_id')
        util_args = args.pop('util_args')

        file_prefix = f"trial_{task_id}"  # append to all file names created during this trial. Do NOT change!
        model = util_args.model  # the model object must be passed into model_trial() here
        model.name = model.name + os.path.sep + file_prefix
        model.set_contexts(path_context=model.path_root + model.name + os.path.sep)

        X_train, Y_train = load_pkl.load(util_args.directory + util_args.dataset_train_filename)
        X_val, Y_val = load_pkl.load(util_args.directory + util_args.dataset_val_filename)

        time_start = util_args.pop('time_start')
        time_limit = util_args.pop('time_limit', None)
        time_current = time.time()
        time_elapsed = time_current - time_start
        if time_limit is not None:
            time_left = time_limit - time_elapsed
            if time_left <= 0:
                raise TimeLimitExceeded
        else:
            time_left = None

        model.params.update(args)
        time_fit_start = time.time()
        model.fit(X_train, Y_train, X_val, Y_val, time_limit=time_left)
        time_fit_end = time.time()
        val_perf = model.score(X_val, Y_val)
        time_pred_end = time.time()
        model.fit_time = time_fit_end - time_fit_start
        model.predict_time = time_pred_end - time_fit_end
        model.val_score = val_perf
        model.save()

    except Exception as e:
        if not isinstance(e, TimeLimitExceeded):
            logger.exception(e, exc_info=True)
        reporter.terminate()
    else:
        reporter(epoch=0, validation_performance=val_perf)

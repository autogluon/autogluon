import os
import time
import logging

from .tabular_nn_dataset import TabularNNDataset
from ....utils.exceptions import TimeLimitExceeded
from ......core import args

logger = logging.getLogger(__name__)


@args()
def tabular_nn_trial(args, reporter):
    """ Training and evaluation function used during a single trial of HPO """
    try:
        task_id = args.pop('task_id')
        util_args = args.pop('util_args')

        train_dataset = TabularNNDataset.load(util_args.train_path)
        test_dataset = TabularNNDataset.load(util_args.test_path)
        y_test = test_dataset.get_labels()

        # Note: args.task_id may not start at 0 if HPO has been run for other models with same scheduler
        file_prefix = "trial_" + str(task_id)  # append to all file names created during this trial. Do NOT change!

        model = util_args.model
        model.name = model.name + os.path.sep + file_prefix
        model.set_contexts(path_context=model.path_root + model.name + os.path.sep)

        time_current = time.time()
        time_elapsed = time_current - util_args.time_start
        if 'time_limit' in util_args:
            time_left = util_args.time_limit - time_elapsed
            if time_left <= 0:
                raise TimeLimitExceeded
        else:
            time_left = None

        model.params.update(args)
        time_fit_start = time.time()
        model.fit(X_train=train_dataset, Y_train=None, X_test=test_dataset, time_limit=time_left, reporter=reporter)
        time_fit_end = time.time()
        score = model.score(X=test_dataset, y=y_test)
        time_pred_end = time.time()
        model.fit_time = time_fit_end - time_fit_start
        model.predict_time = time_pred_end - time_fit_end
        model.val_score = score
        model.save()
    except Exception as e:
        logger.exception(e, exc_info=True)
        reporter.terminate()

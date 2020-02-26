import logging
import os
import time

from .....utils.loaders import load_pkl
from .....utils.exceptions import TimeLimitExceeded
from ......try_import import try_import_lightgbm
from .......core import args

logger = logging.getLogger(__name__)


# TODO: Further simplify to align with model_trial.py
# FIXME: Hyperband does not work with LightGBM
# FIXME: If stopping metric != eval_metric, score will be wrong!
@args()
def lgb_trial(args, reporter):
    """ Training script for hyperparameter evaluation of Gradient Boosting model """
    try:
        try_import_lightgbm()
        import lightgbm as lgb
        task_id = args.pop('task_id')
        util_args = args.pop('util_args')
        dataset_train = lgb.Dataset(util_args.directory + util_args.dataset_train_filename)
        dataset_val = lgb.Dataset(util_args.directory + util_args.dataset_val_filename)
        X_val, Y_val = load_pkl.load(util_args.directory + util_args.dataset_val_pkl_filename)

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
        model.fit(dataset_train=dataset_train, dataset_val=dataset_val, time_limit=time_left, reporter=reporter)
        time_fit_end = time.time()
        score = model.score(X=X_val, y=Y_val)
        time_pred_end = time.time()
        model.fit_time = time_fit_end - time_fit_start
        model.predict_time = time_pred_end - time_fit_end
        model.save()
    except Exception as e:
        if not isinstance(e, TimeLimitExceeded):
            logger.exception(e, exc_info=True)
        reporter.terminate()

    # FIXME: If stopping metric and eval metric differ, the previous reported scores will not align as they will be evaluated with stopping_metric, whereas this is evaluated with eval_metric
    #  This should only impact if the reporter data is used
    # FIXME: If stopping metric score > eval metric score, stopping metric score will be recorded as best score, this is a defect!
    # FIXME: It might be the case that if a reporter has been recorded and the model crash, AutoGluon will try to access the invalid model and fail.
    # reporter(epoch=model.params_trained['num_boost_round'] + 1, validation_performance=score)

import os
import time
import logging

from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.core import args
from autogluon.core.scheduler.reporter import LocalStatusReporter

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
        model, args, util_args = prepare_inputs(args=args)

        X_train, y_train = load_pkl.load(util_args.directory + util_args.dataset_train_filename)
        X_val, y_val = load_pkl.load(util_args.directory + util_args.dataset_val_filename)

        fit_model_args = dict(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
        predict_proba_args = dict(X=X_val)
        model = fit_and_save_model(model=model, params=args, fit_args=fit_model_args, predict_proba_args=predict_proba_args, y_val=y_val,
                                   time_start=util_args.time_start, time_limit=util_args.get('time_limit', None), reporter=None)
    except Exception as e:
        if not isinstance(e, TimeLimitExceeded):
            logger.exception(e, exc_info=True)
        reporter.terminate()
    else:
        reporter(epoch=1, validation_performance=model.val_score)


def prepare_inputs(args):
    task_id = args.pop('task_id')
    util_args = args.pop('util_args')

    file_prefix = f"trial_{task_id}"  # append to all file names created during this trial. Do NOT change!
    model = util_args.model  # the model object must be passed into model_trial() here
    model.name = model.name + os.path.sep + file_prefix
    model.set_contexts(path_context=model.path_root + model.name + os.path.sep)
    return model, args, util_args


def fit_and_save_model(model, params, fit_args, predict_proba_args, y_val, time_start, time_limit=None, reporter=None):
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
    model.fit(**fit_args, time_limit=time_left, reporter=reporter)
    time_fit_end = time.time()
    y_pred_proba = model.predict_proba(**predict_proba_args)
    time_pred_end = time.time()
    model.val_score = model.score_with_y_pred_proba(y=y_val, y_pred_proba=y_pred_proba)
    model.fit_time = time_fit_end - time_fit_start
    model.predict_time = time_pred_end - time_fit_end
    model.save()
    return model


def skip_hpo(model, X_train, y_train, X_val, y_val, scheduler_options=None, **kwargs):
    """Skips HPO and simply trains the model once with the provided HPO time budget. Returns model artifacts as if from HPO."""
    fit_model_args = dict(X_train=X_train, y_train=y_train, **kwargs)
    predict_proba_args = dict(X=X_val)
    time_limit = scheduler_options[1]['time_out']
    fit_and_save_model(model=model, params=dict(), fit_args=fit_model_args, predict_proba_args=predict_proba_args, y_val=y_val, time_start=time.time(), time_limit=time_limit)
    hpo_results = {'total_time': model.fit_time}
    hpo_model_performances = {model.name: model.val_score}
    hpo_models = {model.name: model.path}
    return hpo_models, hpo_model_performances, hpo_results

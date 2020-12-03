import os
import time
import logging

from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.core import args
from autogluon.core.scheduler.reporter import LocalStatusReporter

from autogluon.core.utils.savers import save_pkl

from autogluon.core.utils import AutoGluonEarlyStop
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
    model = None
    try:
        model, args, util_args = prepare_inputs(args=args)

        epochs = util_args.pop("epochs")

        save_val_pred = util_args.pop('save_val_pred')

        X_train, y_train = load_pkl.load(util_args.directory + util_args.dataset_train_filename)
        X_val, y_val = load_pkl.load(util_args.directory + util_args.dataset_val_filename)

        fit_model_args = dict(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, save_val_pred=None)
        predict_proba_args = dict(X=X_val)
        model = fit_and_save_model(model=model, params=args, fit_args=fit_model_args, predict_proba_args=predict_proba_args,
                                   y_val=y_val, time_start=util_args.time_start, time_limit=util_args.get('time_limit'),
                                   epochs=epochs, reporter=reporter, save_val_pred=save_val_pred)
    except AutoGluonEarlyStop:
        if model is not None:
            model.save()
    except TimeLimitExceeded:
        pass  # this is intended to be silent for model trials
    except Exception as e:
        logger.exception(e, exc_info=True)
    finally:
        reporter.terminate()


def prepare_inputs(args):
    task_id = args.pop('task_id')
    util_args = args.pop('util_args')

    args.pop("epochs")

    file_prefix = f"trial_{task_id}"  # append to all file names created during this trial. Do NOT change!
    model = util_args.model  # the model object must be passed into model_trial() here
    model.name = model.name + os.path.sep + file_prefix
    model.set_contexts(path_context=model.path_root + model.name + os.path.sep)
    return model, args, util_args


def fit_and_save_model(model, params, fit_args, predict_proba_args, y_val, time_start, time_limit=None, epochs=None, reporter=None, save_val_pred=None):
    time_current = time.time()
    time_elapsed = time_current - time_start
    if time_limit is not None:
        time_left = time_limit - time_elapsed
        if time_left <= 0:
            raise TimeLimitExceeded
    else:
        time_left = None

    if epochs is None:
        epochs = 1

    model.params.update(params)
    time_fit_start = time.time()
    incremental = epochs > 1
    for i in range(epochs):
        model.fit(**fit_args, time_limit=time_left, reporter=reporter, warm_start=incremental)
        time_fit_end = time.time()
        y_pred_proba = model.predict_proba(**predict_proba_args)
        time_pred_end = time.time()
        val_score = model.score_with_y_pred_proba(y=y_val, y_pred_proba=y_pred_proba)
        model.val_score = val_score
        model.fit_time = time_fit_end - time_fit_start
        model.predict_time = time_pred_end - time_fit_end
        # save validation set prediction to disk
        if save_val_pred:
            save_pkl.save(path=os.path.join(model.path, "val_pred.save"), object=y_pred_proba)
        reporter(epoch=i + 1, validation_performance=val_score, model_path=model.path)

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

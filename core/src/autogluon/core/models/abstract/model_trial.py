import os
import time
import logging

from ...utils.loaders import load_pkl
from ...utils.exceptions import TimeLimitExceeded

logger = logging.getLogger(__name__)


def model_trial(args,
                reporter,
                model,
                train_path,
                val_path,
                time_start,
                time_limit=None,
                fit_kwargs=None,
                ):
    """ Training script for hyperparameter evaluation of an arbitrary model that subclasses AbstractModel.

        Notes:
            - Model object itself must be passed as kwarg: model
            - All model hyperparameters must be stored in model.params dict that may contain special keys such as:
                'seed_value' to ensure reproducibility
                'num_threads', 'num_gpus' to set specific resources in model.fit()
            - model.save() must have return_filename, file_prefix, directory options
    """
    try:
        if fit_kwargs is None:
            fit_kwargs = dict()

        model, args = prepare_inputs(args=args, model=model)

        X, y = load_pkl.load(train_path)
        X_val, y_val = load_pkl.load(val_path)

        fit_model_args = dict(X=X, y=y, X_val=X_val, y_val=y_val, **fit_kwargs)
        predict_proba_args = dict(X=X_val)
        model = fit_and_save_model(
            model=model,
            params=args,
            fit_args=fit_model_args,
            predict_proba_args=predict_proba_args,
            y_val=y_val,
            time_start=time_start,
            time_limit=time_limit,
            reporter=None,
        )
    except Exception as e:
        if not isinstance(e, TimeLimitExceeded):
            logger.exception(e, exc_info=True)
        reporter.terminate()
    else:
        reporter(epoch=1, validation_performance=model.val_score)


def prepare_inputs(args, model):
    task_id = args.pop('task_id')

    file_prefix = f"T{task_id}"  # append to all file names created during this trial. Do NOT change!
    model.name = model.name + os.path.sep + file_prefix
    model.set_contexts(path_context=model.path_root + model.name + os.path.sep)
    return model, args


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

    if model._get_tags().get('valid_oof', False):
        oof_pred_proba = model.get_oof_pred_proba(X=fit_args['X'], y=fit_args['y'])
        time_pred_end = time.time()
        # TODO: use sample_weight?
        # sample_weight = fit_args.get('sample_weight', None)
        model.val_score = model.score_with_y_pred_proba(y=fit_args['y'], y_pred_proba=oof_pred_proba)
    else:
        y_pred_proba = model.predict_proba(**predict_proba_args)
        time_pred_end = time.time()
        sample_weight_val = fit_args.get('sample_weight_val', None)
        model.val_score = model.score_with_y_pred_proba(y=y_val, y_pred_proba=y_pred_proba, sample_weight=sample_weight_val)

    model.fit_time = time_fit_end - time_fit_start
    model.predict_time = time_pred_end - time_fit_end
    model.save()
    return model


def skip_hpo(model, X, y, X_val, y_val, scheduler_options=None, time_limit=None, **kwargs):
    """Skips HPO and simply trains the model once with the provided HPO time budget. Returns model artifacts as if from HPO."""
    fit_model_args = dict(X=X, y=y, **kwargs)
    predict_proba_args = dict(X=X_val)
    fit_and_save_model(model=model, params=dict(), fit_args=fit_model_args, predict_proba_args=predict_proba_args, y_val=y_val, time_start=time.time(), time_limit=time_limit)
    hpo_results = {'total_time': model.fit_time}
    hpo_model_performances = {model.name: model.val_score}
    hpo_models = {model.name: model.path}
    return hpo_models, hpo_model_performances, hpo_results

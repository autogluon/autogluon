import logging
import os
import time
import traceback

from autogluon.core.models.abstract.model_trial import init_model
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.core.utils.loaders import load_pkl

logger = logging.getLogger("autogluon.timeseries.trainer")


def model_trial(
    args,
    model_cls,
    init_params,
    train_path,
    val_path,
    time_start,
    hpo_executor,
    is_bagged_model=False,
    reporter=None,  # reporter only used by custom strategy, hence optional
    time_limit=None,
    fit_kwargs=None,
):
    """Runs a single trial of a hyperparameter tuning. Replaces
    `core.models.abstract.model_trial.model_trial` for timeseries models.
    """
    try:
        model = init_model(
            args, model_cls, init_params, backend=hpo_executor.executor_type, is_bagged_model=is_bagged_model
        )
        model.set_contexts(path_context=os.path.join(model.path_root, model.name))

        train_data = load_pkl.load(train_path)
        val_data = load_pkl.load(val_path)

        eval_metric = model.eval_metric

        model = fit_and_save_model(
            model,
            fit_kwargs,
            train_data,
            val_data,
            eval_metric,
            time_start=time_start,
            time_limit=time_limit,
        )

    except Exception as err:
        if not isinstance(err, TimeLimitExceeded):
            logger.error(f"\tWarning: Exception caused {model.name} to fail during training... Skipping this model.")
            logger.error(f"\t{err}")
            logger.debug(traceback.format_exc())
        # In case of TimeLimitExceed, val_score could be None
        hpo_executor.report(
            reporter=reporter,
            epoch=1,
            validation_performance=model.val_score if model.val_score is not None else float("-inf"),
        )
        if reporter is not None:
            reporter.terminate()
    else:
        hpo_executor.report(reporter=reporter, epoch=1, validation_performance=model.val_score)


def fit_and_save_model(model, fit_kwargs, train_data, val_data, eval_metric, time_start, time_limit=None):
    time_current = time.time()
    time_elapsed = time_current - time_start
    if time_limit is not None:
        time_left = time_limit - time_elapsed
        if time_left <= 0:
            raise TimeLimitExceeded
    else:
        time_left = None

    time_fit_start = time.time()
    model.fit(train_data=train_data, val_data=val_data, time_limit=time_left, **fit_kwargs)
    model.fit_time = time.time() - time_fit_start
    model.score_and_cache_oof(val_data, store_val_score=True, store_predict_time=True)

    logger.debug(f"\tHyperparameter tune run: {model.name}")
    logger.debug(f"\t\t{model.val_score:<7.4f}".ljust(15) + f"= Validation score ({eval_metric})")
    logger.debug(f"\t\t{model.fit_time:<7.3f} s".ljust(15) + "= Training runtime")
    logger.debug(f"\t\t{model.predict_time:<7.3f} s".ljust(15) + "= Training runtime")
    model.save()
    return model


def skip_hpo(model, train_data, val_data, time_limit=None):
    """Skip hyperparameter optimization and train model with given parameters.
    Replaces `core.models.abstract.model_trial.skip_hpo` for timeseries
    models.
    """
    fit_and_save_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        fit_kwargs={},
        eval_metric=model.eval_metric,
        time_start=time.time(),
        time_limit=time_limit,
    )
    hpo_results = {"total_time": model.fit_time}
    hpo_model_performances = {model.name: model.val_score}
    hpo_results["hpo_model_performances"] = hpo_model_performances
    hpo_models = {model.name: model.path}
    return hpo_models, hpo_results

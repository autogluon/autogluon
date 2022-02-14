import os
import time
import logging

from autogluon.core.models.abstract.model_trial import init_model
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.core.utils.loaders import load_pkl

from ....utils.metric_utils import METRIC_COEFFICIENTS


logger = logging.getLogger(__name__)


def model_trial(args,
                reporter,
                model_cls,
                init_params,
                train_path,
                val_path,
                time_start,
                time_limit=None,
                fit_kwargs=None,
                ):
    try:
        model = init_model(args, model_cls, init_params)
        model.set_contexts(path_context=model.path_root + model.name + os.path.sep)

        train_data = load_pkl.load(train_path)
        val_data = load_pkl.load(val_path)

        eval_metric = model.eval_metric

        model = fit_and_save_model(model,
                                   fit_kwargs,
                                   train_data,
                                   val_data,
                                   eval_metric,
                                   time_start=time_start,
                                   time_limit=time_limit,
                                   )

    except Exception as e:
        if not isinstance(e, TimeLimitExceeded):
            logger.exception(e, exc_info=True)
        reporter.terminate()
    else:
        reporter(epoch=1, validation_performance=model.val_score)


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
    model.fit(train_data, val_data=val_data, time_limit=time_left, **fit_kwargs)
    time_fit_end = time.time()
    logger.log(30, f"Evaluating model {model.name} with metric {eval_metric} on validation data...")
    model.val_score = model.score(val_data, eval_metric) * METRIC_COEFFICIENTS[eval_metric]
    time_pred_end = time.time()
    logger.log(30, f"Validation score for model {model.name} is {model.val_score}")
    model.fit_time = time_fit_end - time_fit_start
    model.predict_time = time_pred_end - time_fit_end
    model.save()
    return model

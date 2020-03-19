import logging

from ...abstract import model_trial
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
        model, args, util_args = model_trial.prepare_inputs(args=args)

        try_import_lightgbm()
        import lightgbm as lgb

        dataset_train = lgb.Dataset(util_args.directory + util_args.dataset_train_filename)
        dataset_val = lgb.Dataset(util_args.directory + util_args.dataset_val_filename)
        X_val, y_val = load_pkl.load(util_args.directory + util_args.dataset_val_pkl_filename)

        fit_model_args = dict(dataset_train=dataset_train, dataset_val=dataset_val)
        predict_proba_args = dict(X=X_val)
        model_trial.fit_and_save_model(model=model, params=args, fit_args=fit_model_args, predict_proba_args=predict_proba_args, y_test=y_val,
                                       time_start=util_args.time_start, time_limit=util_args.get('time_limit', None), reporter=reporter)
    except Exception as e:
        if not isinstance(e, TimeLimitExceeded):
            logger.exception(e, exc_info=True)
        reporter.terminate()

    # FIXME: If stopping metric and eval metric differ, the previous reported scores will not align as they will be evaluated with stopping_metric, whereas this is evaluated with eval_metric
    #  This should only impact if the reporter data is used
    # FIXME: If stopping metric score > eval metric score, stopping metric score will be recorded as best score, this is a defect!
    # FIXME: It might be the case that if a reporter has been recorded and the model crash, AutoGluon will try to access the invalid model and fail.
    # reporter(epoch=model.params_trained['num_boost_round'] + 1, validation_performance=score)

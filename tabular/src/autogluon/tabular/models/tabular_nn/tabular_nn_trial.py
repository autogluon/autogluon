import logging

from .tabular_nn_dataset import TabularNNDataset
from autogluon.core.models.abstract import model_trial
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.core import args

logger = logging.getLogger(__name__)


@args()
def tabular_nn_trial(args, reporter):
    """ Training and evaluation function used during a single trial of HPO """
    try:
        model, args, util_args = model_trial.prepare_inputs(args=args)

        train_dataset = TabularNNDataset.load(util_args.train_path)
        val_dataset = TabularNNDataset.load(util_args.val_path)
        y_val = val_dataset.get_labels()

        fit_model_args = dict(X_train=train_dataset, y_train=None, X_val=val_dataset, **util_args.get('fit_kwargs', dict()))
        predict_proba_args = dict(X=val_dataset)
        model_trial.fit_and_save_model(model=model, params=args, fit_args=fit_model_args, predict_proba_args=predict_proba_args, y_val=y_val,
                                       time_start=util_args.time_start, time_limit=util_args.get('time_limit', None), reporter=reporter)
    except Exception as e:
        if not isinstance(e, TimeLimitExceeded):
            logger.exception(e, exc_info=True)
        reporter.terminate()

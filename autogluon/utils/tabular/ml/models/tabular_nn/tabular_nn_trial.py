import logging

from .tabular_nn_dataset import TabularNNDataset
from ..abstract import model_trial
from ....utils.exceptions import TimeLimitExceeded
from ......core import args

logger = logging.getLogger(__name__)


@args()
def tabular_nn_trial(args, reporter):
    """ Training and evaluation function used during a single trial of HPO """
    try:
        model, args, util_args = model_trial.prepare_inputs(args=args)

        train_dataset = TabularNNDataset.load(util_args.train_path)
        test_dataset = TabularNNDataset.load(util_args.test_path)
        y_test = test_dataset.get_labels()

        fit_model_args = dict(X_train=train_dataset, Y_train=None, X_test=test_dataset)
        predict_proba_args = dict(X=test_dataset)
        model_trial.fit_and_save_model(model=model, params=args, fit_args=fit_model_args, predict_proba_args=predict_proba_args, y_test=y_test,
                                       time_start=util_args.time_start, time_limit=util_args.get('time_limit', None), reporter=reporter)
    except Exception as e:
        if not isinstance(e, TimeLimitExceeded):
            logger.exception(e, exc_info=True)
        reporter.terminate()

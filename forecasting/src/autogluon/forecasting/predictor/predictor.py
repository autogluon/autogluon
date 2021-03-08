import copy
import logging
import pprint
import pandas as pd

from autogluon.core.dataset import TabularDataset
from autogluon.core.task.base.base_task import schedulers
from autogluon.core.utils import set_logger_verbosity
from autogluon.core.utils.utils import setup_outputdir
from autogluon.forecasting.utils.dataset_utils import rebuild_tabular, train_test_split_gluonts, \
    train_test_split_dataframe
from autogluon.forecasting.task.forecasting.dataset import TimeSeriesDataset
from ..task.forecasting.predictor_legacy import ForecastingPredictorV1
from ..learner import AbstractLearner, DefaultLearner
from gluonts.dataset.common import FileDataset, ListDataset
from ..trainer import AbstractTrainer

logger = logging.getLogger()  # return root logger


class ForecastingPredictor(ForecastingPredictorV1):
    Dataset = TabularDataset
    predictor_file_name = 'predictor.pkl'

    def __init__(
            self,
            eval_metric=None,
            path=None,
            verbosity=2,
            **kwargs
    ):
        self.verbosity = verbosity
        set_logger_verbosity(self.verbosity, logger=logger)
        # self._validate_init_kwargs(kwargs)
        path = setup_outputdir(path)

        learner_type = kwargs.pop('learner_type', DefaultLearner)
        learner_kwargs = kwargs.pop('learner_kwargs', dict())

        self._learner: AbstractLearner = learner_type(path_context=path, eval_metric=eval_metric, **learner_kwargs)
        self._learner_type = type(self._learner)
        self._trainer = None

    def fit(self,
            train_data,
            prediction_length,
            index_column="index_column",
            time_column="time_column",
            target_column="target_column",
            val_data=None,
            hyperparameters=None,
            time_limits=None,
            **kwargs):
        if self._learner.is_fit:
            raise AssertionError(
                'Predictor is already fit! To fit additional models, refer to `predictor.fit_extra`, or create a new '
                '`Predictor`.')

        self.index_column = index_column
        self.time_column = time_column
        self.target_column = target_column
        kwargs_orig = kwargs.copy()
        kwargs = self._validate_fit_kwargs(kwargs)
        kwargs = self._validate_hyperparameter_tune_kwargs(kwargs)
        verbosity = kwargs.get('verbosity', self.verbosity)
        set_logger_verbosity(verbosity, logger=logger)
        hyperparameter_tune = kwargs["hyperparameter_tune"]
        num_trials = kwargs["num_trials"]
        search_strategy = kwargs["search_strategy"]

        if verbosity >= 3:
            logger.log(20, '============ fit kwarg info ============')
            logger.log(20, 'User Specified kwargs:')
            logger.log(20, f'{pprint.pformat(kwargs_orig)}')
            logger.log(20, 'Full kwargs:')
            logger.log(20, f'{pprint.pformat(kwargs)}')
            logger.log(20, '========================================')

        freq = kwargs.get("freq", None)
        set_logger_verbosity(verbosity, logger)
        if isinstance(train_data, pd.DataFrame):
            logger.log(30, "Training with dataset in tabular format...")
            train_data = rebuild_tabular(train_data,
                                         index_column=index_column,
                                         target_column=target_column,
                                         time_column=time_column)
            logger.log(30, "Finish rebuilding the data, showing the top five rows.")
            logger.log(30, train_data.head())
            if val_data is None:
                logger.log(30, "Validation data is None, will do auto splitting...")
                train_data, val_data = train_test_split_dataframe(train_data, prediction_length)
            else:
                val_data = rebuild_tabular(val_data,
                                           index_column=index_column,
                                           target_column=target_column,
                                           time_column=time_column)
            train_data = TimeSeriesDataset(train_data, index_column=index_column)
            freq = train_data.get_freq()
            val_data = TimeSeriesDataset(val_data, index_column=index_column)
        elif isinstance(train_data, FileDataset) or isinstance(train_data, ListDataset):
            logger.log(30, "Training with dataset in gluon-ts format...")
            if val_data is None:
                logger.log(30, "Validation data is not specified, will do auto splitting...")
                train_data, val_data = train_test_split_gluonts(train_data, prediction_length, freq)
        else:
            raise TypeError("Does not support dataset type:", type(train_data))

        refit_full = kwargs["refit_full"]
        save_data = kwargs["save_data"]
        if not save_data:
            logger.log(30,
                       'Warning: `save_data=False` will disable or limit advanced functionality after training such '
                       'as feature importance calculations. It is recommended to set `save_data=True` unless you '
                       'explicitly wish to not have the data saved to disk.')
            if refit_full:
                raise ValueError(
                    '`refit_full=True` is only available when `cache_data=True`. Set `cache_data=True` to utilize '
                    '`refit_full`.')

        set_best_to_refit_full = kwargs.get('set_best_to_refit_full', False)
        if set_best_to_refit_full and not refit_full:
            raise ValueError(
                '`set_best_to_refit_full=True` is only available when `refit_full=True`. Set `refit_full=True` to '
                'utilize `set_best_to_refit_full`.')

        random_seed = kwargs.get('random_seed', 0)
        logger.log(30, f"Random seed set to {random_seed}")
        quantiles = kwargs.get("quantiles", ["0.5"])
        logger.log(30, f"All models will be trained for quantiles {quantiles}.")
        scheduler_options = {"searcher": search_strategy,
                             "resource": {"num_cpus": 1, "num_gpus": 0},
                             "num_trials": num_trials,
                             "reward_attr": "validation_performance",
                             "time_attr": "epoch",
                             "time_out": time_limits}

        scheduler_cls = schedulers[search_strategy.lower()]

        scheduler_options = (scheduler_cls, scheduler_options)

        self._learner.fit(train_data=train_data,
                          freq=freq,
                          prediction_length=prediction_length,
                          val_data=val_data,
                          scheduler_options=scheduler_options,
                          hyperparameters=hyperparameters,
                          hyperparameter_tune=hyperparameter_tune,
                          quantiles=quantiles,
                          time_limits=time_limits,)

        self._set_post_fit_vars()
        self._post_fit(
            keep_only_best=kwargs["keep_only_best"],
            refit_full=kwargs['refit_full'],
            set_best_to_refit_full=kwargs["set_best_to_refit_full"],
        )
        self.save()
        return self

    def _validate_hyperparameter_tune_kwargs(self, kwargs):
        hyperparameter_kwargs_default = {
            "search_strategy": "random",
            "num_trials": 1,
        }
        copied_hyperparameter_tune_kwargs = copy.deepcopy(hyperparameter_kwargs_default)
        copied_hyperparameter_tune_kwargs.update(kwargs)
        return copied_hyperparameter_tune_kwargs

    def _validate_fit_kwargs(self, kwargs):
        kwargs_default = {
            "hyperparameter_tune": False,
            "set_best_to_refit_full": True,
            "keep_only_best": True,
            "refit_full": False,
            "save_data": True,
            "freq": None,
            "quantiles": ["0.5"]
        }
        copied_kwargs = copy.deepcopy(kwargs_default)
        copied_kwargs.update(kwargs)
        return copied_kwargs

    def _set_post_fit_vars(self, learner: AbstractLearner = None):
        if learner is not None:
            self._learner: AbstractLearner = learner
        self._learner_type = type(self._learner)
        if self._learner.trainer_path is not None:
            self._trainer: AbstractTrainer = self._learner.load_trainer()

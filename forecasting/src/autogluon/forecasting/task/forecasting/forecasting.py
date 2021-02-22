from autogluon.core.task.base.base_task import BaseTask, schedulers, compile_scheduler_options
from .dataset import TimeSeriesDataset
from autogluon.core.dataset import TabularDataset
from ...utils.dataset_utils import rebuild_tabular, train_test_split_dataframe, train_test_split_gluonts
from ...learner import DefaultLearner as Learner
from .predictor_legacy import ForecastingPredictorV1
from ...trainer import AutoTrainer
import pandas as pd
from gluonts.dataset.common import Dataset, ListDataset, FileDataset
from autogluon.core.utils.utils import setup_outputdir
from autogluon.core.utils.miscs import set_logger_verbosity
import logging

__all__ = ['Forecasting']

logger = logging.getLogger()


class Forecasting(BaseTask):
    Dataset = TabularDataset
    Predictor = ForecastingPredictorV1

    @staticmethod
    def load(output_directory, verbosity=2):
        return ForecastingPredictorV1.load(output_directory=output_directory, verbosity=verbosity)

    @staticmethod
    def fit(train_data,
            prediction_length,
            index_column="index",
            target_column=None,
            time_column="date",
            val_data=None,
            time_limits=None,
            output_directory=None,
            eval_metric=None,
            hyperparameter_tune=False,
            hyperparameters=None,
            num_trials=None,
            search_strategy='random',
            verbosity=2,
            **kwargs):

        # TODO: allow user to input more scheduler options and use compile_scheduler_options()
        output_directory = setup_outputdir(output_directory)
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
            if val_data is not None:
                val_data = TimeSeriesDataset(val_data, index_column=index_column)
        elif isinstance(train_data, FileDataset) or isinstance(train_data, ListDataset):
            logger.log(30, "Training with dataset in gluon-ts format...")
            if val_data is None:
                logger.log(30, "Validation data is not specified, will do auto splitting...")
                train_data, val_data = train_test_split_gluonts(train_data, prediction_length, freq)
        else:
            raise TypeError("Does not support dataset type:", type(train_data))
        refit_full = kwargs.get('refit_full', False)
        save_data = kwargs.get('save_data', True)
        if not save_data:
            logger.log(30,
                       'Warning: `save_data=False` will disable or limit advanced functionality after training such as feature importance calculations. It is recommended to set `cache_data=True` unless you explicitly wish to not have the data saved to disk.')
            if refit_full:
                raise ValueError(
                    '`refit_full=True` is only available when `cache_data=True`. Set `cache_data=True` to utilize `refit_full`.')

        set_best_to_refit_full = kwargs.get('set_best_to_refit_full', True)
        if set_best_to_refit_full and not refit_full:
            raise ValueError(
                '`set_best_to_refit_full=True` is only available when `refit_full=True`. Set `refit_full=True` to utilize `set_best_to_refit_full`.')

        trainer_type = kwargs.get('trainer_type', AutoTrainer)
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
        learner = Learner(path_context=output_directory,
                          eval_metric=eval_metric,
                          trainer_type=trainer_type,
                          random_seed=random_seed, )

        learner.fit(train_data=train_data,
                    freq=freq,
                    prediction_length=prediction_length,
                    val_data=val_data,
                    scheduler_options=scheduler_options,
                    hyperparameters=hyperparameters,
                    hyperparameter_tune=hyperparameter_tune,
                    quantiles=quantiles)

        # TODO: refit full
        predictor = ForecastingPredictorV1(learner,
                                           index_column=index_column,
                                           target_column=target_column,
                                           time_column=time_column)

        keep_only_best = kwargs.get('keep_only_best', True)
        if refit_full is True:
            if keep_only_best is True:
                if set_best_to_refit_full is True:
                    refit_full = 'best'
                else:
                    logger.warning(
                        f'refit_full was set to {refit_full}, but keep_only_best=True and set_best_to_refit_full=False. Disabling refit_full to avoid training models which would be automatically deleted.')
                    refit_full = False
            else:
                refit_full = 'all'

        if refit_full is not False:
            logger.log(30, f"refit_full was set, start training models on the whole dataset...")
            trainer = predictor._trainer
            trainer_model_best = trainer.get_model_best()
            predictor.refit_full(models=refit_full)
            if set_best_to_refit_full:
                if trainer_model_best in trainer.model_full_dict.keys():
                    trainer.model_best = trainer.model_full_dict[trainer_model_best]
                    # Note: model_best will be overwritten if additional training is done with new models, since model_best will have validation score of None and any new model will have a better validation score.
                    # This has the side-effect of having the possibility of model_best being overwritten by a worse model than the original model_best.
                    trainer.save()
                else:
                    logger.warning(
                        f'Best model ({trainer_model_best}) is not present in refit_full dictionary. Training may have failed on the refit model. AutoGluon will default to using {trainer_model_best} for predictions.')
            logger.log(30, "End of refit_full")
        predictor.save()

        return predictor

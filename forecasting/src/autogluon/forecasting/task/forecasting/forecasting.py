from core.task.base.base_task import BaseTask, schedulers, compile_scheduler_options
from .dataset import TimeSeriesDataset

from ...learner.abstract_learner import AbstractLearner
from ...learner import DefaultLearner as Learner
from .predictor import ForecastingPredictor
from ...trainer import AutoTrainer
from core.utils.utils import setup_outputdir
__all__ = ['Forecasting']


class Forecasting(BaseTask):

    Dataset = TimeSeriesDataset
    Predictor = ForecastingPredictor

    @staticmethod
    def load(output_directory, verbosity=2):
        return ForecastingPredictor.load(output_directory=output_directory, verbosity=verbosity)

    @staticmethod
    def fit(train_data,
            freq,
            prediction_length,
            test_data=None,
            time_limits=None,
            output_directory=None,
            eval_metric=None,
            hyperparameter_tune=False,
            hyperparameters=None,
            num_trials=None,
            search_strategy='random',
            verbosity=2,
            **kwargs):

        # TODO: Maybe we can infer freq and prediction length inside fit from train_data and test data?
        # TODO: allow user to input more scheduler options and use compile_scheduler_options()
        output_directory = setup_outputdir(output_directory)

        trainer_type = kwargs.get('trainer_type', AutoTrainer)
        random_seed = kwargs.get('random_seed', 0)
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
                          random_seed=random_seed,)

        learner.fit(train_data=train_data,
                    freq=freq,
                    prediction_length=prediction_length,
                    test_data=test_data,
                    scheduler_options=scheduler_options,
                    hyperparameters=hyperparameters,
                    hyperparameter_tune=hyperparameter_tune,)

        # TODO: refit full
        predictor = ForecastingPredictor(learner)
        return predictor







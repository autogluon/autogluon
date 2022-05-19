import logging
import random
import time
from typing import Type, Optional, Any, Dict

import pandas as pd

from autogluon.core.learner import AbstractLearner
from autogluon.forecasting.dataset import TimeSeriesDataFrame
from autogluon.forecasting.models.abstract import AbstractForecastingModel

from autogluon.forecasting.trainer import (
    AutoForecastingTrainer,
    AbstractForecastingTrainer,
)
from autogluon.forecasting.utils.metric_utils import check_get_evaluation_metric

logger = logging.getLogger(__name__)


class ForecastingLearner(AbstractLearner):
    def __init__(
        self,
        path_context: str,
        is_trainer_present: bool = False,
        random_state: int = 0,
        trainer_type: Type[AbstractForecastingTrainer] = AutoForecastingTrainer,
        eval_metric: Optional[str] = None,
    ):
        super().__init__()
        self.path, self.model_context, self.save_path = self.create_contexts(
            path_context
        )
        self.eval_metric: str = check_get_evaluation_metric(eval_metric)
        self.is_trainer_present = is_trainer_present

        if random_state is None:
            random_state = random.randint(0, 1000000)
        self.random_state: int = random_state

        self.trainer: Optional[AbstractForecastingTrainer] = None
        self.trainer_type: Type[AbstractForecastingTrainer] = trainer_type
        self.trainer_path: Optional[str] = None
        self.reset_paths: bool = False

    def fit(self, train_data, freq, prediction_length, val_data=None, **kwargs):
        return self._fit(
            train_data=train_data,
            freq=freq,
            prediction_length=prediction_length,
            val_data=val_data,
            **kwargs,
        )

    def load_trainer(self) -> AbstractForecastingTrainer:
        return super().load_trainer()  # noqa

    def refit_full(self, models="all"):
        return self.load_trainer().refit_full(models=models)

    def predict(
        self,
        data: TimeSeriesDataFrame,
        model: Optional[AbstractForecastingModel] = None,
        **kwargs,
    ):
        predict_target = self.load_trainer().predict(data=data, model=model, **kwargs)
        return predict_target

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        freq: str,
        prediction_length: int,
        val_data: Optional[TimeSeriesDataFrame] = None,
        scheduler_options=None,
        hyperparameter_tune=False,
        hyperparameters=None,
        time_limit=None,
        use_feat_static_cat=False,
        use_feat_static_real=False,
        cardinality=None,
        **kwargs,
    ):
        self._time_limit = time_limit
        time_start = time.time()
        if time_limit:
            logger.log(
                20, f"Beginning AutoGluon training ... Time limit = {time_limit}s"
            )
        else:
            logger.log(20, "Beginning AutoGluon training ...")
        logger.log(20, f"AutoGluon will save models to {self.path}")

        trainer = self.trainer_type(
            path=self.model_context,
            freq=freq,
            prediction_length=prediction_length,
            eval_metric=self.eval_metric,
            scheduler_options=scheduler_options,
            use_feat_static_cat=use_feat_static_cat,
            use_feat_static_real=use_feat_static_real,
            cardinality=cardinality,
            **kwargs,
        )

        self.trainer_path = trainer.path
        if self.eval_metric is None:
            self.eval_metric = trainer.eval_metric

        self.save()
        trainer.fit(
            train_data=train_data,
            val_data=val_data,
            hyperparameter_tune=hyperparameter_tune,
            hyperparameters=hyperparameters,
            time_limit=time_limit,
        )
        self.save_trainer(trainer=trainer)
        time_end = time.time()
        self._time_fit_training = time_end - time_start
        logger.log(
            20,
            f"AutoGluon training complete, total runtime = {round(self._time_fit_training, 2)}s ...",
        )

    def score(self, data: TimeSeriesDataFrame, model: AbstractForecastingModel = None) -> float:
        trainer = self.load_trainer()
        return trainer.score(data=data, model=model)

    def leaderboard(self, data: Optional[TimeSeriesDataFrame] = None) -> pd.DataFrame:
        trainer = self.load_trainer()
        return trainer.leaderboard(data)

    def get_info(self, include_model_info: bool = False, **kwargs) -> Dict[str, Any]:
        learner_info = super().get_info(include_model_info=include_model_info)
        trainer = self.load_trainer()
        trainer_info = trainer.get_info(include_model_info=include_model_info)
        learner_info.update(
            {
                "time_fit_training": self._time_fit_training,
                "time_limit": self._time_limit,
            }
        )

        learner_info.update(trainer_info)
        return learner_info

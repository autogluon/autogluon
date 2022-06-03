import logging
import time
from typing import Type, Optional, Any, Dict, Tuple, Union

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
    """ForecastingLearner encompasses a full time series learning problem for a
    training run, and keeps track of datasets, features, random seeds, and the
    trainer object.
    """

    def __init__(
        self,
        path_context: str,
        target: str = "target",
        random_state: int = 0,
        trainer_type: Type[AbstractForecastingTrainer] = AutoForecastingTrainer,
        eval_metric: Optional[str] = None,
        prediction_length: int = 1,
        **kwargs,
    ):
        super().__init__(path_context=path_context, random_state=random_state)
        self.eval_metric: str = check_get_evaluation_metric(eval_metric)
        self.trainer_type = trainer_type
        self.target = target
        self.prediction_length = prediction_length
        self.quantile_levels = kwargs.get(
            "quantile_levels",
            kwargs.get("quantiles", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        )
        logger.info(f"Learner random seed set to {random_state}")

    def load_trainer(self) -> AbstractForecastingTrainer:
        """Return the trainer object corresponding to the learner."""
        return super().load_trainer()  # noqa

    def fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: TimeSeriesDataFrame = None,
        hyperparameters: Union[str, Dict] = None,
        hyperparameter_tune: bool = False,
        **kwargs,
    ) -> None:
        return self._fit(
            train_data=train_data,
            val_data=val_data,
            hyperparameters=hyperparameters,
            hyperparameter_tune=hyperparameter_tune,
            **kwargs,
        )

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        hyperparameters: Union[str, Dict] = None,
        hyperparameter_tune: bool = False,
        scheduler_options: Tuple[Type, Dict[str, Any]] = None,
        time_limit: Optional[int] = None,
        **kwargs,
    ) -> None:
        self._time_limit = time_limit
        time_start = time.time()

        logger.debug(
            "Beginning AutoGluon training with ForecastingLearner "
            + (f"Time limit = {time_limit}" if time_limit else "")
        )
        logger.info(f"AutoGluon will save models to {self.path}")

        trainer_init_kwargs = kwargs.copy()
        trainer_init_kwargs.update(
            dict(
                path=self.model_context,
                prediction_length=self.prediction_length,
                eval_metric=self.eval_metric,
                scheduler_options=scheduler_options,
                target=self.target,
                quantile_levels=self.quantile_levels,
            )
        )
        self.trainer = self.trainer_type(**trainer_init_kwargs)
        self.trainer_path = self.trainer.path
        self.save()

        self.trainer.fit(
            train_data=train_data,
            val_data=val_data,
            hyperparameter_tune=hyperparameter_tune,
            hyperparameters=hyperparameters,
            time_limit=time_limit,
        )
        self.save_trainer(trainer=self.trainer)

        self._time_fit_training = time.time() - time_start
        logger.info(
            f"AutoGluon ForecastingLearner complete, total runtime = {round(self._time_fit_training, 2)}s",
        )

    def predict(
        self,
        data: TimeSeriesDataFrame,
        model: Optional[AbstractForecastingModel] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        return self.load_trainer().predict(data=data, model=model, **kwargs)

    def score(
        self, data: TimeSeriesDataFrame, model: AbstractForecastingModel = None
    ) -> float:
        return self.load_trainer().score(data=data, model=model)

    def leaderboard(self, data: Optional[TimeSeriesDataFrame] = None) -> pd.DataFrame:
        return self.load_trainer().leaderboard(data)

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

    def refit_full(self, models="all"):
        # TODO: Implement refitting
        # return self.load_trainer().refit_full(models=models)
        raise NotImplementedError(
            "refitting logic currently not implemented in autogluon.forecasting"
        )

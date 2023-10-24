import logging
import reprlib
import time
from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd

from autogluon.core.learner import AbstractLearner
from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame
from autogluon.timeseries.metrics import TimeSeriesScorer, check_get_evaluation_metric
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.splitter import AbstractWindowSplitter
from autogluon.timeseries.trainer import AbstractTimeSeriesTrainer, AutoTimeSeriesTrainer
from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator
from autogluon.timeseries.utils.forecast import get_forecast_horizon_index_ts_dataframe

logger = logging.getLogger(__name__)


class TimeSeriesLearner(AbstractLearner):
    """TimeSeriesLearner encompasses a full time series learning problem for a
    training run, and keeps track of datasets, features, and the trainer object.
    """

    def __init__(
        self,
        path_context: str,
        target: str = "target",
        known_covariates_names: Optional[List[str]] = None,
        trainer_type: Type[AbstractTimeSeriesTrainer] = AutoTimeSeriesTrainer,
        eval_metric: Union[str, TimeSeriesScorer, None] = None,
        eval_metric_seasonal_period: Optional[int] = None,
        prediction_length: int = 1,
        cache_predictions: bool = True,
        **kwargs,
    ):
        super().__init__(path_context=path_context)
        self.eval_metric: TimeSeriesScorer = check_get_evaluation_metric(eval_metric)
        self.eval_metric_seasonal_period = eval_metric_seasonal_period
        self.trainer_type = trainer_type
        self.target = target
        self.known_covariates_names = [] if known_covariates_names is None else known_covariates_names
        self.prediction_length = prediction_length
        self.quantile_levels = kwargs.get("quantile_levels", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        self.cache_predictions = cache_predictions

        self.feature_generator = TimeSeriesFeatureGenerator(
            target=self.target, known_covariates_names=self.known_covariates_names
        )

    def load_trainer(self) -> AbstractTimeSeriesTrainer:
        """Return the trainer object corresponding to the learner."""
        return super().load_trainer()  # noqa

    def fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: TimeSeriesDataFrame = None,
        hyperparameters: Union[str, Dict] = None,
        hyperparameter_tune_kwargs: Optional[Union[str, dict]] = None,
        **kwargs,
    ) -> None:
        return self._fit(
            train_data=train_data,
            val_data=val_data,
            hyperparameters=hyperparameters,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            **kwargs,
        )

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        hyperparameters: Union[str, Dict] = None,
        hyperparameter_tune_kwargs: Optional[Union[str, dict]] = None,
        time_limit: Optional[int] = None,
        val_splitter: Optional[AbstractWindowSplitter] = None,
        refit_every_n_windows: Optional[int] = 1,
        **kwargs,
    ) -> None:
        self._time_limit = time_limit
        time_start = time.time()

        logger.debug(
            "Beginning AutoGluon training with TimeSeriesLearner "
            + (f"Time limit = {time_limit}" if time_limit else "")
        )
        logger.info(f"AutoGluon will save models to {self.path}")

        logger.info(f"AutoGluon will gauge predictive performance using evaluation metric: '{self.eval_metric}'")
        if not self.eval_metric.greater_is_better_internal:
            logger.info(
                "\tThis metric's sign has been flipped to adhere to being 'higher is better'. "
                "The reported score can be multiplied by -1 to get the metric value.",
            )
        train_data = self.feature_generator.fit_transform(train_data, data_frame_name="train_data")
        if val_data is not None:
            val_data = self.feature_generator.transform(val_data, data_frame_name="tuning_data")

        trainer_init_kwargs = kwargs.copy()
        trainer_init_kwargs.update(
            dict(
                path=self.model_context,
                prediction_length=self.prediction_length,
                eval_metric=self.eval_metric,
                eval_metric_seasonal_period=self.eval_metric_seasonal_period,
                target=self.target,
                quantile_levels=self.quantile_levels,
                verbosity=kwargs.get("verbosity", 2),
                enable_ensemble=kwargs.get("enable_ensemble", True),
                metadata=self.feature_generator.covariate_metadata,
                val_splitter=val_splitter,
                refit_every_n_windows=refit_every_n_windows,
                cache_predictions=self.cache_predictions,
            )
        )
        self.trainer = self.trainer_type(**trainer_init_kwargs)
        self.trainer_path = self.trainer.path
        self.save()

        self.trainer.fit(
            train_data=train_data,
            val_data=val_data,
            hyperparameters=hyperparameters,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            excluded_model_types=kwargs.get("excluded_model_types"),
            time_limit=time_limit,
        )
        self.save_trainer(trainer=self.trainer)

        self._time_fit_training = time.time() - time_start

    def _align_covariates_with_forecast_index(
        self,
        known_covariates: Optional[TimeSeriesDataFrame],
        data: TimeSeriesDataFrame,
    ) -> Optional[TimeSeriesDataFrame]:
        """Select the relevant item_ids and timestamps from the known_covariates dataframe.

        If some of the item_ids or timestamps are missing, an exception is raised.
        """
        if len(self.known_covariates_names) == 0:
            return None
        if len(self.known_covariates_names) > 0 and known_covariates is None:
            raise ValueError(
                f"known_covariates {self.known_covariates_names} for the forecast horizon should be provided at prediction time."
            )

        if self.target in known_covariates.columns:
            known_covariates = known_covariates.drop(self.target, axis=1)

        missing_item_ids = data.item_ids.difference(known_covariates.item_ids)
        if len(missing_item_ids) > 0:
            raise ValueError(
                f"known_covariates are missing information for the following item_ids: {reprlib.repr(missing_item_ids.to_list())}."
            )

        forecast_index = get_forecast_horizon_index_ts_dataframe(data, prediction_length=self.prediction_length)
        try:
            known_covariates = known_covariates.loc[forecast_index]
        except KeyError:
            raise ValueError(
                f"known_covariates should include the values for prediction_length={self.prediction_length} "
                "many time steps into the future."
            )
        return known_covariates

    def predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        model: Optional[Union[str, AbstractTimeSeriesModel]] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        data = self.feature_generator.transform(data)
        known_covariates = self.feature_generator.transform_future_known_covariates(known_covariates)
        known_covariates = self._align_covariates_with_forecast_index(known_covariates=known_covariates, data=data)
        return self.load_trainer().predict(
            data=data, known_covariates=known_covariates, model=model, use_cache=use_cache, **kwargs
        )

    def score(
        self,
        data: TimeSeriesDataFrame,
        model: AbstractTimeSeriesModel = None,
        metric: Union[str, TimeSeriesScorer, None] = None,
        use_cache: bool = True,
    ) -> float:
        data = self.feature_generator.transform(data)
        return self.load_trainer().score(data=data, model=model, metric=metric, use_cache=use_cache)

    def leaderboard(self, data: Optional[TimeSeriesDataFrame] = None, use_cache: bool = True) -> pd.DataFrame:
        if data is not None:
            data = self.feature_generator.transform(data)
        return self.load_trainer().leaderboard(data, use_cache=use_cache)

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
        # self.random_state not used during fitting, so we don't include it in the summary
        # TODO: Report random seed passed to predictor.fit?
        learner_info.pop("random_state", None)
        return learner_info

    def refit_full(self, model: str = "all") -> Dict[str, str]:
        return self.load_trainer().refit_full(model=model)

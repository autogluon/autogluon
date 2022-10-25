import logging
import time
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.core.learner import AbstractLearner
from autogluon.timeseries.utils.features import (
    ContinuousAndCategoricalFeatureGenerator,
    convert_numerical_features_to_float,
)

from .dataset import TimeSeriesDataFrame
from .evaluator import TimeSeriesEvaluator
from .models.abstract import AbstractTimeSeriesModel
from .splitter import AbstractTimeSeriesSplitter, LastWindowSplitter
from .trainer import AbstractTimeSeriesTrainer, AutoTimeSeriesTrainer

logger = logging.getLogger(__name__)


class TimeSeriesLearner(AbstractLearner):
    """TimeSeriesLearner encompasses a full time series learning problem for a
    training run, and keeps track of datasets, features, random seeds, and the
    trainer object.
    """

    def __init__(
        self,
        path_context: str,
        target: str = "target",
        random_state: int = 0,
        trainer_type: Type[AbstractTimeSeriesTrainer] = AutoTimeSeriesTrainer,
        eval_metric: Optional[str] = None,
        prediction_length: int = 1,
        validation_splitter: AbstractTimeSeriesSplitter = LastWindowSplitter(),
        **kwargs,
    ):
        super().__init__(path_context=path_context, random_state=random_state)
        self.eval_metric: str = TimeSeriesEvaluator.check_get_evaluation_metric(eval_metric)
        self.trainer_type = trainer_type
        self.target = target
        self.prediction_length = prediction_length
        self.quantile_levels = kwargs.get(
            "quantile_levels",
            kwargs.get("quantiles", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        )
        self.validation_splitter = validation_splitter
        logger.info(f"Learner random seed set to {random_state}")
        self.feature_pipeline: ContinuousAndCategoricalFeatureGenerator = None

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
        **kwargs,
    ) -> None:
        self._time_limit = time_limit
        time_start = time.time()

        logger.debug(
            "Beginning AutoGluon training with TimeSeriesLearner "
            + (f"Time limit = {time_limit}" if time_limit else "")
        )
        logger.info(f"AutoGluon will save models to {self.path}")

        train_data, val_data = self._preprocess_static_features(train_data=train_data, val_data=val_data)

        # Process dynamic features
        # TODO: Handle dynamic features
        extra_columns = [c for c in train_data.columns.copy() if c != self.target]
        if len(extra_columns) > 0:
            logger.warning(f"Provided columns {extra_columns} will not be used.")

        # Train / validation split
        if val_data is None:
            logger.warning(
                "Validation data is None. "
                + self.validation_splitter.describe_validation_strategy(prediction_length=self.prediction_length)
            )
            train_data, val_data = self.validation_splitter.split(
                ts_dataframe=train_data, prediction_length=self.prediction_length
            )

        trainer_init_kwargs = kwargs.copy()
        trainer_init_kwargs.update(
            dict(
                path=self.model_context,
                prediction_length=self.prediction_length,
                eval_metric=self.eval_metric,
                target=self.target,
                quantile_levels=self.quantile_levels,
                verbosity=kwargs.get("verbosity", 2),
                enable_ensemble=kwargs.get("enable_ensemble", True),
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
            time_limit=time_limit,
        )
        self.save_trainer(trainer=self.trainer)

        self._time_fit_training = time.time() - time_start

    def _preprocess_static_features(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        # TODO: Fix unused argument
        static_feature_metadata: Optional[FeatureMetadata] = None,
    ) -> Tuple[TimeSeriesDataFrame, Optional[TimeSeriesDataFrame]]:
        """Convert static features to categorical & float dtypes, and check if val data has compatible features.

        If train_data has static features, then one of the following is guaranteed to be true:
        - val_data is None (split will be done automatically)
        - val_data has static features with the same columns as train_data

        If train_data doesn't have static features, then one of the following is guaranteed to be true:
        - val_data is None (split will be done automatically)
        - val_data doesn't have static features
        """
        # Avoid modifying data inplace
        train_data = train_data.copy(deep=False)
        if val_data is not None:
            val_data = val_data.copy(deep=False)

        self.feature_pipeline = ContinuousAndCategoricalFeatureGenerator(
            feature_metadata=static_feature_metadata, verbosity=0
        )

        if train_data.static_features is None:
            if val_data is not None and val_data.static_features is not None:
                val_data.static_features = None
        else:
            original_static_feature_columns = train_data.static_features.columns
            train_data.static_features = self.feature_pipeline.fit_transform(train_data.static_features)
            train_data.static_features = convert_numerical_features_to_float(train_data.static_features)

            mapped_to_categorical = []
            mapped_to_continuous = []
            unused = []
            for col_name in original_static_feature_columns:
                if train_data.static_features[col_name].dtype == "category":
                    mapped_to_categorical.append(col_name)
                elif train_data.static_features[col_name].dtype == np.float64:
                    mapped_to_continuous.append(col_name)
                else:
                    unused.append(col_name)

            logger.info("Following types of static features have been inferred:")
            logger.info(f"\tcategorical: {mapped_to_categorical}")
            logger.info(f"\tcontinuous (float): {mapped_to_continuous}")
            if len(unused) > 0:
                logger.info(f"\tremoved (neither categorical nor continuous): {unused}")
            logger.info(
                "To learn how to fix incorrectly inferred types, please see documentation for TimeSeriesPredictor.fit "
            )

            if val_data is not None:
                if val_data.static_features is None:
                    raise ValueError(
                        "Provided train_data has static_features, but tuning_data has no static features. "
                        "Please set `tuning_data=None` to automatically generate tuning_data, or make sure that the given "
                        "tuning_data also has static_features."
                    )
                try:
                    val_data.static_features = val_data.static_features[original_static_feature_columns]
                    val_data.static_features = self.feature_pipeline.transform(val_data.static_features)
                    val_data.static_features = convert_numerical_features_to_float(val_data.static_features)
                except:
                    raise ValueError(
                        "tuning_data.static_features should have the same columns as train_data.static_features. "
                        "Please set `tuning_data=None` to automatically generate tuning_data, or make sure that "
                        "names & dtypes of columns in tuning_data.static_features exactly match train_data.static_features"
                    )

        return train_data, val_data

    def predict(
        self,
        data: TimeSeriesDataFrame,
        model: Optional[Union[str, AbstractTimeSeriesModel]] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        if self.feature_pipeline.is_fit:
            if data.static_features is None:
                raise ValueError(
                    "Cannot predict since data has no static features (but training data had static features)"
                )
            data.static_features = self.feature_pipeline.transform(data.static_features)
            data.static_features = convert_numerical_features_to_float(data.static_features)
        prediction = self.load_trainer().predict(data=data, model=model, **kwargs)
        if prediction is None:
            raise RuntimeError("Prediction failed, please provide a different model to the `predict` method.")
        return prediction

    def score(
        self, data: TimeSeriesDataFrame, model: AbstractTimeSeriesModel = None, metric: Optional[str] = None
    ) -> float:
        return self.load_trainer().score(data=data, model=model, metric=metric)

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
        raise NotImplementedError("refitting logic currently not implemented in autogluon.timeseries")

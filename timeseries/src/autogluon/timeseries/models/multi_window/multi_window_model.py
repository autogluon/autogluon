import logging
import copy
import time
from typing import Optional
import numpy as np
import pandas as pd

from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel

logger = logging.getLogger(__name__)


class MultiWindowModel(AbstractTimeSeriesModel):
    def __init__(self, model_base: AbstractTimeSeriesModel, num_val_windows: int = 1, **kwargs):
        super().__init__(
            freq=model_base.freq,
            name=model_base.name,
            prediction_length=model_base.prediction_length,
            path=model_base.path,
            metadata=model_base.metadata,
            eval_metric=model_base.eval_metric,
            eval_metric_seasonal_period=model_base.eval_metric_seasonal_period,
        )
        self.model_base = model_base
        self.num_val_windows = num_val_windows
        self.most_recent_model: AbstractTimeSeriesModel = None
        self.info_per_fold = []

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[int] = None,
        **kwargs,
    ):
        if val_data is not None:
            raise ValueError(f"{self.name}.fit() does not ")

        trained_models = []
        fit_start_time = time.time()
        for window_idx in range(self.num_val_windows):
            model_fit_start_time = time.time()
            train_fold, val_fold = train_data.train_test_split(
                prediction_length=self.prediction_length,
                window_idx=window_idx,
                suffix=f"_fold_{window_idx}",
            )
            model = copy.deepcopy(self.model_base)
            logger.debug(f"\tFitting fold {window_idx + 1}")

            model.fit(
                train_data=train_fold,
                val_data=val_fold,
                time_limit=None if time_limit is None else time_limit - (model_fit_start_time - fit_start_time),
            )
            model.fit_time = time.time() - model_fit_start_time

            model.score_and_cache_oof(val_fold)
            trained_models.append(model)
            self.info_per_fold.append(
                {
                    "window_idx": window_idx,
                    "fit_time": model.fit_time,
                    "val_score": model.val_score,
                    "predict_time": model.predict_time,
                }
            )

        self.fit_time = sum(model.fit_time for model in trained_models)
        self.most_recent_model = trained_models[0]
        self.predict_time = self.most_recent_model.predict_time
        self._oof_predictions = pd.concat([model.get_oof_predictions() for model in trained_models])
        self.val_score = np.mean([model.val_score for model in trained_models])

    def get_info(self) -> dict:
        info = super().get_info()
        info["info_per_fold"] = self.info_per_fold
        return info

    def predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        if self.most_recent_model is None:
            raise ValueError(f"{self.name} must be fit before predicting")
        return self.most_recent_model.predict(data, known_covariates, **kwargs)

    def score_and_cache_oof(self, val_data: TimeSeriesDataFrame) -> None:
        # self.val_score, self.predict_time, self._oof_predictions saved during _fit()
        pass

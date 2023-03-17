import copy
import inspect
import logging
import os
import time
from typing import Dict, Optional, Type, Union

import numpy as np
import pandas as pd

from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel

logger = logging.getLogger(__name__)


class MultiWindowBacktestingModel(AbstractTimeSeriesModel):
    """
    A meta-model that trains the base model multiple times using different train/validation splits.

    Parameters
    ----------
    model_base : Union[AbstractTimeSeriesModel, Type[AbstractTimeSeriesModel]]
        The base model to repeatedly train. If a AbstractTimeSeriesModel class, then also provide model_base_kwargs
        which will be used to initialize the model via model_base(**model_base_kwargs).
    model_base_kwargs : Dict[str, any], default = None
        kwargs used to initialize model_base if model_base is a class.
    num_val_windows : int, default = 1
        Number of windows to use for backtesting, starting from the end of the training data.
    """

    def __init__(
        self,
        model_base: Union[AbstractTimeSeriesModel, Type[AbstractTimeSeriesModel]],
        model_base_kwargs: Optional[Dict[str, any]] = None,
        **kwargs,
    ):
        if inspect.isclass(model_base):
            if model_base_kwargs is None:
                model_base_kwargs = dict()
            self.model_base: AbstractTimeSeriesModel = model_base(**model_base_kwargs)
        elif model_base_kwargs is not None:
            raise AssertionError(
                f"model_base_kwargs must be None if model_base was passed as an object! "
                f"(model_base: {model_base}, model_base_kwargs: {model_base_kwargs})"
            )
        else:
            self.model_base: AbstractTimeSeriesModel = model_base
        self.most_recent_model: AbstractTimeSeriesModel = None
        self.info_per_val_window = []
        super().__init__(**kwargs)

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[int] = None,
        num_val_windows: int = 1,
        **kwargs,
    ):
        # TODO: use incremental training for GluonTS models?
        # TODO: implement parallel fitting similar to ParallelLocalFoldFittingStrategy in tabular?
        verbosity = kwargs.get("verbosity", 2)
        set_logger_verbosity(verbosity, logger=logger)

        if val_data is not None:
            raise ValueError(f"val_data should not be passed to {self.name}.fit()")

        trained_models = []
        global_fit_start_time = time.time()
        for window_idx in range(num_val_windows):
            train_fold, val_fold = train_data.train_test_split(
                prediction_length=self.prediction_length,
                window_idx=window_idx,
                suffix=f"_F{window_idx}",
            )

            logger.debug(f"\tWindow {window_idx + 1}")
            model = self.get_child_model(window_idx)
            model_fit_start_time = time.time()
            model.fit(
                train_data=train_fold,
                val_data=val_fold,
                time_limit=None if time_limit is None else time_limit - (model_fit_start_time - global_fit_start_time),
                **kwargs,
            )
            model.fit_time = time.time() - model_fit_start_time
            model.score_and_cache_oof(val_fold)
            trained_models.append(model)

            logger.debug(f"\t\t{model.val_score:<7.4f}".ljust(15) + f"= Validation score ({model.eval_metric})")
            logger.debug(f"\t\t{model.fit_time:<7.3f} s".ljust(15) + "= Training runtime")
            logger.debug(f"\t\t{model.predict_time:<7.3f} s".ljust(15) + "= Training runtime")

            self.info_per_val_window.append(
                {
                    "window_idx": window_idx,
                    "fit_time": model.fit_time,
                    "val_score": model.val_score,
                    "predict_time": model.predict_time,
                }
            )

        # Only the model trained on most recent data is saved & used for prediction
        self.most_recent_model = trained_models[0]
        self.predict_time = self.most_recent_model.predict_time
        self.fit_time = time.time() - global_fit_start_time - self.predict_time
        self._oof_predictions = pd.concat([model.get_oof_predictions() for model in trained_models])
        self.val_score = np.mean([model.val_score for model in trained_models])

    def get_info(self) -> dict:
        info = super().get_info()
        info["info_per_val_window"] = self.info_per_val_window
        return info

    def get_child_model(self, window_idx: int) -> AbstractTimeSeriesModel:
        model = copy.deepcopy(self.model_base)
        model.set_contexts(self.path + f"W{window_idx + 1}" + os.sep)
        return model

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
        # self.val_score, self.predict_time, self._oof_predictions already saved during _fit()
        pass

    def get_user_params(self) -> dict:
        return self.model_base.get_user_params()

    def _get_search_space(self):
        return self.model_base._get_search_space()

    def _initialize(self, **kwargs) -> None:
        super()._initialize(**kwargs)
        self.model_base.initialize(**kwargs)

    def _update_hpo_train_fn_kwargs(self, train_fn_kwargs: dict) -> dict:
        train_fn_kwargs["is_bagged_model"] = True
        train_fn_kwargs["init_params"]["model_base"] = self.model_base.__class__
        train_fn_kwargs["init_params"]["model_base_kwargs"] = self.get_params()
        return train_fn_kwargs

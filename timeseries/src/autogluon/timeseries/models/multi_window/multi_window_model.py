import copy
import inspect
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Type, Union

import numpy as np
import pandas as pd

import autogluon.core as ag
from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel

logger = logging.getLogger(__name__)


class MultiWindowBacktestingModel(AbstractTimeSeriesModel):
    """
    A meta-model that trains the base model multiple times using different train/validation splits.

    Follows the logic of autogluon.core.models.ensembles.BaggedEnsembleModel.

    Parameters
    ----------
    model_base : Union[AbstractTimeSeriesModel, Type[AbstractTimeSeriesModel]]
        The base model to repeatedly train. If a AbstractTimeSeriesModel class, then also provide model_base_kwargs
        which will be used to initialize the model via model_base(**model_base_kwargs).
    model_base_kwargs : Optional[Dict[str, any]], default = None
        kwargs used to initialize model_base if model_base is a class.
    num_val_windows : int, default = 1
        Number of windows to use for backtesting, starting from the end of the training data.
    """
    _most_recent_model_folder: str = "W0"

    def __init__(
        self,
        model_base: Union[AbstractTimeSeriesModel, Type[AbstractTimeSeriesModel]],
        model_base_kwargs: Optional[Dict[str, any]] = None,
        **kwargs,
    ):
        if inspect.isclass(model_base) and issubclass(model_base, AbstractTimeSeriesModel):
            if model_base_kwargs is None:
                model_base_kwargs = dict()
            self.model_base: AbstractTimeSeriesModel = model_base(**model_base_kwargs)
        elif model_base_kwargs is not None:
            raise AssertionError(
                f"model_base_kwargs must be None if model_base was passed as an object! "
                f"(model_base: {model_base}, model_base_kwargs: {model_base_kwargs})"
            )
        elif isinstance(model_base, AbstractTimeSeriesModel):
            self.model_base: AbstractTimeSeriesModel = model_base
        else:
            raise AssertionError(f"model_base must be an instance of AbstractTimeSeriesModel (got {type(model_base)})")
        self.model_base_type = type(self.model_base)
        self.info_per_val_window = []

        self.most_recent_model: AbstractTimeSeriesModel = None
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
        if num_val_windows == 0:
            raise ValueError("MultiWindowBacktestingModel can only be trained with num_val_windows > 0")

        trained_models = []
        global_fit_start_time = time.time()
        for window_index in range(num_val_windows):
            if window_index == 0:
                end_index = None
            else:
                end_index = -self.prediction_length * window_index
            train_fold, val_fold = train_data.train_test_split(
                prediction_length=self.prediction_length,
                end_index=end_index,
                suffix=f"_W{window_index}",
            )

            logger.debug(f"\tWindow {window_index}")
            model = self.get_child_model(window_index)
            model_fit_start_time = time.time()
            model.fit(
                train_data=train_fold,
                val_data=val_fold,
                time_limit=None if time_limit is None else time_limit - (model_fit_start_time - global_fit_start_time),
                **kwargs,
            )
            model.fit_time = time.time() - model_fit_start_time
            model.score_and_cache_oof(val_fold, store_val_score=True, store_predict_time=True)
            trained_models.append(model)

            logger.debug(f"\t\t{model.val_score:<7.4f}".ljust(15) + f"= Validation score ({model.eval_metric})")
            logger.debug(f"\t\t{model.fit_time:<7.3f} s".ljust(15) + "= Training runtime")
            logger.debug(f"\t\t{model.predict_time:<7.3f} s".ljust(15) + "= Training runtime")

            self.info_per_val_window.append(
                {
                    "window_index": window_index,
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

    def get_child_model(self, window_index: int) -> AbstractTimeSeriesModel:
        model = copy.deepcopy(self.model_base)
        model.rename(model.name + os.sep + f"W{window_index}")
        return model

    def predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        if self.most_recent_model is None:
            raise ValueError(f"{self.name} must be fit before predicting")
        return self.most_recent_model.predict(data, known_covariates=known_covariates, **kwargs)

    def score_and_cache_oof(
        self,
        val_data: TimeSeriesDataFrame,
        store_val_score: bool = False,
        store_predict_time: bool = False,
    ) -> None:
        # self.val_score, self.predict_time, self._oof_predictions already saved during _fit()
        assert self._oof_predictions is not None
        if store_val_score:
            assert self.val_score is not None
        if store_predict_time:
            assert self.predict_time is not None

    def get_user_params(self) -> dict:
        return self.model_base.get_user_params()

    def _get_search_space(self):
        return self.model_base._get_search_space()

    def _initialize(self, **kwargs) -> None:
        super()._initialize(**kwargs)
        self.model_base.initialize(**kwargs)

    def _get_hpo_train_fn_kwargs(self, **train_fn_kwargs) -> dict:
        train_fn_kwargs["is_bagged_model"] = True
        train_fn_kwargs["init_params"]["model_base"] = self.model_base.__class__
        train_fn_kwargs["init_params"]["model_base_kwargs"] = self.get_params()
        return train_fn_kwargs

    def save(self, path: str = None, verbose=True) -> str:
        most_recent_model = self.most_recent_model
        self.most_recent_model = None
        save_path = super().save(path, verbose)

        self.most_recent_model = most_recent_model
        if most_recent_model is not None:
            most_recent_model._oof_predictions = None
            most_recent_model.save()
        return save_path

    @classmethod
    def load(
        cls, path: str, reset_paths: bool = True, load_oof: bool = False, verbose: bool = True
    ) -> AbstractTimeSeriesModel:
        model = super().load(path=path, reset_paths=reset_paths, load_oof=load_oof, verbose=verbose)
        most_recent_model_path = model.path + os.sep + cls._most_recent_model_folder + os.sep
        model.most_recent_model = model.model_base_type.load(
            most_recent_model_path,
            reset_paths=reset_paths,
            verbose=verbose,
        )
        return model

    def convert_to_refit_full_template(self) -> AbstractTimeSeriesModel:
        return self.most_recent_model.convert_to_refit_full_template()

    def convert_to_refit_full_via_copy(self) -> AbstractTimeSeriesModel:
        most_recent_model_full = copy.deepcopy(self.most_recent_model)
        self.most_recent_model = None
        model_full = copy.deepcopy(self)
        model_full.rename(self.name + ag.constants.REFIT_FULL_SUFFIX)
        most_recent_model_full.rename(model_full.name + os.sep + self._most_recent_model_folder)
        model_full.most_recent_model = most_recent_model_full
        return model_full

    def _more_tags(self) -> dict:
        return self.most_recent_model._get_tags()

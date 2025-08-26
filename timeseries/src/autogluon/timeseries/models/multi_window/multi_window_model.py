import copy
import inspect
import logging
import math
import os
import time
from typing import Any, Optional, Type, Union

import numpy as np

import autogluon.core as ag
from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.models.local.abstract_local_model import AbstractLocalModel
from autogluon.timeseries.splitter import AbstractWindowSplitter, ExpandingWindowSplitter

logger = logging.getLogger(__name__)


class MultiWindowBacktestingModel(AbstractTimeSeriesModel):
    """
    A meta-model that trains the base model multiple times using different train/validation splits.

    Follows the logic of autogluon.core.models.ensembles.BaggedEnsembleModel.

    Parameters
    ----------
    model_base
        The base model to repeatedly train. If a AbstractTimeSeriesModel class, then also provide model_base_kwargs
        which will be used to initialize the model via model_base(**model_base_kwargs).
    model_base_kwargs
        kwargs used to initialize model_base if model_base is a class.
    """

    # TODO: Remove the MultiWindowBacktestingModel class, move the logic to TimeSeriesTrainer
    default_max_time_limit_ratio = 1.0

    def __init__(
        self,
        model_base: Union[AbstractTimeSeriesModel, Type[AbstractTimeSeriesModel]],
        model_base_kwargs: Optional[dict[str, Any]] = None,
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

        self.most_recent_model: Optional[AbstractTimeSeriesModel] = None
        self.most_recent_model_folder: Optional[str] = None
        super().__init__(**kwargs)

    @property
    def supports_static_features(self) -> bool:
        return self.model_base.supports_static_features

    @property
    def supports_known_covariates(self) -> bool:
        return self.model_base.supports_known_covariates

    @property
    def supports_past_covariates(self) -> bool:
        return self.model_base.supports_past_covariates

    def _get_model_base(self):
        return self.model_base

    def _get_hpo_backend(self) -> str:
        return self._get_model_base()._get_hpo_backend()

    def _is_gpu_available(self) -> bool:
        return self._get_model_base()._is_gpu_available()

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, Union[int, float]]:
        return self._get_model_base().get_minimum_resources(is_gpu_available)

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[float] = None,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
        verbosity: int = 2,
        val_splitter: Optional[AbstractWindowSplitter] = None,
        refit_every_n_windows: Optional[int] = 1,
        **kwargs,
    ):
        # TODO: use incremental training for GluonTS models?
        # TODO: implement parallel fitting similar to ParallelLocalFoldFittingStrategy in tabular?
        if val_data is not None:
            raise ValueError(f"val_data should not be passed to {self.name}.fit()")
        if val_splitter is None:
            val_splitter = ExpandingWindowSplitter(prediction_length=self.prediction_length)
        if not isinstance(val_splitter, AbstractWindowSplitter) or val_splitter.num_val_windows <= 0:
            raise ValueError(f"{self.name}.fit expects an AbstractWindowSplitter with num_val_windows > 0")
        if refit_every_n_windows is None:
            refit_every_n_windows = val_splitter.num_val_windows + 1  # only fit model for the first window

        oof_predictions_per_window = []
        global_fit_start_time = time.time()
        model: Optional[AbstractTimeSeriesModel] = None

        for window_index, (train_fold, val_fold) in enumerate(val_splitter.split(train_data)):
            logger.debug(f"\tWindow {window_index}")

            # refit_this_window is always True for the 0th window
            refit_this_window = window_index % refit_every_n_windows == 0
            assert not window_index == 0 or refit_this_window

            if time_limit is None:
                time_left_for_window = None
            else:
                time_left = time_limit - (time.time() - global_fit_start_time)
                if issubclass(self.model_base_type, AbstractLocalModel):
                    # For local models we call `fit` for every window to ensure that the time_limit is respected.
                    refit_this_window = True
                    # Local models cannot early stop, we allocate all remaining time and hope that they finish in time
                    time_left_for_window = time_left
                else:
                    num_refits_remaining = math.ceil(
                        (val_splitter.num_val_windows - window_index) / refit_every_n_windows
                    )
                    time_left_for_window = time_left / num_refits_remaining

            if refit_this_window:
                model = self.get_child_model(window_index)
                model_fit_start_time = time.time()
                model.fit(
                    train_data=train_fold,
                    val_data=val_fold,
                    time_limit=time_left_for_window,
                    **kwargs,
                )
                model.fit_time = time.time() - model_fit_start_time
                most_recent_refit_window = f"W{window_index}"

            if time_limit is None:
                time_left_for_prediction = None
            else:
                time_left_for_prediction = time_limit - (time.time() - global_fit_start_time)

            assert model is not None
            model.score_and_cache_oof(
                val_fold, store_val_score=True, store_predict_time=True, time_limit=time_left_for_prediction
            )

            oof_predictions_per_window.append(model.get_oof_predictions()[0])

            logger.debug(
                f"\t\t{model.val_score:<7.4f}".ljust(15) + f"= Validation score ({model.eval_metric.name_with_sign})"
            )
            logger.debug(f"\t\t{model.fit_time:<7.3f} s".ljust(15) + "= Training runtime")
            logger.debug(f"\t\t{model.predict_time:<7.3f} s".ljust(15) + "= Prediction runtime")

            self.info_per_val_window.append(
                {
                    "window_index": window_index,
                    "refit_this_window": refit_this_window,
                    "fit_time": model.fit_time if refit_this_window else float("nan"),
                    "val_score": model.val_score,
                    "predict_time": model.predict_time,
                }
            )

        # Only the model trained on most recent data is saved & used for prediction
        self.most_recent_model = model
        assert self.most_recent_model is not None

        self.most_recent_model_folder = most_recent_refit_window  # type: ignore
        self.predict_time = self.most_recent_model.predict_time
        self.fit_time = time.time() - global_fit_start_time - self.predict_time  # type: ignore
        self._oof_predictions = oof_predictions_per_window
        self.val_score = np.mean([info["val_score"] for info in self.info_per_val_window])  # type: ignore

    def get_info(self) -> dict:
        info = super().get_info()
        info["info_per_val_window"] = self.info_per_val_window
        return info

    def get_child_model(self, window_index: int) -> AbstractTimeSeriesModel:
        model = copy.deepcopy(self.model_base)
        model.rename(self.name + os.sep + f"W{window_index}")
        return model

    def _predict(
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
        **predict_kwargs,
    ) -> None:
        # self.val_score, self.predict_time, self._oof_predictions already saved during _fit()
        assert self._oof_predictions is not None
        if store_val_score:
            assert self.val_score is not None
        if store_predict_time:
            assert self.predict_time is not None

    def _get_search_space(self):
        return self.model_base._get_search_space()

    def _initialize_transforms_and_regressor(self) -> None:
        # Do not initialize the target_scaler and covariate_regressor in the multi window model!
        self.target_scaler = None
        self.covariate_scaler = None
        self.covariate_regressor = None

    def _get_hpo_train_fn_kwargs(self, **train_fn_kwargs) -> dict:
        train_fn_kwargs["is_bagged_model"] = True
        train_fn_kwargs["init_params"]["model_base"] = self.model_base.__class__
        train_fn_kwargs["init_params"]["model_base_kwargs"] = self.get_params()
        return train_fn_kwargs

    def save(self, path: Optional[str] = None, verbose: bool = True) -> str:
        most_recent_model = self.most_recent_model
        self.most_recent_model = None
        save_path = super().save(path, verbose)

        self.most_recent_model = most_recent_model
        if most_recent_model is not None:
            most_recent_model._oof_predictions = None
            most_recent_model.save()
        return save_path

    def persist(self) -> AbstractTimeSeriesModel:
        if self.most_recent_model is None:
            raise ValueError(f"{self.name} must be fit before persisting")
        return self.most_recent_model.persist()

    @classmethod
    def load(
        cls, path: str, reset_paths: bool = True, load_oof: bool = False, verbose: bool = True
    ) -> AbstractTimeSeriesModel:
        model = super().load(path=path, reset_paths=reset_paths, load_oof=load_oof, verbose=verbose)
        if model.most_recent_model_folder is not None:
            most_recent_model_path = os.path.join(model.path, model.most_recent_model_folder)
            model.most_recent_model = model.model_base_type.load(
                most_recent_model_path,
                reset_paths=reset_paths,
                verbose=verbose,
            )
        return model

    def convert_to_refit_full_template(self) -> AbstractTimeSeriesModel:
        # refit_model is an instance of base model type, not MultiWindowBacktestingModel
        assert self.most_recent_model is not None, "Most recent model is None. Model must be fit first."
        refit_model = self.most_recent_model.convert_to_refit_full_template()
        refit_model.rename(self.name + ag.constants.REFIT_FULL_SUFFIX)
        return refit_model

    def convert_to_refit_full_via_copy(self) -> AbstractTimeSeriesModel:
        # refit_model is an instance of base model type, not MultiWindowBacktestingModel
        assert self.most_recent_model is not None, "Most recent model is None. Model must be fit first."
        refit_model = self.most_recent_model.convert_to_refit_full_via_copy()
        refit_model.rename(self.name + ag.constants.REFIT_FULL_SUFFIX)
        return refit_model

    def _more_tags(self) -> dict:
        tags = self.model_base._get_tags()
        tags["can_use_val_data"] = False
        return tags

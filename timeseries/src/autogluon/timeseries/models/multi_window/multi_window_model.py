import copy
import inspect
import logging
import os
import time
from typing import Dict, Optional, Type, Union

import numpy as np

import autogluon.core as ag
from autogluon.common.utils.log_utils import set_logger_verbosity
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
    model_base : Union[AbstractTimeSeriesModel, Type[AbstractTimeSeriesModel]]
        The base model to repeatedly train. If a AbstractTimeSeriesModel class, then also provide model_base_kwargs
        which will be used to initialize the model via model_base(**model_base_kwargs).
    model_base_kwargs : Optional[Dict[str, any]], default = None
        kwargs used to initialize model_base if model_base is a class.
    """

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
        self.most_recent_model_folder: Optional[str] = None
        super().__init__(**kwargs)

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[int] = None,
        val_splitter: AbstractWindowSplitter = None,
        refit_every_n_windows: Optional[int] = 1,
        **kwargs,
    ):
        # TODO: use incremental training for GluonTS models?
        # TODO: implement parallel fitting similar to ParallelLocalFoldFittingStrategy in tabular?
        verbosity = kwargs.get("verbosity", 2)
        set_logger_verbosity(verbosity, logger=logger)

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

        for window_index, (train_fold, val_fold) in enumerate(val_splitter.split(train_data)):
            logger.debug(f"\tWindow {window_index}")
            # refit_this_window is always True for the 0th window
            refit_this_window = window_index % refit_every_n_windows == 0
            # For local models we call `fit` for every window to ensure that the time_limit is respected
            if refit_this_window or issubclass(self.model_base_type, AbstractLocalModel):
                model = self.get_child_model(window_index)
                model_fit_start_time = time.time()
                model.fit(
                    train_data=train_fold,
                    val_data=val_fold,
                    time_limit=None
                    if time_limit is None
                    else time_limit - (model_fit_start_time - global_fit_start_time),
                    **kwargs,
                )
                model.fit_time = time.time() - model_fit_start_time
            model.score_and_cache_oof(val_fold, store_val_score=True, store_predict_time=True)

            oof_predictions_per_window.append(model.get_oof_predictions()[0])

            logger.debug(
                f"\t\t{model.val_score:<7.4f}".ljust(15) + f"= Validation score ({model.eval_metric.name_with_sign})"
            )
            logger.debug(f"\t\t{model.fit_time:<7.3f} s".ljust(15) + "= Training runtime")
            logger.debug(f"\t\t{model.predict_time:<7.3f} s".ljust(15) + "= Training runtime")

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
        self.most_recent_model_folder = f"W{window_index}"
        self.predict_time = self.most_recent_model.predict_time
        self.fit_time = time.time() - global_fit_start_time - self.predict_time
        self._oof_predictions = oof_predictions_per_window
        self.val_score = np.mean([info["val_score"] for info in self.info_per_val_window])

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
        most_recent_model_path = os.path.join(model.path, model.most_recent_model_folder)
        model.most_recent_model = model.model_base_type.load(
            most_recent_model_path,
            reset_paths=reset_paths,
            verbose=verbose,
        )
        return model

    def convert_to_refit_full_template(self) -> AbstractTimeSeriesModel:
        # refit_model is an instance of base model type, not MultiWindowBacktestingModel
        refit_model = self.most_recent_model.convert_to_refit_full_template()
        refit_model.rename(self.name + ag.constants.REFIT_FULL_SUFFIX)
        return refit_model

    def convert_to_refit_full_via_copy(self) -> AbstractTimeSeriesModel:
        # refit_model is an instance of base model type, not MultiWindowBacktestingModel
        refit_model = self.most_recent_model.convert_to_refit_full_via_copy()
        refit_model.rename(self.name + ag.constants.REFIT_FULL_SUFFIX)
        return refit_model

    def _more_tags(self) -> dict:
        return self.most_recent_model._get_tags()

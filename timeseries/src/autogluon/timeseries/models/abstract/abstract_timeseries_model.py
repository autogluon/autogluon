import logging
import os
import re
import time
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from autogluon.common import space
from autogluon.common.loaders import load_pkl
from autogluon.common.savers import save_pkl
from autogluon.core.hpo.exceptions import EmptySearchSpace
from autogluon.core.hpo.executors import HpoExecutor, RayHpoExecutor
from autogluon.core.models import AbstractModel
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.metrics import TimeSeriesScorer, check_get_evaluation_metric
from autogluon.timeseries.regressor import CovariateRegressor
from autogluon.timeseries.transforms import (
    CovariateScaler,
    LocalTargetScaler,
    get_covariate_scaler_from_name,
    get_target_scaler_from_name,
)
from autogluon.timeseries.utils.features import CovariateMetadata
from autogluon.timeseries.utils.forecast import get_forecast_horizon_index_ts_dataframe
from autogluon.timeseries.utils.warning_filters import disable_stdout, warning_filter

from .model_trial import model_trial, skip_hpo

logger = logging.getLogger(__name__)


class AbstractTimeSeriesModel(AbstractModel):
    """Abstract class for all `Model` objects in autogluon.timeseries.

    Parameters
    ----------
    path : str, default = None
        Directory location to store all outputs.
        If None, a new unique time-stamped directory is chosen.
    freq: str
        Frequency string (cf. gluonts frequency strings) describing the frequency
        of the time series data. For example, "h" for hourly or "D" for daily data.
    prediction_length: int
        Length of the prediction horizon, i.e., the number of time steps the model
        is fit to forecast.
    name : str, default = None
        Name of the subdirectory inside path where model will be saved.
        The final model directory will be os.path.join(path, name)
        If None, defaults to the model's class name: self.__class__.__name__
    metadata: CovariateMetadata
        A mapping of different covariate types known to autogluon.timeseries to column names
        in the data set.
    eval_metric : Union[str, TimeSeriesScorer], default = "WQL"
        Metric by which predictions will be ultimately evaluated on future test data. This only impacts
        ``model.score()``, as eval_metric is not used during training. Available metrics can be found in
        ``autogluon.timeseries.metrics``.
    eval_metric_seasonal_period : int, optional
        Seasonal period used to compute some evaluation metrics such as mean absolute scaled error (MASE). Defaults to
        ``None``, in which case the seasonal period is computed based on the data frequency.
    hyperparameters : dict, default = None
        Hyperparameters that will be used by the model (can be search spaces instead of fixed values).
        If None, model defaults are used. This is identical to passing an empty dictionary.
    """

    _oof_filename = "oof.pkl"
    # TODO: For which models should we override this parameter?
    _covariate_regressor_fit_time_fraction: float = 0.5
    default_max_time_limit_ratio: float = 0.9

    # TODO: refactor "pruned" methods after AbstractModel is refactored
    predict_proba = None
    score_with_y_pred_proba = None
    disk_usage = None  # disk / memory size
    estimate_memory_usage = None
    reduce_memory_size = None
    compute_feature_importance = None  # feature processing and importance
    get_features = None
    _apply_conformalization = None
    _apply_temperature_scaling = None
    _predict_proba = None
    _convert_proba_to_unified_form = None
    _compute_permutation_importance = None
    _estimate_memory_usage = None
    _preprocess = None
    _preprocess_nonadaptive = None
    _preprocess_set_features = None

    supports_known_covariates: bool = False
    supports_past_covariates: bool = False
    supports_static_features: bool = False

    def __init__(
        self,
        freq: Optional[str] = None,
        prediction_length: int = 1,
        path: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[CovariateMetadata] = None,
        eval_metric: Union[str, TimeSeriesScorer, None] = None,
        eval_metric_seasonal_period: Optional[int] = None,
        hyperparameters: Dict[str, Union[int, float, str, space.Space]] = None,
        **kwargs,
    ):
        name = name or re.sub(r"Model$", "", self.__class__.__name__)
        super().__init__(
            path=path,
            name=name,
            problem_type=None,
            eval_metric=None,
            hyperparameters=hyperparameters,
        )
        self.eval_metric: TimeSeriesScorer = check_get_evaluation_metric(eval_metric)
        self.eval_metric_seasonal_period = eval_metric_seasonal_period
        self.stopping_metric = None
        self.problem_type = "timeseries"
        self.conformalize = False
        self.target: str = kwargs.get("target", "target")
        self.metadata = metadata or CovariateMetadata()

        self.freq: str = freq
        self.prediction_length: int = prediction_length
        self.quantile_levels = kwargs.get("quantile_levels", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        if not all(0 < q < 1 for q in self.quantile_levels):
            raise ValueError("Invalid quantile_levels specified. Quantiles must be between 0 and 1 (exclusive).")

        # We ensure that P50 forecast is always among the "raw" predictions generated by _predict.
        # We remove P50 from the final predictions if P50 wasn't present among the specified quantile_levels.
        if 0.5 not in self.quantile_levels:
            self.must_drop_median = True
            self.quantile_levels = sorted(set([0.5] + self.quantile_levels))
        else:
            self.must_drop_median = False

        self._oof_predictions: Optional[List[TimeSeriesDataFrame]] = None
        self.target_scaler: Optional[LocalTargetScaler] = None
        self.covariate_scaler: Optional[CovariateScaler] = None
        self.covariate_regressor: Optional[CovariateRegressor] = None

    def __repr__(self) -> str:
        return self.name

    def save(self, path: str = None, verbose=True) -> str:
        # Save self._oof_predictions as a separate file, not model attribute
        if self._oof_predictions is not None:
            save_pkl.save(
                path=os.path.join(self.path, "utils", self._oof_filename),
                object=self._oof_predictions,
                verbose=verbose,
            )
        oof_predictions = self._oof_predictions
        self._oof_predictions = None
        save_path = super().save(path=path, verbose=verbose)
        self._oof_predictions = oof_predictions
        return save_path

    @classmethod
    def load(
        cls, path: str, reset_paths: bool = True, load_oof: bool = False, verbose: bool = True
    ) -> "AbstractTimeSeriesModel":
        model = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        if load_oof and model._oof_predictions is None:
            model._oof_predictions = cls.load_oof_predictions(path=path, verbose=verbose)
        return model

    @classmethod
    def load_oof_predictions(cls, path: str, verbose: bool = True) -> List[TimeSeriesDataFrame]:
        """Load the cached OOF predictions from disk."""
        return load_pkl.load(path=os.path.join(path, "utils", cls._oof_filename), verbose=verbose)

    def get_oof_predictions(self):
        if self._oof_predictions is None:
            self._oof_predictions = self.load_oof_predictions(self.path)
        return self._oof_predictions

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params["max_time_limit_ratio"] = self.default_max_time_limit_ratio
        return default_auxiliary_params

    def _initialize(self, **kwargs) -> None:
        self._init_params_aux()
        self._init_params()
        self.target_scaler = self._create_target_scaler()
        self.covariate_scaler = self._create_covariate_scaler()
        self.covariate_regressor = self._create_covariate_regressor()

    def _compute_fit_metadata(self, val_data: TimeSeriesDataFrame = None, **kwargs):
        fit_metadata = dict(
            val_in_fit=val_data is not None,
        )
        return fit_metadata

    def _validate_fit_memory_usage(self, **kwargs):
        # memory usage handling not implemented for timeseries models
        pass

    def get_params(self) -> dict:
        params = super().get_params()
        params.update(
            dict(
                freq=self.freq,
                prediction_length=self.prediction_length,
                quantile_levels=self.quantile_levels,
                metadata=self.metadata,
                target=self.target,
            )
        )
        return params

    def get_info(self) -> dict:
        """
        Returns a dictionary of numerous fields describing the model.
        """
        # TODO: Include self.metadata
        info = {
            "name": self.name,
            "model_type": type(self).__name__,
            "eval_metric": self.eval_metric,
            "fit_time": self.fit_time,
            "predict_time": self.predict_time,
            "freq": self.freq,
            "prediction_length": self.prediction_length,
            "quantile_levels": self.quantile_levels,
            "val_score": self.val_score,
            "hyperparameters": self.params,
        }
        return info

    def fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[float] = None,
        **kwargs,
    ) -> "AbstractTimeSeriesModel":
        """Fit timeseries model.

        Models should not override the `fit` method, but instead override the `_fit` method which
        has the same arguments.

        Other Parameters
        ----------------
        train_data : TimeSeriesDataFrame
            The training data provided in the library's `autogluon.timeseries.dataset.TimeSeriesDataFrame`
            format.
        val_data : TimeSeriesDataFrame, optional
            The validation data set in the same format as training data.
        time_limit : float, default = None
            Time limit in seconds to adhere to when fitting model.
            Ideally, model should early stop during fit to avoid going over the time limit if specified.
        num_cpus : int, default = 'auto'
            How many CPUs to use during fit.
            This is counted in virtual cores, not in physical cores.
            If 'auto', model decides.
        num_gpus : int, default = 'auto'
            How many GPUs to use during fit.
            If 'auto', model decides.
        verbosity : int, default = 2
            Verbosity levels range from 0 to 4 and control how much information is printed.
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
        **kwargs :
            Any additional fit arguments a model supports.

        Returns
        -------
        model: AbstractTimeSeriesModel
            The fitted model object
        """
        start_time = time.monotonic()
        self.initialize(**kwargs)

        if self.target_scaler is not None:
            train_data = self.target_scaler.fit_transform(train_data)

        if self.covariate_scaler is not None:
            train_data = self.covariate_scaler.fit_transform(train_data)

        if self.covariate_regressor is not None:
            covariate_regressor_time_limit = (
                self._covariate_regressor_fit_time_fraction * time_limit if time_limit is not None else None
            )
            self.covariate_regressor.fit(
                train_data,
                time_limit=covariate_regressor_time_limit,
                verbosity=kwargs.get("verbosity", 2) - 1,
            )

        if self._get_tags()["can_use_train_data"]:
            if self.covariate_regressor is not None:
                train_data = self.covariate_regressor.transform(train_data)
            train_data, _ = self.preprocess(train_data, is_train=True)

        if self._get_tags()["can_use_val_data"] and val_data is not None:
            if self.target_scaler is not None:
                val_data = self.target_scaler.transform(val_data)
            if self.covariate_scaler is not None:
                val_data = self.covariate_scaler.transform(val_data)
            if self.covariate_regressor is not None:
                val_data = self.covariate_regressor.transform(val_data)
            val_data, _ = self.preprocess(val_data, is_train=False)

        if time_limit is not None:
            time_limit = time_limit - (time.monotonic() - start_time)
        return super().fit(train_data=train_data, val_data=val_data, time_limit=time_limit, **kwargs)

    @property
    def allowed_hyperparameters(self) -> List[str]:
        """List of hyperparameters allowed by the model."""
        return ["target_scaler", "covariate_regressor"]

    def _create_target_scaler(self) -> Optional[LocalTargetScaler]:
        """Create a LocalTargetScaler object based on the value of the `target_scaler` hyperparameter."""
        # TODO: Add support for custom target transforms (e.g., Box-Cox, log1p, ...)
        target_scaler_type = self._get_model_params().get("target_scaler")
        if target_scaler_type is not None:
            return get_target_scaler_from_name(target_scaler_type, target=self.target)
        else:
            return None

    def _create_covariate_scaler(self) -> Optional[CovariateScaler]:
        """Create a CovariateScaler object based on the value of the `covariate_scaler` hyperparameter."""
        covariate_scaler_type = self._get_model_params().get("covariate_scaler")
        if covariate_scaler_type is not None:
            return get_covariate_scaler_from_name(
                covariate_scaler_type,
                metadata=self.metadata,
                use_static_features=self.supports_static_features,
                use_known_covariates=self.supports_known_covariates,
                use_past_covariates=self.supports_past_covariates,
            )
        else:
            return None

    def _create_covariate_regressor(self) -> Optional[CovariateRegressor]:
        """Create a CovariateRegressor object based on the value of the `covariate_regressor` hyperparameter."""
        covariate_regressor = self._get_model_params().get("covariate_regressor")
        if covariate_regressor is not None:
            if len(self.metadata.known_covariates + self.metadata.static_features) == 0:
                logger.info(
                    "\tSkipping covariate_regressor since the dataset contains no covariates or static features."
                )
                return None
            else:
                if isinstance(covariate_regressor, str):
                    return CovariateRegressor(covariate_regressor, target=self.target, metadata=self.metadata)
                elif isinstance(covariate_regressor, dict):
                    return CovariateRegressor(**covariate_regressor, target=self.target, metadata=self.metadata)
                elif isinstance(covariate_regressor, CovariateRegressor):
                    logger.warning(
                        "\tUsing a custom covariate_regressor is experimental functionality that may break in the future!"
                    )
                    covariate_regressor.target = self.target
                    covariate_regressor.metadata = self.metadata
                    return covariate_regressor
                else:
                    raise ValueError(
                        f"Invalid value for covariate_regressor {covariate_regressor} of type {type(covariate_regressor)}"
                    )
        else:
            return None

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[int] = None,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
        verbosity: int = 2,
        **kwargs,
    ) -> None:
        """Private method for `fit`. See `fit` for documentation of arguments. Apart from
        the model training logic, `fit` additionally implements other logic such as keeping
        track of the time limit, etc.
        """
        # TODO: Make the models respect `num_cpus` and `num_gpus` parameters
        raise NotImplementedError

    def _check_fit_params(self):
        # gracefully handle hyperparameter specifications if they are provided to fit instead
        if any(isinstance(v, space.Space) for v in self.params.values()):
            raise ValueError(
                "Hyperparameter spaces provided to `fit`. Please provide concrete values "
                "as hyperparameters when initializing or use `hyperparameter_tune` instead."
            )

    def predict(
        self,
        data: Union[TimeSeriesDataFrame, Dict[str, TimeSeriesDataFrame]],
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        """Given a dataset, predict the next `self.prediction_length` time steps.
        This method produces predictions for the forecast horizon *after* the individual time series.

        For example, if the data set includes 24 hour time series, of hourly data, starting from
        00:00 on day 1, and forecast horizon is set to 5. The forecasts are five time steps 00:00-04:00
        on day 2.

        Parameters
        ----------
        data: Union[TimeSeriesDataFrame, Dict[str, TimeSeriesDataFrame]]
            The dataset where each time series is the "context" for predictions. For ensemble models that depend on
            the predictions of other models, this method may accept a dictionary of previous models' predictions.
        known_covariates : Optional[TimeSeriesDataFrame]
            A TimeSeriesDataFrame containing the values of the known covariates during the forecast horizon.

        Returns
        -------
        predictions: TimeSeriesDataFrame
            pandas data frames with a timestamp index, where each input item from the input
            data is given as a separate forecast item in the dictionary, keyed by the `item_id`s
            of input items.
        """
        if self.target_scaler is not None:
            data = self.target_scaler.fit_transform(data)
        if self.covariate_scaler is not None:
            data = self.covariate_scaler.fit_transform(data)
            known_covariates = self.covariate_scaler.transform_known_covariates(known_covariates)
        if self.covariate_regressor is not None:
            data = self.covariate_regressor.fit_transform(data)

        data, known_covariates = self.preprocess(data, known_covariates, is_train=False)

        # FIXME: Set self.covariate_regressor=None so to avoid copying it across processes during _predict
        # FIXME: The clean solution is to convert all methods executed in parallel to @classmethod
        covariate_regressor = self.covariate_regressor
        self.covariate_regressor = None
        predictions = self._predict(data=data, known_covariates=known_covariates, **kwargs)
        self.covariate_regressor = covariate_regressor

        # "0.5" might be missing from the quantiles if self is a wrapper (MultiWindowBacktestingModel or ensemble)
        if "0.5" in predictions.columns:
            if self.eval_metric.optimized_by_median:
                predictions["mean"] = predictions["0.5"]
            if self.must_drop_median:
                predictions = predictions.drop("0.5", axis=1)

        if self.covariate_regressor is not None:
            if known_covariates is None:
                forecast_index = get_forecast_horizon_index_ts_dataframe(
                    data, prediction_length=self.prediction_length, freq=self.freq
                )
                known_covariates = pd.DataFrame(index=forecast_index, dtype="float32")

            predictions = self.covariate_regressor.inverse_transform(
                predictions,
                known_covariates=known_covariates,
                static_features=data.static_features,
            )

        if self.target_scaler is not None:
            predictions = self.target_scaler.inverse_transform(predictions)
        return predictions

    def _predict(
        self,
        data: Union[TimeSeriesDataFrame, Dict[str, TimeSeriesDataFrame]],
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        """Private method for `predict`. See `predict` for documentation of arguments."""
        raise NotImplementedError

    def _score_with_predictions(
        self,
        data: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        metric: Optional[str] = None,
    ) -> float:
        """Compute the score measuring how well the predictions align with the data."""
        eval_metric = self.eval_metric if metric is None else check_get_evaluation_metric(metric)
        return eval_metric.score(
            data=data,
            predictions=predictions,
            prediction_length=self.prediction_length,
            target=self.target,
            seasonal_period=self.eval_metric_seasonal_period,
        )

    def score(self, data: TimeSeriesDataFrame, metric: Optional[str] = None) -> float:
        """Return the evaluation scores for given metric and dataset. The last
        `self.prediction_length` time steps of each time series in the input data set
        will be held out and used for computing the evaluation score. Time series
        models always return higher-is-better type scores.

        Parameters
        ----------
        data: TimeSeriesDataFrame
            Dataset used for scoring.
        metric: str
            String identifier of evaluation metric to use, from one of
            `autogluon.timeseries.utils.metric_utils.AVAILABLE_METRICS`.

        Other Parameters
        ----------------
        num_samples: int
            Number of samples to use for making evaluation predictions if the probabilistic
            forecasts are generated by forward sampling from the fitted model.

        Returns
        -------
        score: float
            The computed forecast evaluation score on the last `self.prediction_length`
            time steps of each time series.
        """
        past_data, known_covariates = data.get_model_inputs_for_scoring(
            prediction_length=self.prediction_length, known_covariates_names=self.metadata.known_covariates
        )
        predictions = self.predict(past_data, known_covariates=known_covariates)
        return self._score_with_predictions(data=data, predictions=predictions, metric=metric)

    def score_and_cache_oof(
        self,
        val_data: TimeSeriesDataFrame,
        store_val_score: bool = False,
        store_predict_time: bool = False,
        **predict_kwargs,
    ) -> None:
        """Compute val_score, predict_time and cache out-of-fold (OOF) predictions."""
        past_data, known_covariates = val_data.get_model_inputs_for_scoring(
            prediction_length=self.prediction_length, known_covariates_names=self.metadata.known_covariates
        )
        predict_start_time = time.time()
        oof_predictions = self.predict(past_data, known_covariates=known_covariates, **predict_kwargs)
        self._oof_predictions = [oof_predictions]
        if store_predict_time:
            self.predict_time = time.time() - predict_start_time
        if store_val_score:
            self.val_score = self._score_with_predictions(val_data, oof_predictions)

    def _get_hpo_train_fn_kwargs(self, **train_fn_kwargs) -> dict:
        """Update kwargs passed to model_trial depending on the model configuration.

        These kwargs need to be updated, for example, by MultiWindowBacktestingModel.
        """
        return train_fn_kwargs

    def _is_gpu_available(self) -> bool:
        return False

    def hyperparameter_tune(
        self, hyperparameter_tune_kwargs="auto", hpo_executor: HpoExecutor = None, time_limit: float = None, **kwargs
    ):
        if hpo_executor is None:
            hpo_executor = self._get_default_hpo_executor()
            default_num_trials = kwargs.pop("default_num_trials", None)
            hpo_executor.initialize(
                hyperparameter_tune_kwargs, default_num_trials=default_num_trials, time_limit=time_limit
            )

        kwargs = self.initialize(time_limit=time_limit, **kwargs)

        self._register_fit_metadata(**kwargs)
        self._validate_fit_memory_usage(**kwargs)

        kwargs = self._preprocess_fit_resources(
            parallel_hpo=hpo_executor.executor_type == "ray", silent=True, **kwargs
        )
        self.validate_fit_resources(**kwargs)

        # autogluon.core runs a complicated logic to determine the final number of gpus
        # used in trials, which results in unintended setting of num_gpus=0. We override this
        # logic here, and set to minimum num_gpus to 1 if it is set to 0 when GPUs are available
        kwargs["num_gpus"] = 0 if not self._is_gpu_available() else max(kwargs.get("num_gpus", 1), 1)

        # we use k_fold=1 to circumvent autogluon.core logic to manage resources during parallelization
        # of different folds
        hpo_executor.register_resources(self, k_fold=1, **kwargs)
        return self._hyperparameter_tune(hpo_executor=hpo_executor, **kwargs)

    def persist(self) -> "AbstractTimeSeriesModel":
        """Ask the model to persist its assets in memory, i.e., to predict with low latency. In practice
        this is used for pretrained models that have to lazy-load model parameters to device memory at
        prediction time.
        """
        return self

    def _hyperparameter_tune(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: TimeSeriesDataFrame,
        hpo_executor: HpoExecutor,
        **kwargs,
    ):
        time_start = time.time()
        logger.debug(f"\tStarting AbstractTimeSeriesModel hyperparameter tuning for {self.name}")
        search_space = self._get_search_space()

        try:
            hpo_executor.validate_search_space(search_space, self.name)
        except EmptySearchSpace:
            return skip_hpo(self, train_data, val_data, time_limit=hpo_executor.time_limit)

        self.set_contexts(os.path.abspath(self.path))
        directory = self.path
        dataset_train_filename = "dataset_train.pkl"
        train_path = os.path.join(self.path, dataset_train_filename)
        save_pkl.save(path=train_path, object=train_data)

        dataset_val_filename = "dataset_val.pkl"
        val_path = os.path.join(self.path, dataset_val_filename)
        save_pkl.save(path=val_path, object=val_data)

        fit_kwargs = dict(
            val_splitter=kwargs.get("val_splitter"),
            refit_every_n_windows=kwargs.get("refit_every_n_windows", 1),
        )
        train_fn_kwargs = self._get_hpo_train_fn_kwargs(
            model_cls=self.__class__,
            init_params=self.get_params(),
            time_start=time_start,
            time_limit=hpo_executor.time_limit,
            fit_kwargs=fit_kwargs,
            train_path=train_path,
            val_path=val_path,
            hpo_executor=hpo_executor,
        )

        model_estimate_memory_usage = None
        if self.estimate_memory_usage is not None:
            model_estimate_memory_usage = self.estimate_memory_usage(**kwargs)

        minimum_resources = self.get_minimum_resources(is_gpu_available=self._is_gpu_available())
        hpo_context = disable_stdout if isinstance(hpo_executor, RayHpoExecutor) else nullcontext
        with hpo_context(), warning_filter():  # prevent Ray from outputting its results to stdout with print
            hpo_executor.execute(
                model_trial=model_trial,
                train_fn_kwargs=train_fn_kwargs,
                directory=directory,
                minimum_cpu_per_trial=minimum_resources.get("num_cpus", 1),
                minimum_gpu_per_trial=minimum_resources.get("num_gpus", 0),
                model_estimate_memory_usage=model_estimate_memory_usage,
                adapter_type="timeseries",
            )

            hpo_models, analysis = hpo_executor.get_hpo_results(
                model_name=self.name,
                model_path_root=self.path_root,
                time_start=time_start,
            )

        return hpo_models, analysis

    def preprocess(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        is_train: bool = False,
        **kwargs,
    ) -> Tuple[TimeSeriesDataFrame, Optional[TimeSeriesDataFrame]]:
        """Method that implements model-specific preprocessing logic."""
        return data, known_covariates

    def get_memory_size(self, **kwargs) -> Optional[int]:
        return None

    def convert_to_refit_full_via_copy(self) -> "AbstractTimeSeriesModel":
        refit_model = super().convert_to_refit_full_via_copy()
        refit_model.val_score = None
        refit_model.predict_time = None
        return refit_model

    def get_user_params(self) -> dict:
        """Used to access user-specified parameters for the model before initialization."""
        if self._user_params is None:
            return {}
        else:
            return self._user_params.copy()

    def _more_tags(self) -> dict:
        """Encode model properties using tags, similar to sklearn & autogluon.tabular.

        For more details, see `autogluon.core.models.abstract.AbstractModel._get_tags()` and https://scikit-learn.org/stable/_sources/developers/develop.rst.txt.

        List of currently supported tags:
        - allow_nan: Can the model handle data with missing values represented by np.nan?
        - can_refit_full: Does it make sense to retrain the model without validation data?
            See `autogluon.core.models.abstract._tags._DEFAULT_TAGS` for more details.
        - can_use_train_data: Can the model use train_data if it's provided to model.fit()?
        - can_use_val_data: Can the model use val_data if it's provided to model.fit()?
        """
        return {
            "allow_nan": False,
            "can_refit_full": False,
            "can_use_train_data": True,
            "can_use_val_data": False,
        }

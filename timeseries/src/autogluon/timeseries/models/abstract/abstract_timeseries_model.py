import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Union

from autogluon.common import space
from autogluon.common.loaders import load_pkl
from autogluon.common.savers import save_pkl
from autogluon.core.hpo.exceptions import EmptySearchSpace
from autogluon.core.hpo.executors import HpoExecutor
from autogluon.core.models import AbstractModel
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.evaluator import TimeSeriesEvaluator
from autogluon.timeseries.utils.features import CovariateMetadata

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
        of the time series data. For example, "H" for hourly or "D" for daily data.
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
    eval_metric : str, default
        Metric by which predictions will be ultimately evaluated on test data.
        This only impacts `model.score()`, as eval_metric is not used during training.
        Available metrics can be found in `autogluon.timeseries.utils.metric_utils.AVAILABLE_METRICS`, and
        detailed documentation can be found in `gluonts.evaluation.Evaluator`. By default, `WQL`
        will be used.
    eval_metric_seasonal_period : int, optional
        Seasonal period used to compute the mean absolute scaled error (MASE) evaluation metric. This parameter is only
        used if ``eval_metric="MASE"`. See https://en.wikipedia.org/wiki/Mean_absolute_scaled_error for more details.
        Defaults to ``None``, in which case the seasonal period is computed based on the data frequency.
    hyperparameters : dict, default = None
        Hyperparameters that will be used by the model (can be search spaces instead of fixed values).
        If None, model defaults are used. This is identical to passing an empty dictionary.
    """

    _oof_filename = "oof.pkl"

    # TODO: refactor "pruned" methods after AbstractModel is refactored
    predict_proba = None
    score_with_y_pred_proba = None
    get_disk_size = None  # disk / memory size
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

    def __init__(
        self,
        freq: Optional[str] = None,
        prediction_length: int = 1,
        path: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[CovariateMetadata] = None,
        eval_metric: Optional[str] = None,
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
        self.eval_metric: str = TimeSeriesEvaluator.check_get_evaluation_metric(eval_metric)
        self.eval_metric_seasonal_period = eval_metric_seasonal_period
        self.stopping_metric = None
        self.problem_type = "timeseries"
        self.conformalize = False
        self.target: str = kwargs.get("target", "target")
        self.metadata = metadata or CovariateMetadata()

        self.freq: str = freq
        self.prediction_length: int = prediction_length
        self.quantile_levels = kwargs.get("quantile_levels", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        self._oof_predictions: Optional[List[TimeSeriesDataFrame]] = None

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

    def _initialize(self, **kwargs) -> None:
        self._init_params_aux()
        self._init_params()

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

    def fit(self, **kwargs) -> "AbstractTimeSeriesModel":
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
            verbosity 4: logs every training iteration, and logs the most detailed information.
            verbosity 3: logs training iterations periodically, and logs more detailed information.
            verbosity 2: logs only important information.
            verbosity 1: logs only warnings and exceptions.
            verbosity 0: logs only exceptions.
        **kwargs :
            Any additional fit arguments a model supports.

        Returns
        -------
        model: AbstractTimeSeriesModel
            The fitted model object
        """
        return super().fit(**kwargs)

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

    def _check_predict_inputs(
        self,
        data: TimeSeriesDataFrame,
        quantile_levels: Optional[List[float]] = None,
        **kwargs,  # noqa: F841
    ):
        logger.debug(f"Predicting with time series model {self.name}")
        logger.debug(
            f"\tProvided data for prediction with {len(data)} rows, {data.num_items} items. "
            f"Average time series length is {len(data) / data.num_items:.1f}."
        )

        quantiles = quantile_levels or self.quantile_levels
        if not all(0 < q < 1 for q in quantiles):
            raise ValueError("Invalid quantile value specified. Quantiles must be between 0 and 1 (exclusive).")

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

        Other Parameters
        ----------------
        quantile_levels
            Quantiles of probabilistic forecasts, if probabilistic forecasts are implemented by the
            corresponding subclass. If None, `self.quantile_levels` will be used instead,
            if provided during initialization.

        Returns
        -------
        predictions: TimeSeriesDataFrame
            pandas data frames with a timestamp index, where each input item from the input
            data is given as a separate forecast item in the dictionary, keyed by the `item_id`s
            of input items.
        """
        raise NotImplementedError

    def _score_with_predictions(
        self,
        data: TimeSeriesDataFrame,
        predictions: TimeSeriesDataFrame,
        metric: Optional[str] = None,
    ) -> float:
        """Compute the score measuring how well the predictions align with the data."""
        eval_metric = self.eval_metric if metric is None else metric
        evaluator = TimeSeriesEvaluator(
            eval_metric=eval_metric,
            eval_metric_seasonal_period=self.eval_metric_seasonal_period,
            prediction_length=self.prediction_length,
            target_column=self.target,
        )
        return evaluator(data, predictions) * evaluator.coefficient

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
            prediction_length=self.prediction_length, known_covariates_names=self.metadata.known_covariates_real
        )
        predictions = self.predict(past_data, known_covariates=known_covariates)
        return self._score_with_predictions(data=data, predictions=predictions, metric=metric)

    def score_and_cache_oof(
        self,
        val_data: TimeSeriesDataFrame,
        store_val_score: bool = False,
        store_predict_time: bool = False,
    ) -> None:
        """Compute val_score, predict_time and cache out-of-fold (OOF) predictions."""
        past_data, known_covariates = val_data.get_model_inputs_for_scoring(
            prediction_length=self.prediction_length, known_covariates_names=self.metadata.known_covariates_real
        )
        predict_start_time = time.time()
        oof_predictions = self.predict(past_data, known_covariates=known_covariates)
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

    def _hyperparameter_tune(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: TimeSeriesDataFrame,
        hpo_executor: HpoExecutor,
        **kwargs,
    ):
        # verbosity = kwargs.get('verbosity', 2)
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
        hpo_executor.execute(
            model_trial=model_trial,
            train_fn_kwargs=train_fn_kwargs,
            directory=directory,
            minimum_cpu_per_trial=self.get_minimum_resources().get("num_cpus", 1),
            minimum_gpu_per_trial=self.get_minimum_resources().get("num_gpus", 0),
            model_estimate_memory_usage=model_estimate_memory_usage,
            adapter_type="timeseries",
        )

        return hpo_executor.get_hpo_results(
            model_name=self.name,
            model_path_root=self.path_root,
            time_start=time_start,
        )

    def preprocess(self, data: Any, **kwargs) -> Any:
        return data

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

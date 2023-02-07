import copy
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import autogluon.core as ag
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
        The final model directory will be path+name+os.path.sep()
        If None, defaults to the model's class name: self.__class__.__name__
    metadata: CovariateMetadata
        A mapping of different covariate types known to autogluon.timeseries to column names
        in the data set.
    eval_metric : str, default
        Metric by which predictions will be ultimately evaluated on test data.
        This only impacts `model.score()`, as eval_metric is not used during training.
        Available metrics can be found in `autogluon.timeseries.utils.metric_utils.AVAILABLE_METRICS`, and
        detailed documentation can be found in `gluonts.evaluation.Evaluator`. By default, `mean_wQuantileLoss`
        will be used.
    hyperparameters : dict, default = None
        Hyperparameters that will be used by the model (can be search spaces instead of fixed values).
        If None, model defaults are used. This is identical to passing an empty dictionary.
    """

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
        hyperparameters: Dict[str, Union[int, float, str, ag.Space]] = None,
        **kwargs,
    ):
        super().__init__(
            path=path,
            name=name,
            problem_type=None,
            eval_metric=None,
            hyperparameters=hyperparameters,
        )
        self.eval_metric: str = TimeSeriesEvaluator.check_get_evaluation_metric(eval_metric)
        self.stopping_metric = None
        self.problem_type = "timeseries"
        self.conformalize = False
        self.target: str = kwargs.get("target", "target")
        self.metadata = metadata or CovariateMetadata()

        self.freq: str = freq
        self.prediction_length: int = prediction_length
        self.quantile_levels = kwargs.get(
            "quantile_levels",
            kwargs.get("quantiles", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        )

    def __repr__(self) -> str:
        return self.name

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
        val_data : TimeSeriesDataFrame
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
        train_data,
        val_data=None,
        time_limit=None,
        num_cpus=None,
        num_gpus=None,
        verbosity=2,
        **kwargs,
    ) -> None:
        """Private method for `fit`. See `fit` for documentation of arguments. Apart from
        the model training logic, `fit` additionally implements other logic such as keeping
        track of the time limit, etc.
        """
        raise NotImplementedError

    def _check_fit_params(self):
        # gracefully handle hyperparameter specifications if they are provided to fit instead
        if any(isinstance(v, ag.Space) for v in self.params.values()):
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

    def predict_for_scoring(self, data: TimeSeriesDataFrame, **kwargs):
        """Given a dataset, truncate the last `self.prediction_length` time steps and forecast these
        steps with previous history. This method produces predictions for the *last* `self.prediction_length`
        steps of the *given* time series, in order to be used for validation or scoring.

        Parameters
        ----------
        data: TimeSeriesDataFrame
            The dataset where each time series is the "context" for predictions.

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
        past_data = data.slice_by_timestep(None, -self.prediction_length)
        if len(self.metadata.known_covariates_real) > 0:
            future_data = data.slice_by_timestep(-self.prediction_length, None)
            known_covariates = future_data[self.metadata.known_covariates_real]
        else:
            known_covariates = None
        return self.predict(past_data, known_covariates=known_covariates, **kwargs)

    def score(self, data: TimeSeriesDataFrame, metric: str = None, **kwargs) -> float:
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
        metric = self.eval_metric if metric is None else metric
        evaluator = TimeSeriesEvaluator(
            eval_metric=metric,
            prediction_length=self.prediction_length,
            target_column=self.target,
        )
        predictions = self.predict_for_scoring(data)
        metric_value = evaluator(data, predictions)

        return metric_value * evaluator.coefficient

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

        self.set_contexts(os.path.abspath(self.path) + os.path.sep)
        directory = self.path
        dataset_train_filename = "dataset_train.pkl"
        train_path = os.path.join(self.path, dataset_train_filename)
        save_pkl.save(path=train_path, object=train_data)

        dataset_val_filename = "dataset_val.pkl"
        val_path = os.path.join(self.path, dataset_val_filename)
        save_pkl.save(path=val_path, object=val_data)

        fit_kwargs = dict()
        train_fn_kwargs = dict(
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

    def convert_to_refit_full_template(self):
        params = copy.deepcopy(self.get_params())

        # TODO: Time series models currently do not support incremental training
        params["hyperparameters"].update(self.params_trained)
        params["name"] = params["name"] + ag.constants.REFIT_FULL_SUFFIX

        template = self.__class__(**params)

        return template

    def get_user_params(self) -> dict:
        """Used to access user-specified parameters for the model before initialization."""
        if self._user_params is None:
            return {}
        else:
            return self._user_params.copy()


class AbstractTimeSeriesModelFactory:
    """Factory class interface for callable objects that produce timeseries models"""

    def __call__(self, *args, **kwargs) -> AbstractTimeSeriesModel:
        raise NotImplementedError

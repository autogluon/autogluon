import copy
import logging
import time
from typing import Any, Dict, Union, Tuple, Optional

import pandas as pd
from gluonts.dataset.common import (
    Dataset,
)  # TODO: this interface should not depend on GluonTS datasets

import autogluon.core as ag
from autogluon.core.models import AbstractModel
from autogluon.common.savers import save_pkl

from ...utils.metadata import get_prototype_metadata_dict
from ...utils.metric_utils import check_get_evaluation_metric
from .model_trial import skip_hpo, model_trial

logger = logging.getLogger(__name__)


class AbstractForecastingModel(AbstractModel):
    """Abstract class for all `Model` objects in autogluon.forecasting.

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
    metadata: MetadataDict
        A dictionary mapping different feature types known to autogluon.forecasting to column names
        in the data set.
    eval_metric : str, default
        Metric by which predictions will be ultimately evaluated on test data.
        This only impacts `model.score()`, as eval_metric is not used during training.
        Available metrics can be found in `autogluon.forecasting.utils.metric_utils.AVAILABLE_METRICS`, and
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

    # TODO: handle static features in the models and Dataset API
    def __init__(
        self,
        freq: Optional[str] = None,
        prediction_length: int = 1,
        path: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
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
        self.eval_metric: str = check_get_evaluation_metric(eval_metric)
        self.stopping_metric = None
        self.problem_type = "forecasting"
        self.conformalize = False
        self.metadata = metadata or get_prototype_metadata_dict()

        self.freq: str = freq
        self.prediction_length: int = prediction_length
        self.quantile_levels = kwargs.get(
            "quantile_levels", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )

    def _initialize(self, **kwargs) -> None:
        self._init_params_aux()
        self._init_params()

    def _compute_fit_metadata(self, val_data: Dataset = None, **kwargs):
        fit_metadata = dict(
            val_in_fit=val_data is not None,
        )
        return fit_metadata

    def _validate_fit_memory_usage(self, **kwargs):
        # memory usage handling not implemented for forecasting models
        pass

    def get_params(self) -> dict:
        params = super().get_params()
        params.update(
            dict(
                freq=self.freq,
                prediction_length=self.prediction_length,
                quantile_levels=self.quantile_levels,
                metadata=self.metadata,
            )
        )
        return params

    def get_info(self) -> dict:
        info_dict = super().get_info()
        info_dict.update(
            {
                "freq": self.freq,
                "prediction_length": self.prediction_length,
                "quantile_levels": self.quantile_levels,
                "metadata": self.metadata,
            }
        )
        return info_dict

    def fit(self, **kwargs) -> "AbstractForecastingModel":
        """Fit forecasting model.

        Models should not override the `fit` method, but instead override the `_fit` method which
        has the same arguments.

        Other Parameters
        ----------------
        train_data : gluonts.dataset.common.Dataset
            The training data. Forecasting models expect data to be in GluonTS format, i.e., an
            iterator of dictionaries with `start` and `target` keys. `start` is a timestamp object
            for the first data point in a time series while `target` is the time series which the
            model will be fit to predict. The data set can optionally have other features which may
            be used by the model. See individual model documentation and GluonTS dataset documentation
            for details.
        val_data : gluonts.dataset.common.Dataset
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
        model: AbstractForecastingModel
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

    def predict(self, data: Dataset, **kwargs) -> Dict[Any, pd.DataFrame]:
        """Given a dataset, predict the next `self.prediction_length` time steps. The data
        set is a GluonTS data set, an iterator over time series represented as python dictionaries.
        This method produces predictions for the forecast horizon *after* the individual time series.

        For example, if the data set includes 24 hour time series, of hourly data, starting from
        00:00 on day 1, and forecast horizon is set to 5. The forecasts are five time steps 00:00-04:00
        on day 2.

        # TODO: The function currently returns a pandas.DataFrame instead of a GluonTS dataset, however
        # TODO: this API will be aligned with the rest of the library.

        Parameters
        ----------
        data: gluonts.dataset.common.Dataset
            The dataset where each time series is the "context" for predictions.

        Other Parameters
        ----------------
        quantile_levels
            Quantiles of probabilistic forecasts, if probabilistic forecasts are implemented by the
            corresponding subclass. If None, `self.quantile_levels` will be used instead,
            if provided during initialization.

        Returns
        -------
        predictions: Dict[any, pandas.DataFrame]
            pandas data frames with an timestamp index, where each input item from the input
            data is given as a separate forecast item in the dictionary, keyed by the `item_id`s
            of input items.
        """
        raise NotImplementedError

    def score(self, data: Dataset, metric: str = None, **kwargs) -> float:
        """Return the evaluation scores for given metric and dataset. The last
        `self.prediction_length` time steps of each time series in the input data set
        will be held out and used for computing the evaluation score. Forecasting
        models always return higher-is-better type scores.

        Parameters
        ----------
        data: gluonts.dataset.common.Dataset
            Dataset used for scoring.
        metric: str
            String identifier of evaluation metric to use, from one of
            `autogluon.forecasting.utils.metric_utils.AVAILABLE_METRICS`.

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
        raise NotImplementedError

    def _hyperparameter_tune(
        self,
        train_data: Dataset,
        val_data: Dataset,
        scheduler_options: Tuple[Any, Dict],
        **kwargs,
    ):
        # verbosity = kwargs.get('verbosity', 2)
        time_start = time.time()
        logger.log(
            15,
            "Starting generic AbstractForecastingModel hyperparameter tuning for %s model..."
            % self.name,
        )
        search_space = self._get_search_space()

        scheduler_cls, scheduler_params = scheduler_options
        if scheduler_cls is None or scheduler_params is None:
            raise ValueError(
                "scheduler_cls and scheduler_params cannot be None for hyperparameter tuning"
            )
        time_limit = scheduler_params.get("time_out", None)
        if time_limit is None:
            scheduler_params["num_trials"] = scheduler_params.get("num_trials", 9999)
        else:
            scheduler_params.pop("num_trials", None)

        if not any(
            isinstance(search_space[hyperparameter], ag.Space)
            for hyperparameter in search_space
        ):
            logger.warning(
                f"\tNo hyperparameter search space specified for {self.name}. Skipping HPO. "
                f"Will train one model based on the provided hyperparameters."
            )
            return skip_hpo(self, train_data, val_data, time_limit=time_limit)
        else:
            logger.log(15, f"\tHyperparameter search space for {self.name}: ")
            for hyperparameter in search_space:
                if isinstance(search_space[hyperparameter], ag.Space):
                    logger.log(
                        15, f"{hyperparameter}:   {search_space[hyperparameter]}"
                    )

        dataset_train_filename = "dataset_train.pkl"
        train_path = self.path + dataset_train_filename
        save_pkl.save(path=train_path, object=train_data)

        dataset_val_filename = "dataset_val.pkl"
        val_path = self.path + dataset_val_filename
        save_pkl.save(path=val_path, object=val_data)

        train_fn_kwargs = dict(
            model_cls=self.__class__,
            init_params=self.get_params(),
            time_start=time_start,
            time_limit=time_limit,
            fit_kwargs=scheduler_params["resource"].copy(),
            train_path=train_path,
            val_path=val_path,
        )

        scheduler = scheduler_cls(
            model_trial,
            search_space=search_space,
            train_fn_kwargs=train_fn_kwargs,
            **scheduler_params,
        )

        scheduler.run()
        scheduler.join_jobs()

        return self._get_hpo_results(scheduler, scheduler_params, time_start)

    def preprocess(self, data: Any, **kwargs) -> Any:
        return data

    def get_memory_size(self, **kwargs) -> Optional[int]:
        return None

    def convert_to_refit_full_template(self):
        params = copy.deepcopy(self.get_params())

        # TODO: Forecasting models currently do not support incremental training
        params["hyperparameters"].update(self.params_trained)
        params["name"] = params["name"] + ag.constants.REFIT_FULL_SUFFIX

        template = self.__class__(**params)

        return template


class AbstractForecastingModelFactory:
    """Factory class interface for callable objects that produce forecasting models"""

    def __call__(self, *args, **kwargs) -> AbstractForecastingModel:
        raise NotImplementedError

import logging
import time
from typing import Any, Dict

import autogluon.core as ag
from autogluon.core.models import AbstractModel
from autogluon.common.savers import save_pkl

from ...utils.metric_utils import check_get_evaluation_metric
from .model_trial import skip_hpo, model_trial

logger = logging.getLogger(__name__)


# TODO: TYPING
# TODO: Docstrings
# TODO: Override the fit method to include train_data. (Is the API correct)?
class AbstractForecastingModel(AbstractModel):
    # following methods will not be available in forecasting models
    # TODO: check usage to see if higher level modules are dependent on these
    predict_proba = None
    score_with_y_pred_proba = None
    convert_to_refit_full_template = None
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
        path: str,
        freq: str,
        prediction_length: int,
        name: str,
        eval_metric: str = None,
        hyperparameters=None,
        **kwargs,
    ):
        """
        Create a new model
        Args:
            path(str): directory where to store all the model
            freq(str): frequency
            name(str): name of subdirectory inside path where model will be saved.
            eval_metric(str): objective function the model intends to optimize, will use mean_wQuantileLoss by default
            hyperparameters: various hyperparameters that will be used by model (can be search spaces instead of fixed
                values).

        """
        super().__init__(
            path=path,
            name=name,
            problem_type=None,
            eval_metric=None,
            hyperparameters=hyperparameters,
        )
        self.params = {}

        self.eval_metric = check_get_evaluation_metric(eval_metric)
        self.stopping_metric = None
        self.problem_type = "forecasting"
        self.conformalize = False

        self.quantiles = kwargs.get(
            "quantiles", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
        self.freq = freq
        self.prediction_length = prediction_length

    def _initialize(self, X=None, y=None, **kwargs):
        self._init_params_aux()
        self._init_params()

    def _init_misc(self, **kwargs):
        pass  # noop

    def _compute_fit_metadata(self, val_data=None, **kwargs):
        fit_metadata = dict(
            val_in_fit=val_data is not None,
        )
        return fit_metadata

    def _validate_fit_memory_usage(self, **kwargs):
        # memory usage handling not implemented for forecasting models
        pass  # noop

    def fit(self, **kwargs):
        """TODO: UPDATE DOCSTRING WITH NEW INTERFACE"""
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
    ):
        """TODO: UPDATE DOCSTRING WITH NEW INTERFACE"""
        raise NotImplementedError

    def predict(self, data, quantiles=None, **kwargs) -> Dict:
        raise NotImplementedError

    def score(self, data, metric=None, num_samples=100) -> float:
        raise NotImplementedError

    def _hyperparameter_tune(self, train_data, val_data, scheduler_options, **kwargs):
        """
        Hyperparameter tune the model.

        This usually does not need to be overwritten by models.
        """
        # verbosity = kwargs.get('verbosity', 2)
        time_start = time.time()
        logger.log(
            15,
            "Starting generic AbstractForecastingModel hyperparameter tuning for %s model..."
            % self.name,
        )
        search_space = self._get_search_space()

        scheduler_cls, scheduler_params = scheduler_options  # Unpack tuple
        if scheduler_cls is None or scheduler_params is None:
            raise ValueError(
                "scheduler_cls and scheduler_params cannot be None for hyperparameter tuning"
            )

        time_limit = scheduler_params.get("time_out", None)

        if not any(
            isinstance(search_space[hyperparameter], ag.Space)
            for hyperparameter in search_space
        ):
            logger.warning(
                f"\tNo hyperparameter search space specified for {self.name}. Skipping HPO. "
                f"Will train one model based on the provided hyperparameters."
            )
            return skip_hpo(self, train_data, val_data, time_limit=time_limit, **kwargs)
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
            search_space=self._get_search_space(),
            train_fn_kwargs=train_fn_kwargs,
            **scheduler_params,
        )

        scheduler.run()
        scheduler.join_jobs()

        return self._get_hpo_results(scheduler, scheduler_params, time_start)

    # OTHER DUMMY METHODS
    def preprocess(self, X: Any, **kwargs):
        return X

    def reset_metrics(self):
        super().reset_metrics()
        # TODO: reset the epoch counter?

    def get_memory_size(self, **kwargs):
        return None

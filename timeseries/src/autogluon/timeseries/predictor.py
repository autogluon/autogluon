import logging
import pprint
import time
from typing import Optional, Type, Any, Union, Dict, Tuple, List

import pandas as pd

from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.common.utils.utils import setup_outputdir
from autogluon.core.scheduler.scheduler_factory import scheduler_factory
from autogluon.core.utils.decorators import apply_presets
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl

from .configs import TIMESERIES_PRESETS_CONFIGS
from .dataset import TimeSeriesDataFrame
from .learner import AbstractLearner, TimeSeriesLearner
from .trainer import AbstractTimeSeriesTrainer

logger = logging.getLogger(__name__)


class TimeSeriesPredictor:
    """autogluon.timeseries's TimeSeriesPredictor predicts future values of multiple related time-series by fitting
    global timeseries models.

    autogluon.timeseries provides probabilistic (distributional) forecasts for univariate time series, where the
    timeseries model is essentially a mapping from the past of the time series to its future of length (i.e., forecast
    horizon) defined by the user. The model learned "globally" from a collection of time series i.e., it is a
    set of time series model parameters that are shared across all time series to be predicted, in contrast to
    classical "local" approaches such as ARIMA.

    Parameters
    ----------
    target: str, default = "target"
        Name of column that contains the target values to forecast (ie. the numeric observations of the
        time series). This column must contain numeric values, and missing target values
        should be in a pandas compatible format:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html
    eval_metric: str, default = None
        Metric by which predictions will be ultimately evaluated on future test data. AutoGluon tunes hyperparameters
        in order to improve this metric on validation data, and ranks models (on validation data) according to this
        metric. Available options include: "MASE", "MAPE", "sMAPE", "mean_wQuantileLoss".

        If `eval_metric = None`, it is set by default as "mean_wQuantileLoss".
        For more information about these options, see `autogluon.timeseries.utils.metric_utils` and GluonTS docs at
        https://ts.gluon.ai/api/gluonts/gluonts.evaluation.metrics.html
    path: str, default = None
        Path to directory where models and intermediate outputs should be saved. If unspecified, a timestamped folder
        "AutogluonModels/ag-[TIMESTAMP]" will be created in the working directory to store all models.
    verbosity : int, default = 2
        Verbosity levels range from 0 to 4 and control how much information is printed to stdout. Higher levels
        correspond to more detailed print statements, and verbosity=0 suppresses output including warnings.
        If using `logging`, you can alternatively control amount of information printed via `logger.setLevel(L)`,
        where `L` ranges from 0 to 50 (Note: higher values of `L` correspond to fewer print statements,
        opposite of verbosity levels).
    prediction_length: int, default = 1
        The forecast horizon, i.e., How many time points into the future forecasters should be trained to predict.
        For example, if our time series contain daily observations, setting `prediction_length=3` will train
        models that predict up to 3 days in the future from the most recent observation.
    quantile_levels: List[float], default = None
        List of increasing decimals that specifies which quantiles should be estimated
        when making distributional forecasts. Can alternatively be provided with the keyword
        argument `quantiles`. If None, defaults to [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].

    Other Parameters
    ----------------
    learner_type : AbstractLearner, default = TimeSeriesLearner
        A class which inherits from `AbstractLearner`. The learner specifies the inner logic of the TimeSeriesPredictor
    label: str
        Alias for `target`.
        for training models and preprocessing data.
    learner_kwargs : dict, default = None
        Keyword arguments to send to the learner (for advanced users only). Options include `trainer_type`, a
        class inheriting from `AbstractTrainer` which controls training of multiple models. If `path` and `eval_metric`
        are re-specified within `learner_kwargs`, these are ignored.
    quantiles: List[float]
        Alias for `quantile_levels`.

    Attributes
    ----------
    target_column: str
        Name of column in training/validation data that contains the target time-series value to be predicted. If
        not specified explicitly during `fit()`, this will default to "target".
    """

    predictor_file_name = "predictor.pkl"

    def __init__(
        self,
        target: Optional[str] = None,
        eval_metric: Optional[str] = None,
        path: Optional[str] = None,
        verbosity: int = 2,
        prediction_length: int = 1,
        quantile_levels: Optional[List[float]] = None,
        **kwargs,
    ):
        self.verbosity = verbosity
        set_logger_verbosity(self.verbosity, logger=logger)
        self.path = setup_outputdir(path)

        if target is not None and kwargs.get("label") is not None:
            raise ValueError(
                "Both `label` and `target` are specified. Please specify at most one of these. "
                "arguments."
            )
        self.target = target or kwargs.get("label", "target")

        self.prediction_length = prediction_length
        self.eval_metric = eval_metric
        self.quantile_levels = quantile_levels or kwargs.get(
            "quantiles", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )

        learner_type = kwargs.pop("learner_type", TimeSeriesLearner)
        learner_kwargs = kwargs.pop("learner_kwargs", dict())
        learner_kwargs = learner_kwargs.copy()
        learner_kwargs.update(
            dict(
                path_context=self.path,
                eval_metric=eval_metric,
                target=self.target,
                prediction_length=self.prediction_length,
                quantile_levels=self.quantile_levels,
            )
        )
        self._learner: AbstractLearner = learner_type(**learner_kwargs)
        self._learner_type = type(self._learner)

    @property
    def _trainer(self) -> AbstractTimeSeriesTrainer:
        return self._learner.load_trainer()  # noqa

    @apply_presets(TIMESERIES_PRESETS_CONFIGS)
    def fit(
        self,
        train_data: TimeSeriesDataFrame,
        tuning_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[int] = None,
        presets: Optional[str] = None,
        hyperparameters: Dict[Union[str, Type], Any] = None,
        hyperparameter_tune_kwargs: Optional[Union[str, Dict]] = None,
        **kwargs,
    ) -> "TimeSeriesPredictor":
        """Fit models to predict (distributional) forecasts of multiple related time series
        based on historical observations.

        Parameters
        ----------
        train_data: TimeSeriesDataFrame
            Training data in the TimeSeriesDataFrame format. See documentation for `autogluon.timeseries.dataset` for
            further information.
        tuning_data: TimeSeriesDataFrame, default = None
            Data reserved for model selection or hyperparameter tuning, rather than training individual
            models. If None, AutoGluon will reserve the most recent portion of `train_data` for tuning. Validation
            scores will by default be computed over the last `prediction_length` time points in the tuning data.
        time_limit: int, default = None
            Approximately how long fit() will run for (wall-clock time in seconds). If not specified, `fit()` will
            run until all models have completed training.
        presets: str, default = None
            Optional preset configurations for various arguments in `fit()`. Can significantly impact predictive
            accuracy, memory footprint, inference latency of trained models, and various other properties of the
            returned predictor. It is recommended to specify presets and avoid specifying most other `fit()` arguments
            or model hyperparameters prior to becoming familiar with AutoGluon. For example, set
            `presets="best_quality"` to get a high-accuracy predictor, or set `presets="low_quality"` to get a
            toy predictor that trains quickly but lacks accuracy. Any user-specified arguments in `fit()` will
            override the values used by presets.

            Available presets are "best_quality", "high_quality", "good_quality", "medium_quality", "low_quality",
            and "low_quality_hpo". Details for these presets can be found in
            `autogluon/timeseries/configs/presets_configs.py`. If not provided, user-provided values for other
            arguments (specifically, `hyperparameters` and `hyperparameter_tune_kwargs` will be used (defaulting
            to their default values specified below).
        hyperparameters: str or dict, default = "default"
            Determines the hyperparameters used by each model.
            If str is passed, will use a preset hyperparameter configuration, can be one of "default", "default_hpo",
            "toy", or "toy_hpo", where "toy" settings correspond to models only intended for prototyping.
            If dict is provided, the keys are strings or Types that indicate which model types to train. In this case,
            the predictor will only train the given model types. Stable model options include: 'DeepAR', 'MQCNN', and
            'SFF' (SimpleFeedForward). See References for more detail on these models.

            Values in the `hyperparameters` dict are themselves dictionaries of hyperparameter settings for each model
            type. Each hyperparameter can either be a single fixed value or a search space containing many possible
            values. A search space should only be provided when `hyperparameter_tune_kwargs` is specified (i.e.,
            hyperparameter-tuning is utilized). Any omitted hyperparameters not specified here will be set to default
            values which are given in`autogluon/timeseries/trainer/models/presets.py`. Specific hyperparameter
            choices for each of the recommended models can be found in the references.
        hyperparameter_tune_kwargs: str or dict, default = None
            # TODO

        References
        ----------
            - DeepAR: https://ts.gluon.ai/api/gluonts/gluonts.model.deepar.html
            - MQCNN: https://ts.gluon.ai/api/gluonts/gluonts.model.seq2seq.html
            - SFF: https://ts.gluon.ai/api/gluonts/gluonts.model.simple_feedforward.html
        """
        time_start = time.time()
        if self._learner.is_fit:
            raise AssertionError(
                "Predictor is already fit! To fit additional models create a new `Predictor`."
            )

        if self.target not in train_data.columns:
            raise ValueError(
                f"Target column `{self.target}` not found in the training data set."
            )
        if tuning_data is not None and self.target not in tuning_data.columns:
            raise ValueError(
                f"Target column `{self.target}` not found in the tuning data set."
            )
        if hyperparameters is None:
            hyperparameters = "default"

        verbosity = kwargs.get("verbosity", self.verbosity)
        set_logger_verbosity(verbosity, logger=logger)
        if presets is not None:
            logger.info(f"presets is set to {presets}")

        if verbosity >= 3:
            fit_args = dict(
                prediction_length=self.prediction_length,
                target_column=self.target,
                hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                presets=presets,
                time_limit=time_limit,
                **kwargs,
            )
            logger.info("============ fit arguments ============")
            logger.info("fit() called with arguments:")
            logger.info(f"{pprint.pformat(fit_args)}")
            logger.info("=======================================")

        # Inform the user extra columns in dataset will not be used.
        extra_columns = [c for c in train_data.columns.copy() if c != self.target]
        if len(extra_columns) > 0:
            logger.warning(f"Provided columns {extra_columns} will not be used.")

        if tuning_data is None:
            logger.warning(
                f"Validation data is None, will hold the last prediction_length {self.prediction_length} "
                f"time steps out to use as validation set.",
            )
            tuning_data = train_data
            train_data = train_data.slice_by_timestep(
                slice(None, -self.prediction_length)
            )

        scheduler_options = self._get_scheduler_options(
            hyperparameter_tune_kwargs, time_limit=time_limit
        )
        time_left = (
            None if time_limit is None else time_limit - (time.time() - time_start)
        )
        self._learner.fit(
            train_data=train_data,
            val_data=tuning_data,
            scheduler_options=scheduler_options,
            hyperparameters=hyperparameters,
            hyperparameter_tune=all(scheduler_options),
            time_limit=time_left,
        )

        self.save()
        return self

    # TODO: to be changed after ray tune integration
    def _get_scheduler_options(
        self,
        hyperparameter_tune_kwargs: Optional[Union[str, Dict]],
        time_limit: Optional[int] = None,
    ) -> Tuple[Optional[Type], Optional[Dict[str, Any]]]:
        """Validation logic for `hyperparameter_tune_kwargs`. Returns True if `hyperparameter_tune_kwargs` is None or
        can construct a valid scheduler. Returns False if hyperparameter_tune_kwargs results in an invalid scheduler.
        """
        if hyperparameter_tune_kwargs is None:
            return None, None

        num_trials: Optional[int] = None
        if isinstance(hyperparameter_tune_kwargs, dict):
            num_trials = hyperparameter_tune_kwargs.get("num_trials")
            if time_limit is None and num_trials is None:
                logger.warning(
                    "None of time_limit and num_trials are set, defaulting to num_trials=2",
                )
                num_trials = 2
            else:
                num_trials = hyperparameter_tune_kwargs.get("num_trials", 999)
        elif isinstance(hyperparameter_tune_kwargs, str):
            num_trials = 999

        scheduler_cls, scheduler_params = scheduler_factory(
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            time_out=time_limit,
            nthreads_per_trial="auto",
            ngpus_per_trial="auto",
            num_trials=num_trials,
        )

        if scheduler_params["num_trials"] == 1:
            logger.warning(
                "Warning: Specified num_trials == 1 for hyperparameter tuning, disabling HPO. "
            )
            return None, None

        scheduler_ngpus = scheduler_params["resource"].get("num_gpus", 0)
        if (
            scheduler_ngpus is not None
            and isinstance(scheduler_ngpus, int)
            and scheduler_ngpus > 1
        ):
            logger.warning(
                f"Warning: TimeSeriesPredictor currently doesn't use >1 GPU per training run. "
                f"Detected {scheduler_ngpus} GPUs."
            )
        return scheduler_cls, scheduler_params

    def get_model_names(self) -> List[str]:
        """Returns the list of model names trained in this `predictor` object."""
        return self._trainer.get_model_names()

    def predict(
        self,
        data: TimeSeriesDataFrame,
        model: Optional[str] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        """Return quantile and mean forecasts given a dataset to condition on.

        Parameters
        ----------
        data: TimeSeriesDataFrame
            Time series data to forecast with.
        model: str, default=None
            Name of the model that you would like to use for timeseries. If None, it will by default use the
            best model from trainer.
        """
        return self._learner.predict(data, model=model, **kwargs)

    def evaluate(self, data: TimeSeriesDataFrame, **kwargs):
        """Evaluate the performance for given dataset, computing the score determined by `self.eval_metric`
        on the given data set, and with the same `prediction_length` used when training models."""
        return self._learner.score(data, **kwargs)

    def score(self, data: TimeSeriesDataFrame, **kwargs):
        """See `evaluate`"""
        return self.evaluate(data, **kwargs)

    @classmethod
    def load(cls, path: str) -> "TimeSeriesPredictor":
        """Load an existing TimeSeriesPredictor from output_directory."""
        if not path:
            raise ValueError("`path` cannot be None or empty in load().")
        path = setup_outputdir(path, warn_if_exist=False)

        logger.info(f"Loading predictor from path {path}")
        learner = AbstractLearner.load(path)
        predictor = load_pkl.load(path=learner.path + cls.predictor_file_name)
        predictor._learner = learner
        return predictor

    def save(self) -> None:
        """Save this predictor to file in directory specified by this Predictor's `output_directory`.
        Note that `fit()` already saves the predictor object automatically
        (we do not recommend modifying the Predictor object yourself as it tracks many trained models).
        """
        tmp_learner = self._learner
        self._learner = None
        save_pkl.save(path=tmp_learner.path + self.predictor_file_name, object=self)
        self._learner = tmp_learner

    def info(self) -> Dict[str, Any]:
        return self._learner.get_info(include_model_info=True)

    def get_model_best(self) -> str:
        """Returns the name of the best model from trainer"""
        return self._trainer.get_model_best()

    def leaderboard(self, data: Optional[TimeSeriesDataFrame] = None) -> pd.DataFrame:
        """Return a leaderboard showing the performance of every trained model, the output is a
        pandas data frame containing the columns,
        * 'model': The name of the model.
        * 'score_val': The validation score of the model on the 'eval_metric'.
            NOTE: Metrics scores always show in higher is better form.
            This means that metrics such as RMSE or MAPE will have their signs FLIPPED, and values will be negative.
            This is necessary to avoid the user needing to know the metric to understand if higher is better when
            looking at leaderboard.
        * 'fit_time_marginal': The fit time required to train the model (ignoring base models for ensembles).
        * 'fit_order': The order in which models were fit. The first model fit has `fit_order=1`, and the Nth
            model fit has `fit_order=N`.

        Parameters
        ----------
        data: TimeSeriesDataFrame
            dataset used for additional evaluation. If None, the validation set used during training will
            be used.

        Returns
        -------
        leaderboard: pandas.DataFrame
            The leaderboard containing information on all models and in order of best model to worst in terms of
            validation performance.
        """
        return self._learner.leaderboard(data)

    def fit_summary(self, verbosity: int = 1) -> Dict[str, Any]:
        """Output summary of information about models produced during `fit()`.

        Parameters
        ----------
        verbosity : int, default = 1
            Controls the detail level of summary to output. Set 0 for no output printing.

        Returns
        -------
            Dict containing various detailed information. We do not recommend directly printing this dict as it may
            be very large.
        """
        # TODO: HPO-specific information currently not reported in fit_summary
        # TODO: Revisit after ray tune integration

        model_types = self._trainer.get_models_attribute_dict(attribute="type")
        model_typenames = {key: model_types[key].__name__ for key in model_types}
        unique_model_types = set(model_typenames.values())  # no more class info

        # all fit() information that is returned:
        results = {
            "model_types": model_typenames,  # dict with key = model-name, value = type of model (class-name)
            "model_performance": self._trainer.get_models_attribute_dict("score"),
            "model_best": self._trainer.model_best,  # the name of the best model (on validation data)
            "model_paths": self._trainer.get_models_attribute_dict("path"),
            "model_fit_times": self._trainer.get_models_attribute_dict("fit_time"),
        }
        # get dict mapping model name to final hyperparameter values for each model:
        model_hyperparams = {}
        for model_name in self.get_model_names():
            model_obj = self._trainer.load_model(model_name)
            model_hyperparams[model_name] = model_obj.params

        results["model_hyperparams"] = model_hyperparams
        results["leaderboard"] = self._learner.leaderboard()

        if verbosity > 0:  # print stuff
            print("****************** Summary of fit() ******************")
            print("Estimated performance of each model:")
            print(results["leaderboard"])
            print(f"Number of models trained: {len(results['model_performance'])}")
            print("Types of models trained:")
            print(unique_model_types)
            print("****************** End of fit() summary ******************")
        return results

    # TODO
    def refit_full(self, models="all"):
        raise NotImplementedError(
            "Refitting logic not yet implemented in autogluon.timeseries"
        )

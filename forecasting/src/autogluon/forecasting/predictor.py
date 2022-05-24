import copy
import logging
import pprint
import time
from typing import Optional, Type, Any, Union, Dict

import pandas as pd
import numpy as np
from gluonts.evaluation import Evaluator
from gluonts.model.forecast import QuantileForecast

from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.common.utils.utils import setup_outputdir
from autogluon.core.scheduler.scheduler_factory import scheduler_factory
from autogluon.core.utils.decorators import apply_presets
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl

from .configs import FORECASTING_PRESETS_CONFIGS
from .dataset import TimeSeriesDataFrame
from .learner import AbstractLearner, ForecastingLearner
from .trainer import AbstractForecastingTrainer

# from ..utils.dataset_utils import (
#     time_series_dataset,
#     rebuild_tabular,
#     train_test_split_gluonts,
#     train_test_split_dataframe,
#     TimeSeriesDataset,
# )
from autogluon.forecasting.utils.warning_filters import evaluator_warning_filter

logger = logging.getLogger()


class ForecastingPredictor:
    """autogluon.forecasting's ForecastingPredictor predicts future values of multiple related time-series by fitting
    and ensembling global forecasting models.

    autogluon.forecasting provide probabilistic (distributional) forecasts for univariate time series, where the
    forecasting model, essentially a mapping from the past of the time series to its future of length (i.e., forecast
    horizon) defined by the user, is learned globally from a collection of time series. Here, a "global" model can
    be understood as a set of time series model parameters that are shared across all time series to be predicted,
    in contrast to classical "local" approaches such as ARIMA.

    Parameters
    ----------
    eval_metric: str, default = None
        Metric by which predictions will be ultimately evaluated on future test data. AutoGluon tunes hyperparameters
        in order to improve this metric on validation data, and ranks models (on validation data) according to this
        metric. Available options include: "MASE", "MAPE", "sMAPE", "mean_wQuantileLoss".

        If `eval_metric = None`, it is set by default as "mean_wQuantileLoss".
        For more information about these options, see `autogluon.forecasting.utils.metric_utils` and GluonTS docs at
        https://ts.gluon.ai/api/gluonts/gluonts.evaluation.metrics.html
    path: str, default = None
        Path to directory where models and intermediate outputs should be saved. If unspecified, a time-stamped folder
        called "AutogluonModels/ag-[TIMESTAMP]" will be created in the working directory to store all models.
        Note: To call `fit()` twice and save all results of each fit, you must specify different `path` locations or
        not specify `path` at all. Otherwise, files from first `fit()` call will be overwritten by second `fit()`.
    verbosity : int, default = 2
        Verbosity levels range from 0 to 4 and control how much information is printed to stdout. Higher levels
        correspond to more detailed print statements, and verbosity=0 suppresses output including warnings.
        If using `logging`, you can alternatively control amount of information printed via `logger.setLevel(L)`,
        where `L` ranges from 0 to 50 (Note: higher values of `L` correspond to fewer print statements,
        opposite of verbosity levels).

    Other Parameters
    ----------------
    learner_type : AbstractLearner, default = ForecastingLearner
        A class which inherits from `AbstractLearner`. The learner dictates the inner logic of the ForecastingPredictor
        for training models and preprocessing data.
    learner_kwargs : dict, default = None
        Keyword arguments to send to the learner (for advanced users only). Options include `trainer_type`, a
        class inheriting from `AbstractTrainer` that controls training of multiple models.

    Attributes
    ----------
    target_column: str
        Name of column in training/validation data that contains the target time-series value to be predicted. If
        not specified explicitly during `fit()`, this will default to "target".
    """

    predictor_file_name = "predictor.pkl"

    def __init__(
        self,
        eval_metric: Optional[str] = None,
        path: Optional[str] = None,
        verbosity: int = 2,
        **kwargs,
    ):
        self.verbosity = verbosity
        set_logger_verbosity(self.verbosity, logger=logger)
        path = setup_outputdir(path)

        learner_type = kwargs.pop("learner_type", ForecastingLearner)
        learner_kwargs = kwargs.pop("learner_kwargs", dict())
        learner_kwargs.pop("eval_metric", None)
        learner_kwargs.pip("path", None)

        self.eval_metric = eval_metric
        self._learner: AbstractLearner = learner_type(
            path_context=path, eval_metric=eval_metric, **learner_kwargs
        )
        self._learner_type = type(self._learner)
        # self._trainer = None

        self.target_column = "target"

    @apply_presets(FORECASTING_PRESETS_CONFIGS)
    def fit(
        self,
        train_data: TimeSeriesDataFrame,
        prediction_length: int = 1,
        target_column: str = "target",
        val_data: Optional[TimeSeriesDataFrame] = None,
        presets: Optional[str] = None,
        hyperparameters: Dict[Union[str, Type], Any] = None,
        hyperparameter_tune_kwargs: Optional[Union[str, Dict]] = None,
        time_limit: Optional[int] = None,
        **kwargs,
    ) -> "ForecastingPredictor":
        """Fit models to predict (distributional) forecasts of multiple related time series
        based on historical observations.

        Parameters
        ----------
        train_data: TimeSeriesDataFrame
            Training data in the TimeSeriesDataFrame format. See documentation for `autogluon.forecasting.dataset` for
            further information.
        prediction_length: int
            The forecast horizon, i.e., How many time points into the future forecasters should be trained to predict.
            For example, if our time-series contains daily observations, setting `prediction_length=3` will train
            models that predict up to 3 days in the future from the most recent observation.
        target_column: str
            Name of column that contains the target values to forecast (ie. the numeric observations of the
            time-series). This column must contain numeric values, and missing target values
            should be in a pandas compatible format:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html
            By default, `target_column="target"`.
        val_data: TimeSeriesDataFrame
            Validation data reserved for model selection or hyperparameter tuning, rather than training individual
            models. If provided, it should have the same format as `train_data`. If None, AutoGluon will reserve the
            most recent portion of `train_data` for validation. Validation scores will by default be computed over
            the last `prediction_length` time points in the validation data.
        presets: str, default = None
            Optional preset configurations for various arguments in `fit()`. Can significantly impact predictive
            accuracy, memory-footprint, and inference latency of trained models, and various other properties of the
            returned predictor. It is recommended to specify presets and avoid specifying most other `fit()` arguments
            or model hyperparameters prior to becoming familiar with AutoGluon. For example, set
            `presets="best_quality"` to get a high-accuracy predictor, or set `presets="low_quality"` to get a
            toy predictor that trains very quick but lacks accuracy. Any user-specified arguments in `fit()` will
            override the values used by presets.

            Available presets are "best_quality", "high_quality", "good_quality", "medium_quality", "low_quality",
            and "low_quality_hpo". Details for these presets can be found in
            `autogluon/forecasting/configs/presets_configs.py`
        hyperparameter_tune_kwargs: Optional, None by default, can be str or dict
            # TODO
        hyperparameters: str or dict, default = None
            Determines the hyperparameters used by each model.
            If str is passed, will use a preset hyperparameter configuration, can be one of "default", "default_hpo",
            "toy", or "toy_hpo", where "toy" settings correspond to tiny models only intended for prototyping.
            If dict is provided, the keys are strings that indicate which model types to train. In this case,
            the predictor will only train the given model types. Stable model options include: 'DeepAR', 'MQCNN', and
            'SFF' (SimpleFeedForward). See the references section for more detail on these models.

            Values in the `hyperparameters` dict are themselves dictionaries of hyperparameter settings for each model
            type. Each hyperparameter can either be a single fixed value or a search space containing many possible
            values. A search space should only be provided when `hyperparameter_tune_kwargs` is specified (i.e.,
            hyperparameter-tuning is utilized). Any omitted hyperparameters not specified here will be set to default
            values which are given in`autogluon/forecasting/trainer/models/presets.py`. Specific hyperparameter
            choices for each of the recommended models can be found in the references.
        time_limit: int, default=None
            Approximately how long fit() will run for (wall-clock time in seconds). If not specified, `fit()` will
            run until all models have completed training.

        Other Parameters
        ----------------
        quantiles: List[float], default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            List of increasing decimals that specifies which quantiles should be estimated
            when making distributional forecasts.

        References
        ----------
            - DeepAR: https://ts.gluon.ai/api/gluonts/gluonts.model.deepar.html
            - MQCNN: https://ts.gluon.ai/api/gluonts/gluonts.model.seq2seq.html
            - SFF: https://ts.gluon.ai/api/gluonts/gluonts.model.simple_feedforward.html
        """
        start_time = time.time()
        if self._learner.is_fit:
            raise AssertionError(
                "Predictor is already fit! To fit additional models create a new `Predictor`."
            )

        if presets is not None:
            logger.log(30, f"presets is set to {presets}")

        self.target_column = target_column

        kwargs_orig = kwargs.copy()
        kwargs = self._validate_fit_kwargs(kwargs)
        if not self._validate_hyperparameter_tune_kwargs(
            hyperparameter_tune_kwargs, time_limit=time_limit
        ):
            hyperparameter_tune_kwargs = None
            logger.warning(
                30,
                "Invalid hyperparameter_tune_kwarg, disabling hyperparameter tuning.",
            )

        verbosity = kwargs.get("verbosity", self.verbosity)
        set_logger_verbosity(verbosity, logger=logger)

        hyperparameter_tune = hyperparameter_tune_kwargs is not None

        if verbosity >= 3:
            logger.log(20, "============ fit kwarg info ============")
            logger.log(20, "User Specified kwargs:")
            logger.log(20, f"{pprint.pformat(kwargs_orig)}")
            logger.log(20, "Full kwargs:")
            logger.log(20, f"{pprint.pformat(kwargs)}")
            logger.log(20, "========================================")

        set_logger_verbosity(verbosity, logger)

        # Inform the user extra columns in dataset will not be used.
        extra_columns = [c for c in train_data.columns.copy() if c != target_column]
        if len(extra_columns) > 0:
            logger.log(30, f"Provided columns {extra_columns} will not be used.")

        if val_data is None:
            logger.log(30, "Validation data is None, will split the data frame..")
            train_data, val_data = train_test_split_dataframe(
                train_data, prediction_length
            )

        random_state = kwargs.get("random_state", 0)
        logger.log(30, f"Random seed set to {random_state}")
        quantiles = kwargs.get(
            "quantiles", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
        logger.log(30, f"All models will be trained for quantiles {quantiles}.")
        if hyperparameter_tune_kwargs is not None:
            if (
                time_limit is None
                and hyperparameter_tune_kwargs.get("num_trials", None) is None
            ):
                logger.log(
                    30,
                    "None of time_limit and num_tirals are set, by default setting num_tirals=2",
                )
                num_trials = 2
            else:
                if isinstance(hyperparameter_tune_kwargs, str):
                    num_trials = 9999
                elif isinstance(hyperparameter_tune_kwargs, dict):
                    num_trials = hyperparameter_tune_kwargs.get("num_trials", 9999)
            scheduler_cls, scheduler_params = scheduler_factory(
                hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                nthreads_per_trial="auto",
                ngpus_per_trial="auto",
                num_trials=num_trials,
            )
            scheduler_options = (scheduler_cls, scheduler_params)
        else:
            scheduler_options = (None, None)
        if time_limit is not None:
            time_left = time_limit - processing_time
        else:
            time_left = time_limit

        self._learner.fit(
            train_data=train_data,
            prediction_length=prediction_length,
            val_data=val_data,
            scheduler_options=scheduler_options,
            hyperparameters=hyperparameters,
            hyperparameter_tune=hyperparameter_tune,
            quantiles=quantiles,
            time_limit=time_left,
        )

        self._set_post_fit_vars()
        self._post_fit(
            keep_only_best=kwargs["keep_only_best"],
            refit_full=kwargs["refit_full"],
            set_best_to_refit_full=kwargs["set_best_to_refit_full"],
        )
        self.save()
        return self

    def _validate_fit_kwargs(self, kwargs):
        """Validate kwargs given in .fit()"""
        kwargs_default = {
            "set_best_to_refit_full": False,
            "keep_only_best": False,
            "refit_full": False,
            "save_data": True,
            "freq": None,
            "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }
        if kwargs.get("refit_full", False):
            if "set_best_to_refit_full" not in kwargs:
                kwargs["set_best_to_refit_full"] = True
                logger.log(
                    30,
                    "refit_full is set while set_best_to_refit_full is not set, automatically setting set_best_to_refit_full=True"
                    "to make sure that the model will predict with refit full model by default.",
                )
        copied_kwargs = copy.deepcopy(kwargs_default)
        copied_kwargs.update(kwargs)
        return copied_kwargs

    def _set_post_fit_vars(self, learner: AbstractLearner = None):
        """
        Variable settings after fitting.
        """
        if learner is not None:
            self._learner: AbstractLearner = learner
        self._learner_type = type(self._learner)
        if self._learner.trainer_path is not None:
            self._trainer: AbstractTrainer = self._learner.load_trainer()

    def get_model_names(self):
        """Returns the list of model names trained in this `predictor` object."""
        return self._trainer.get_model_names_all()

    def predict(
        self,
        data,
        model=None,
        **kwargs,
    ):
        """
        Return forecasts given a dataset

        Parameters
        ----------
        data: dataset to forecast,
              should be in the same format as train_data when you call .fit()
        model: str, default=None
              Name of the model that you would like to use for forecasting. If None, it will by default use the best model from trainer.
        """
        predict_targets = self._learner.predict(
            data, model=model, **kwargs
        )
        return predict_targets

    def evaluate(self, data, **kwargs):
        """
        Evaluate the performance for given dataset.
        """
        perf = self._learner.score(data, **kwargs)
        return perf

    @classmethod
    def load(cls, output_directory, verbosity=2):
        """
        Load an existing ForecastingPredictor from output_directory
        """
        if output_directory is None:
            raise ValueError("output_directory cannot be None in load()")
        output_directory = setup_outputdir(
            output_directory, warn_if_exist=False
        )  # replace ~ with absolute path if it exists
        logger.log(30, f"Loading predictor from path {output_directory}")
        learner = AbstractLearner.load(output_directory)
        predictor = load_pkl.load(path=learner.path + cls.predictor_file_name)
        predictor._learner = learner
        predictor._trainer = learner.load_trainer()
        return predictor

    def save(self):
        """Save this predictor to file in directory specified by this Predictor's `output_directory`.
        Note that `fit()` already saves the predictor object automatically
        (we do not recommend modifying the Predictor object yourself as it tracks many trained models).
        """
        tmp_learner = self._learner
        tmp_trainer = self._trainer
        self._learner = None
        self._trainer = None
        save_pkl.save(path=tmp_learner.path + self.predictor_file_name, object=self)
        self._learner = tmp_learner
        self._trainer = tmp_trainer

    def info(self):
        """
        Get information from learner.
        """
        return self._learner.get_info(include_model_info=True)

    def get_model_best(self):
        """
        Get the best model from trainer.
        """
        return self._trainer.get_model_best()

    def leaderboard(self, data=None, static_features=None):
        """
        Return a leaderboard showing the performance of every trained model

        Parameters
        ----------
        data: a dataset in the same format of the train_data input input ForecastingPredictor().fit()
              used for additional evaluation aside from the validation set.
        static_features: a Dataframe containing static_features,
              must be provided if static_features is provided when calling ForecastingPredictor().fit()
        """
        return self._learner.leaderboard(data)

    def fit_summary(self, verbosity=3):
        """
        Output summary of information about models produced during `fit()`.
        May create various generated summary plots and store them in folder: `Predictor.output_directory`.

        Parameters
        ----------
        verbosity : int, default = 3
            Controls how detailed of a summary to ouput.
            Set <= 0 for no output printing, 1 to print just high-level summary,
            2 to print summary and create plots, >= 3 to print all information produced during fit().

        Returns
        -------
        Dict containing various detailed information. We do not recommend directly printing this dict as it may be very large.
        """
        hpo_used = len(self._trainer.hpo_results) > 0
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
            "hyperparameter_tune": hpo_used,
            "hyperparameters_userspecified": self._trainer.hyperparameters,
        }
        if hpo_used:
            results["hpo_results"] = self._trainer.hpo_results
        # get dict mapping model name to final hyperparameter values for each model:
        model_hyperparams = {}
        for model_name in self._trainer.get_model_names_all():
            model_obj = self._trainer.load_model(model_name)
            model_hyperparams[model_name] = model_obj.params
        results["model_hyperparams"] = model_hyperparams
        results["leaderboard"] = self._learner.leaderboard()
        if verbosity > 0:  # print stuff
            print("*** Summary of fit() ***")
            print("Estimated performance of each model:")
            print(results["leaderboard"])
            print("Number of models trained: %s" % len(results["model_performance"]))
            print("Types of models trained:")
            print(unique_model_types)
            hpo_str = ""
            if hpo_used and verbosity <= 2:
                hpo_str = (
                    " (call fit_summary() with verbosity >= 3 to see detailed HPO info)"
                )
            print("Hyperparameter-tuning used: %s %s" % (hpo_used, hpo_str))
            print("User-specified hyperparameters:")
            print(results["hyperparameters_userspecified"])
            print("Feature Metadata (Processed):")
            print("(raw dtype, special dtypes):")
        if verbosity > 2:  # print detailed information
            if hpo_used:
                hpo_results = results["hpo_results"]
                print("*** Details of Hyperparameter optimization ***")
                for model_type in hpo_results:
                    hpo_model = hpo_results[model_type]
                    if "trial_info" in hpo_model:
                        print(
                            f"HPO for {model_type} model:  Num. configurations tried = {len(hpo_model['trial_info'])}, Time spent = {hpo_model['total_time']}s"
                        )
                        print(
                            f"Best hyperparameter-configuration (validation-performance: {self.eval_metric} = {hpo_model['validation_performance']}):"
                        )
                        print(hpo_model["best_config"])
        if verbosity > 0:
            print("*** End of fit() summary ***")
        return results

    # TODO
    def refit_full(self, models="all"):
        raise NotImplementedError(
            "Refitting logic not yet implemented in autogluon.forecasting"
        )

    def _validate_hyperparameter_tune_kwargs(
        self, hyperparameter_tune_kwargs, time_limit=None
    ):
        """
        Returns True if hyperparameter_tune_kwargs is None or can construct a valid scheduler.
        Returns False if hyperparameter_tune_kwargs results in an invalid scheduler.
        """
        if hyperparameter_tune_kwargs is None:
            return True

        scheduler_cls, scheduler_params = scheduler_factory(
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            time_out=time_limit,
            nthreads_per_trial="auto",
            ngpus_per_trial="auto",
        )
        assert (
            scheduler_params["searcher"] != "bayesopt_hyperband"
        ), "searcher == 'bayesopt_hyperband' not yet supported"
        if scheduler_params.get("dist_ip_addrs", None):
            logger.warning(
                "Warning: dist_ip_addrs does not currently work for Tabular. Distributed instances will not be utilized."
            )

        if scheduler_params["num_trials"] == 1:
            logger.warning(
                "Warning: Specified num_trials == 1 for hyperparameter tuning, disabling HPO. "
                "This can occur if time_limit was not specified in `fit()`."
            )
            return False

        scheduler_ngpus = scheduler_params["resource"].get("num_gpus", 0)
        if (
            scheduler_ngpus is not None
            and isinstance(scheduler_ngpus, int)
            and scheduler_ngpus > 1
        ):
            logger.warning(
                f"Warning: ForecastingPredictor currently doesn't use >1 GPU per training run. Detected {scheduler_ngpus} GPUs."
            )
        return True

    @classmethod
    def evaluate_predictions(
        cls,
        forecasts,
        targets,
        index_column,
        time_column,
        target_column,
        eval_metric=None,
        quantiles=None,
    ):
        """
        Evaluate predictions once future targets are received.

        Parameters
        ----------
        forecasts: dict, produced by ForecastingPredictor().predict()
            a dictionary containing predictions for different targets.
            Keys are time series index
            Values are pandas Dataframe containing predictions for different quantiles.

        targets: a Dataframe which has the same format as what you have for train_data/test_data,
            must contain targets for all time presented in forecasts.

        index_column: str or None
            Name of column in targets that contains an index ID specifying which time series is being observed at each time-point (for datasets containing multiple time-series).
            If None, we will assume that there is only one time series in the dataset.

        time_column: str
            Name of column in targets that lists the time of each observation.

        target_column: str
            Name of column in targets that contains the target time-series value to be predicted.
        """
        with evaluator_warning_filter():
            targets = (
                rebuild_tabular(
                    targets,
                    index_column=index_column,
                    target_column=target_column,
                    time_column=time_column,
                )
                .set_index(index_column)
                .transpose()
            )

            required_time = list(forecasts.values())[0].index
            targets_time = pd.DatetimeIndex(
                targets.index, freq=pd.infer_freq(targets.index)
            )
            targets.index = targets_time

            for time in required_time:
                if time not in targets_time:
                    raise ValueError(
                        f"Time {time} is presented in predictions but not given in targets. Please check your targets."
                    )

            formated_targets = []
            quantile_forecasts = []
            for ts_id, forecast in forecasts.items():
                tmp_targets = targets.loc[required_time, ts_id]
                formated_targets.append(tmp_targets)

                tmp = []
                for quantile in forecast.columns:
                    tmp.append(forecast[quantile])
                quantile_forecasts.append(
                    QuantileForecast(
                        forecast_arrays=np.array(tmp),
                        start_date=forecast.index[0],
                        freq=pd.infer_freq(forecast.index),
                        forecast_keys=forecast.columns,
                        item_id=ts_id,
                    )
                )
            evaluator = Evaluator(quantile_forecasts[0].forecast_keys)
            num_series = len(formated_targets)
            agg_metrics, item_metrics = evaluator(
                iter(formated_targets), iter(quantile_forecasts), num_series=num_series
            )
            if eval_metric is None:
                return agg_metrics
            else:
                return agg_metrics[eval_metric]

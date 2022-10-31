import logging
import pprint
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd

from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.common.utils.utils import setup_outputdir
from autogluon.core.utils.decorators import apply_presets
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl

from .configs import TIMESERIES_PRESETS_CONFIGS
from .dataset import TimeSeriesDataFrame
from .learner import AbstractLearner, TimeSeriesLearner
from .splitter import AbstractTimeSeriesSplitter, LastWindowSplitter, MultiWindowSplitter
from .trainer import AbstractTimeSeriesTrainer

logger = logging.getLogger(__name__)

DEPRECATED_PRESETS_TO_FALLBACK = {
    "low_quality": "fast_training",
    "high_quality": "medium_quality",
    "good_quality": "medium_quality",
}


class TimeSeriesPredictor:
    """AutoGluon ``TimeSeriesPredictor`` predicts future values of multiple related time series.

    ``TimeSeriesPredictor`` provides probabilistic (distributional) multi-step-ahead forecasts for univariate time
    series. The forecast includes both the mean (i.e., conditional expectation of future values given the past), as
    well as the quantiles of the forecast distribution, indicating the range of possible future outcomes.

    ``TimeSeriesPredictor`` fits both "global" deep learning models that are shared across all time series
    (e.g., DeepAR, Transformer), as well as "local" statistical models that are fit to each individual time series
    (e.g., ARIMA, ETS).

    ``TimeSeriesPredictor`` expects input data and makes predictions in the
    :class:`~autogluon.timeseries.TimeSeriesDataFrame` format.


    Parameters
    ----------
    target : str, default = "target"
        Name of column that contains the target values to forecast (i.e., numeric observations of the time series).
    prediction_length : int, default = 1
        The forecast horizon, i.e., How many time steps into the future the models should be trained to predict.
        For example, if time series contain daily observations, setting ``prediction_length = 3`` will train
        models that predict up to 3 days into the future from the most recent observation.
    eval_metric : str, default = "mean_wQuantileLoss"
        Metric by which predictions will be ultimately evaluated on future test data. AutoGluon tunes hyperparameters
        in order to improve this metric on validation data, and ranks models (on validation data) according to this
        metric. Available options:

        - ``"mean_wQuantileLoss"``: mean weighted quantile loss, defined as average of quantile losses for the specified ``quantile_levels`` scaled by the total value of the time series
        - ``"MAPE"``: mean absolute percentage error
        - ``"sMAPE"``: "symmetric" mean absolute percentage error
        - ``"MASE"``: mean absolute scaled error
        - ``"MSE"``: mean squared error
        - ``"RMSE"``: root mean squared error

        For more information about these metrics, see https://docs.aws.amazon.com/forecast/latest/dg/metrics.html.
    quantile_levels : List[float], optional
        List of increasing decimals that specifies which quantiles should be estimated when making distributional
        forecasts. Defaults to ``[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]``.
        Can alternatively be provided with the keyword argument ``quantiles``.
    path : str, optional
        Path to the directory where models and intermediate outputs will be saved. Defaults to a timestamped folder
        ``AutogluonModels/ag-[TIMESTAMP]`` that will be created in the working directory.
    verbosity : int, default = 2
        Verbosity levels range from 0 to 4 and control how much information is printed to stdout. Higher levels
        correspond to more detailed print statements, and ``verbosity=0`` suppresses output including warnings.
        If using ``logging``, you can alternatively control amount of information printed via ``logger.setLevel(L)``,
        where ``L`` ranges from 0 to 50 (Note: higher values of ``L`` correspond to fewer print statements, opposite
        of verbosity levels).
    ignore_time_index : bool, default = False
        If True, the predictor will ignore the datetime indexes during both training and testing, and will replace
        the data indexes with dummy timestamps in second frequency. In this case, the forecast output time indexes will
        be arbitrary values, and seasonality will be turned off for local models.
    validation_splitter : Union[str, AbstractTimeSeriesSplitter], default = "last_window"
        Strategy for splitting ``train_data`` into training and validation parts during
        :meth:`~autogluon.timeseries.TimeSeriesPredictor.fit`. If ``tuning_data`` is passed to
        :meth:`~autogluon.timeseries.TimeSeriesPredictor.fit`, validation_splitter is ignored. Possible choices:

        - ``"last_window"`` - use last ``prediction_length`` time steps of each time series for validation.
        - ``"multi_window"`` - use last 3 non-overlapping windows of length ``prediction_length`` of each time series for validation.
        - object of type :class:`~autogluon.timeseries.splitter.AbstractTimeSeriesSplitter` implementing a custom splitting strategy (for advanced users only).

    Other Parameters
    ----------------
    learner_type : AbstractLearner, default = TimeSeriesLearner
        A class which inherits from ``AbstractLearner``. The learner specifies the inner logic of the
        ``TimeSeriesPredictor``.
    label : str
        Alias for :attr:`target`.
    learner_kwargs : dict, optional
        Keyword arguments to send to the learner (for advanced users only). Options include ``trainer_type``, a
        class inheriting from ``AbstractTrainer`` which controls training of multiple models.
        If ``path`` and ``eval_metric`` are re-specified within ``learner_kwargs``, these are ignored.
    quantiles : List[float]
        Alias for :attr:`quantile_levels`.
    """

    # TODO: Update description of presets after the presets are finalized
    # TODO: Update docstring for predict

    predictor_file_name = "predictor.pkl"

    def __init__(
        self,
        target: Optional[str] = None,
        prediction_length: int = 1,
        eval_metric: Optional[str] = None,
        path: Optional[str] = None,
        verbosity: int = 2,
        quantile_levels: Optional[List[float]] = None,
        ignore_time_index: bool = False,
        validation_splitter: Union[str, AbstractTimeSeriesSplitter] = "last_window",
        **kwargs,
    ):
        self.verbosity = verbosity
        set_logger_verbosity(self.verbosity, logger=logger)
        self.path = setup_outputdir(path)

        self.ignore_time_index = ignore_time_index
        if target is not None and kwargs.get("label") is not None:
            raise ValueError("Both `label` and `target` are specified. Please specify at most one of these arguments.")
        self.target = target or kwargs.get("label", "target")

        self.prediction_length = prediction_length
        self.eval_metric = eval_metric
        self.quantile_levels = quantile_levels or kwargs.get(
            "quantiles", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
        if validation_splitter == "last_window":
            splitter = LastWindowSplitter()
        elif validation_splitter == "multi_window":
            splitter = MultiWindowSplitter()
        elif isinstance(validation_splitter, AbstractTimeSeriesSplitter):
            splitter = validation_splitter
        else:
            raise ValueError(
                f"`validation_splitter` must be one of 'last_window', 'multi_window', or an object of type "
                f"`autogluon.timeseries.splitter.AbstractTimeSeriesSplitter` "
                f"(received {validation_splitter} of type {type(validation_splitter)})."
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
                validation_splitter=splitter,
            )
        )
        self._learner: AbstractLearner = learner_type(**learner_kwargs)
        self._learner_type = type(self._learner)

    @property
    def _trainer(self) -> AbstractTimeSeriesTrainer:
        return self._learner.load_trainer()  # noqa

    @property
    def validation_splitter(self) -> AbstractTimeSeriesSplitter:
        return self._learner.validation_splitter

    def _check_and_prepare_data_frame(self, df: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        """Ensure that TimeSeriesDataFrame has a frequency, or replace its time index with a dummy if
        ``self.ignore_time_index`` is True.
        """
        if df is None:
            return df
        if self.ignore_time_index:
            df = df.get_reindexed_view(freq="S")
        if df.freq is None:
            raise ValueError(
                "Frequency not provided and cannot be inferred. This is often due to the "
                "time index of the data being irregularly sampled. Please ensure that the "
                "data set used has a uniform time index, or create the `TimeSeriesPredictor` "
                "setting `ignore_time_index=True`."
            )
        return df

    @apply_presets(TIMESERIES_PRESETS_CONFIGS)
    def fit(
        self,
        train_data: TimeSeriesDataFrame,
        tuning_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[int] = None,
        presets: Optional[str] = None,
        hyperparameters: Dict[Union[str, Type], Any] = None,
        hyperparameter_tune_kwargs: Optional[Union[str, Dict]] = None,
        enable_ensemble: bool = True,
        **kwargs,
    ) -> "TimeSeriesPredictor":
        """Fit probabilistic forecasting models to the given time series dataset.

        Parameters
        ----------
        train_data : TimeSeriesDataFrame
            Training data in the :class:`~autogluon.timeseries.TimeSeriesDataFrame` format.

            If ``train_data`` has static features (i.e., ``train_data.static_features`` is a pandas DataFrame), the
            predictor will interpret columns with ``int`` and ``float`` dtypes as continuous (real-valued) features,
            columns with ``object`` and ``str`` dtypes as categorical features, and will ignore the rest of columns.

            For example, to ensure that column "store_id" with dtype ``int`` is interpreted as a category,
            we need to change its type to ``category``::

                train_data.static_features["store_id"] = train_data.static_features["store_id"].astype("category")

        tuning_data : TimeSeriesDataFrame, optional
            Data reserved for model selection and hyperparameter tuning, rather than training individual models. Also
            used to compute the validation scores. Note that only the last ``prediction_length`` time steps of each
            time series are used for computing the validation score.

            Leaving this argument empty and letting AutoGluon automatically generate the validation set from
            ``train_data`` is a good default.

            If not provided, AutoGluon will split :attr:`train_data` into training and tuning subsets using
            ``validation_splitter``. If ``tuning_data`` is provided, ``validation_splitter`` will be ignored.
            See the description of ``validation_splitter`` in the docstring for
            :class:`~autogluon.timeseries.TimeSeriesPredictor` for more details.

            If ``train_data`` has static features, ``tuning_data`` must have also have static features with the same
            column names and dtypes.
        time_limit : int, optional
            Approximately how long :meth:`~autogluon.timeseries.TimeSeriesPredictor.fit` will run (wall-clock time in
            seconds). If not specified, :meth:`~autogluon.timeseries.TimeSeriesPredictor.fit` will run until all models
            have completed training.
        presets : str, optional
            Optional preset configurations for various arguments in
            :meth:`~autogluon.timeseries.TimeSeriesPredictor.fit`.

            Can significantly impact predictive accuracy, memory footprint, inference latency of trained models,
            and various other properties of the returned predictor. It is recommended to specify presets and avoid
            specifying most other :meth:`~autogluon.timeseries.TimeSeriesPredictor.fit` arguments or model
            hyperparameters prior to becoming familiar with AutoGluon. For example, set ``presets="best_quality"``
            to get a high-accuracy predictor, or set ``presets="fast_training"`` to quickly fit multiple simple
            statistical models.
            Any user-specified arguments in :meth:`~autogluon.timeseries.TimeSeriesPredictor.fit` will
            override the values used by presets.

            Available presets are "best_quality", "medium_quality", and "fast_training".
            Details for these presets can be found in ``autogluon/timeseries/configs/presets_configs.py``. If not
            provided, user-provided values for other arguments (specifically, ``hyperparameters`` and
            ``hyperparameter_tune_kwargs`` will be used (defaulting to their default values specified below).
        hyperparameters : str or dict, default = "default"
            Determines what models are trained and what hyperparameters are used by each model.

            If str is passed, will use a preset hyperparameter configuration. Can be one of "default", "default_hpo",
            or "local_only". These configurations are defined in ``autogluon/timeseries/trainer/models/presets.py``.

            If dict is provided, the keys are strings or Types that indicate which models to train. Each value is
            itself a dict containing hyperparameters for each of the trained models. Any omitted hyperparameters not
            specified here will be set to default. For example::

                predictor.fit(
                    ...
                    hyperparameters={
                        "DeepAR": {},
                        "ETS": {"seasonal_period": 7},
                    }
                )

            The above example will only train two models:

            * ``DeepAR`` (with default hyperparameters)
            * ``ETS`` (with the given `seasonal_period`; all other parameters set to their defaults)

            Full list of available models and their hyperparameters is provided in :ref:`forecasting_zoo`.

            The hyperparameters for each model can be fixed values (as shown above), or search spaces over which
            hyperparameter optimization is performed. A search space should only be provided when
            ``hyperparameter_tune_kwargs`` is given (i.e., hyperparameter-tuning is utilized). For example::

                import autogluon.core as ag

                predictor.fit(
                    ...
                    hyperparameters={
                        "DeepAR": {
                            "num_cells": ag.space.Int(20, 100),
                            "cell_type": ag.space.Categorical("lstm", "gru")
                        },
                    },
                    hyperparameter_tune_kwargs="auto",
                )

            In the above example, multiple versions of the DeepAR model with different values of the parameters
            "num_cells" and "cell_type" will be trained.
        hyperparameter_tune_kwargs : str or dict, optional
            Hyperparameter tuning strategy and kwargs (for example, how many HPO trials to run). If ``None``, then
            hyperparameter tuning will not be performed.

            Ray Tune backend is used to tune deep-learning forecasting models from GluonTS. All other models use a
            custom HPO backed based on random search.

            Can be set to a string to choose one of available presets:

            * ``"random"`` - 10 trials of random search
            * ``"auto"`` - 10 trials of bayesian optimization GluonTS models, 10 trials of random search for other models

            Alternatively, a dict can be passed for more fine-grained control. The dict must include the following keys

            * ``"num_trials"`` - int, number of configurations to train for each tuned model
            * ``"searcher"`` - one of ``"random"`` (random search), ``"bayes"`` (bayesian optimization for GluonTS models, random search for other models) and ``"auto"`` (same as ``"bayes"``).
            * ``"scheduler"`` - the only supported option is ``"local"`` (all models trained on the same machine)

            Example::

                predictor.fit(
                    ...
                    hyperparameter_tune_kwargs={
                        "scheduler": "local",
                        "searcher": "auto",
                        "num_trials": 5,
                    }
                )

        enable_ensemble : bool, default = True
            If True, the ``TimeSeriesPredictor`` will fit a simple weighted ensemble on top of the models specified via
            ``hyperparameters``.

        """
        time_start = time.time()
        if self._learner.is_fit:
            raise AssertionError("Predictor is already fit! To fit additional models create a new `Predictor`.")

        if self.target not in train_data.columns:
            raise ValueError(f"Target column `{self.target}` not found in the training data set.")
        if tuning_data is not None and self.target not in tuning_data.columns:
            raise ValueError(f"Target column `{self.target}` not found in the tuning data set.")
        if hyperparameters is None:
            hyperparameters = "default"

        train_data = self._check_and_prepare_data_frame(train_data)
        tuning_data = self._check_and_prepare_data_frame(tuning_data)

        verbosity = kwargs.get("verbosity", self.verbosity)
        set_logger_verbosity(verbosity)

        fit_args = dict(
            prediction_length=self.prediction_length,
            target_column=self.target,
            time_limit=time_limit,
            evaluation_metric=self.eval_metric,
            hyperparameters=hyperparameters,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            enable_ensemble=enable_ensemble,
            **kwargs,
        )
        logger.info("================ TimeSeriesPredictor ================")
        logger.info("TimeSeriesPredictor.fit() called")
        if presets is not None:
            if presets in DEPRECATED_PRESETS_TO_FALLBACK:
                new_presets = DEPRECATED_PRESETS_TO_FALLBACK[presets]
                warnings.warn(
                    f"Presets {presets} are deprecated as of version 0.6.0. Please see the documentation for "
                    f"TimeSeriesPredictor.fit for the list of available presets. "
                    f"Falling back to presets='{new_presets}'."
                )
                presets = new_presets
            logger.info(f"Setting presets to: {presets}")
        logger.info("Fitting with arguments:")
        logger.info(f"{pprint.pformat(fit_args)}")
        logger.info(
            f"Provided training data set with {len(train_data)} rows, {train_data.num_items} items (item = single time series). "
            f"Average time series length is {len(train_data) / train_data.num_items:.1f}."
        )
        if tuning_data is not None:
            logger.info(
                f"Provided tuning data set with {len(tuning_data)} rows, {tuning_data.num_items} items. "
                f"Average time series length is {len(tuning_data) / tuning_data.num_items:.1f}."
            )
        logger.info(f"Training artifacts will be saved to: {Path(self.path).resolve()}")
        logger.info("=====================================================")

        time_left = None if time_limit is None else time_limit - (time.time() - time_start)
        self._learner.fit(
            train_data=train_data,
            val_data=tuning_data,
            hyperparameters=hyperparameters,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            time_limit=time_left,
            verbosity=verbosity,
            enable_ensemble=enable_ensemble,
        )

        self.save()
        return self

    def get_model_names(self) -> List[str]:
        """Returns the list of model names trained by this predictor object."""
        return self._trainer.get_model_names()

    def predict(
        self,
        data: TimeSeriesDataFrame,
        model: Optional[str] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        """Return quantile and mean forecasts for the given dataset, starting from the end of each time series.

        Parameters
        ----------
        data : TimeSeriesDataFrame
            Time series data to forecast with.

            If ``train_data`` used to train the predictor contained static features, then ``data`` must also contain
            static features that have the same columns and dtypes.
        model : str, optional
            Name of the model that you would like to use for prediction. By default, the best model during training
            (with highest validation score) will be used.
        """
        if "quantile_levels" in kwargs:
            warnings.warn(
                "Passing `quantile_levels` as a keyword argument to `TimeSeriesPredictor.predict` is deprecated and "
                "will be removed in v0.7.0. This might also lead to some models not working properly. "
                "Please specify the desired quantile levels when creating the predictor as "
                "`TimeSeriesPredictor(..., quantile_levels=quantile_levels)`.",
                category=DeprecationWarning,
            )
        data = self._check_and_prepare_data_frame(data)
        return self._learner.predict(data, model=model, **kwargs)

    def evaluate(self, data: TimeSeriesDataFrame, **kwargs):
        """Evaluate the performance for given dataset, computing the score determined by ``self.eval_metric``
        on the given data set, and with the same ``prediction_length`` used when training models.

        Parameters
        ----------
        data : TimeSeriesDataFrame
            The data to evaluate the best model on. The last ``prediction_length`` time steps of the
            data set, for each item, will be held out for prediction and forecast accuracy will be calculated
            on these time steps.

        Other Parameters
        ----------------
        model : str, optional
            Name of the model that you would like to evaluate. By default, the best model during training
            (with highest validation score) will be used.
        metric : str, optional
            Name of the evaluation metric to compute scores with. Defaults to ``self.eval_metric``

        Returns
        -------
        score : float
            A forecast accuracy score, where higher values indicate better quality. For consistency, error metrics
            will have their signs flipped to obey this convention. For example, negative MAPE values will be reported.
        """
        data = self._check_and_prepare_data_frame(data)
        return self._learner.score(data, **kwargs)

    def score(self, data: TimeSeriesDataFrame, **kwargs):
        """See, :meth:`~autogluon.timeseries.TimeSeriesPredictor.evaluate`."""
        return self.evaluate(data, **kwargs)

    @classmethod
    def load(cls, path: str) -> "TimeSeriesPredictor":
        """Load an existing ``TimeSeriesPredictor`` from given ``path``.

        Parameters
        ----------
        path : str
            Path where the predictor was saved via :meth:`~autogluon.timeseries.TimeSeriesPredictor.save`.

        Returns
        -------
        predictor : TimeSeriesPredictor
        """
        if not path:
            raise ValueError("`path` cannot be None or empty in load().")
        path = setup_outputdir(path, warn_if_exist=False)

        logger.info(f"Loading predictor from path {path}")
        learner = AbstractLearner.load(path)
        predictor = load_pkl.load(path=learner.path + cls.predictor_file_name)
        predictor._learner = learner
        return predictor

    def save(self) -> None:
        """Save this predictor to file in directory specified by this Predictor's ``path``.

        Note that :meth:`~autogluon.timeseries.TimeSeriesPredictor.fit` already saves the predictor object automatically
        (we do not recommend modifying the Predictor object yourself as it tracks many trained models).
        """
        tmp_learner = self._learner
        self._learner = None
        save_pkl.save(path=tmp_learner.path + self.predictor_file_name, object=self)
        self._learner = tmp_learner

    def info(self) -> Dict[str, Any]:
        """Returns a dictionary of objects each describing an attribute of the training process and trained models."""
        return self._learner.get_info(include_model_info=True)

    def get_model_best(self) -> str:
        """Returns the name of the best model from trainer."""
        return self._trainer.get_model_best()

    def leaderboard(self, data: Optional[TimeSeriesDataFrame] = None, silent=False) -> pd.DataFrame:
        """Return a leaderboard showing the performance of every trained model, the output is a
        pandas data frame with columns:

        * ``model``: The name of the model.
        * ``score_test``: The test score of the model on ``data``, if provided. Computed according to ``eval_metric``.
        * ``score_val``: The validation score of the model using the internal validation data. Computed according
            to ``eval_metric``.

            **NOTE:** Metrics scores are always shown in higher is better form.
            This means that metrics such as MASE or MAPE will have their signs `flipped`, and values will be negative.
            This is necessary to avoid the user needing to know the metric to understand if higher is better when
            looking at leaderboard.

        * ``pred_time_val``: Time taken by the model to predict on the validation data set
        * ``fit_time_marginal``: The fit time required to train the model (ignoring base models for ensembles).
        * ``fit_order``: The order in which models were fit. The first model fit has ``fit_order=1``, and the Nth
          model fit has ``fit_order=N``.

        Parameters
        ----------
        data : TimeSeriesDataFrame, optional
            dataset used for additional evaluation. If not provided, the validation set used during training will be
            used.
        silent : bool, default = False
            If False, the leaderboard DataFrame will be printed.

        Returns
        -------
        leaderboard : pandas.DataFrame
            The leaderboard containing information on all models and in order of best model to worst in terms of
            test performance.
        """
        data = self._check_and_prepare_data_frame(data)
        leaderboard = self._learner.leaderboard(data)
        if not silent:
            with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
                print(leaderboard)
        return leaderboard

    def fit_summary(self, verbosity: int = 1) -> Dict[str, Any]:
        """Output summary of information about models produced during
        :meth:`~autogluon.timeseries.TimeSeriesPredictor.fit`.

        Parameters
        ----------
        verbosity : int, default = 1
            Controls the detail level of summary to output. Set 0 for no output printing.

        Returns
        -------
        summary_dict : Dict[str, Any]
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
        raise NotImplementedError("Refitting logic not yet implemented in autogluon.timeseries")

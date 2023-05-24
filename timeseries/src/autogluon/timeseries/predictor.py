import logging
import pprint
import time
import warnings
from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd

from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.common.utils.utils import check_saved_predictor_version, setup_outputdir
from autogluon.core.utils.decorators import apply_presets
from autogluon.core.utils.loaders import load_pkl, load_str
from autogluon.core.utils.savers import save_pkl, save_str
from autogluon.timeseries import __version__ as current_ag_version
from autogluon.timeseries.configs import TIMESERIES_PRESETS_CONFIGS
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from autogluon.timeseries.learner import AbstractLearner, TimeSeriesLearner
from autogluon.timeseries.trainer import AbstractTimeSeriesTrainer
from autogluon.timeseries.utils.random import set_random_seed

logger = logging.getLogger(__name__)

SUPPORTED_FREQUENCIES = {"D", "W", "M", "Q", "A", "Y", "H", "T", "min", "S"}


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
    eval_metric_seasonal_period : int, optional
        Seasonal period used to compute the mean absolute scaled error (MASE) evaluation metric. This parameter is only
        used if ``eval_metric="MASE"`. See https://en.wikipedia.org/wiki/Mean_absolute_scaled_error for more details.
        Defaults to ``None``, in which case the seasonal period is computed based on the data frequency.
    known_covariates_names: List[str], optional
        Names of the covariates that are known in advance for all time steps in the forecast horizon. These are also
        known as dynamic features, exogenous variables, additional regressors or related time series. Examples of such
        covariates include holidays, promotions or weather forecasts.

        Currently, only numeric (float of integer dtype) are supported.

        If ``known_covariates_names`` are provided, then:

        - :meth:`~autogluon.timeseries.TimeSeriesPredictor.fit`, :meth:`~autogluon.timeseries.TimeSeriesPredictor.evaluate`, and :meth:`~autogluon.timeseries.TimeSeriesPredictor.leaderboard` will expect a data frame with columns listed in ``known_covariates_names`` (in addition to the ``target`` column).
        - :meth:`~autogluon.timeseries.TimeSeriesPredictor.predict` will expect an additional keyword argument ``known_covariates`` containing the future values of the known covariates in ``TimeSeriesDataFrame`` format.

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
    learner_type : AbstractLearner, default = TimeSeriesLearner
        A class which inherits from ``AbstractLearner``. The learner specifies the inner logic of the
        ``TimeSeriesPredictor``.
    learner_kwargs : dict, optional
        Keyword arguments to send to the learner (for advanced users only). Options include ``trainer_type``, a
        class inheriting from ``AbstractTrainer`` which controls training of multiple models.
        If ``path`` and ``eval_metric`` are re-specified within ``learner_kwargs``, these are ignored.
    label : str, optional
        Alias for :attr:`target`.
    """

    predictor_file_name = "predictor.pkl"
    _predictor_version_file_name = "__version__"

    def __init__(
        self,
        target: Optional[str] = None,
        known_covariates_names: Optional[List[str]] = None,
        prediction_length: int = 1,
        eval_metric: Optional[str] = None,
        eval_metric_seasonal_period: Optional[int] = None,
        path: Optional[str] = None,
        verbosity: int = 2,
        quantile_levels: Optional[List[float]] = None,
        ignore_time_index: bool = False,
        learner_type: Type[AbstractLearner] = TimeSeriesLearner,
        learner_kwargs: Optional[dict] = None,
        label: Optional[str] = None,
        quantiles: Optional[List[float]] = None,
        validation_splitter: Optional[Any] = None,
    ):
        self.verbosity = verbosity
        set_logger_verbosity(self.verbosity, logger=logger)
        self.path = setup_outputdir(path)

        self.ignore_time_index = ignore_time_index
        if target is not None and label is not None:
            raise ValueError("Both `label` and `target` are specified. Please specify at most one of these arguments.")
        self.target = target or label or "target"

        if known_covariates_names is None:
            known_covariates_names = []
        if isinstance(known_covariates_names, str):
            known_covariates_names = [known_covariates_names]
        if not all(isinstance(name, str) for name in known_covariates_names):
            raise ValueError(
                "known_covariates_names must be a list of strings (names of columns that are known at prediction time)."
            )
        if self.target in known_covariates_names:
            raise ValueError(f"Target column {self.target} cannot be one of the known covariates.")
        self.known_covariates_names = known_covariates_names

        self.prediction_length = prediction_length
        self.eval_metric = eval_metric
        self.eval_metric_seasonal_period = eval_metric_seasonal_period
        if quantile_levels is None:
            quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.quantile_levels = sorted(quantile_levels)

        if validation_splitter is not None:
            warnings.warn(
                "`validation_splitter` argument has been deprecated as of v0.8.0. "
                "Please use the `num_val_windows` argument of `TimeSeriesPredictor.fit` instead."
            )
        if quantiles is not None:
            warnings.warn(
                "`quantiles` argument has been deprecated as of v0.8.0. "
                "Please use the `quantile_levels` argument instead."
            )

        if learner_kwargs is None:
            learner_kwargs = {}
        learner_kwargs = learner_kwargs.copy()
        learner_kwargs.update(
            dict(
                path_context=self.path,
                eval_metric=eval_metric,
                eval_metric_seasonal_period=eval_metric_seasonal_period,
                target=self.target,
                known_covariates_names=self.known_covariates_names,
                prediction_length=self.prediction_length,
                quantile_levels=self.quantile_levels,
                ignore_time_index=ignore_time_index,
            )
        )
        self._learner: AbstractLearner = learner_type(**learner_kwargs)
        self._learner_type = type(self._learner)

    @property
    def _trainer(self) -> AbstractTimeSeriesTrainer:
        return self._learner.load_trainer()  # noqa

    def _check_and_prepare_data_frame(self, df: Union[TimeSeriesDataFrame, pd.DataFrame]) -> TimeSeriesDataFrame:
        """Ensure that TimeSeriesDataFrame has a frequency, or replace its time index with a dummy if
        ``self.ignore_time_index`` is True.
        """
        if df is None:
            return df
        if not isinstance(df, TimeSeriesDataFrame):
            if isinstance(df, pd.DataFrame):
                try:
                    df = TimeSeriesDataFrame(df)
                except:
                    raise ValueError(
                        f"Provided data of type {type(df)} cannot be automatically converted to a TimeSeriesDataFrame."
                    )
            else:
                raise ValueError(
                    f"Please provide data in TimeSeriesDataFrame format (received an object of type {type(df)})."
                )
        if self.ignore_time_index:
            df = df.get_reindexed_view(freq="S")
        # MultiIndex.is_monotonic_increasing checks if index is sorted by ["item_id", "timestamp"]
        if not df.index.is_monotonic_increasing:
            logger.info("Provided data is not sorted by item_id & timestamp, sorting...")
            df = df.sort_index()
            df._cached_freq = None  # in case frequency was incorrectly cached as IRREGULAR_TIME_INDEX_FREQSTR
        if df.freq is None:
            raise ValueError(
                "Frequency not provided and cannot be inferred. This is often due to the "
                "time index of the data being irregularly sampled. Please ensure that the "
                "data set used has a uniform time index, or create the `TimeSeriesPredictor` "
                "setting `ignore_time_index=True`."
            )
        # Check if frequency is supported
        offset = pd.tseries.frequencies.to_offset(df.freq)
        norm_freq_str = offset.name.split("-")[0]
        if norm_freq_str not in SUPPORTED_FREQUENCIES:
            warnings.warn(
                f"Detected frequency '{norm_freq_str}' is not supported by TimeSeriesPredictor. This may lead to some "
                f"models not working as intended. "
                f"Please convert the timestamps to one of the supported frequencies: {SUPPORTED_FREQUENCIES}. "
                f"See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases for details."
            )
        if df.isna().values.any():
            raise ValueError(
                "TimeSeriesPredictor does not yet support missing values. "
                "Please make sure that the provided data contains no NaNs."
            )
        if (df.num_timesteps_per_item() <= 2).any():
            # Time series with length <= 2 make frequency inference impossible
            raise ValueError("Detected time series with length <= 2 in data. Please remove them from the dataset.")
        return df

    def _validate_num_val_windows(
        self,
        train_data: TimeSeriesDataFrame,
        tuning_data: Optional[TimeSeriesDataFrame],
        num_val_windows: int,
    ) -> int:
        """Check if given num_val_windows is suitable for given data and validate length of training time series.

        Returns
        -------
        num_val_windows : int
            Number of cross validation windows adjusted based on the length of training time series.
        """
        shortest_ts_length = train_data.num_timesteps_per_item().min()
        if tuning_data is None:
            recommended_ts_length = 2 * self.prediction_length + 1
            recommended_ts_length_str = "2 * prediction_length + 1"
        else:
            recommended_ts_length = self.prediction_length + 1
            recommended_ts_length_str = "prediction_length + 1"

        if shortest_ts_length < recommended_ts_length:
            logger.warning(
                f"\nIt is recommended that all time series in train_data have length >= {recommended_ts_length_str} "
                f"(at least {recommended_ts_length}). "
                "Otherwise the predictor may not work as expected. "
                "\nPlease reduce prediction_length or provide longer time series as train_data. "
            )

        # Ensure that after splitting off the last prediction_length timesteps all time series have length >= 1
        max_possible_num_val_windows = int((shortest_ts_length - 1) / self.prediction_length)
        if num_val_windows > max_possible_num_val_windows:
            logger.warning(
                f"\nTime series in train_data are too short for the given num_val_windows = {num_val_windows}. "
                f"Setting num_val_windows = {max_possible_num_val_windows}"
            )
            num_val_windows = max_possible_num_val_windows
        if num_val_windows == 0 and tuning_data is None:
            raise ValueError("Training is impossible because num_val_windows = 0 and no tuning_data is provided")

        return num_val_windows

    @apply_presets(TIMESERIES_PRESETS_CONFIGS)
    def fit(
        self,
        train_data: Union[TimeSeriesDataFrame, pd.DataFrame],
        tuning_data: Optional[Union[TimeSeriesDataFrame, pd.DataFrame]] = None,
        time_limit: Optional[int] = None,
        presets: Optional[str] = None,
        hyperparameters: Dict[Union[str, Type], Any] = None,
        hyperparameter_tune_kwargs: Optional[Union[str, Dict]] = None,
        excluded_model_types: Optional[List[str]] = None,
        num_val_windows: int = 1,
        refit_full: bool = False,
        enable_ensemble: bool = True,
        random_seed: Optional[int] = None,
        verbosity: Optional[int] = None,
    ) -> "TimeSeriesPredictor":
        """Fit probabilistic forecasting models to the given time series dataset.

        Parameters
        ----------
        train_data : Union[TimeSeriesDataFrame, pd.DataFrame]
            Training data in the :class:`~autogluon.timeseries.TimeSeriesDataFrame` format. For best performance, all
            time series should have length ``> 2 * prediction_length``.

            If ``known_covariates_names`` were specified when creating the predictor, ``train_data`` must include the
            columns listed in ``known_covariates_names`` with the covariates values aligned with the target time series.
            The known covariates must have a numeric (float or integer) dtype.

            Columns of ``train_data`` except ``target`` and those listed in ``known_covariates_names`` will be
            interpreted as ``past_covariates`` - covariates that are known only in the past.

            If ``train_data`` has static features (i.e., ``train_data.static_features`` is a pandas DataFrame), the
            predictor will interpret columns with ``int`` and ``float`` dtypes as continuous (real-valued) features,
            columns with ``object`` and ``str`` dtypes as categorical features, and will ignore the rest of columns.

            For example, to ensure that column "store_id" with dtype ``int`` is interpreted as a category,
            we need to change its type to ``category``::

                data.static_features["store_id"] = data.static_features["store_id"].astype("category")

            If provided data is an instance of pandas DataFrame, AutoGluon will attempt to automatically convert it
            to a ``TimeSeriesDataFrame``.

        tuning_data : Union[TimeSeriesDataFrame, pd.DataFrame], optional
            Data reserved for model selection and hyperparameter tuning, rather than training individual models. Also
            used to compute the validation scores. Note that only the last ``prediction_length`` time steps of each
            time series are used for computing the validation score.

            If ``tuning_data`` is provided, multi-window backtesting on training data will be disabled, the
            ``num_val_windows`` argument will be ignored, and ``refit_full`` will be set to ``False``.

            Leaving this argument empty and letting AutoGluon automatically generate the validation set from
            ``train_data`` is a good default.

            If ``known_covariates_names`` were specified when creating the predictor, ``tuning_data`` must also include
            the columns listed in ``known_covariates_names`` with the covariates values aligned with the target time
            series.

            If ``train_data`` has past covariates or static features, ``tuning_data`` must have also include them (with
            same columns names and dtypes).

            If provided data is an instance of pandas DataFrame, AutoGluon will attempt to automatically convert it
            to a ``TimeSeriesDataFrame``.

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
            hyperparameters prior to becoming familiar with AutoGluon. For example, set ``presets="high_quality"``
            to get a high-accuracy predictor, or set ``presets="fast_training"`` to quickly fit multiple simple
            statistical models.
            Any user-specified arguments in :meth:`~autogluon.timeseries.TimeSeriesPredictor.fit` will
            override the values used by presets.

            Available presets:

            - ``"fast_training"``: fit simple "local" statistical models (``ETS``, ``ARIMA``, ``Theta``, ``Naive``, ``SeasonalNaive``). These models are fast to train, but cannot capture more complex patterns in the data.
            - ``"medium_quality"``: all models mentioned above + tree-based model ``DirectTabular`` + deep learning model ``DeepAR``. Default setting that produces good forecasts with reasonable training time.
            - ``"high_quality"``: all models mentioned above + hyperparameter optimization for local statistical models + deep learning models ``TemporalFusionTransformerMXNet`` (if MXNet is available) and ``SimpleFeedForward``. Usually more accurate than ``medium_quality``, but takes longer to train.
            - ``"best_quality"``: all models mentioned above + deep learning model ``TransformerMXNet`` (if MXNet is available) + hyperparameter optimization for deep learning models. Usually better than ``high_quality``, but takes much longer to train.

            Details for these presets can be found in ``autogluon/timeseries/configs/presets_configs.py``. If not
            provided, user-provided values for ``hyperparameters`` and ``hyperparameter_tune_kwargs`` will be used
            (defaulting to their default values specified below).
        hyperparameters : str or dict, default = "medium_quality"
            Determines what models are trained and what hyperparameters are used by each model.

            If str is passed, will use a preset hyperparameter configuration defined in`
            `autogluon/timeseries/trainer/models/presets.py``.

            If dict is provided, the keys are strings or types that indicate which models to train. Each value is
            itself a dict containing hyperparameters for each of the trained models, or a list of such dicts. Any
            omitted hyperparameters not specified here will be set to default. For example::

                predictor.fit(
                    ...
                    hyperparameters={
                        "DeepAR": {},
                        "ETS": [
                            {"seasonal": "add"},
                            {"seasonal": None},
                        ],
                    }
                )

            The above example will train three models:

            * ``DeepAR`` with default hyperparameters
            * ``ETS`` with additive seasonality (all other parameters set to their defaults)
            * ``ETS`` with seasonality disabled (all other parameters set to their defaults)

            Full list of available models and their hyperparameters is provided in :ref:`forecasting_zoo`.

            The hyperparameters for each model can be fixed values (as shown above), or search spaces over which
            hyperparameter optimization is performed. A search space should only be provided when
            ``hyperparameter_tune_kwargs`` is given (i.e., hyperparameter-tuning is utilized). For example::

                from autogluon.common import space

                predictor.fit(
                    ...
                    hyperparameters={
                        "DeepAR": {
                            "hidden_size": space.Int(20, 100),
                            "dropout_rate": space.Categorical(0.1, 0.3),
                        },
                    },
                    hyperparameter_tune_kwargs="auto",
                )

            In the above example, multiple versions of the DeepAR model with different values of the parameters
            "hidden_size" and "dropout_rate" will be trained.
        hyperparameter_tune_kwargs : str or dict, optional
            Hyperparameter tuning strategy and kwargs (for example, how many HPO trials to run). If ``None``, then
            hyperparameter tuning will not be performed.

            Ray Tune backend is used to tune deep-learning forecasting models from GluonTS implemented in MXNet. All
            other models use a custom HPO backed based on random search.

            Can be set to a string to choose one of available presets:

            - ``"random"``: 10 trials of random search
            - ``"auto"``: 10 trials of bayesian optimization GluonTS MXNet models, 10 trials of random search for other models

            Alternatively, a dict can be passed for more fine-grained control. The dict must include the following keys

            - ``"num_trials"``: int, number of configurations to train for each tuned model
            - ``"searcher"``: one of ``"random"`` (random search), ``"bayes"`` (bayesian optimization for GluonTS MXNet models, random search for other models) and ``"auto"`` (same as ``"bayes"``).
            - ``"scheduler"``: the only supported option is ``"local"`` (all models trained on the same machine)

            Example::

                predictor.fit(
                    ...
                    hyperparameter_tune_kwargs={
                        "scheduler": "local",
                        "searcher": "auto",
                        "num_trials": 5,
                    },
                )
        excluded_model_types: List[str], optional
            Banned subset of model types to avoid training during ``fit()``, even if present in ``hyperparameters``.
            For example, the following code will train all models included in the ``high_quality`` presets except ``DeepAR``::

                predictor.fit(
                    ...,
                    presets="high_quality",
                    excluded_model_types=["DeepAR"],
                )
        num_val_windows : int, default = 1
            Number of backtests done on ``train_data`` for each trained model to estimate the validation performance.
            A separate copy of each model is trained for each validation window.
            When ``num_val_windows = k``, training time is increased roughly by a factor of ``k``.
        refit_full : bool, default = False
            If True, after training is complete, AutoGluon will attempt to re-train all models using all of training
            data (including the data initially reserved for validation). This argument has no effect if ``tuning_data``
            is provided.
        enable_ensemble : bool, default = True
            If True, the ``TimeSeriesPredictor`` will fit a simple weighted ensemble on top of the models specified via
            ``hyperparameters``.
        random_seed : int, optional
            If provided, fixes the seed of the random number generator for all models. This guarantees reproducible
            results for most models (except those trained on GPU because of the non-determinism of GPU operations).
        verbosity : int, optional
            If provided, overrides the ``verbosity`` value used when creating the ``TimeSeriesPredictor``. See
            documentation for :class:`~autogluon.timeseries.TimeSeriesPredictor` for more details.

        """
        time_start = time.time()
        if self._learner.is_fit:
            raise AssertionError("Predictor is already fit! To fit additional models create a new `Predictor`.")

        if hyperparameters is None:
            hyperparameters = "default"

        train_data = self._check_and_prepare_data_frame(train_data)
        tuning_data = self._check_and_prepare_data_frame(tuning_data)

        if verbosity is None:
            verbosity = self.verbosity
        set_logger_verbosity(verbosity)

        fit_args = dict(
            prediction_length=self.prediction_length,
            target=self.target,
            time_limit=time_limit,
            evaluation_metric=self.eval_metric,
            hyperparameters=hyperparameters,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            excluded_model_types=excluded_model_types,
            num_val_windows=num_val_windows,
            enable_ensemble=enable_ensemble,
            random_seed=random_seed,
            verbosity=verbosity,
        )
        logger.info("================ TimeSeriesPredictor ================")
        logger.info("TimeSeriesPredictor.fit() called")
        if presets is not None:
            logger.info(f"Setting presets to: {presets}")
        logger.info("Fitting with arguments:")
        logger.info(f"{pprint.pformat(fit_args)}")
        logger.info(
            f"Provided training data set with {len(train_data)} rows, {train_data.num_items} items (item = single time series). "
            f"Average time series length is {len(train_data) / train_data.num_items:.1f}. "
            f"Data frequency is '{train_data.freq}'."
        )
        if tuning_data is not None:
            logger.info(
                f"Provided tuning data set with {len(tuning_data)} rows, {tuning_data.num_items} items. "
                f"Average time series length is {len(tuning_data) / tuning_data.num_items:.1f}."
            )
            if num_val_windows > 0:
                logger.warning("Multi-window backtesting is disabled (setting num_val_windows = 0)")
                num_val_windows = 0
        num_val_windows = self._validate_num_val_windows(train_data, tuning_data, num_val_windows)

        logger.info("=====================================================")

        if random_seed is not None:
            set_random_seed(random_seed)

        time_left = None if time_limit is None else time_limit - (time.time() - time_start)
        self._learner.fit(
            train_data=train_data,
            val_data=tuning_data,
            hyperparameters=hyperparameters,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            excluded_model_types=excluded_model_types,
            time_limit=time_left,
            verbosity=verbosity,
            num_val_windows=num_val_windows,
            enable_ensemble=enable_ensemble,
        )
        if refit_full:
            if tuning_data is None:
                self.refit_full()
            else:
                logger.warning("Skipping `refit_full` because custom `tuning_data` was provided during `fit`.")

        self.save()
        return self

    def get_model_names(self) -> List[str]:
        """Returns the list of model names trained by this predictor object."""
        return self._trainer.get_model_names()

    def predict(
        self,
        data: Union[TimeSeriesDataFrame, pd.DataFrame],
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        model: Optional[str] = None,
        random_seed: Optional[int] = 123,
    ) -> TimeSeriesDataFrame:
        """Return quantile and mean forecasts for the given dataset, starting from the end of each time series.

        Parameters
        ----------
        data : Union[TimeSeriesDataFrame, pd.DataFrame]
            Time series data to forecast with.

            If ``known_covariates_names`` were specified when creating the predictor, ``data`` must include the columns
            listed in ``known_covariates_names`` with the covariates values aligned with the target time series.

            If ``train_data`` used to train the predictor contained past covariates or static features, then ``data``
            must also include them (with same column names and dtypes).

            If provided data is an instance of pandas DataFrame, AutoGluon will attempt to automatically convert it
            to a ``TimeSeriesDataFrame``.
        known_covariates : TimeSeriesDataFrame, optional
            If ``known_covariates_names`` were specified when creating the predictor, it is necessary to provide the
            values of the known covariates for each time series during the forecast horizon. That is:

            - The columns must include all columns listed in ``known_covariates_names``
            - The ``item_id`` index must include all item ids present in ``data``
            - The ``timestamp`` index must include the values for ``prediction_length`` many time steps into the future from the end of each time series in ``data``

            See example below.
        model : str, optional
            Name of the model that you would like to use for prediction. By default, the best model during training
            (with highest validation score) will be used.
        random_seed : int or None, default = 123
            If provided, fixes the seed of the random number generator for all models. This guarantees reproducible
            results for most models (except those trained on GPU because of the non-determinism of GPU operations).


        Examples
        --------
        >>> print(data)
                            target  promotion  price
        item_id timestamp
        A       2020-01-05      20          0   19.9
                2020-01-06      40          1    9.9
                2020-01-07      32          0   15.0
        B       2020-03-01      13          0    5.0
                2020-03-02      44          1    2.9
                2020-03-03      72          1    2.9
        >>> predictor = TimeSeriesPredictor(prediction_length=2, known_covariates_names=["promotion", "price"]).fit(data)
        >>> print(future_known_covariates)
                            promotion  price
        item_id timestamp
        A       2020-01-08          1   12.9
                2020-01-09          1   12.9
        B       2020-03-04          0    5.0
                2020-03-05          0    7.0
        >>> predictor.predict(data, known_covariates=future_known_covariates)
                              mean
        item_id timestamp
        A       2020-01-08    30.2
                2020-01-09    27.0
        B       2020-03-04    17.1
                2020-03-05     8.3
        """
        if random_seed is not None:
            set_random_seed(random_seed)
        data = self._check_and_prepare_data_frame(data)
        return self._learner.predict(data, known_covariates=known_covariates, model=model)

    def evaluate(self, data: Union[TimeSeriesDataFrame, pd.DataFrame], **kwargs):
        """Evaluate the performance for given dataset, computing the score determined by ``self.eval_metric``
        on the given data set, and with the same ``prediction_length`` used when training models.

        Parameters
        ----------
        data : Union[TimeSeriesDataFrame, pd.DataFrame]
            The data to evaluate the best model on. The last ``prediction_length`` time steps of the data set, for each
            item, will be held out for prediction and forecast accuracy will be calculated on these time steps.

            If ``known_covariates_names`` were specified when creating the predictor, ``data`` must include the columns
            listed in ``known_covariates_names`` with the covariates values aligned with the target time series.

            If ``train_data`` used to train the predictor contained past covariates or static features, then ``data``
            must also include them (with same column names and dtypes).

            If provided data is an instance of pandas DataFrame, AutoGluon will attempt to automatically convert it
            to a ``TimeSeriesDataFrame``.

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

    def score(self, data: Union[TimeSeriesDataFrame, pd.DataFrame], **kwargs):
        """See, :meth:`~autogluon.timeseries.TimeSeriesPredictor.evaluate`."""
        return self.evaluate(data, **kwargs)

    @classmethod
    def _load_version_file(cls, path: str) -> str:
        version_file_path = path + cls._predictor_version_file_name
        version = load_str.load(path=version_file_path)
        return version

    @classmethod
    def load(cls, path: str, require_version_match: bool = True) -> "TimeSeriesPredictor":
        """Load an existing ``TimeSeriesPredictor`` from given ``path``.

        Parameters
        ----------
        path : str
            Path where the predictor was saved via :meth:`~autogluon.timeseries.TimeSeriesPredictor.save`.
        require_version_match : bool, default = True
            If True, will raise an AssertionError if the ``autogluon.timeseries`` version of the loaded predictor does
            not match the installed version of ``autogluon.timeseries``.
            If False, will allow loading of models trained on incompatible versions, but is NOT recommended. Users may
            run into numerous issues if attempting this.

        Returns
        -------
        predictor : TimeSeriesPredictor
        """
        if not path:
            raise ValueError("`path` cannot be None or empty in load().")
        path = setup_outputdir(path, warn_if_exist=False)

        try:
            version_saved = cls._load_version_file(path=path)
        except:
            logger.warning(
                f'WARNING: Could not find version file at "{path + cls._predictor_version_file_name}".\n'
                f"This means that the predictor was fit in a version `<=0.7.0`."
            )
            version_saved = "Unknown (Likely <=0.7.0)"

        check_saved_predictor_version(
            version_current=current_ag_version,
            version_saved=version_saved,
            require_version_match=require_version_match,
            logger=logger,
        )

        logger.info(f"Loading predictor from path {path}")
        learner = AbstractLearner.load(path)
        predictor = load_pkl.load(path=learner.path + cls.predictor_file_name)
        predictor._learner = learner
        predictor.path = learner.path
        return predictor

    def _save_version_file(self):
        version_file_contents = current_ag_version
        version_file_path = self.path + self._predictor_version_file_name
        save_str.save(path=version_file_path, data=version_file_contents, verbose=False)

    def save(self) -> None:
        """Save this predictor to file in directory specified by this Predictor's ``path``.

        Note that :meth:`~autogluon.timeseries.TimeSeriesPredictor.fit` already saves the predictor object automatically
        (we do not recommend modifying the Predictor object yourself as it tracks many trained models).
        """
        tmp_learner = self._learner
        self._learner = None
        save_pkl.save(path=tmp_learner.path + self.predictor_file_name, object=self)
        self._learner = tmp_learner
        self._save_version_file()

    def info(self) -> Dict[str, Any]:
        """Returns a dictionary of objects each describing an attribute of the training process and trained models."""
        return self._learner.get_info(include_model_info=True)

    def get_model_best(self) -> str:
        """Returns the name of the best model from trainer."""
        if self._trainer.model_best is not None:
            models = self._trainer.get_model_names()
            if self._trainer.model_best in models:
                return self._trainer.model_best
        return self._trainer.get_model_best()

    def leaderboard(
        self, data: Optional[Union[TimeSeriesDataFrame, pd.DataFrame]] = None, silent=False
    ) -> pd.DataFrame:
        """Return a leaderboard showing the performance of every trained model, the output is a
        pandas data frame with columns:

        * ``model``: The name of the model.
        * ``score_test``: The test score of the model on ``data``, if provided. Computed according to ``eval_metric``.
        * ``score_val``: The validation score of the model using the internal validation data. Computed according to ``eval_metric``.

            **NOTE:** Metrics scores are always shown in 'higher is better' format.
            This means that metrics such as MASE or MAPE will be multiplied by -1, so their values will be negative.
            This is necessary to avoid the user needing to know the metric to understand if higher is better when
            looking at leaderboard.

        * ``pred_time_val``: Time taken by the model to predict on the validation data set
        * ``fit_time_marginal``: The fit time required to train the model (ignoring base models for ensembles).
        * ``fit_order``: The order in which models were fit. The first model fit has ``fit_order=1``, and the Nth
          model fit has ``fit_order=N``.

        Parameters
        ----------
        data : Union[TimeSeriesDataFrame, pd.DataFrame], optional
            dataset used for additional evaluation. If not provided, the validation set used during training will be
            used.

            If ``known_covariates_names`` were specified when creating the predictor, ``data`` must include the columns
            listed in ``known_covariates_names`` with the covariates values aligned with the target time series.

            If ``train_data`` used to train the predictor contained past covariates or static features, then ``data``
            must also include them (with same column names and dtypes).

            If provided data is an instance of pandas DataFrame, AutoGluon will attempt to automatically convert it
            to a ``TimeSeriesDataFrame``.

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
            "model_performance": self._trainer.get_models_attribute_dict("val_score"),
            "model_best": self._trainer.get_model_best(),  # the name of the best model (on validation data)
            "model_paths": self._trainer.get_models_attribute_dict("path"),
            "model_fit_times": self._trainer.get_models_attribute_dict("fit_time"),
            "model_pred_times": self._trainer.get_models_attribute_dict("predict_time"),
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

    def refit_full(self, model: str = "all", set_best_to_refit_full: bool = True) -> Dict[str, str]:
        """Retrain model on all of the data (training + validation).

        This method can only be used if no ``tuning_data`` was passed to :meth:`~autogluon.timeseries.TimeSeriesPredictor.fit`.

        .. warning::
            This is experimental functionality, many time series models do not yet support ``refit_full`` and will
            simply be copied.


        Parameters
        ----------
        model : str, default = "all"
            Name of the model to refit.
            All ancestor models will also be refit in the case that the selected model is a weighted ensemble.
            Valid models are listed in this ``predictor`` by calling :meth:`~autogluon.timeseries.TimeSeriesPredictor.get_model_names`.

            * If "all" then all models are refitted.
            * If "best" then the model with the highest validation score is refit.

        set_best_to_refit_full : bool, default = True
            If True, sets best model to the refit_full version of the prior best model. This means the model used when
            ``predictor.predict(data)`` is called will be the refit_full version instead of the original version of the
            model. Has no effect if ``model`` is not the best model.
        """
        logger.warning(
            "\tWARNING: refit_full functionality for TimeSeriesPredictor is experimental "
            "and is not yet supported by all models."
        )

        logger.info(
            "Refitting models via `refit_full` using all of the data (combined train and validation)...\n"
            "\tModels trained in this way will have the suffix '_FULL' and have NaN validation score.\n"
            "\tThis process is not bound by time_limit, but should take less time than the original `fit` call."
        )
        model_best = self.get_model_best()
        refit_full_dict = self._learner.refit_full(model=model)

        if set_best_to_refit_full:
            if model_best in refit_full_dict:
                self._trainer.model_best = refit_full_dict[model_best]
                self._trainer.save()
                logger.info(
                    f"Updated best model to '{self._trainer.model_best}' (Previously '{model_best}'). "
                    f"AutoGluon will default to using '{self._trainer.model_best}' for predict()."
                )
            elif model_best in refit_full_dict.values():
                # Model best is already a refit full model
                prev_best = self._trainer.model_best
                self._trainer.model_best = model_best
                self._trainer.save()
                logger.info(
                    f"Updated best model to '{self._trainer.model_best}' (Previously '{prev_best}'). "
                    f"AutoGluon will default to using '{self._trainer.model_best}' for predict()."
                )
            else:
                logger.warning(
                    f"Best model ('{model_best}') is not present in refit_full dictionary. "
                    f"Training may have failed on the refit model. AutoGluon will default to using '{model_best}' for predict()."
                )
        return refit_full_dict

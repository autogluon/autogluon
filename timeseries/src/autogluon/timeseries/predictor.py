import json
import logging
import os
import pprint
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pandas as pd

from autogluon.common.utils.deprecated_utils import Deprecated
from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.common.utils.system_info import get_ag_system_info
from autogluon.common.utils.utils import check_saved_predictor_version, seed_everything, setup_outputdir
from autogluon.core.utils.decorators import apply_presets
from autogluon.core.utils.loaders import load_pkl, load_str
from autogluon.core.utils.savers import save_pkl, save_str
from autogluon.timeseries import __version__ as current_ag_version
from autogluon.timeseries.configs import TIMESERIES_PRESETS_CONFIGS
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TimeSeriesDataFrame
from autogluon.timeseries.learner import AbstractLearner, TimeSeriesLearner
from autogluon.timeseries.metrics import TimeSeriesScorer, check_get_evaluation_metric
from autogluon.timeseries.splitter import ExpandingWindowSplitter
from autogluon.timeseries.trainer import AbstractTimeSeriesTrainer

logger = logging.getLogger(__name__)


class TimeSeriesPredictor:
    """AutoGluon ``TimeSeriesPredictor`` predicts future values of multiple related time series.

    ``TimeSeriesPredictor`` provides probabilistic (quantile) multi-step-ahead forecasts for univariate time series.
    The forecast includes both the mean (i.e., conditional expectation of future values given the past), as well as the
    quantiles of the forecast distribution, indicating the range of possible future outcomes.

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
    freq : str, optional
        Frequency of the time series data (see `pandas documentation <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
        for available frequencies). For example, ``"D"`` for daily data or ``"H"`` for hourly data.

        By default, the predictor will attempt to automatically infer the frequency from the data. This argument should
        only be set in two cases:

        1. The time series data has irregular timestamps, so frequency cannot be inferred automatically.
        2. You would like to resample the original data at a different frequency (for example, convert hourly measurements into daily measurements).

        If ``freq`` is provided when creating the predictor, all data passed to the predictor will be automatically
        resampled at this frequency.
    eval_metric : Union[str, TimeSeriesScorer], default = "WQL"
        Metric by which predictions will be ultimately evaluated on future test data. AutoGluon tunes hyperparameters
        in order to improve this metric on validation data, and ranks models (on validation data) according to this
        metric.

        Probabilistic forecast metrics (evaluated on quantile forecasts for the specified ``quantile_levels``):

        - ``"SQL"``: scaled quantile loss, defined as average of quantile losses divided by the in-sample seasonal error
        - ``"WQL"``: mean weighted quantile loss, defined as average of quantile losses divided by the sum of absolute time series values in the forecast horizon

        Point forecast metrics (these are always evaluated on the ``"mean"`` column of the predictions):

        - ``"MAE"``: mean absolute error
        - ``"MAPE"``: mean absolute percentage error
        - ``"MASE"``: mean absolute scaled error
        - ``"MSE"``: mean squared error
        - ``"RMSE"``: root mean squared error
        - ``"RMSSE"``: root mean squared scaled error
        - ``"SMAPE"``: "symmetric" mean absolute percentage error
        - ``"WAPE"``: weighted absolute percentage error

        For more information about these metrics, see https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-metrics.html.
    eval_metric_seasonal_period : int, optional
        Seasonal period used to compute some evaluation metrics such as mean absolute scaled error (MASE). Defaults to
        ``None``, in which case the seasonal period is computed based on the data frequency.
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
    path : str or pathlib.Path, optional
        Path to the directory where models and intermediate outputs will be saved. Defaults to a timestamped folder
        ``AutogluonModels/ag-[TIMESTAMP]`` that will be created in the working directory.
    verbosity : int, default = 2
        Verbosity levels range from 0 to 4 and control how much information is printed to stdout. Higher levels
        correspond to more detailed print statements, and ``verbosity=0`` suppresses output including warnings.
        If using ``logging``, you can alternatively control amount of information printed via ``logger.setLevel(L)``,
        where ``L`` ranges from 0 to 50 (Note: higher values of ``L`` correspond to fewer print statements, opposite
        of verbosity levels).
    cache_predictions : bool, default = True
        If True, the predictor will cache and reuse the predictions made by individual models whenever
        :meth:`~autogluon.timeseries.TimeSeriesPredictor.predict`, :meth:`~autogluon.timeseries.TimeSeriesPredictor.leaderboard`,
        or :meth:`~autogluon.timeseries.TimeSeriesPredictor.evaluate` methods are called. This allows to significantly
        speed up these methods. If False, caching will be disabled. You can set this argument to False to reduce disk
        usage at the cost of longer prediction times.
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
        freq: str = None,
        eval_metric: Union[str, TimeSeriesScorer, None] = None,
        eval_metric_seasonal_period: Optional[int] = None,
        path: Optional[Union[str, Path]] = None,
        verbosity: int = 2,
        quantile_levels: Optional[List[float]] = None,
        cache_predictions: bool = True,
        learner_type: Optional[Type[AbstractLearner]] = None,
        learner_kwargs: Optional[dict] = None,
        label: Optional[str] = None,
        **kwargs,
    ):
        self.verbosity = verbosity
        set_logger_verbosity(self.verbosity, logger=logger)
        self.path = setup_outputdir(path)

        self.cache_predictions = cache_predictions
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
        self.known_covariates_names = list(known_covariates_names)

        self.prediction_length = prediction_length
        # For each validation fold, all time series in training set must have length >= _min_train_length
        self._min_train_length = max(self.prediction_length + 1, 5)
        self.freq = freq
        if self.freq is not None:
            # Standardize frequency string (e.g., "min" -> "T", "Y" -> "A-DEC")
            std_freq = pd.tseries.frequencies.to_offset(self.freq).freqstr
            if std_freq != str(self.freq):
                logger.info(f"Frequency '{self.freq}' stored as '{std_freq}'")
            self.freq = std_freq
        self.eval_metric = check_get_evaluation_metric(eval_metric)
        self.eval_metric_seasonal_period = eval_metric_seasonal_period
        if quantile_levels is None:
            quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.quantile_levels = sorted(quantile_levels)

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
                cache_predictions=self.cache_predictions,
            )
        )
        # Using `TimeSeriesLearner` as default argument breaks doc generation with Sphnix
        if learner_type is None:
            learner_type = TimeSeriesLearner
        self._learner: AbstractLearner = learner_type(**learner_kwargs)
        self._learner_type = type(self._learner)

        if "ignore_time_index" in kwargs:
            raise TypeError(
                "`ignore_time_index` argument to TimeSeriesPredictor.__init__() has been deprecated.\n"
                "If your data has irregular timestamps, please either 1) specify the desired regular frequency when "
                "creating the predictor as `TimeSeriesPredictor(freq=...)` or 2) manually convert timestamps to "
                "regular frequency with `data.convert_frequency(freq=...)`."
            )
        if len(kwargs) > 0:
            for key in kwargs:
                raise TypeError(f"TimeSeriesPredictor.__init__() got an unexpected keyword argument '{key}'")

    @property
    def _trainer(self) -> AbstractTimeSeriesTrainer:
        return self._learner.load_trainer()  # noqa

    def _to_data_frame(
        self,
        data: Union[TimeSeriesDataFrame, pd.DataFrame, str],
        name: str = "data",
    ) -> "TimeSeriesDataFrame":
        if isinstance(data, TimeSeriesDataFrame):
            return data
        elif isinstance(data, (pd.DataFrame, str)):
            try:
                data = TimeSeriesDataFrame(data)
            except:
                raise ValueError(
                    f"Provided {name} of type {type(data)} cannot be automatically converted to a TimeSeriesDataFrame."
                )
            return data
        else:
            raise TypeError(
                f"{name} must be a TimeSeriesDataFrame or pandas.DataFrame or string (path to data) "
                f"but received an object of type {type(data)}."
            )

    def _check_and_prepare_data_frame(
        self,
        data: Union[TimeSeriesDataFrame, pd.DataFrame, str],
        name: str = "data",
    ) -> TimeSeriesDataFrame:
        """Ensure that TimeSeriesDataFrame has a sorted index, valid frequency, and contains no missing values.

        If self.freq is None, then self.freq of the predictor will be set to the frequency of the data.

        Parameters
        ----------
        data : Union[TimeSeriesDataFrame, pd.DataFrame, str]
            Data as a data frame or path to file storing the data.
        name : str
            Name of the data that will be used in log messages (e.g., 'train_data', 'tuning_data', or 'data').

        Returns
        -------
        df : TimeSeriesDataFrame
            Preprocessed data in TimeSeriesDataFrame format.
        """
        df = self._to_data_frame(data, name=name)
        df = df.astype({self.target: float})
        # MultiIndex.is_monotonic_increasing checks if index is sorted by ["item_id", "timestamp"]
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
            df._cached_freq = None  # in case frequency was incorrectly cached as IRREGULAR_TIME_INDEX_FREQSTR

        # Ensure that data has a regular frequency that matches the predictor frequency
        if self.freq is None:
            if df.freq is None:
                raise ValueError(
                    f"Frequency of {name} is not provided and cannot be inferred. Please set the expected data "
                    f"frequency when creating the predictor with `TimeSeriesPredictor(freq=...)` or ensure that "
                    f"the data has a regular time index with `{name}.convert_frequency(freq=...)`"
                )
            else:
                self.freq = df.freq
                logger.info(f"Inferred time series frequency: '{df.freq}'")
        else:
            if df.freq != self.freq:
                logger.warning(f"{name} with frequency '{df.freq}' has been resampled to frequency '{self.freq}'.")
                df = df.convert_frequency(freq=self.freq)

        # Fill missing values
        if df.isna().values.any():
            # FIXME: Do not automatically fill NaNs here, handle missing values at the level of individual models.
            # FIXME: Current solution leads to incorrect metric computation if missing values are present
            logger.warning(
                f"{name} contains missing values represented by NaN. "
                f"They have been filled by carrying forward the last valid observation."
            )
            df = df.fill_missing_values()
            if df.isna().values.any():
                raise ValueError(f"Some time series in {name} consist completely of NaN values. Please remove them.")
        return df

    def _check_data_for_evaluation(self, data: TimeSeriesDataFrame, name: str = "data"):
        """Make sure that provided evaluation data includes both historic and future time series values."""
        if data.num_timesteps_per_item().min() <= self.prediction_length:
            raise ValueError(
                f"Cannot reserve last prediction_length={self.prediction_length} time steps for evaluation in some "
                f"time series in {name}. Please make sure that {name} includes both historic and future data, and that"
                f"all time series have length > prediction_length (at least {self.prediction_length + 1})"
            )

    @staticmethod
    def _get_dataset_stats(data: TimeSeriesDataFrame) -> str:
        ts_lengths = data.num_timesteps_per_item()
        median_length = int(ts_lengths.median())
        min_length = ts_lengths.min()
        max_length = ts_lengths.max()
        return (
            f"{len(data)} rows, {data.num_items} time series. "
            f"Median time series length is {median_length} (min={min_length}, max={max_length}). "
        )

    def _reduce_num_val_windows_if_necessary(
        self,
        train_data: TimeSeriesDataFrame,
        original_num_val_windows: int,
        val_step_size: int,
    ) -> int:
        """Adjust num_val_windows based on the length of time series in train_data.

        Chooses num_val_windows such that TS with median length is long enough to perform num_val_windows validations
        (at least 1, at most `original_num_val_windows`).

        In other words, find largest `num_val_windows` that satisfies
        median_length >= min_train_length + prediction_length + (num_val_windows - 1) * val_step_size
        """
        median_length = train_data.num_timesteps_per_item().median()
        num_val_windows_for_median_ts = int(
            (median_length - self._min_train_length - self.prediction_length) // val_step_size + 1
        )
        new_num_val_windows = min(original_num_val_windows, max(1, num_val_windows_for_median_ts))
        if new_num_val_windows < original_num_val_windows:
            logger.warning(
                f"Time series in train_data are too short for chosen num_val_windows={original_num_val_windows}. "
                f"Reducing num_val_windows to {new_num_val_windows}."
            )
        return new_num_val_windows

    def _filter_short_series(
        self,
        train_data: TimeSeriesDataFrame,
        num_val_windows: int,
        val_step_size: int,
    ) -> Tuple[TimeSeriesDataFrame, Optional[TimeSeriesDataFrame]]:
        """Remove time series from train_data that are too short for chosen prediction_length and validation settings.

        This method ensures that for each validation fold, all train series have length >= max(prediction_length + 1, 5).

        In other words, this method removes from train_data all time series with length less than
        min_train_length + prediction_length + (num_val_windows - 1) * val_step_size
        """
        min_length = self._min_train_length + self.prediction_length + (num_val_windows - 1) * val_step_size

        train_lengths = train_data.num_timesteps_per_item()
        train_items_to_drop = train_lengths.index[train_lengths < min_length]
        if len(train_items_to_drop) > 0:
            logger.info(
                f"\tRemoving {len(train_items_to_drop)} short time series from train_data. Only series with length "
                f">= {min_length} will be used for training."
            )
            filtered_train_data = train_data.query("item_id not in @train_items_to_drop")
            if len(filtered_train_data) == 0:
                raise ValueError(
                    f"At least some time series in train_data must have length >= {min_length}. Please provide longer "
                    f"time series as train_data or reduce prediction_length, num_val_windows, or val_step_size."
                )
            logger.info(
                f"\tAfter removing short series, train_data has {self._get_dataset_stats(filtered_train_data)}"
            )
        else:
            filtered_train_data = train_data

        return filtered_train_data

    @apply_presets(TIMESERIES_PRESETS_CONFIGS)
    def fit(
        self,
        train_data: Union[TimeSeriesDataFrame, pd.DataFrame, str],
        tuning_data: Optional[Union[TimeSeriesDataFrame, pd.DataFrame, str]] = None,
        time_limit: Optional[int] = None,
        presets: Optional[str] = None,
        hyperparameters: Dict[Union[str, Type], Any] = None,
        hyperparameter_tune_kwargs: Optional[Union[str, Dict]] = None,
        excluded_model_types: Optional[List[str]] = None,
        num_val_windows: int = 1,
        val_step_size: Optional[int] = None,
        refit_every_n_windows: int = 1,
        refit_full: bool = False,
        enable_ensemble: bool = True,
        random_seed: Optional[int] = 123,
        verbosity: Optional[int] = None,
    ) -> "TimeSeriesPredictor":
        """Fit probabilistic forecasting models to the given time series dataset.

        Parameters
        ----------
        train_data : Union[TimeSeriesDataFrame, pd.DataFrame, str]
            Training data in the :class:`~autogluon.timeseries.TimeSeriesDataFrame` format.

            Time series with length ``<= (num_val_windows + 1) * prediction_length`` will be ignored during training.
            See :attr:`num_val_windows` for details.

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

        tuning_data : Union[TimeSeriesDataFrame, pd.DataFrame, str], optional
            Data reserved for model selection and hyperparameter tuning, rather than training individual models. Also
            used to compute the validation scores. Note that only the last ``prediction_length`` time steps of each
            time series are used for computing the validation score.

            If ``tuning_data`` is provided, multi-window backtesting on training data will be disabled, the
            :attr:`num_val_windows` will be set to ``0``, and :attr:`refit_full` will be set to ``False``.

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
            to get a high-accuracy predictor, or set ``presets="fast_training"`` to quickly get the results.
            Any user-specified arguments in :meth:`~autogluon.timeseries.TimeSeriesPredictor.fit` will
            override the values used by presets.

            Available presets:

            - ``"fast_training"``: fit simple statistical models (``ETS``, ``Theta``, ``Naive``, ``SeasonalNaive``) + fast tree-based models ``RecursiveTabular`` and ``DirectTabular``. These models are fast to train but may not be very accurate.
            - ``"medium_quality"``: all models mentioned above + deep learning model ``TemporalFusionTransformer``. Default setting that produces good forecasts with reasonable training time.
            - ``"high_quality"``: All ML models available in AutoGluon + additional statistical models (``NPTS``, ``AutoETS``, ``AutoARIMA``, ``CrostonSBA``, ``DynamicOptimizedTheta``). Much more accurate than ``medium_quality``, but takes longer to train.
            - ``"best_quality"``: Same models as in ``"high_quality"`, but performs validation with multiple backtests. Usually better than ``high_quality``, but takes even longer to train.

            Details for these presets can be found in ``autogluon/timeseries/configs/presets_configs.py``. If not
            provided, user-provided values for ``hyperparameters`` and ``hyperparameter_tune_kwargs`` will be used
            (defaulting to their default values specified below).
        hyperparameters : str or dict, optional
            Determines what models are trained and what hyperparameters are used by each model.

            If str is passed, will use a preset hyperparameter configuration defined in
            ``autogluon/timeseries/trainer/models/presets.py``. Supported values are ``"default"``, ``"light"`` and
            ``"very_light"``.

            If dict is provided, the keys are strings or types that indicate which models to train. Each value is
            itself a dict containing hyperparameters for each of the trained models, or a list of such dicts. Any
            omitted hyperparameters not specified here will be set to default. For example::

                predictor.fit(
                    ...
                    hyperparameters={
                        "DeepAR": {},
                        "Theta": [
                            {"decomposition_type": "additive"},
                            {"seasonal_period": 1},
                        ],
                    }
                )

            The above example will train three models:

            * ``DeepAR`` with default hyperparameters
            * ``Theta`` with additive seasonal decomposition (all other parameters set to their defaults)
            * ``Theta`` with seasonality disabled (all other parameters set to their defaults)

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
            Hyperparameter tuning strategy and kwargs (for example, how many HPO trials to run).
            If None, then hyperparameter tuning will not be performed.

            If type is ``str``, then this argument specifies a preset.
            Valid preset values:

            * "auto": Performs HPO via bayesian optimization search on GluonTS-backed neural forecasting models and
                random search on other models using local scheduler.
            * "random": Performs HPO via random search.

            You can also provide a dict to specify searchers and schedulers
            Valid keys:

            * "num_trials": How many HPO trials to run
            * "scheduler": Which scheduler to use. Valid values:
                * "local": Local shceduler that schedules trials FIFO
            * "searcher": Which searching algorithm to use. Valid values:
                * "local_random": Uses the "random" searcher
                * "random": Perform random search
                * "bayes": Perform HPO with HyperOpt on GluonTS-backed models via Ray tune. Perform random search on other models.
                * "auto": alias for "bayes"

            The "scheduler" and "searcher" key are required when providing a dict.

            Example::

                predictor.fit(
                    ...
                    hyperparameter_tune_kwargs={
                        "num_trials": 5,
                        "searcher": "auto",
                        "scheduler": "local",
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
            If ``num_val_windows > 1`` is provided, this value may be automatically reduced to ensure that the majority
            of time series in ``train_data`` are long enough for the chosen number of backtests.

            Increasing this parameter increases the training time roughly by a factor of ``num_val_windows // refit_every_n_windows``.
            See :attr:`refit_every_n_windows` and :attr:`val_step_size`: for details.

            For example, for ``prediction_length=2``, ``num_val_windows=3`` and ``val_step_size=1`` the folds are::

                |-------------------|
                | x x x x x y y - - |
                | x x x x x x y y - |
                | x x x x x x x y y |

            where ``x`` are the train time steps and ``y`` are the validation time steps.

            This argument has no effect if ``tuning_data`` is provided.
        val_step_size : int or None, default = None
            Step size between consecutive validation windows. If set to ``None``, defaults to ``prediction_length``
            provided when creating the predictor.

            This argument has no effect if ``tuning_data`` is provided.
        refit_every_n_windows: int or None, default = 1
            When performing cross validation, each model will be retrained every ``refit_every_n_windows`` validation
            windows. If set to ``None``, model will only be fit once for the first validation window.
        refit_full : bool, default = False
            If True, after training is complete, AutoGluon will attempt to re-train all models using all of training
            data (including the data initially reserved for validation). This argument has no effect if ``tuning_data``
            is provided.
        enable_ensemble : bool, default = True
            If True, the ``TimeSeriesPredictor`` will fit a simple weighted ensemble on top of the models specified via
            ``hyperparameters``.
        random_seed : int or None, default = 123
            If provided, fixes the seed of the random number generator for all models. This guarantees reproducible
            results for most models (except those trained on GPU because of the non-determinism of GPU operations).
        verbosity : int, optional
            If provided, overrides the ``verbosity`` value used when creating the ``TimeSeriesPredictor``. See
            documentation for :class:`~autogluon.timeseries.TimeSeriesPredictor` for more details.

        """
        time_start = time.time()
        if self._learner.is_fit:
            raise AssertionError("Predictor is already fit! To fit additional models create a new `Predictor`.")

        logger.info("Beginning AutoGluon training..." + (f" Time limit = {time_limit}s" if time_limit else ""))
        logger.info(f"AutoGluon will save models to '{self.path}'")
        logger.info(get_ag_system_info(path=self.path, include_gpu_count=True))

        if hyperparameters is None:
            hyperparameters = "default"

        if verbosity is None:
            verbosity = self.verbosity
        set_logger_verbosity(verbosity)

        fit_args = dict(
            prediction_length=self.prediction_length,
            target=self.target,
            known_covariates_names=self.known_covariates_names,
            eval_metric=self.eval_metric,
            eval_metric_seasonal_period=self.eval_metric_seasonal_period,
            quantile_levels=self.quantile_levels,
            freq=self.freq,
            time_limit=time_limit,
            hyperparameters=hyperparameters,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            excluded_model_types=excluded_model_types,
            num_val_windows=num_val_windows,
            val_step_size=val_step_size,
            refit_every_n_windows=refit_every_n_windows,
            refit_full=refit_full,
            enable_ensemble=enable_ensemble,
            random_seed=random_seed,
            verbosity=verbosity,
        )
        if presets is not None:
            logger.info(f"Setting presets to: {presets}")
        logger.info("\nFitting with arguments:")
        logger.info(f"{pprint.pformat({k: v for k, v in fit_args.items() if v is not None})}\n")

        train_data = self._check_and_prepare_data_frame(train_data, name="train_data")
        logger.info(f"Provided train_data has {self._get_dataset_stats(train_data)}")

        if val_step_size is None:
            val_step_size = self.prediction_length

        if num_val_windows > 0:
            num_val_windows = self._reduce_num_val_windows_if_necessary(
                train_data, original_num_val_windows=num_val_windows, val_step_size=val_step_size
            )

        if tuning_data is not None:
            tuning_data = self._check_and_prepare_data_frame(tuning_data, name="tuning_data")
            self._check_data_for_evaluation(tuning_data, name="tuning_data")
            logger.info(f"Provided tuning_data has {self._get_dataset_stats(train_data)}")
            # TODO: Use num_val_windows to perform multi-window backtests on tuning_data
            if num_val_windows > 0:
                logger.warning(
                    "\tSetting num_val_windows = 0 (disabling backtesting on train_data) because tuning_data is provided."
                )
                num_val_windows = 0

        if num_val_windows == 0 and tuning_data is None:
            raise ValueError("Please set num_val_windows >= 1 or provide custom tuning_data")

        train_data = self._filter_short_series(
            train_data, num_val_windows=num_val_windows, val_step_size=val_step_size
        )

        val_splitter = ExpandingWindowSplitter(
            prediction_length=self.prediction_length, num_val_windows=num_val_windows, val_step_size=val_step_size
        )

        if random_seed is not None:
            seed_everything(random_seed)

        time_left = None if time_limit is None else time_limit - (time.time() - time_start)
        self._learner.fit(
            train_data=train_data,
            val_data=tuning_data,
            hyperparameters=hyperparameters,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            excluded_model_types=excluded_model_types,
            time_limit=time_left,
            verbosity=verbosity,
            val_splitter=val_splitter,
            refit_every_n_windows=refit_every_n_windows,
            enable_ensemble=enable_ensemble,
        )
        if refit_full:
            if tuning_data is None:
                self.refit_full()
            else:
                logger.warning("Skipping `refit_full` because custom `tuning_data` was provided during `fit`.")

        self.save()
        return self

    def model_names(self) -> List[str]:
        """Returns the list of model names trained by this predictor object."""
        return self._trainer.get_model_names()

    def predict(
        self,
        data: Union[TimeSeriesDataFrame, pd.DataFrame, str],
        known_covariates: Optional[Union[TimeSeriesDataFrame, pd.DataFrame, str]] = None,
        model: Optional[str] = None,
        use_cache: bool = True,
        random_seed: Optional[int] = 123,
    ) -> TimeSeriesDataFrame:
        """Return quantile and mean forecasts for the given dataset, starting from the end of each time series.

        Parameters
        ----------
        data : Union[TimeSeriesDataFrame, pd.DataFrame, str]
            Time series data to forecast with.

            If ``known_covariates_names`` were specified when creating the predictor, ``data`` must include the columns
            listed in ``known_covariates_names`` with the covariates values aligned with the target time series.

            If ``train_data`` used to train the predictor contained past covariates or static features, then ``data``
            must also include them (with same column names and dtypes).

            If provided data is an instance of pandas DataFrame, AutoGluon will attempt to automatically convert it
            to a ``TimeSeriesDataFrame``.
        known_covariates : Union[TimeSeriesDataFrame, pd.DataFrame, str], optional
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
        use_cache : bool, default = True
            If True, will attempt to use the cached predictions. If False, cached predictions will be ignored.
            This argument is ignored if ``cache_predictions`` was set to False when creating the ``TimeSeriesPredictor``.


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
            seed_everything(random_seed)
        # Don't use data.item_ids in case data is not a TimeSeriesDataFrame
        original_item_id_order = data.reset_index()[ITEMID].unique()
        data = self._check_and_prepare_data_frame(data)
        if known_covariates is not None:
            known_covariates = self._to_data_frame(known_covariates)
        predictions = self._learner.predict(data, known_covariates=known_covariates, model=model, use_cache=use_cache)
        return predictions.reindex(original_item_id_order, level=ITEMID)

    def evaluate(
        self,
        data: Union[TimeSeriesDataFrame, pd.DataFrame, str],
        model: Optional[str] = None,
        metrics: Optional[Union[str, TimeSeriesScorer, List[Union[str, TimeSeriesScorer]]]] = None,
        display: bool = False,
        use_cache: bool = True,
    ) -> Dict[str, float]:
        """Evaluate the forecast accuracy for given dataset.

        This method measures the forecast accuracy using the last ``self.prediction_length`` time steps of each time
        series in ``data`` as a hold-out set.

        Parameters
        ----------
        data : Union[TimeSeriesDataFrame, pd.DataFrame, str]
            The data to evaluate the best model on. The last ``prediction_length`` time steps of the data set, for each
            item, will be held out for prediction and forecast accuracy will be calculated on these time steps.

            If ``known_covariates_names`` were specified when creating the predictor, ``data`` must include the columns
            listed in ``known_covariates_names`` with the covariates values aligned with the target time series.

            If ``train_data`` used to train the predictor contained past covariates or static features, then ``data``
            must also include them (with same column names and dtypes).

            If provided data is an instance of pandas DataFrame, AutoGluon will attempt to automatically convert it
            to a ``TimeSeriesDataFrame``.
        model : str, optional
            Name of the model that you would like to evaluate. By default, the best model during training
            (with highest validation score) will be used.
        metrics : str, TimeSeriesScorer or List[Union[str, TimeSeriesScorer]], optional
            Metric or a list of metrics to compute scores with. Defaults to ``self.eval_metric``. Supports both
            metric names as strings and custom metrics based on TimeSeriesScorer.
        display : bool, default = False
            If True, the scores will be printed.
        use_cache : bool, default = True
            If True, will attempt to use the cached predictions. If False, cached predictions will be ignored.
            This argument is ignored if ``cache_predictions`` was set to False when creating the ``TimeSeriesPredictor``.

        Returns
        -------
        scores_dict : Dict[str, float]
            Dictionary where keys = metrics, values = performance along each metric. For consistency, error metrics
            will have their signs flipped to obey this convention. For example, negative MAPE values will be reported.
            To get the ``eval_metric`` score, do ``output[predictor.eval_metric.name]``.
        """
        data = self._check_and_prepare_data_frame(data)
        self._check_data_for_evaluation(data)
        scores_dict = self._learner.evaluate(data, model=model, metrics=metrics, use_cache=use_cache)
        if display:
            logger.info("Evaluations on test data:")
            logger.info(json.dumps(scores_dict, indent=4))
        return scores_dict

    @classmethod
    def _load_version_file(cls, path: str) -> str:
        version_file_path = os.path.join(path, cls._predictor_version_file_name)
        version = load_str.load(path=version_file_path)
        return version

    @classmethod
    def load(cls, path: Union[str, Path], require_version_match: bool = True) -> "TimeSeriesPredictor":
        """Load an existing ``TimeSeriesPredictor`` from given ``path``.

        Parameters
        ----------
        path : str or pathlib.Path
            Path where the predictor was saved via :meth:`~autogluon.timeseries.TimeSeriesPredictor.save`.
        require_version_match : bool, default = True
            If True, will raise an AssertionError if the ``autogluon.timeseries`` version of the loaded predictor does
            not match the installed version of ``autogluon.timeseries``.
            If False, will allow loading of models trained on incompatible versions, but is NOT recommended. Users may
            run into numerous issues if attempting this.

        Returns
        -------
        predictor : TimeSeriesPredictor

        Examples
        --------
        >>> predictor = TimeSeriesPredictor.load(path_to_predictor)

        """
        if not path:
            raise ValueError("`path` cannot be None or empty in load().")
        path: str = setup_outputdir(path, warn_if_exist=False)

        try:
            version_saved = cls._load_version_file(path=path)
        except:
            logger.warning(
                f'WARNING: Could not find version file at "{os.path.join(path, cls._predictor_version_file_name)}".\n'
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
        predictor = load_pkl.load(path=os.path.join(learner.path, cls.predictor_file_name))
        predictor._learner = learner
        predictor.path = learner.path
        return predictor

    def _save_version_file(self):
        version_file_contents = current_ag_version
        version_file_path = os.path.join(self.path, self._predictor_version_file_name)
        save_str.save(path=version_file_path, data=version_file_contents, verbose=False)

    def save(self) -> None:
        """Save this predictor to file in directory specified by this Predictor's ``path``.

        Note that :meth:`~autogluon.timeseries.TimeSeriesPredictor.fit` already saves the predictor object automatically
        (we do not recommend modifying the Predictor object yourself as it tracks many trained models).
        """
        tmp_learner = self._learner
        self._learner = None
        save_pkl.save(path=os.path.join(tmp_learner.path, self.predictor_file_name), object=self)
        self._learner = tmp_learner
        self._save_version_file()

    def info(self) -> Dict[str, Any]:
        """Returns a dictionary of objects each describing an attribute of the training process and trained models."""
        return self._learner.get_info(include_model_info=True)

    def model_best(self) -> str:
        """Returns the name of the best model from trainer."""
        if self._trainer.model_best is not None:
            models = self._trainer.get_model_names()
            if self._trainer.model_best in models:
                return self._trainer.model_best
        return self._trainer.get_model_best()

    def leaderboard(
        self,
        data: Optional[Union[TimeSeriesDataFrame, pd.DataFrame, str]] = None,
        display: bool = False,
        use_cache: bool = True,
        **kwargs,
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
        data : Union[TimeSeriesDataFrame, pd.DataFrame, str], optional
            dataset used for additional evaluation. If not provided, the validation set used during training will be
            used.

            If ``known_covariates_names`` were specified when creating the predictor, ``data`` must include the columns
            listed in ``known_covariates_names`` with the covariates values aligned with the target time series.

            If ``train_data`` used to train the predictor contained past covariates or static features, then ``data``
            must also include them (with same column names and dtypes).

            If provided data is an instance of pandas DataFrame, AutoGluon will attempt to automatically convert it
            to a ``TimeSeriesDataFrame``.

        display : bool, default = False
            If True, the leaderboard DataFrame will be printed.
        use_cache : bool, default = True
            If True, will attempt to use the cached predictions. If False, cached predictions will be ignored.
            This argument is ignored if ``cache_predictions`` was set to False when creating the ``TimeSeriesPredictor``.

        Returns
        -------
        leaderboard : pandas.DataFrame
            The leaderboard containing information on all models and in order of best model to worst in terms of
            test performance.
        """
        if "silent" in kwargs:
            # keep `silent` logic for backwards compatibility
            assert isinstance(kwargs["silent"], bool)
            display = not kwargs.pop("silent")
        if len(kwargs) > 0:
            for key in kwargs:
                raise TypeError(f"TimeSeriesPredictor.leaderboard() got an unexpected keyword argument '{key}'")

        if data is not None:
            data = self._check_and_prepare_data_frame(data)
            self._check_data_for_evaluation(data)
        leaderboard = self._learner.leaderboard(data, use_cache=use_cache)
        if display:
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
        for model_name in self.model_names():
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
            Valid models are listed in this ``predictor`` by calling :meth:`~autogluon.timeseries.TimeSeriesPredictor.model_names`.

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
        model_best = self.model_best()
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

    def __dir__(self) -> List[str]:
        # This hides method from IPython autocomplete, but not VSCode autocomplete
        deprecated = ["score", "get_model_best", "get_model_names"]
        return [d for d in super().__dir__() if d not in deprecated]

    @Deprecated(min_version_to_warn="0.8.3", min_version_to_error="1.2", version_to_remove="1.2", new="evaluate")
    def score(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    @Deprecated(min_version_to_warn="0.8.3", min_version_to_error="1.2", version_to_remove="1.2", new="model_best")
    def get_model_best(self) -> str:
        return self.model_best()

    @Deprecated(min_version_to_warn="0.8.3", min_version_to_error="1.2", version_to_remove="1.2", new="model_names")
    def get_model_names(self) -> str:
        return self.model_names()

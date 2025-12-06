import json
import logging
import math
import os
import pprint
import time
from pathlib import Path
from typing import Any, Literal, Type, cast, overload

import numpy as np
import pandas as pd

from autogluon.common.utils.log_utils import (
    add_log_to_file,
    set_logger_verbosity,
    warn_if_mlflow_autologging_is_enabled,
)
from autogluon.common.utils.system_info import get_ag_system_info
from autogluon.common.utils.utils import check_saved_predictor_version, setup_outputdir
from autogluon.core.utils.decorators import apply_presets
from autogluon.core.utils.loaders import load_pkl, load_str
from autogluon.core.utils.savers import save_pkl, save_str
from autogluon.timeseries import __version__ as current_ag_version
from autogluon.timeseries.configs import get_predictor_presets
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.learner import TimeSeriesLearner
from autogluon.timeseries.metrics import TimeSeriesScorer, check_get_evaluation_metric
from autogluon.timeseries.trainer import TimeSeriesTrainer
from autogluon.timeseries.utils.forecast import make_future_data_frame

logger = logging.getLogger("autogluon.timeseries")


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
        for available frequencies). For example, ``"D"`` for daily data or ``"h"`` for hourly data.

        By default, the predictor will attempt to automatically infer the frequency from the data. This argument should
        only be set in two cases:

        1. The time series data has irregular timestamps, so frequency cannot be inferred automatically.
        2. You would like to resample the original data at a different frequency (for example, convert hourly measurements into daily measurements).

        If ``freq`` is provided when creating the predictor, all data passed to the predictor will be automatically
        resampled at this frequency.
    eval_metric : str | TimeSeriesScorer, default = "WQL"
        Metric by which predictions will be ultimately evaluated on future test data. AutoGluon tunes hyperparameters
        in order to improve this metric on validation data, and ranks models (on validation data) according to this
        metric.

        Probabilistic forecast metrics (evaluated on quantile forecasts for the specified ``quantile_levels``):

        - ``"SQL"``: scaled quantile loss
        - ``"WQL"``: weighted quantile loss

        Point forecast metrics (these are always evaluated on the ``"mean"`` column of the predictions):

        - ``"MAE"``: mean absolute error
        - ``"MAPE"``: mean absolute percentage error
        - ``"MASE"``: mean absolute scaled error
        - ``"MSE"``: mean squared error
        - ``"RMSE"``: root mean squared error
        - ``"RMSLE"``: root mean squared logarithmic error
        - ``"RMSSE"``: root mean squared scaled error
        - ``"SMAPE"``: "symmetric" mean absolute percentage error
        - ``"WAPE"``: weighted absolute percentage error

        For more information about these metrics, see :ref:`Forecasting Time Series - Evaluation Metrics <forecasting_metrics>`.
    eval_metric_seasonal_period : int, optional
        Seasonal period used to compute some evaluation metrics such as mean absolute scaled error (MASE). Defaults to
        ``None``, in which case the seasonal period is computed based on the data frequency.
    horizon_weight : list[float], optional
        Weight assigned to each time step in the forecast horizon when computing the ``eval_metric``. If provided, this
        must be a list with ``prediction_length`` non-negative values, where at least some values are greater than zero.
        AutoGluon will automatically normalize the weights so that they sum up to ``prediction_length``. By default, all
        time steps in the forecast horizon have the same weight, which is equivalent to setting ``horizon_weight = [1] * prediction_length``.

        This parameter only affects model selection and ensemble construction; it has no effect on the loss function of
        the individual forecasting models.
    known_covariates_names: list[str], optional
        Names of the covariates that are known in advance for all time steps in the forecast horizon. These are also
        known as dynamic features, exogenous variables, additional regressors or related time series. Examples of such
        covariates include holidays, promotions or weather forecasts.

        If ``known_covariates_names`` are provided, then:

        - :meth:`~autogluon.timeseries.TimeSeriesPredictor.fit`, :meth:`~autogluon.timeseries.TimeSeriesPredictor.evaluate`, and :meth:`~autogluon.timeseries.TimeSeriesPredictor.leaderboard` will expect a dataframe with columns listed in ``known_covariates_names`` (in addition to the ``target`` column).
        - :meth:`~autogluon.timeseries.TimeSeriesPredictor.predict` will expect an additional keyword argument ``known_covariates`` containing the future values of the known covariates in ``TimeSeriesDataFrame`` format.

    quantile_levels : list[float], optional
        List of increasing decimals that specifies which quantiles should be estimated when making distributional
        forecasts. Defaults to ``[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]``.
    path : str or pathlib.Path, optional
        Path to the local directory where models and intermediate outputs will be saved. Defaults to a timestamped folder
        ``AutogluonModels/ag-[TIMESTAMP]`` that will be created in the working directory.
    verbosity : int, default = 2
        Verbosity levels range from 0 to 4 and control how much information is printed to stdout. Higher levels
        correspond to more detailed print statements, and ``verbosity=0`` suppresses output including warnings.
        Verbosity 0 corresponds to Python's ERROR log level, where only error outputs will be logged. Verbosity 1 and 2
        will additionally log warnings and info outputs, respectively. Verbosity 4 enables all logging output including
        debug messages from AutoGluon and all logging in dependencies (GluonTS, PyTorch Lightning, AutoGluon-Tabular, etc.)
    log_to_file: bool, default = True
        Whether to save the logs into a file for later reference
    log_file_path: str | Path, default = "auto"
        File path to save the logs.
        If auto, logs will be saved under ``predictor_path/logs/predictor_log.txt``.
        Will be ignored if ``log_to_file`` is set to False
    cache_predictions : bool, default = True
        If True, the predictor will cache and reuse the predictions made by individual models whenever
        :meth:`~autogluon.timeseries.TimeSeriesPredictor.predict`, :meth:`~autogluon.timeseries.TimeSeriesPredictor.leaderboard`,
        or :meth:`~autogluon.timeseries.TimeSeriesPredictor.evaluate` methods are called. This allows to significantly
        speed up these methods. If False, caching will be disabled. You can set this argument to False to reduce disk
        usage at the cost of longer prediction times.
    label : str, optional
        Alias for :attr:`target`.
    """

    _learner_type = TimeSeriesLearner
    predictor_file_name = "predictor.pkl"
    _predictor_version_file_name = "version.txt"
    _predictor_log_file_name = "predictor_log.txt"

    def __init__(
        self,
        target: str | None = None,
        known_covariates_names: list[str] | None = None,
        prediction_length: int = 1,
        freq: str | None = None,
        eval_metric: str | TimeSeriesScorer | None = None,
        eval_metric_seasonal_period: int | None = None,
        horizon_weight: list[float] | None = None,
        path: str | Path | None = None,
        verbosity: int = 2,
        log_to_file: bool = True,
        log_file_path: str | Path = "auto",
        quantile_levels: list[float] | None = None,
        cache_predictions: bool = True,
        label: str | None = None,
        **kwargs,
    ):
        self.verbosity = verbosity
        set_logger_verbosity(self.verbosity, logger=logger)
        self.path = setup_outputdir(path)
        if self.path.lower().startswith("s3://"):
            logger.warning(
                "Warning: S3 paths are not supported for the `path` argument in TimeSeriesPredictor. "
                "Use a local path and upload the trained predictor to S3 manually if needed"
            )
        self._setup_log_to_file(log_to_file=log_to_file, log_file_path=log_file_path)

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

        self.prediction_length = int(prediction_length)
        # For each validation fold, all time series in training set must have length >= _min_train_length
        self._min_train_length = max(self.prediction_length + 1, 5)
        self.freq = freq
        if self.freq is not None:
            # Standardize frequency string (e.g., "T" -> "min", "Y" -> "YE")
            offset = pd.tseries.frequencies.to_offset(self.freq)
            assert offset is not None
            std_freq = offset.freqstr
            if std_freq != str(self.freq):
                logger.info(f"Frequency '{self.freq}' stored as '{std_freq}'")
            self.freq = std_freq
        self.eval_metric: TimeSeriesScorer = check_get_evaluation_metric(
            eval_metric,
            prediction_length=prediction_length,
            seasonal_period=eval_metric_seasonal_period,
            horizon_weight=horizon_weight,
        )
        if quantile_levels is None:
            quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.quantile_levels = sorted(quantile_levels)
        self._learner: TimeSeriesLearner = self._learner_type(
            path_context=self.path,
            eval_metric=self.eval_metric,
            target=self.target,
            known_covariates_names=self.known_covariates_names,
            prediction_length=self.prediction_length,
            quantile_levels=self.quantile_levels,
            cache_predictions=self.cache_predictions,
            ensemble_model_type=kwargs.pop("ensemble_model_type", None),
        )

        if len(kwargs) > 0:
            for key in kwargs:
                raise TypeError(f"TimeSeriesPredictor.__init__() got an unexpected keyword argument '{key}'")

    @property
    def _trainer(self) -> TimeSeriesTrainer:
        return self._learner.load_trainer()  # noqa

    @property
    def is_fit(self) -> bool:
        return self._learner.is_fit

    def _assert_is_fit(self, method_name: str) -> None:
        """Check if predictor is fit and raise AssertionError with informative message if not."""
        if not self.is_fit:
            raise AssertionError(f"Predictor is not fit. Call `.fit` before calling `.{method_name}`. ")

    def _setup_log_to_file(self, log_to_file: bool, log_file_path: str | Path) -> None:
        if log_to_file:
            if log_file_path == "auto":
                log_file_path = os.path.join(self.path, "logs", self._predictor_log_file_name)
            log_file_path = os.path.abspath(os.path.normpath(log_file_path))
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            add_log_to_file(log_file_path)

    def _to_data_frame(
        self,
        data: TimeSeriesDataFrame | pd.DataFrame | Path | str,
        name: str = "data",
    ) -> TimeSeriesDataFrame:
        if isinstance(data, TimeSeriesDataFrame):
            return data
        elif isinstance(data, (pd.DataFrame, Path, str)):
            try:
                data = TimeSeriesDataFrame(data)  # type: ignore
            except:
                raise ValueError(
                    f"Provided {name} of type {type(data)} cannot be automatically converted to a TimeSeriesDataFrame."
                )
            return data
        else:
            raise TypeError(
                f"{name} must be a TimeSeriesDataFrame, pandas.DataFrame, pathlib.Path or string (path to data) "
                f"but received an object of type {type(data)}."
            )

    def _check_and_prepare_data_frame(
        self,
        data: TimeSeriesDataFrame | pd.DataFrame | Path | str,
        name: str = "data",
    ) -> TimeSeriesDataFrame:
        """Ensure that TimeSeriesDataFrame has a sorted index and a valid frequency.

        If self.freq is None, then self.freq of the predictor will be set to the frequency of the data.

        Parameters
        ----------
        data : TimeSeriesDataFrame | pd.DataFrame | Path | str
            Data as a dataframe or path to file storing the data.
        name : str
            Name of the data that will be used in log messages (e.g., 'train_data', 'tuning_data', or 'data').

        Returns
        -------
        df : TimeSeriesDataFrame
            Preprocessed data in TimeSeriesDataFrame format.
        """
        df: TimeSeriesDataFrame = self._to_data_frame(data, name=name)
        if not pd.api.types.is_numeric_dtype(df[self.target]):
            raise ValueError(f"Target column {name}['{self.target}'] has a non-numeric dtype {df[self.target].dtype}")
        # Assign makes a copy, so future operations can be performed in-place
        df = df.assign(**{self.target: df[self.target].astype("float64")})
        df.replace(to_replace=[float("-inf"), float("inf")], value=float("nan"), inplace=True)

        # MultiIndex.is_monotonic_increasing checks if index is sorted by ["item_id", "timestamp"]
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        # Ensure that data has a regular frequency that matches the predictor frequency
        if self.freq is None:
            try:
                # Use all items for inferring the frequency
                data_freq = df.infer_frequency(num_items=None, raise_if_irregular=True)
            except ValueError:
                raise ValueError(
                    f"Frequency of {name} is not provided and cannot be inferred. Please set the expected data "
                    f"frequency when creating the predictor with `TimeSeriesPredictor(freq=...)` or ensure that "
                    f"the data has a regular time index with `{name}.convert_frequency(freq=...)`"
                )
            else:
                self.freq = data_freq
                logger.info(f"Inferred time series frequency: '{data_freq}'")
        else:
            data_freq = df.infer_frequency(num_items=None)
            if data_freq != self.freq:
                logger.warning(f"{name} with frequency '{data_freq}' has been resampled to frequency '{self.freq}'.")
                df = df.convert_frequency(freq=self.freq)
        return df

    def _check_and_prepare_data_frame_for_evaluation(
        self, data: TimeSeriesDataFrame, cutoff: int | None = None, name: str = "data"
    ) -> TimeSeriesDataFrame:
        """
        Make sure that provided evaluation data includes both historical and future time series values.
        Slices the dataframe based on cutoff, if needed.
        """
        cutoff = -1 * self.prediction_length if cutoff is None else cutoff
        if not (isinstance(cutoff, int) and cutoff <= -self.prediction_length):
            raise ValueError(f"`cutoff` should be a negative integer <= -prediction_length, got: {cutoff=}")

        expected_length = -cutoff

        if data.num_timesteps_per_item().min() <= expected_length:
            var_name = "-cutoff" if expected_length > self.prediction_length else "prediction_length"
            raise ValueError(
                f"Cannot reserve last {expected_length} time steps for evaluation in some "
                f"time series in {name}. Please make sure that {name} includes both historical and future data, and that"
                f"all time series have length > {var_name} (at least {expected_length + 1})"
            )

        if cutoff < -self.prediction_length:
            data = data.slice_by_timestep(None, cutoff + self.prediction_length)

        return data

    def _get_dataset_stats(self, data: TimeSeriesDataFrame) -> str:
        ts_lengths = data.num_timesteps_per_item()
        median_length = ts_lengths.median()
        min_length = ts_lengths.min()
        max_length = ts_lengths.max()
        missing_value_fraction = data[self.target].isna().mean()
        if missing_value_fraction > 0:
            missing_value_fraction_str = f" (NaN fraction={missing_value_fraction:.1%})"
        else:
            missing_value_fraction_str = ""
        return (
            f"{len(data)} rows{missing_value_fraction_str}, {data.num_items} time series. "
            f"Median time series length is {median_length:.0f} (min={min_length}, max={max_length}). "
        )

    def _filter_useless_train_data(
        self,
        train_data: TimeSeriesDataFrame,
        num_val_windows: tuple[int, ...],
        val_step_size: int,
    ) -> TimeSeriesDataFrame:
        """Remove time series from train_data that either contain all NaNs or are too short for chosen settings.

        This method ensures that 1) no time series consist of all NaN values and 2) for each validation fold, all train
        series have length >= max(prediction_length + 1, 5).

        In other words, this method removes from train_data all time series with only NaN values or length less than
        min_train_length + prediction_length + (num_val_windows - 1) * val_step_size
        """
        total_num_val_windows = sum(num_val_windows)
        min_length = self._min_train_length + self.prediction_length + (total_num_val_windows - 1) * val_step_size
        train_lengths = train_data.num_timesteps_per_item()
        too_short_items = train_lengths.index[train_lengths < min_length]

        if len(too_short_items) > 0:
            logger.info(
                f"\tRemoving {len(too_short_items)} short time series from train_data. Only series with length "
                f">= {min_length} will be used for training."
            )
            train_data = train_data.query("item_id not in @too_short_items")

        all_nan_items = train_data.item_ids[
            train_data[self.target].isna().groupby(TimeSeriesDataFrame.ITEMID, sort=False).all()
        ]
        if len(all_nan_items) > 0:
            logger.info(f"\tRemoving {len(all_nan_items)} time series consisting of only NaN values from train_data.")
            train_data = train_data.query("item_id not in @all_nan_items")

        if len(too_short_items) or len(all_nan_items):
            logger.info(f"\tAfter filtering, train_data has {self._get_dataset_stats(train_data)}")

        if len(train_data) == 0:
            raise ValueError(
                f"At least some time series in train_data must have >= {min_length} observations. Please provide "
                f"longer time series as train_data or reduce prediction_length, num_val_windows, or val_step_size."
            )
        return train_data

    @apply_presets(get_predictor_presets())
    def fit(
        self,
        train_data: TimeSeriesDataFrame | pd.DataFrame | Path | str,
        tuning_data: TimeSeriesDataFrame | pd.DataFrame | Path | str | None = None,
        time_limit: int | None = None,
        presets: str | None = None,
        hyperparameters: str | dict[str | Type, Any] | None = None,
        hyperparameter_tune_kwargs: str | dict | None = None,
        excluded_model_types: list[str] | None = None,
        ensemble_hyperparameters: dict[str, Any] | list[dict[str, Any]] | None = None,
        num_val_windows: int | tuple[int, ...] = 1,
        val_step_size: int | None = None,
        refit_every_n_windows: int | None = 1,
        refit_full: bool = False,
        enable_ensemble: bool = True,
        skip_model_selection: bool = False,
        random_seed: int | None = 123,
        verbosity: int | None = None,
    ) -> "TimeSeriesPredictor":
        """Fit probabilistic forecasting models to the given time series dataset.

        Parameters
        ----------
        train_data : TimeSeriesDataFrame | pd.DataFrame | Path | str
            Training data in the :class:`~autogluon.timeseries.TimeSeriesDataFrame` format.

            Time series with length ``<= (num_val_windows + 1) * prediction_length`` will be ignored during training.
            See :attr:`num_val_windows` for details.

            If ``known_covariates_names`` were specified when creating the predictor, ``train_data`` must include the
            columns listed in ``known_covariates_names`` with the covariates values aligned with the target time series.

            Columns of ``train_data`` except ``target`` and those listed in ``known_covariates_names`` will be
            interpreted as ``past_covariates`` - covariates that are known only in the past.

            If ``train_data`` contains covariates or static features, they will be interpreted as follows:

            * columns with ``int``, ``bool`` and ``float`` dtypes are interpreted as continuous (real-valued) features
            * columns with ``object``, ``str`` and ``category`` dtypes are as interpreted as categorical features
            * columns with other dtypes are ignored

            To ensure that the column type is interpreted correctly, please convert it to one of the above dtypes.
            For example, to ensure that column "store_id" with dtype ``int`` is interpreted as a category, change
            its dtype to ``category``::

                data.static_features["store_id"] = data.static_features["store_id"].astype("category")

            If provided data is a ``pandas.DataFrame``, AutoGluon will attempt to convert it to a ``TimeSeriesDataFrame``.
            If a ``str`` or a ``Path`` is provided, AutoGluon will attempt to load this file.
        tuning_data : TimeSeriesDataFrame | pd.DataFrame | Path | str, optional
            Data reserved for model selection and hyperparameter tuning, rather than training individual models. Also
            used to compute the validation scores. Note that only the last ``prediction_length`` time steps of each
            time series are used for computing the validation score.

            If ``tuning_data`` is provided, multi-window backtesting on training data will be disabled, the
            ``num_val_windows`` will be set to ``0``, and ``refit_full`` will be set to ``False``.

            Leaving this argument empty and letting AutoGluon automatically generate the validation set from
            ``train_data`` is a good default.

            The names and dtypes of columns and static features in ``tuning_data`` must match the ``train_data``.

            If provided data is a ``pandas.DataFrame``, AutoGluon will attempt to convert it to a ``TimeSeriesDataFrame``.
            If a ``str`` or a ``Path`` is provided, AutoGluon will attempt to load this file.
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

            - ``"fast_training"``: Simple statistical and tree-based ML models. These models are fast to train but may not be very accurate.
            - ``"medium_quality"``: Same models as above, plus deep learning models ``TemporalFusionTransformer`` and Chronos-Bolt (small). Produces good forecasts with reasonable training time.
            - ``"high_quality"``: A mix of multiple DL, ML and statistical forecasting models available in AutoGluon that offers the best forecast accuracy. Much more accurate than ``medium_quality``, but takes longer to train.
            - ``"best_quality"``: Same models as in ``"high_quality"``, but performs validation with multiple backtests. Usually better than ``high_quality``, but takes even longer to train.

            Available presets with the `Chronos-Bolt <https://github.com/amazon-science/chronos-forecasting>`_ model:

            - ``"bolt_{model_size}"``: where model size is one of ``tiny,mini,small,base``. Uses the Chronos-Bolt pretrained model for zero-shot forecasting.
              See the documentation for ``ChronosModel`` or see `Hugging Face <https://huggingface.co/collections/amazon/chronos-models-65f1791d630a8d57cb718444>`_ for more information.

            Exact definitions of these presets can be found in the source code
            [`1 <https://github.com/autogluon/autogluon/blob/stable/timeseries/src/autogluon/timeseries/configs/presets_configs.py>`_,
            `2 <https://github.com/autogluon/autogluon/blob/stable/timeseries/src/autogluon/timeseries/models/presets.py>`_].

            If no ``presets`` are selected, user-provided values for ``hyperparameters`` will be used (defaulting to their
            default values specified below).
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

            Full list of available models and their hyperparameters is provided in :ref:`Forecasting Time Series - Model Zoo <forecasting_model_zoo>`.

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
                * "local": Local scheduler that schedules trials FIFO
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
        excluded_model_types: list[str], optional
            Banned subset of model types to avoid training during ``fit()``, even if present in ``hyperparameters``.
            For example, the following code will train all models included in the ``high_quality`` presets except ``DeepAR``::

                predictor.fit(
                    ...,
                    presets="high_quality",
                    excluded_model_types=["DeepAR"],
                )
        ensemble_hyperparameters : dict or list of dict, optional
            Hyperparameters for ensemble models. Can be a single dict for one ensemble layer, or a list of dicts
            for multiple ensemble layers (multi-layer stacking).

            For single-layer ensembling (default)::

                predictor.fit(
                    ...,
                    ensemble_hyperparameters={"GreedyEnsemble": {"ensemble_size": 10}},
                )

            For multi-layer ensembling, provide a list where each element configures one ensemble layer::

                predictor.fit(
                    ...,
                    num_val_windows=(2, 3),
                    ensemble_hyperparameters=[
                        {"GreedyEnsemble": {"ensemble_size": 5}},  # Layer 1
                        {"PerformanceWeightedEnsemble": {}},       # Layer 2
                    ],
                )

            When using multi-layer ensembling, ``len(ensemble_hyperparameters)`` must match ``len(num_val_windows)``.
        num_val_windows : int | tuple[int, ...], default = 1
            Number of backtests done on ``train_data`` for each trained model to estimate the validation performance.
            This parameter is also used to control multi-layer ensembling.

            Increasing this parameter increases the training time roughly by a factor of
            ``num_val_windows // refit_every_n_windows``. See ``refit_every_n_windows`` and ``val_step_size`` for
            details.

            For example, for ``prediction_length=2``, ``num_val_windows=3`` and ``val_step_size=1`` the folds are::

                |-------------------|
                | x x x x x y y - - |
                | x x x x x x y y - |
                | x x x x x x x y y |

            where ``x`` are the train time steps and ``y`` are the validation time steps.

            This parameter can also be used to control how many of the backtesting windows are reserved for training
            multiple layers of ensemble models. By default, AutoGluon-TimeSeries uses only a single layer of ensembles
            trained on the backtest windows specified by the ``num_val_windows`` parameter. However, the
            ``ensemble_hyperparameters`` argument can be used to specify multiple layers of ensembles. In this case,
            a tuple of integers can be provided in ``num_val_windows`` to control how many of the backtesting windows
            will be used to train which ensemble layers.

            For example, if ``len(ensemble_hyperparameters) == 2``, a 2-tuple ``num_val_windows=(2, 3)`` is analogous
            to ``num_val_windows=5``, except the first layer of ensemble models will be trained on the first two
            backtest windows, and the second layer will be trained on the latter three. Validation scores of all models
            will be computed on the last three windows.

            If ``len(ensemble_hyperparameters) == 1``, then ``num_val_windows=(5,)`` has the same effect as
            ``num_val_windows=5``.

            If ``tuning_data`` is provided and ``len(ensemble_hyperparameters) == 1``, then this parameter is ignored.
            Validation and ensemble training will be performed on ``tuning_data``.

            If ``tuning_data`` is provided and ``len(ensemble_hyperparameters) > 1``, then this method expects that
            ``len(num_val_windows) > 1``. In this case, the last element of ``num_val_windows`` will be ignored. The
            last layer of ensemble training will be performed on ``tuning_data``. Validation scores will likewise be
            computed on ``tuning_data``.

        val_step_size : int or None, default = None
            Step size between consecutive validation windows. If set to ``None``, defaults to ``prediction_length``
            provided when creating the predictor.

            If ``tuning_data`` is provided and ``len(ensemble_hyperparameters) == 1``, then this parameter is ignored.
        refit_every_n_windows: int or None, default = 1
            When performing cross validation, each model will be retrained every ``refit_every_n_windows`` validation
            windows, where the number of validation windows is specified by ``num_val_windows``. Note that in the
            default setting where ``num_val_windows=1``, this argument has no effect.

            If set to ``None``, models will only be fit once for the first (oldest) validation window. By default,
            ``refit_every_n_windows=1``, i.e., all models will be refit for each validation window.
        refit_full : bool, default = False
            If True, after training is complete, AutoGluon will attempt to re-train all models using all of training
            data (including the data initially reserved for validation). This argument has no effect if ``tuning_data``
            is provided.
        enable_ensemble : bool, default = True
            If True, the ``TimeSeriesPredictor`` will fit a simple weighted ensemble on top of the models specified via
            ``hyperparameters``.
        skip_model_selection : bool, default = False
            If True, predictor will not compute the validation score. For example, this argument is useful if we want
            to use the predictor as a wrapper for a single pre-trained model. If set to True, then the ``hyperparameters``
            dict must contain exactly one model without hyperparameter search spaces or an exception will be raised.
        random_seed : int or None, default = 123
            If provided, fixes the seed of the random number generator for all models. This guarantees reproducible
            results for most models (except those trained on GPU because of the non-determinism of GPU operations).
        verbosity : int, optional
            If provided, overrides the ``verbosity`` value used when creating the ``TimeSeriesPredictor``. See
            documentation for :class:`~autogluon.timeseries.TimeSeriesPredictor` for more details.

        """
        time_start = time.time()
        if self.is_fit:
            raise AssertionError(
                "Predictor is already fit! To fit additional models create a new `TimeSeriesPredictor`."
            )

        if verbosity is None:
            verbosity = self.verbosity
        set_logger_verbosity(verbosity, logger=logger)
        warn_if_mlflow_autologging_is_enabled(logger=logger)

        logger.info("Beginning AutoGluon training..." + (f" Time limit = {time_limit}s" if time_limit else ""))
        logger.info(f"AutoGluon will save models to '{self.path}'")
        logger.info(get_ag_system_info(path=self.path, include_gpu_count=True))

        if hyperparameters is None:
            hyperparameters = "default"

        fit_args = dict(
            prediction_length=self.prediction_length,
            target=self.target,
            known_covariates_names=self.known_covariates_names,
            eval_metric=self.eval_metric,
            eval_metric_seasonal_period=self.eval_metric.seasonal_period,
            horizon_weight=self.eval_metric.horizon_weight,
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
            skip_model_selection=skip_model_selection,
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

        num_val_windows, ensemble_hyperparameters = self._validate_and_normalize_validation_and_ensemble_inputs(
            num_val_windows=num_val_windows,
            ensemble_hyperparameters=ensemble_hyperparameters,
            val_step_size=val_step_size,
            median_timeseries_length=train_data.num_timesteps_per_item().median(),
        )

        if tuning_data is not None:
            tuning_data = self._check_and_prepare_data_frame(tuning_data, name="tuning_data")
            tuning_data = self._check_and_prepare_data_frame_for_evaluation(tuning_data, name="tuning_data")
            logger.info(f"Provided tuning_data has {self._get_dataset_stats(tuning_data)}")

            # TODO: Use num_val_windows to perform multi-window backtests on tuning_data
            num_val_windows = num_val_windows[:-1] + (1,)
            logger.warning(
                f"\tTuning data is provided. Setting num_val_windows = {num_val_windows}. Validation scores will"
                " be computed on a single window of tuning_data."
            )

        if sum(num_val_windows) <= 1 and refit_every_n_windows is not None and refit_every_n_windows > 1:
            logger.warning(
                f"\trefit_every_n_windows provided as {refit_every_n_windows} but num_val_windows is set to "
                f"{num_val_windows}. refit_every_n_windows will have no effect."
            )

        if not skip_model_selection:
            # When tuning_data is provided, ignore the last element of num_val_windows for filtering purposes
            filter_num_val_windows = num_val_windows[:-1] if tuning_data is not None else num_val_windows
            train_data = self._filter_useless_train_data(train_data, filter_num_val_windows, val_step_size)

        time_left = None if time_limit is None else time_limit - (time.time() - time_start)
        self._learner.fit(
            train_data=train_data,
            hyperparameters=hyperparameters,
            val_data=tuning_data,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            excluded_model_types=excluded_model_types,
            ensemble_hyperparameters=ensemble_hyperparameters,
            time_limit=time_left,
            verbosity=verbosity,
            num_val_windows=(num_val_windows,) if isinstance(num_val_windows, int) else num_val_windows,
            val_step_size=val_step_size,
            refit_every_n_windows=refit_every_n_windows,
            skip_model_selection=skip_model_selection,
            enable_ensemble=enable_ensemble,
            random_seed=random_seed,
        )
        if refit_full:
            if tuning_data is None:
                self.refit_full()
            else:
                logger.warning("Skipping `refit_full` because custom `tuning_data` was provided during `fit`.")

        self.save()
        return self

    def _validate_and_normalize_validation_and_ensemble_inputs(
        self,
        num_val_windows: int | tuple[int, ...],
        ensemble_hyperparameters: dict[str, Any] | list[dict[str, Any]] | None,
        val_step_size: int,
        median_timeseries_length: float,
    ) -> tuple[tuple[int, ...], dict[str, Any] | list[dict[str, Any]] | None]:
        """Validate and normalize num_val_windows and ensemble_hyperparameters for multilayer ensembling."""
        original_num_val_windows = num_val_windows if isinstance(num_val_windows, tuple) else (num_val_windows,)

        if ensemble_hyperparameters is not None:
            if isinstance(ensemble_hyperparameters, dict):
                ensemble_hyperparameters = [ensemble_hyperparameters]

            if len(ensemble_hyperparameters) != len(original_num_val_windows):
                raise ValueError(
                    f"Length mismatch: num_val_windows has {len(original_num_val_windows)} layers but "
                    f"ensemble_hyperparameters has {len(ensemble_hyperparameters)} layers. "
                    f"These must match for multilayer ensembling."
                )

        num_val_windows = self._normalize_num_val_windows_input(
            num_val_windows, val_step_size, median_timeseries_length
        )

        if ensemble_hyperparameters is not None and len(num_val_windows) < len(ensemble_hyperparameters):
            logger.warning(
                f"Time series too short: reducing ensemble layers from {len(ensemble_hyperparameters)} to "
                f"{len(num_val_windows)}. Only the first {len(num_val_windows)} ensemble layer(s) will be trained."
            )
            ensemble_hyperparameters = ensemble_hyperparameters[: len(num_val_windows)]

        return num_val_windows, ensemble_hyperparameters

    def _normalize_num_val_windows_input(
        self,
        num_val_windows: int | tuple[int, ...],
        val_step_size: int,
        median_timeseries_length: float,
    ) -> tuple[int, ...]:
        if isinstance(num_val_windows, int):
            num_val_windows = (num_val_windows,)
        if not isinstance(num_val_windows, tuple):
            raise TypeError(f"num_val_windows must be int or tuple[int, ...], got {type(num_val_windows)}")
        if len(num_val_windows) == 0:
            raise ValueError("num_val_windows tuple cannot be empty")
        if not all(isinstance(n, int) and n > 0 for n in num_val_windows):
            raise ValueError(f"All elements of num_val_windows must be positive integers, got {num_val_windows}")
        num_val_windows = self._reduce_num_val_windows_if_necessary(
            num_val_windows, val_step_size, median_timeseries_length
        )
        return num_val_windows

    def _reduce_num_val_windows_if_necessary(
        self,
        num_val_windows: tuple[int, ...],
        val_step_size: int,
        median_time_series_length: float,
    ) -> tuple[int, ...]:
        """Adjust num_val_windows based on the length of time series in train_data.

        Chooses num_val_windows such that TS with median length is long enough to perform num_val_windows validations
        (at least 1, at most `original_num_val_windows`).

        In other words, find largest `num_val_windows` that satisfies
        median_length >= min_train_length + prediction_length + (num_val_windows - 1) * val_step_size
        """
        num_val_windows_for_median_ts = int(
            (median_time_series_length - self._min_train_length - self.prediction_length) // val_step_size + 1
        )
        max_allowed = max(1, num_val_windows_for_median_ts)
        total_requested = sum(num_val_windows)

        if max_allowed >= total_requested:
            return num_val_windows

        logger.warning(
            f"Time series in train_data are too short for chosen num_val_windows={num_val_windows}. "
            f"Reducing num_val_windows to {max_allowed} total windows."
        )

        result = list(num_val_windows)
        for i in range(len(result) - 1, -1, -1):
            while result[i] > 1 and sum(result) > max_allowed:
                result[i] -= 1
            if sum(result) <= max_allowed:
                break

        if sum(result) > max_allowed:
            result = [1] * max_allowed

        return tuple(result)

    def model_names(self) -> list[str]:
        """Returns the list of model names trained by this predictor object."""
        self._assert_is_fit("model_names")
        return self._trainer.get_model_names()

    def predict(
        self,
        data: TimeSeriesDataFrame | pd.DataFrame | Path | str,
        known_covariates: TimeSeriesDataFrame | pd.DataFrame | Path | str | None = None,
        model: str | None = None,
        use_cache: bool = True,
        random_seed: int | None = 123,
    ) -> TimeSeriesDataFrame:
        """Return quantile and mean forecasts for the given dataset, starting from the end of each time series.

        Parameters
        ----------
        data : TimeSeriesDataFrame | pd.DataFrame | Path | str
            Historical time series data for which the forecast needs to be made.

            The names and dtypes of columns and static features in ``data`` must match the ``train_data`` used to train
            the predictor.

            If provided data is a ``pandas.DataFrame``, AutoGluon will attempt to convert it to a ``TimeSeriesDataFrame``.
            If a ``str`` or a ``Path`` is provided, AutoGluon will attempt to load this file.
        known_covariates : TimeSeriesDataFrame | pd.DataFrame | Path | str, optional
            If ``known_covariates_names`` were specified when creating the predictor, it is necessary to provide the
            values of the known covariates for each time series during the forecast horizon. Specifically:

            - Must contain all columns listed in ``known_covariates_names``.
            - Must include all ``item_id`` values present in the input ``data``.
            - Must include ``timestamp`` values for the full forecast horizon (i.e., ``prediction_length`` time steps) following the end of each series in the input ``data``.

            You can use :meth:`autogluon.timeseries.TimeSeriesPredictor.make_future_data_frame` to generate a template
            containing the required ``item_id`` and ``timestamp`` combinations for the ``known_covariates`` dataframe.

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
        self._assert_is_fit("predict")
        # Save original item_id order to return predictions in the same order as input data
        data = self._to_data_frame(data)
        original_item_id_order = data.item_ids
        data = self._check_and_prepare_data_frame(data)
        if known_covariates is not None:
            known_covariates = self._to_data_frame(known_covariates)
        predictions = self._learner.predict(
            data,
            known_covariates=known_covariates,
            model=model,
            use_cache=use_cache,
            random_seed=random_seed,
        )
        return cast(TimeSeriesDataFrame, predictions.reindex(original_item_id_order, level=TimeSeriesDataFrame.ITEMID))

    @overload
    def backtest_predictions(
        self,
        data: TimeSeriesDataFrame | None = None,
        *,
        model: str | None = None,
        num_val_windows: int | None = None,
        val_step_size: int | None = None,
        use_cache: bool = True,
    ) -> list[TimeSeriesDataFrame]: ...

    @overload
    def backtest_predictions(
        self,
        data: TimeSeriesDataFrame | None = None,
        *,
        model: list[str],
        num_val_windows: int | None = None,
        val_step_size: int | None = None,
        use_cache: bool = True,
    ) -> dict[str, list[TimeSeriesDataFrame]]: ...

    def backtest_predictions(
        self,
        data: TimeSeriesDataFrame | None = None,
        *,
        model: str | list[str] | None = None,
        num_val_windows: int | None = None,
        val_step_size: int | None = None,
        use_cache: bool = True,
    ) -> list[TimeSeriesDataFrame] | dict[str, list[TimeSeriesDataFrame]]:
        """Return predictions for multiple validation windows.

        When ``data=None``, returns the predictions that were saved during training. Otherwise, generates new
        predictions by splitting ``data`` into multiple windows using an expanding window strategy.

        The corresponding target values for each window can be obtained using
        :meth:`~autogluon.timeseries.TimeSeriesPredictor.backtest_targets`.

        Parameters
        ----------
        data : TimeSeriesDataFrame, optional
            Time series data to generate predictions for. If ``None``, returns the predictions that were saved
            during training on ``train_data``.

            If provided, all time series in ``data`` must have length at least
            ``prediction_length + (num_val_windows - 1) * val_step_size + 1``.

            The names and dtypes of columns and static features in ``data`` must match the ``train_data`` used to train
            the predictor.
        model : str, list[str], or None, default = None
            Name of the model(s) to generate predictions with. By default, the best model during training
            (with highest validation score) will be used.

            - If ``str``: Returns predictions for a single model as a list.
            - If ``list[str]``: Returns predictions for multiple models as a dict mapping model names to lists.
            - If ``None``: Uses the best model.
        num_val_windows : int, optional
            Number of validation windows to generate. If ``None``, uses the ``num_val_windows`` value from training
            configuration when ``data=None``, otherwise defaults to 1.

            For example, with ``prediction_length=2``, ``num_val_windows=3``, and ``val_step_size=1``, the validation
            windows are::

                |-------------------|
                | x x x x x y y - - |
                | x x x x x x y y - |
                | x x x x x x x y y |

            where ``x`` denotes training time steps and ``y`` denotes validation time steps for each window.
        val_step_size : int, optional
            Number of time steps between the start of consecutive validation windows. If ``None``, defaults to
            ``prediction_length``.
        use_cache : bool, default = True
            If True, will attempt to use cached predictions. If False, cached predictions will be ignored.
            This argument is ignored if ``cache_predictions`` was set to False when creating the ``TimeSeriesPredictor``.

        Returns
        -------
        list[TimeSeriesDataFrame] or dict[str, list[TimeSeriesDataFrame]]
            Predictions for each validation window.

            - If ``model`` is a ``str`` or ``None``: Returns a list of length ``num_val_windows``, where each element
              contains the predictions for one validation window.
            - If ``model`` is a ``list[str]``: Returns a dict mapping each model name to a list of predictions for
              each validation window.

        Examples
        --------
        Make predictions on new data with the best model

        >>> predictor.backtest_predictions(test_data, num_val_windows=2)

        Load validation predictions for all models that were saved during training

        >>> predictor.backtest_predictions(model=predictor.model_names())

        See Also
        --------
        backtest_targets
            Return target values aligned with predictions.
        evaluate
            Evaluate forecast accuracy on a hold-out set.
        predict
            Generate forecasts for future time steps.
        """
        self._assert_is_fit("backtest_predictions")
        if data is not None:
            data = self._check_and_prepare_data_frame(data)

        if model is None:
            model_names = [self.model_best]
        elif isinstance(model, str):
            model_names = [model]
        else:
            model_names = model

        result = self._learner.backtest_predictions(
            data=data,
            model_names=model_names,
            num_val_windows=num_val_windows,
            val_step_size=val_step_size,
            use_cache=use_cache,
        )

        if isinstance(model, list):
            return result
        else:
            return result[model_names[0]]

    def backtest_targets(
        self,
        data: TimeSeriesDataFrame | None = None,
        *,
        num_val_windows: int | None = None,
        val_step_size: int | None = None,
    ) -> list[TimeSeriesDataFrame]:
        """Return target values for each validation window.

        Returns the actual target values corresponding to each validation window used in
        :meth:`~autogluon.timeseries.TimeSeriesPredictor.backtest_predictions`. The returned targets are aligned
        with the predictions, making it easy to compute custom evaluation metrics or analyze forecast errors.

        Parameters
        ----------
        data : TimeSeriesDataFrame, optional
            Time series data to extract targets from. If ``None``, returns the targets from the validation windows
            used during training.

            If provided, all time series in ``data`` must have length at least
            ``prediction_length + (num_val_windows - 1) * val_step_size + 1``.

            The names and dtypes of columns and static features in ``data`` must match the ``train_data`` used to train
            the predictor.
        num_val_windows : int, optional
            Number of validation windows to extract targets for. If ``None``, uses the ``num_val_windows`` value from
            training configuration when ``data=None``, otherwise defaults to 1.

            This should match the ``num_val_windows`` argument passed to
            :meth:`~autogluon.timeseries.TimeSeriesPredictor.backtest_predictions`.
        val_step_size : int, optional
            Number of time steps between the start of consecutive validation windows. If ``None``, defaults to
            ``prediction_length``.

            This should match the ``val_step_size`` argument passed to
            :meth:`~autogluon.timeseries.TimeSeriesPredictor.backtest_predictions`.

        Returns
        -------
        list[TimeSeriesDataFrame]
            Target values for each validation window. Returns a list of length ``num_val_windows``,
            where each element contains the full time series data for one validation window.
            Each dataframe includes both historical context and the last ``prediction_length`` time steps
            that represent the target values to compare against predictions.

            The returned targets are aligned with the output of
            :meth:`~autogluon.timeseries.TimeSeriesPredictor.backtest_predictions`, so ``targets[i]`` corresponds
            to ``predictions[i]`` for the i-th validation window.

        See Also
        --------
        backtest_predictions
            Return predictions for multiple validation windows.
        evaluate
            Evaluate forecast accuracy on a hold-out set.
        """
        self._assert_is_fit("backtest_targets")
        if data is not None:
            data = self._check_and_prepare_data_frame(data)
        return self._learner.backtest_targets(
            data=data,
            num_val_windows=num_val_windows,
            val_step_size=val_step_size,
        )

    def evaluate(
        self,
        data: TimeSeriesDataFrame | pd.DataFrame | Path | str,
        model: str | None = None,
        metrics: str | TimeSeriesScorer | list[str | TimeSeriesScorer] | None = None,
        cutoff: int | None = None,
        display: bool = False,
        use_cache: bool = True,
    ) -> dict[str, float]:
        """Evaluate the forecast accuracy for given dataset.

        This method measures the forecast accuracy using the last ``self.prediction_length`` time steps of each time
        series in ``data`` as a hold-out set.

        .. note::
            Metrics are always reported in 'higher is better' format.
            This means that metrics such as MASE or MAPE will be multiplied by -1, so their values will be negative.
            This is necessary to avoid the user needing to know the metric to understand if higher is better when
            looking at the evaluation results.

        Parameters
        ----------
        data : TimeSeriesDataFrame | pd.DataFrame | Path | str
            The data to evaluate the best model on. If a ``cutoff`` is not provided, the last ``prediction_length``
            time steps of each time series in ``data`` will be held out for prediction and forecast accuracy will
            be calculated on these time steps. When a ``cutoff`` is provided, the ``-cutoff``-th to the
            ``-cutoff + prediction_length``-th time steps of each time series are used for evaluation.

            Must include both historical and future data (i.e., length of all time series in ``data`` must be at least
            ``prediction_length + 1``, if ``cutoff`` is not provided, ``-cutoff + 1`` otherwise).

            The names and dtypes of columns and static features in ``data`` must match the ``train_data`` used to train
            the predictor.

            If provided data is a ``pandas.DataFrame``, AutoGluon will attempt to convert it to a ``TimeSeriesDataFrame``.
            If a ``str`` or a ``Path`` is provided, AutoGluon will attempt to load this file.
        model : str, optional
            Name of the model that you would like to evaluate. By default, the best model during training
            (with highest validation score) will be used.
        metrics : str, TimeSeriesScorer or list[str | TimeSeriesScorer], optional
            Metric or a list of metrics to compute scores with. Defaults to ``self.eval_metric``. Supports both
            metric names as strings and custom metrics based on TimeSeriesScorer.
        cutoff : int, optional
            A *negative* integer less than or equal to ``-1 * prediction_length`` denoting the time step in ``data``
            where the forecast evaluation starts, i.e., time series are evaluated from the ``-cutoff``-th to the
            ``-cutoff + prediction_length``-th time step. Defaults to ``-1 * prediction_length``, using the last
            ``prediction_length`` time steps of each time series for evaluation.
        display : bool, default = False
            If True, the scores will be printed.
        use_cache : bool, default = True
            If True, will attempt to use the cached predictions. If False, cached predictions will be ignored.
            This argument is ignored if ``cache_predictions`` was set to False when creating the ``TimeSeriesPredictor``.

        Returns
        -------
        scores_dict : dict[str, float]
            Dictionary where keys = metrics, values = performance along each metric. For consistency, error metrics
            will have their signs flipped to obey this convention. For example, negative MAPE values will be reported.
            To get the ``eval_metric`` score, do ``output[predictor.eval_metric.name]``.
        """
        self._assert_is_fit("evaluate")
        data = self._check_and_prepare_data_frame(data)
        data = self._check_and_prepare_data_frame_for_evaluation(data, cutoff=cutoff)

        scores_dict = self._learner.evaluate(data, model=model, metrics=metrics, use_cache=use_cache)
        if display:
            logger.info("Evaluations on test data:")
            logger.info(json.dumps(scores_dict, indent=4))
        return scores_dict

    def feature_importance(
        self,
        data: TimeSeriesDataFrame | pd.DataFrame | Path | str | None = None,
        model: str | None = None,
        metric: str | TimeSeriesScorer | None = None,
        features: list[str] | None = None,
        time_limit: float | None = None,
        method: Literal["naive", "permutation"] = "permutation",
        subsample_size: int = 50,
        num_iterations: int | None = None,
        random_seed: int | None = 123,
        relative_scores: bool = False,
        include_confidence_band: bool = True,
        confidence_level: float = 0.99,
    ) -> pd.DataFrame:
        """
        Calculates feature importance scores for the given model via replacing each feature by a shuffled version of the same feature
        (also known as permutation feature importance) or by assigning a constant value representing the median or mode of the feature,
        and computing the relative decrease in the model's predictive performance.

        A feature's importance score represents the performance drop that results when the model makes predictions on a perturbed copy
        of the data where this feature's values have been randomly shuffled across rows. A feature score of 0.01 would indicate that the
        predictive performance dropped by 0.01 when the feature was randomly shuffled or replaced. The higher the score a feature has,
        the more important it is to the model's performance.

        If a feature has a negative score, this means that the feature is likely harmful to the final model, and a model trained with
        the feature removed would be expected to achieve a better predictive performance. Note that calculating feature importance can
        be a computationally expensive process, particularly if the model uses many features. In many cases, this can take longer than
        the original model training. Roughly, this will equal to the number of features in the data multiplied by ``num_iterations``
        (or, 1 when ``method="naive"``) and time taken when ``evaluate()`` is called on a dataset with ``subsample_size``.

        Parameters
        ----------
        data : TimeSeriesDataFrame, pd.DataFrame, Path or str, optional
            The data to evaluate feature importances on. The last ``prediction_length`` time steps of the data set, for each
            item, will be held out for prediction and forecast accuracy will be calculated on these time steps.
            More accurate feature importances will be obtained from new data that was held-out during ``fit()``.

            The names and dtypes of columns and static features in ``data`` must match the ``train_data`` used to train
            the predictor.

            If provided data is a ``pandas.DataFrame``, AutoGluon will attempt to convert it to a ``TimeSeriesDataFrame``.
            If a ``str`` or a ``Path`` is provided, AutoGluon will attempt to load this file.

            If ``data`` is not provided, then validation (tuning) data provided during training (or the held out data used for
            validation if ``tuning_data`` was not explicitly provided ``fit()``) will be used.
        model : str, optional
            Name of the model that you would like to evaluate. By default, the best model during training
            (with highest validation score) will be used.
        metric : str or TimeSeriesScorer, optional
            Metric to be used for computing feature importance. If None, the ``eval_metric`` specified during initialization of
            the ``TimeSeriesPredictor`` will be used.
        features : list[str], optional
            List of feature names that feature importances are calculated for and returned. By default, all feature importances
            will be returned.
        method : {"permutation", "naive"}, default = "permutation"
            Method to be used for computing feature importance.

            * ``naive``: computes feature importance by replacing the values of each feature by a constant value and computing
              feature importances as the relative improvement in the evaluation metric. The constant value is the median for
              real-valued features and the mode for categorical features, for both covariates and static features, obtained from the
              feature values in ``data`` provided.
            * ``permutation``: computes feature importance by naively shuffling the values of the feature across different items
              and time steps. Each feature is shuffled for ``num_iterations`` times and feature importances are computed as the
              relative improvement in the evaluation metric. Refer to https://explained.ai/rf-importance/ for an explanation of
              permutation importance.

        subsample_size : int, default = 50
            The number of items to sample from ``data`` when computing feature importance. Larger values increase the accuracy of
            the feature importance scores. Runtime linearly scales with ``subsample_size``.
        time_limit : float, optional
            Time in seconds to limit the calculation of feature importance. If None, feature importance will calculate without early stopping.
            If ``method="permutation"``, a minimum of 1 full shuffle set will always be evaluated. If a shuffle set evaluation takes longer than
            ``time_limit``, the method will take the length of a shuffle set evaluation to return regardless of the ``time_limit``.
        num_iterations : int, optional
            The number of different iterations of the data that are evaluated. If ``method="permutation"``, this will be interpreted
            as the number of shuffle sets (equivalent to ``num_shuffle_sets`` in :meth:`TabularPredictor.feature_importance`). If ``method="naive"``, the
            constant replacement approach is repeated for ``num_iterations`` times, and a different subsample of data (of size ``subsample_size``) will
            be taken in each iteration.
            Default is 1 for ``method="naive"`` and 5 for ``method="permutation"``. The value will be ignored if ``method="naive"`` and the subsample
            size is greater than the number of items in ``data`` as additional iterations will be redundant.
            Larger values will increase the quality of the importance evaluation.
            It is generally recommended to increase ``subsample_size`` before increasing ``num_iterations``.
            Runtime scales linearly with ``num_iterations``.
        random_seed : int or None, default = 123
            If provided, fixes the seed of the random number generator for all models. This guarantees reproducible
            results for feature importance.
        relative_scores : bool, default = False
            By default, this method will return expected average *absolute* improvement in the eval metric due to the feature. If True, then
            the statistics will be computed over the *relative* (percentage) improvements.
        include_confidence_band: bool, default = True
            If True, returned DataFrame will include two additional columns specifying confidence interval for the true underlying importance value of
            each feature. Increasing ``subsample_size`` and ``num_iterations`` will tighten the confidence interval.
        confidence_level: float, default = 0.99
            This argument is only considered when ``include_confidence_band=True``, and can be used to specify the confidence level used
            for constructing confidence intervals. For example, if ``confidence_level`` is set to 0.99, then the returned DataFrame will include
            columns ``p99_high`` and ``p99_low`` which indicates that the true feature importance will be between ``p99_high`` and ``p99_low`` 99% of
            the time (99% confidence interval). More generally, if ``confidence_level`` = 0.XX, then the columns containing the XX% confidence interval
            will be named ``pXX_high`` and ``pXX_low``.

        Returns
        -------
        :class:`pd.DataFrame` of feature importance scores with 2 columns:
            index: The feature name.
            'importance': The estimated feature importance score.
            'stddev': The standard deviation of the feature importance score. If NaN, then not enough ``num_iterations`` were used.
        """
        self._assert_is_fit("feature_importance")
        if data is not None:
            data = self._check_and_prepare_data_frame(data)
            data = self._check_and_prepare_data_frame_for_evaluation(data)

        fi_df = self._learner.get_feature_importance(
            data=data,
            model=model,
            metric=metric,
            features=features,
            time_limit=time_limit,
            method=method,
            subsample_size=subsample_size,
            num_iterations=num_iterations,
            random_seed=random_seed,
            relative_scores=relative_scores,
            include_confidence_band=include_confidence_band,
            confidence_level=confidence_level,
        )
        return fi_df.sort_values("importance", ascending=False)

    @classmethod
    def _load_version_file(cls, path: str) -> str:
        """
        Loads the version file that is part of the saved predictor artifact.

        Parameters
        ----------
        path: str
            The path that would be used to load the predictor via `predictor.load(path)`

        Returns
        -------
        The version of AutoGluon used to fit the predictor, as a string.

        """
        version_file_path = os.path.join(path, cls._predictor_version_file_name)
        try:
            version = load_str.load(path=version_file_path)
        except:
            # Loads the old version file used in `autogluon.timeseries<=1.1.0`, named `__version__`.
            # This file name was changed because Kaggle does not allow uploading files named `__version__`.
            version_file_path = os.path.join(path, "__version__")
            version = load_str.load(path=version_file_path)
        return version

    @classmethod
    def load(cls, path: str | Path, require_version_match: bool = True) -> "TimeSeriesPredictor":
        """Load an existing ``TimeSeriesPredictor`` from given ``path``.

        .. warning::

            :meth:`autogluon.timeseries.TimeSeriesPredictor.load` uses ``pickle`` module implicitly, which is known to
            be insecure. It is possible to construct malicious pickle data which will execute arbitrary code during
            unpickling. Never load data that could have come from an untrusted source, or that could have been tampered
            with. **Only load data you trust.**

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
        path = setup_outputdir(path, warn_if_exist=False)

        predictor_path = Path(path) / cls.predictor_file_name
        if not predictor_path.exists():
            raise FileNotFoundError(f"No such file '{predictor_path}'")

        try:
            version_saved = cls._load_version_file(path=path)
        except:
            logger.warning(
                f'WARNING: Could not find version file at "{os.path.join(path, cls._predictor_version_file_name)}".\n'
                f"This means that the predictor was fit in an AutoGluon version `<=0.7.0`."
            )
            version_saved = "Unknown (Likely <=0.7.0)"

        check_saved_predictor_version(
            version_current=current_ag_version,
            version_saved=version_saved,
            require_version_match=require_version_match,
            logger=logger,
        )

        logger.info(f"Loading predictor from path {path}")
        learner = cls._learner_type.load(path)
        predictor = load_pkl.load(path=str(predictor_path))
        predictor._learner = learner
        predictor.path = learner.path
        return predictor

    def _save_version_file(self) -> None:
        version_file_contents = current_ag_version
        version_file_path = os.path.join(self.path, self._predictor_version_file_name)
        save_str.save(path=version_file_path, data=version_file_contents, verbose=False)

    def save(self) -> None:
        """Save this predictor to file in directory specified by this Predictor's ``path``.

        Note that :meth:`~autogluon.timeseries.TimeSeriesPredictor.fit` already saves the predictor object automatically
        (we do not recommend modifying the Predictor object yourself as it tracks many trained models).
        """
        tmp_learner = self._learner
        self._learner = None  # type: ignore
        save_pkl.save(path=os.path.join(tmp_learner.path, self.predictor_file_name), object=self)
        self._learner = tmp_learner
        self._save_version_file()

    def info(self) -> dict[str, Any]:
        """Returns a dictionary of objects each describing an attribute of the training process and trained models."""
        return self._learner.get_info(include_model_info=True)

    @property
    def model_best(self) -> str:
        """Returns the name of the best model from trainer."""
        self._assert_is_fit("model_best")
        if self._trainer.model_best is not None:
            models = self._trainer.get_model_names()
            if self._trainer.model_best in models:
                return self._trainer.model_best
        return self._trainer.get_model_best()

    def persist(self, models: Literal["all", "best"] | list[str] = "best", with_ancestors: bool = True) -> list[str]:
        """Persist models in memory for reduced inference latency. This is particularly important if the models are being used for online
        inference where low latency is critical. If models are not persisted in memory, they are loaded from disk every time they are
        asked to make predictions. This is especially cumbersome for large deep learning based models which have to be loaded into
        accelerator (e.g., GPU) memory each time.

        Parameters
        ----------
        models : list of str or str, default = 'best'
            Model names of models to persist.
            If 'best' then the model with the highest validation score is persisted (this is the model used for prediction by default).
            If 'all' then all models are persisted. Valid models are listed in this ``predictor`` by calling ``predictor.model_names()``.
        with_ancestors : bool, default = True
            If True, all ancestor models of the provided models will also be persisted.
            If False, ensemble models will not have the models they depend on persisted unless those models were specified in ``models``.
            This will slow down inference as the ancestor models will still need to be loaded from disk for each predict call.
            Only relevant for ensemble models.

        Returns
        -------
        list_of_models : list[str]
            List of persisted model names.
        """
        self._assert_is_fit("persist")
        return self._learner.persist_trainer(models=models, with_ancestors=with_ancestors)

    def unpersist(self) -> list[str]:
        """Unpersist models in memory for reduced memory usage. If models are not persisted in memory, they are loaded from
        disk every time they are asked to make predictions.

        Note: Another way to reset the predictor and unpersist models is to reload the predictor from disk
        via ``predictor = TimeSeriesPredictor.load(predictor.path)``.

        Returns
        -------
        list_of_models : list[str]
            List of unpersisted model names.
        """
        return self._learner.unpersist_trainer()

    def leaderboard(
        self,
        data: TimeSeriesDataFrame | pd.DataFrame | Path | str | None = None,
        cutoff: int | None = None,
        extra_info: bool = False,
        extra_metrics: list[str | TimeSeriesScorer] | None = None,
        display: bool = False,
        use_cache: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Return a leaderboard showing the performance of every trained model, the output is a
        pandas dataframe with columns:

        * ``model``: The name of the model.
        * ``score_test``: The test score of the model on ``data``, if provided. Computed according to ``eval_metric``.
        * ``score_val``: The validation score of the model using the internal validation data. Computed according to ``eval_metric``.

        .. note::
            Metrics are always reported in 'higher is better' format.
            This means that metrics such as MASE or MAPE will be multiplied by -1, so their values will be negative.
            This is necessary to avoid the user needing to know the metric to understand if higher is better when
            looking at the leaderboard.

        * ``pred_time_val``: Time taken by the model to predict on the validation data set
        * ``fit_time_marginal``: The fit time required to train the model (ignoring base models for ensembles).
        * ``fit_order``: The order in which models were fit. The first model fit has ``fit_order=1``, and the Nth
          model fit has ``fit_order=N``.

        Parameters
        ----------
        data : TimeSeriesDataFrame | pd.DataFrame | Path | str, optional
            dataset used for additional evaluation. Must include both historical and future data (i.e., length of all
            time series in ``data`` must be at least ``prediction_length + 1``, if ``cutoff`` is not provided,
            ``-cutoff + 1`` otherwise).

            The names and dtypes of columns and static features in ``data`` must match the ``train_data`` used to train
            the predictor.

            If provided data is a ``pandas.DataFrame``, AutoGluon will attempt to convert it to a ``TimeSeriesDataFrame``.
            If a ``str`` or a ``Path`` is provided, AutoGluon will attempt to load this file.
        cutoff : int, optional
            A *negative* integer less than or equal to ``-1 * prediction_length`` denoting the time step in ``data``
            where the forecast evaluation starts, i.e., time series are evaluated from the ``-cutoff``-th to the
            ``-cutoff + prediction_length``-th time step. Defaults to ``-1 * prediction_length``, using the last
            ``prediction_length`` time steps of each time series for evaluation.
        extra_info : bool, default = False
            If True, the leaderboard will contain an additional column ``hyperparameters`` with the hyperparameters used
            by each model during training. An empty dictionary ``{}`` means that the model was trained with default
            hyperparameters.
        extra_metrics : list[str | TimeSeriesScorer], optional
            A list of metrics to calculate scores for and include in the output DataFrame.

            Only valid when ``data`` is specified. The scores refer to the scores on ``data`` (same data as used to
            calculate the ``score_test`` column).

            This list can contain any values which would also be valid for ``eval_metric`` when creating a :class:`~autogluon.timeseries.TimeSeriesPredictor`.

            For each provided ``metric``, a column with name ``str(metric)`` will be added to the leaderboard, containing
            the value of the metric computed on ``data``.
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
        self._assert_is_fit("leaderboard")
        if "silent" in kwargs:
            # keep `silent` logic for backwards compatibility
            assert isinstance(kwargs["silent"], bool)
            display = not kwargs.pop("silent")
        if len(kwargs) > 0:
            for key in kwargs:
                raise TypeError(f"TimeSeriesPredictor.leaderboard() got an unexpected keyword argument '{key}'")
        if data is None and extra_metrics is not None:
            raise ValueError("`extra_metrics` is only valid when `data` is specified.")
        if data is None and cutoff is not None:
            raise ValueError("`cutoff` is only valid when `data` is specified.")

        if data is not None:
            data = self._check_and_prepare_data_frame(data)
            data = self._check_and_prepare_data_frame_for_evaluation(data, cutoff=cutoff)

        leaderboard = self._learner.leaderboard(
            data, extra_info=extra_info, extra_metrics=extra_metrics, use_cache=use_cache
        )
        if display:
            with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
                print(leaderboard)
        return leaderboard

    def make_future_data_frame(self, data: TimeSeriesDataFrame | pd.DataFrame | Path | str) -> pd.DataFrame:
        """Generate a dataframe with the ``item_id`` and ``timestamp`` values corresponding to the forecast horizon.

        Parameters
        ----------
        data : TimeSeriesDataFrame | pd.DataFrame | Path | str
            Historical time series data.

        Returns
        -------
        forecast_horizon : pd.DataFrame
            Data frame with columns ``item_id`` and ``timestamp`` corresponding to the forecast horizon. For each item ID
            in ``data``, ``forecast_horizon`` will contain the timestamps for the next ``prediction_length`` time steps,
            following the end of each series in the input data.

        Examples
        --------
        >>> print(data)
                            target
        item_id timestamp
        A       2024-01-01       0
                2024-01-02       1
                2024-01-03       2
        B       2024-04-07       3
                2024-04-08       4
        >>> predictor = TimeSeriesPredictor(prediction_length=2, freq="D")
        >>> print(predictor.make_future_data_frame(data))
          item_id  timestamp
        0       A 2024-01-04
        0       A 2024-01-05
        1       B 2024-04-09
        1       B 2024-04-10
        """
        if self.freq is None:
            raise ValueError("Please fit the predictor before calling `make_future_data_frame`")
        data = self._check_and_prepare_data_frame(data)
        return make_future_data_frame(data, prediction_length=self.prediction_length, freq=self.freq)

    def fit_summary(self, verbosity: int = 1) -> dict[str, Any]:
        """Output summary of information about models produced during
        :meth:`~autogluon.timeseries.TimeSeriesPredictor.fit`.

        Parameters
        ----------
        verbosity : int, default = 1
            Controls the detail level of summary to output. Set 0 for no output printing.

        Returns
        -------
        summary_dict : dict[str, Any]
            Dict containing various detailed information. We do not recommend directly printing this dict as it may
            be very large.
        """
        self._assert_is_fit("fit_summary")
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
            model_hyperparams[model_name] = model_obj.get_hyperparameters()

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

    def refit_full(self, model: str = "all", set_best_to_refit_full: bool = True) -> dict[str, str]:
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
        self._assert_is_fit("refit_full")
        logger.warning(
            "\tWARNING: refit_full functionality for TimeSeriesPredictor is experimental "
            "and is not yet supported by all models."
        )

        logger.info(
            "Refitting models via `refit_full` using all of the data (combined train and validation)...\n"
            "\tModels trained in this way will have the suffix '_FULL' and have NaN validation score.\n"
            "\tThis process is not bound by time_limit, but should take less time than the original `fit` call."
        )
        model_best = self.model_best
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

    def _simulation_artifact(self, test_data: TimeSeriesDataFrame) -> dict:
        """[Advanced] Computes and returns the necessary information to perform offline ensemble simulation."""

        def select_target(ts_df: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
            ts_df = ts_df.copy()
            ts_df.static_features = None
            return cast(TimeSeriesDataFrame, ts_df[[self.target]])

        test_data = self._check_and_prepare_data_frame(test_data)
        test_data = self._check_and_prepare_data_frame_for_evaluation(test_data, name="test_data")
        test_data = self._learner.feature_generator.transform(test_data)

        trainer = self._trainer
        train_data = trainer.load_train_data()
        val_data = trainer.load_val_data()
        base_model_names = trainer.get_model_names(layer=0)
        pred_proba_dict_val: dict[str, list[TimeSeriesDataFrame]] = {
            model_name: trainer._get_model_oof_predictions(model_name)
            for model_name in base_model_names
            if "_FULL" not in model_name
        }

        past_data, known_covariates = test_data.get_model_inputs_for_scoring(
            prediction_length=self.prediction_length,
            known_covariates_names=trainer.covariate_metadata.known_covariates,
        )
        pred_proba_dict_test, _ = trainer.get_model_pred_dict(
            base_model_names, data=past_data, known_covariates=known_covariates
        )

        y_val: list[TimeSeriesDataFrame] = [
            select_target(df) for df in trainer._get_validation_windows(train_data=train_data, val_data=val_data)
        ]
        y_test: TimeSeriesDataFrame = select_target(test_data)

        simulation_dict = dict(
            pred_proba_dict_val=pred_proba_dict_val,
            pred_proba_dict_test=pred_proba_dict_test,
            y_val=y_val,
            y_test=y_test,
            target=self.target,
            prediction_length=self.prediction_length,
            eval_metric=self.eval_metric.name,
            eval_metric_seasonal_period=self.eval_metric.seasonal_period,
            horizon_weight=self.eval_metric.horizon_weight,
            quantile_levels=self.quantile_levels,
        )
        return simulation_dict

    def plot(
        self,
        data: TimeSeriesDataFrame | pd.DataFrame | Path | str,
        predictions: TimeSeriesDataFrame | None = None,
        quantile_levels: list[float] | None = None,
        item_ids: list[str | int] | None = None,
        max_num_item_ids: int = 8,
        max_history_length: int | None = None,
        point_forecast_column: str | None = None,
        matplotlib_rc_params: dict | None = None,
    ):
        """Plot historical time series values and the forecasts.

        Parameters
        ----------
        data : TimeSeriesDataFrame | pd.DataFrame | Path | str
            Observed time series data.
        predictions : TimeSeriesDataFrame, optional
            Predictions generated by calling :meth:`~autogluon.timeseries.TimeSeriesPredictor.predict`.
        quantile_levels : list[float], optional
            Quantile levels for which to plot the prediction intervals. Defaults to lowest & highest quantile levels
            available in ``predictions``.
        item_ids : list[str | int], optional
            If provided, plots will only be generated for time series with these item IDs. By default (if set to
            ``None``), item IDs are selected randomly. In either case, plots are generated for at most
            ``max_num_item_ids`` time series.
        max_num_item_ids : int, default = 8
            At most this many time series will be plotted by the method.
        max_history_length : int, optional
            If provided, at most this many time steps will be shown for each time series in ``data``.
        point_forecast_column : str, optional
            Name of the column in ``predictions`` that will be plotted as the point forecast. Defaults to ``"0.5"``,
            if this column is present in ``predictions``, otherwise ``"mean"``.
        matplotlib_rc_params : dict, optional
            Dictionary describing the plot style that will be passed to `matplotlib.pyplot.rc_context <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.rc_context.html>`_.
            See `matplotlib documentation <https://matplotlib.org/stable/users/explain/customizing.html#the-default-matplotlibrc-file>`_ for the list of available options.
        """
        import matplotlib.pyplot as plt

        data = self._check_and_prepare_data_frame(data)
        if item_ids is None:
            item_ids = list(np.random.choice(data.item_ids, size=min(max_num_item_ids, data.num_items), replace=False))
        else:
            item_ids = list(item_ids)[:max_num_item_ids]

        if predictions is not None:
            if (
                not isinstance(predictions, TimeSeriesDataFrame)
                or "mean" not in predictions.columns
                or predictions.index.nlevels != 2
            ):
                raise ValueError("predictions must be a TimeSeriesDataFrame produced by predictor.predict()")
            if point_forecast_column is None:
                point_forecast_column = "0.5" if "0.5" in predictions.columns else "mean"
            if quantile_levels is None:
                available_quantile_levels = [float(q) for q in predictions.columns if q != "mean"]
                if len(available_quantile_levels) >= 2:
                    quantile_levels = [min(available_quantile_levels), max(available_quantile_levels)]
                else:
                    quantile_levels = []

        if len(item_ids) == 1:
            ncols = 1
            nrows = 1
        else:
            ncols = 2
            nrows = math.ceil(len(item_ids) / ncols)

        rc_params = {
            "font.size": 10,
            "figure.figsize": [20, 3.5 * nrows],
            "figure.dpi": 100,
            "legend.loc": "upper center",
        }
        if matplotlib_rc_params is not None:
            rc_params.update(matplotlib_rc_params)

        with plt.rc_context(rc_params):
            fig, axes = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False)
            fig.tight_layout(h_pad=2.5, w_pad=0.5)
            axes = axes.ravel()

            for i, (item_id, ax) in enumerate(zip(item_ids, axes)):
                ax.set_title(item_id)
                ax.grid()
                # Label the x axis for subplots in the lowest row
                if i // nrows == 1:
                    ax.set_xlabel("Time")
                # Label the y axis for subplots in the leftmost column
                if i % ncols == 0:
                    ax.set_ylabel(self.target)

                ts = data.loc[item_id][self.target]
                if max_history_length is not None:
                    ts = ts.iloc[-max_history_length:]
                ax.plot(ts, label="Observed", color="C0")

                if predictions is not None:
                    forecast: pd.DataFrame = predictions.loc[item_id]  # type: ignore
                    point_forecast = forecast[point_forecast_column]
                    ax.plot(point_forecast, color="C1", label="Forecast")
                    if quantile_levels is not None:
                        for q in quantile_levels:
                            ax.fill_between(forecast.index, point_forecast, forecast[str(q)], color="C1", alpha=0.2)
            if len(axes) > len(item_ids):
                axes[len(item_ids)].set_axis_off()  # type: ignore
            handles, labels = axes[0].get_legend_handles_labels()  # type: ignore
            fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.0), ncols=len(handles))
        return fig

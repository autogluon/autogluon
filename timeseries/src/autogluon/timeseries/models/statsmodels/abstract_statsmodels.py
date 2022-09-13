import logging
import re
from dataclasses import dataclass
from functools import partial
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.base.tsa_model import TimeSeriesModelResults as StatsmodelsTSModelResults

from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.hashing import hash_ts_dataframe_items

logger = logging.getLogger(__name__)


@dataclass
class FittedLocalModel:
    """Stores the parameters of a fitted Statsmodels time series model.

    FittedLocalModel doesn't store intermediate results and therefore takes up very little disk space when pickled. In
    contrast, StatsmodelsFittedModel stores the intermediate results and therefore can often take several gigabytes
    when pickled.

    FittedLocalModel is consumed by AbstractStatsmodelsModel._predict_using_fit_summary to generate predictions.

    Attributes
    ----------
    model_name:
        Name of the class
    sm_model_init_args:
        Arguments passed to the __init__ method of the SM model
    parameters:
        Estimated learnable parameters of the model (after completing the fitting procedure)
    """

    model_name: str
    sm_model_init_args: Dict[str, Any]
    parameters: Dict[str, Any]


class AbstractStatsmodelsModel(AbstractTimeSeriesModel):
    """Abstract class for local models that are fitted for each time series in a dataset.

    Parameters
    ----------
    path: str
        directory to store model artifacts.
    freq: str
        string representation (compatible with GluonTS frequency strings) for the data provided.
        For example, "1D" for daily data, "1H" for hourly data, etc.
    prediction_length: int
        Number of time steps ahead (length of the forecast horizon) the model will be optimized
        to predict. At inference time, this will be the number of time steps the model will
        predict.
    name: str
        Name of the model. Also, name of subdirectory inside path where model will be saved.
    eval_metric: str
        objective function the model will be scored on, will use mean_wQuantileLoss by default.
    hyperparameters:
        various hyperparameters that will be used by model (can be search spaces instead of
        fixed values). See *Other Parameters* in each inheriting model's documentation for
        possible values.
    """

    statsmodels_allowed_init_args: List[str] = []
    statsmodels_allowed_fit_args: List[str] = []
    quantile_method_name: str = "pred_int"

    def __init__(
        self,
        freq: Optional[str] = None,
        prediction_length: int = 1,
        path: Optional[str] = None,
        name: Optional[str] = None,
        eval_metric: str = None,
        hyperparameters: Dict[str, Any] = None,
        **kwargs,  # noqa
    ):
        name = name or re.sub(r"Model$", "", self.__class__.__name__)
        super().__init__(
            path=path,
            freq=freq,
            prediction_length=prediction_length,
            name=name,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            **kwargs,
        )
        if hyperparameters is None:
            hyperparameters = {}
        # Use 50% of the available CPU cores by default
        n_jobs = hyperparameters.get("n_jobs", 0.5)
        if isinstance(n_jobs, float) and 0 < n_jobs <= 1.0:
            self.n_jobs = max(int(cpu_count() * n_jobs), 1)
        elif isinstance(n_jobs, int):
            self.n_jobs = n_jobs
        else:
            raise ValueError(f"n_jobs must be a float between 0 and 1 or an integer (received n_jobs = {n_jobs})")

        self._fitted_models: Dict[str, FittedLocalModel] = {}

    def _get_sm_model_init_and_fit_args(self) -> Tuple[dict, dict]:
        """Separate arguments that will be passed to __init__ and fit methods of the underlying SM model."""
        sm_model_init_args = {}
        sm_model_fit_args = {}
        unused_args = []
        for key, value in self._get_model_params().items():
            if key in self.statsmodels_allowed_init_args:
                sm_model_init_args[key] = value
            elif key in self.statsmodels_allowed_fit_args:
                sm_model_fit_args[key] = value
            elif key == "n_jobs":
                # n_jobs isn't passed to the underlying statsmodels model
                pass
            else:
                unused_args.append(key)
        if len(unused_args) > 0:
            logger.warning(
                f" {self.name} ignores following hyperparameters: {unused_args}. "
                f"See the docstring of {self.name} for the list of supported hyperparameters."
            )
        return sm_model_init_args, sm_model_fit_args

    def _update_sm_model_init_args(self, sm_model_init_args: dict, data: TimeSeriesDataFrame) -> dict:
        """Update the arguments passed to the underlying SM model.

        This method takes care of
         - renaming arguments if AutoGluon API uses a different name than the underlying SM model
         - overriding the default values
         - fixing incompatible configurations (e.g., disabling seasonality if the seasonal period is unknown)
        """
        raise NotImplementedError

    def _fit_local_model(
        self, timeseries: pd.Series, sm_model_init_args: dict, sm_model_fit_args: dict
    ) -> FittedLocalModel:
        """Fit a single local model to the time series."""
        raise NotImplementedError

    def _fit(self, train_data: TimeSeriesDataFrame, time_limit: int = None, **kwargs):
        verbosity = kwargs.get("verbosity", 2)
        set_logger_verbosity(verbosity, logger=logger)
        self._check_fit_params()

        if self.freq is None:
            self.freq = train_data.freq
        else:
            if self.freq != train_data.freq:
                raise RuntimeError(
                    f"Frequency {train_data.freq} of the dataset must match the frequency "
                    f"{self.freq} of the {self.name} model"
                )

        # Select timeseries that don't have a fitted local model
        train_hash = hash_ts_dataframe_items(train_data)
        items_to_fit = [item_id for item_id, ts_hash in train_hash.iteritems() if ts_hash not in self._fitted_models]
        timeseries_to_fit = (train_data.loc[item_id][self.target] for item_id in items_to_fit)

        # Get initialization / fitting args and update them, if necessary
        sm_model_init_args, sm_model_fit_args = self._get_sm_model_init_and_fit_args()
        sm_model_init_args = self._update_sm_model_init_args(sm_model_init_args=sm_model_init_args, data=train_data)

        # Fit models in parallel
        fit_fn = partial(
            self._fit_local_model, sm_model_fit_args=sm_model_fit_args, sm_model_init_args=sm_model_init_args
        )
        fitted_models = Parallel(n_jobs=self.n_jobs)(delayed(fit_fn)(timeseries=ts) for ts in timeseries_to_fit)
        for item_id, model in zip(items_to_fit, fitted_models):
            self._fitted_models[train_hash.loc[item_id]] = model

    def predict(self, data: TimeSeriesDataFrame, quantile_levels: List[float] = None, **kwargs) -> TimeSeriesDataFrame:
        self._check_predict_inputs(data, quantile_levels=quantile_levels, **kwargs)
        if quantile_levels is None:
            quantile_levels = self.quantile_levels

        # Make sure that we fitted a local model to each time series in data
        data_hash = hash_ts_dataframe_items(data)
        items_to_fit = [item_id for item_id, ts_hash in data_hash.iteritems() if ts_hash not in self._fitted_models]
        if len(items_to_fit) > 0:
            logger.info(f"{self.name} received {len(items_to_fit)} items not seen during training, re-running fit")
            self._fit(train_data=data)

        # Make predictions in parallel
        predict_fn = partial(self._predict_with_local_model, quantile_levels=quantile_levels)
        timeseries_with_models = (
            (data.loc[item_id][self.target], self._fitted_models[ts_hash])
            for item_id, ts_hash in data_hash.iteritems()
        )
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(predict_fn)(timeseries=ts, fitted_model=model) for ts, model in timeseries_with_models
        )

        # Combine all predictions into a single DataFrame
        result_df = pd.concat({item_id: pred for item_id, pred in zip(data.item_ids, predictions)})
        result_df.index.rename([ITEMID, TIMESTAMP], inplace=True)
        return TimeSeriesDataFrame(result_df)

    def _predict_with_local_model(
        self, timeseries: pd.Series, fitted_model: FittedLocalModel, quantile_levels: List[float]
    ) -> pd.DataFrame:
        """Make predictions for a single time series using the fitted local model.

        Parameters
        ----------
        timeseries:
            Time series containing target values with timestamp index.
        fitted_model:
            Fitted local model that will be used to make predictions for this time series.
        quantile_levels:
            List of quantiles that should be predicted.

        Returns
        -------
        pred_df:
            DataFrame with timestamp as index and mean & quantile forecasts as columns.
        """
        raise NotImplementedError

    def _get_predictions_from_statsmodels_model(
        self, sm_model: StatsmodelsTSModelResults, cutoff: pd.Timestamp, quantile_levels: List[float], freq: str
    ) -> pd.DataFrame:
        """Make predictions using an initialized statsmodels model.

        This method should be called inside _predict_with_local_model.

        Parameters
        ----------
        sm_model:
            Statsmodels model that was created from a FittedLocalModel and initialized with the observed data.
        cutoff:
            Timestamp of the last observation in the observed time series.
        quantile_levels:
            List of quantiles that should be predicted.
        freq:
            Pandas frequency string.

        Returns
        -------
        pred_df:
            DataFrame with timestamp as index and mean & quantile forecasts as columns.
        """
        start = cutoff + pd.tseries.frequencies.to_offset(freq)
        end = cutoff + self.prediction_length * pd.tseries.frequencies.to_offset(freq)
        predictions = sm_model.get_prediction(start=start, end=end)
        results = [predictions.predicted_mean.rename("mean")]
        for q in quantile_levels:
            if q < 0.5:
                coverage = 2 * q
                column_index = 0
            else:
                coverage = 2 * (1 - q)
                column_index = 1
            # Different statsmodels models call the method that produces probabilistic forecasts differently, we store
            # the correct method name in self.quantile_method_name
            quantile_pred = getattr(predictions, self.quantile_method_name)(alpha=coverage)
            # Select lower bound of the confidence interval if q < 0.5, upper bound otherwise
            results.append(quantile_pred.iloc[:, column_index].rename(str(q)))
        return pd.concat(results, axis=1)

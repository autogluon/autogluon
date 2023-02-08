import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Type

import gluonts
import gluonts.core.settings
import numpy as np
import pandas as pd
from gluonts.dataset.common import Dataset as GluonTSDataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.estimator import Estimator as GluonTSEstimator
from gluonts.model.forecast import Forecast, QuantileForecast, SampleForecast
from gluonts.model.predictor import Predictor as GluonTSPredictor
from gluonts.torch.model.forecast import DistributionForecast
from pandas.tseries.frequencies import to_offset

from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.core.hpo.constants import RAY_BACKEND
from autogluon.core.utils import warning_filter
from autogluon.core.utils.savers import save_pkl
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.forecast import get_forecast_horizon_index_ts_dataframe
from autogluon.timeseries.utils.warning_filters import disable_root_logger

logger = logging.getLogger(__name__)
gts_logger = logging.getLogger(gluonts.__name__)


GLUONTS_SUPPORTED_OFFSETS = ["Y", "Q", "M", "W", "D", "B", "H", "T", "min", "S"]


class SimpleGluonTSDataset(GluonTSDataset):
    """A simple GluonTS dataset that wraps a TimeSeriesDataFrame and implements the
    GluonTS Dataset protocol via lazy iterations.
    """

    def __init__(
        self,
        target_df: TimeSeriesDataFrame,
        target_column: str = "target",
        feat_static_cat: Optional[pd.DataFrame] = None,
        feat_static_real: Optional[pd.DataFrame] = None,
        feat_dynamic_real: Optional[pd.DataFrame] = None,
        past_feat_dynamic_real: Optional[pd.DataFrame] = None,
        float_dtype: Type = np.float64,
        int_dtype: Type = np.int64,
    ):
        assert target_df is not None
        assert target_df.freq, "Initializing GluonTS data sets without freq is not allowed"
        # Convert TimeSeriesDataFrame to pd.Series for faster processing
        self.target_series = target_df[target_column]
        self.item_ids = target_df.item_ids
        self.freq_ = target_df.freq
        self.feat_static_cat = feat_static_cat
        self.feat_static_real = feat_static_real
        self.feat_dynamic_real = feat_dynamic_real
        self.past_feat_dynamic_real = past_feat_dynamic_real

        self.int_dtype = int_dtype
        self.float_dtype = float_dtype

    @property
    def freq(self):
        # FIXME: GluonTS expects a frequency string, but only supports a limited number of such strings
        # for feature generation. If the frequency string doesn't match or is not provided, it raises an exception.
        # Here we bypass this by issuing a default "yearly" frequency, tricking it into not producing
        # any lags or features.
        pd_offset = to_offset(self.freq_)

        # normalize freq str to handle peculiarities such as W-SUN
        offset_base_alias = pd_offset.name.split("-")[0]

        return "A" if offset_base_alias is None or offset_base_alias not in GLUONTS_SUPPORTED_OFFSETS else self.freq_

    def __len__(self):
        return len(self.item_ids)  # noqa

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for item_id in self.item_ids:  # noqa
            ts = self.target_series.loc[item_id]
            time_series = {
                FieldName.ITEM_ID: item_id,
                FieldName.TARGET: ts.to_numpy(dtype=self.float_dtype).ravel(),
                FieldName.START: pd.Period(ts.index[0], freq=self.freq),
            }
            if self.feat_static_cat is not None:
                time_series[FieldName.FEAT_STATIC_CAT] = self.feat_static_cat.loc[item_id].to_numpy(
                    dtype=self.int_dtype
                )
            if self.feat_static_real is not None:
                time_series[FieldName.FEAT_STATIC_REAL] = self.feat_static_real.loc[item_id].to_numpy(
                    dtype=self.float_dtype
                )
            if self.feat_dynamic_real is not None:
                time_series[FieldName.FEAT_DYNAMIC_REAL] = (
                    self.feat_dynamic_real.loc[item_id].to_numpy(dtype=self.float_dtype).T
                )
            if self.past_feat_dynamic_real is not None:
                time_series[FieldName.PAST_FEAT_DYNAMIC_REAL] = (
                    self.past_feat_dynamic_real.loc[item_id].to_numpy(dtype=self.float_dtype).T
                )

            yield time_series


class AbstractGluonTSModel(AbstractTimeSeriesModel):
    """Abstract class wrapping GluonTS estimators for use in autogluon.timeseries.

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
        objective function the model intends to optimize, will use mean_wQuantileLoss by default.
    hyperparameters:
        various hyperparameters that will be used by model (can be search spaces instead of
        fixed values). See *Other Parameters* in each inheriting model's documentation for
        possible values.
    """

    gluonts_model_path = "gluon_ts"
    gluonts_estimator_class: Type[GluonTSEstimator] = None
    # datatype of floating point and integers passed internally to GluonTS
    float_dtype: Type = np.float64
    int_dtype: Type = np.int64
    # default number of samples for prediction
    default_num_samples: int = 1000
    supports_known_covariates: bool = False
    supports_past_covariates: bool = False

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
        name = name or re.sub(r"Model$", "", self.__class__.__name__)  # TODO: look name up from presets
        super().__init__(
            path=path,
            freq=freq,
            prediction_length=prediction_length,
            name=name,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            **kwargs,
        )
        self.gts_predictor: Optional[GluonTSPredictor] = None
        self.callbacks = []
        self.num_feat_static_cat = 0
        self.num_feat_static_real = 0
        self.num_feat_dynamic_real = 0
        self.num_past_feat_dynamic_real = 0
        self.feat_static_cat_cardinality: List[int] = []

    def save(self, path: str = None, **kwargs) -> str:
        if path is None:
            path = self.path
        path = Path(path)
        path.mkdir(exist_ok=True)

        predictor = self.gts_predictor
        self.gts_predictor = None

        with disable_root_logger():
            if predictor:
                Path.mkdir(path / self.gluonts_model_path, exist_ok=True)
                predictor.serialize(path / self.gluonts_model_path)

        save_pkl.save(path=str(path / self.model_file_name), object=self)
        self.gts_predictor = predictor

        return str(path)

    @classmethod
    def load(cls, path: str, reset_paths: bool = True, verbose: bool = True) -> "AbstractGluonTSModel":
        model = super().load(path, reset_paths, verbose)
        model.gts_predictor = GluonTSPredictor.deserialize(Path(path) / cls.gluonts_model_path)
        return model

    def _deferred_init_params_aux(self, **kwargs) -> None:
        """Update GluonTS specific parameters with information available
        only at training time.
        """
        if "dataset" in kwargs:
            ds = kwargs.get("dataset")
            self.freq = ds.freq or self.freq
            if not self.freq:
                raise ValueError(
                    "Dataset frequency not provided in the dataset, fit arguments or "
                    "during initialization. Please provide a `freq` string to `fit`."
                )

            model_params = self._get_model_params()
            disable_static_features = model_params.get("disable_static_features", False)
            if not disable_static_features:
                self.num_feat_static_cat = len(self.metadata.static_features_cat)
                self.num_feat_static_real = len(self.metadata.static_features_real)
                if self.num_feat_static_cat > 0:
                    feat_static_cat = ds.static_features[self.metadata.static_features_cat]
                    self.feat_static_cat_cardinality = feat_static_cat.nunique().tolist()
            disable_known_covariates = model_params.get("disable_known_covariates", False)
            if not disable_known_covariates and self.supports_known_covariates:
                self.num_feat_dynamic_real = len(self.metadata.known_covariates_real)
            disable_past_covariates = model_params.get("disable_past_covariates", False)
            if not disable_past_covariates and self.supports_past_covariates:
                self.num_past_feat_dynamic_real = len(self.metadata.past_covariates_real)

        if "callbacks" in kwargs:
            self.callbacks += kwargs["callbacks"]

    def _get_model_params(self) -> dict:
        """Gets params that are passed to the inner model."""
        args = super()._get_model_params().copy()
        args.setdefault("batch_size", 64)
        args.update(
            dict(
                freq=self.freq,
                prediction_length=self.prediction_length,
                quantiles=self.quantile_levels,
                callbacks=self.callbacks,
            )
        )
        return args

    def _get_estimator_init_args(self) -> Dict[str, Any]:
        """Get GluonTS specific constructor arguments for estimator objects, an alias to
        `self._get_model_params` for better readability."""
        return self._get_model_params()

    def _get_estimator(self) -> GluonTSEstimator:
        """Return the GluonTS Estimator object for the model"""
        with warning_filter():
            return self.gluonts_estimator_class.from_hyperparameters(**self._get_estimator_init_args())

    def _to_gluonts_dataset(
        self, time_series_df: Optional[TimeSeriesDataFrame], known_covariates: Optional[TimeSeriesDataFrame] = None
    ) -> Optional[GluonTSDataset]:
        if time_series_df is not None:
            if self.num_feat_static_cat > 0:
                feat_static_cat = time_series_df.static_features[self.metadata.static_features_cat]
            else:
                feat_static_cat = None

            if self.num_feat_static_real > 0:
                feat_static_real = time_series_df.static_features[self.metadata.static_features_real]
            else:
                feat_static_real = None

            if self.num_feat_dynamic_real > 0:
                # Convert TSDF -> DF to avoid overhead / input validation
                feat_dynamic_real = pd.DataFrame(time_series_df[self.metadata.known_covariates_real])
                # Append future values of known covariates
                if known_covariates is not None:
                    feat_dynamic_real = pd.concat([feat_dynamic_real, known_covariates], axis=0)
                    expected_length = len(time_series_df) + self.prediction_length * time_series_df.num_items
                    if len(feat_dynamic_real) != expected_length:
                        raise ValueError(
                            f"known_covariates must contain values for the next prediction_length = "
                            f"{self.prediction_length} time steps in each time series."
                        )
            else:
                feat_dynamic_real = None

            if self.num_past_feat_dynamic_real > 0:
                # Convert TSDF -> DF to avoid overhead / input validation
                past_feat_dynamic_real = pd.DataFrame(time_series_df[self.metadata.past_covariates_real])
            else:
                past_feat_dynamic_real = None

            return SimpleGluonTSDataset(
                target_df=time_series_df,
                target_column=self.target,
                feat_static_cat=feat_static_cat,
                feat_static_real=feat_static_real,
                feat_dynamic_real=feat_dynamic_real,
                past_feat_dynamic_real=past_feat_dynamic_real,
                float_dtype=self.float_dtype,
                int_dtype=self.int_dtype,
            )
        else:
            return None

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: int = None,
        **kwargs,
    ) -> None:
        verbosity = kwargs.get("verbosity", 2)
        set_logger_verbosity(verbosity, logger=logger)
        gts_logger.setLevel(logging.ERROR if verbosity <= 3 else logging.INFO)

        if verbosity > 3:
            logger.warning(
                "GluonTS logging is turned on during training. Note that losses reported by GluonTS "
                "may not correspond to those specified via `eval_metric`."
            )

        self._check_fit_params()

        # update auxiliary parameters
        self._deferred_init_params_aux(
            dataset=train_data, callbacks=self._get_callbacks(time_limit=time_limit), **kwargs
        )

        estimator = self._get_estimator()
        with warning_filter(), disable_root_logger(), gluonts.core.settings.let(gluonts.env.env, use_tqdm=False):
            self.gts_predictor = estimator.train(
                self._to_gluonts_dataset(train_data),
                validation_data=self._to_gluonts_dataset(val_data),
                cache_data=True,
            )

    def _get_callbacks(self, time_limit: int, *args, **kwargs) -> List[Callable]:
        """Retrieve a list of callback objects for the GluonTS trainer"""
        return []

    def predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        quantile_levels: List[float] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        if self.gts_predictor is None:
            raise ValueError("Please fit the model before predicting.")

        logger.debug(f"Predicting with time series model {self.name}")
        logger.debug(
            f"\tProvided data for prediction with {len(data)} rows, {data.num_items} items. "
            f"Average time series length is {len(data) / data.num_items}."
        )
        with warning_filter(), gluonts.core.settings.let(gluonts.env.env, use_tqdm=False):
            quantiles = quantile_levels or self.quantile_levels
            if not all(0 < q < 1 for q in quantiles):
                raise ValueError("Invalid quantile value specified. Quantiles must be between 0 and 1 (exclusive).")

            predicted_targets = self._predict_gluonts_forecasts(data, known_covariates=known_covariates, **kwargs)

            df = self._gluonts_forecasts_to_data_frame(
                predicted_targets,
                quantile_levels=quantile_levels or self.quantile_levels,
                forecast_index=get_forecast_horizon_index_ts_dataframe(data, self.prediction_length),
            )

        return df

    def _predict_gluonts_forecasts(
        self, data: TimeSeriesDataFrame, known_covariates: Optional[TimeSeriesDataFrame] = None, **kwargs
    ) -> List[Forecast]:
        gts_data = self._to_gluonts_dataset(data, known_covariates=known_covariates)

        predictor_kwargs = dict(dataset=gts_data)
        predictor_kwargs["num_samples"] = kwargs.get("num_samples", self.default_num_samples)

        return list(self.gts_predictor.predict(**predictor_kwargs))

    @staticmethod
    def _sample_to_quantile_forecast(forecast: SampleForecast, quantile_levels: List[float]) -> QuantileForecast:
        forecast_arrays = [forecast.mean]

        quantile_keys = [str(q) for q in quantile_levels]
        for q in quantile_keys:
            forecast_arrays.append(forecast.quantile(q))

        forecast_init_args = dict(
            forecast_arrays=np.array(forecast_arrays),
            start_date=forecast.start_date,
            forecast_keys=["mean"] + quantile_keys,
            item_id=str(forecast.item_id),
        )
        if isinstance(forecast.start_date, pd.Timestamp):  # GluonTS version is <0.10
            forecast_init_args.update({"freq": forecast.freq})
        return QuantileForecast(**forecast_init_args)

    @staticmethod
    def _distribution_to_quantile_forecast(forecast: SampleForecast, quantile_levels: List[float]) -> QuantileForecast:
        raise NotImplementedError

    def _gluonts_forecasts_to_data_frame(
        self,
        forecasts: List[Forecast],
        quantile_levels: List[float],
        forecast_index: pd.MultiIndex,
    ) -> TimeSeriesDataFrame:
        # if predictions are gluonts SampleForecasts, convert to quantile forecasts
        if isinstance(forecasts[0], SampleForecast):
            forecasts = [self._sample_to_quantile_forecast(f, quantile_levels) for f in forecasts]
        elif isinstance(forecasts[0], DistributionForecast):
            forecasts = [self._distribution_to_quantile_forecast(f, quantile_levels) for f in forecasts]
        else:
            assert isinstance(forecasts[0], QuantileForecast), f"Unrecognized forecast type {type(forecasts[0])}"

        # sanity check to ensure all quantiles are accounted for
        assert all(str(q) in forecasts[0].forecast_keys for q in quantile_levels), (
            "Some forecast quantiles are missing from GluonTS forecast outputs. Was"
            " the model trained to forecast all quantiles?"
        )
        item_id_to_forecast = {str(f.item_id): f for f in forecasts}
        result_dfs = []
        for item_id in forecast_index.unique(level=ITEMID):
            # GluonTS always saves item_id as a string
            forecast = item_id_to_forecast[str(item_id)]
            item_forecast_dict = {"mean": forecast.mean}
            for quantile in quantile_levels:
                item_forecast_dict[str(quantile)] = forecast.quantile(str(quantile))
            result_dfs.append(pd.DataFrame(item_forecast_dict))

        result = pd.concat(result_dfs)
        result.index = forecast_index
        return TimeSeriesDataFrame(result)

    def _get_hpo_backend(self):
        return RAY_BACKEND

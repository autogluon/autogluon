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
from pandas.tseries.frequencies import to_offset

from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.core.hpo.constants import RAY_BACKEND
from autogluon.core.utils import warning_filter
from autogluon.core.utils.savers import save_pkl
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.features import get_categorical_and_continuous_features
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
        time_series_df: TimeSeriesDataFrame,
        target_field_name: str = "target",
        feat_static_cat: Optional[pd.DataFrame] = None,
        feat_static_real: Optional[pd.DataFrame] = None,
        float_dtype: Type = np.float64,
        int_dtype: Type = np.int64,
    ):
        assert time_series_df is not None
        assert time_series_df.freq, "Initializing GluonTS data sets without freq is not allowed"
        self.time_series_df = time_series_df
        self.target_field_name = target_field_name
        self.feat_static_cat = feat_static_cat
        self.feat_static_real = feat_static_real

        self.int_dtype = int_dtype
        self.float_dtype = float_dtype
        if self.feat_static_cat is not None:
            self.feat_static_cat = self.feat_static_cat.astype(self.int_dtype)
        if self.feat_static_real is not None:
            self.feat_static_real = self.feat_static_real.astype(self.float_dtype)

    @property
    def freq(self):
        # FIXME: GluonTS expects a frequency string, but only supports a limited number of such strings
        # for feature generation. If the frequency string doesn't match or is not provided, it raises an exception.
        # Here we bypass this by issuing a default "yearly" frequency, tricking it into not producing
        # any lags or features.
        freq_ = self.time_series_df.freq
        pd_offset = to_offset(freq_)

        # normalize freq str to handle peculiarities such as W-SUN
        offset_base_alias = pd_offset.name.split("-")[0]

        return "A" if offset_base_alias is None or offset_base_alias not in GLUONTS_SUPPORTED_OFFSETS else freq_

    def __len__(self):
        return len(self.time_series_df.item_ids)  # noqa

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for item_id in self.time_series_df.item_ids:  # noqa
            df = self.time_series_df.loc[item_id]
            time_series = {
                FieldName.ITEM_ID: item_id,
                FieldName.TARGET: df[self.target_field_name].to_numpy(dtype=self.float_dtype),
                FieldName.START: pd.Period(df.index[0], freq=self.freq),
            }
            if self.feat_static_cat is not None:
                time_series[FieldName.FEAT_STATIC_CAT] = self.feat_static_cat.loc[item_id].to_numpy(
                    dtype=self.int_dtype
                )
            if self.feat_static_real is not None:
                time_series[FieldName.FEAT_STATIC_REAL] = self.feat_static_real.loc[item_id].to_numpy(
                    dtype=self.float_dtype
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
        self.use_feat_static_cat = False
        self.use_feat_static_real = False
        self.use_feat_dynamic_real = False
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

            disable_static_features = self._get_model_params().get("disable_static_features", False)
            if not disable_static_features:
                feat_static_cat, feat_static_real = get_categorical_and_continuous_features(ds.static_features)
                self.use_feat_static_cat = feat_static_cat is not None
                self.use_feat_static_real = feat_static_real is not None
                if self.use_feat_static_cat:
                    self.feat_static_cat_cardinality = feat_static_cat.nunique().tolist()

        if "callbacks" in kwargs:
            self.callbacks += kwargs["callbacks"]

    def _get_model_params(self) -> dict:
        """Gets params that are passed to the inner model."""
        args = super()._get_model_params().copy()
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

    def _to_gluonts_dataset(self, time_series_df: Optional[TimeSeriesDataFrame]) -> Optional[GluonTSDataset]:
        if time_series_df is not None:
            feat_static_cat, feat_static_real = get_categorical_and_continuous_features(time_series_df.static_features)
            if not self.use_feat_static_cat:
                feat_static_cat = None
            if not self.use_feat_static_real:
                feat_static_real = None
            return SimpleGluonTSDataset(
                time_series_df,
                target_field_name=self.target,
                feat_static_cat=feat_static_cat,
                feat_static_real=feat_static_real,
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
        # TODO: reintroduce early stopping callbacks

        # update auxiliary parameters
        self._deferred_init_params_aux(
            dataset=train_data, callbacks=self._get_callbacks(time_limit=time_limit), **kwargs
        )

        estimator = self._get_estimator()
        with warning_filter(), disable_root_logger(), gluonts.core.settings.let(gluonts.env.env, use_tqdm=False):
            self.gts_predictor = estimator.train(
                self._to_gluonts_dataset(train_data),
                validation_data=self._to_gluonts_dataset(val_data),
            )

    def _get_callbacks(self, time_limit: int, *args, **kwargs) -> List[Callable]:
        """Retrieve a list of callback objects for the GluonTS trainer"""
        raise NotImplementedError

    def predict(self, data: TimeSeriesDataFrame, quantile_levels: List[float] = None, **kwargs) -> TimeSeriesDataFrame:
        if self.gts_predictor is None:
            raise ValueError("Please fit the model before predicting.")

        logger.debug(f"Predicting with time series model {self.name}")
        logger.debug(
            f"\tProvided data for prediction with {len(data)} rows, {data.num_items} items. "
            f"Average time series length is {len(data) / data.num_items}."
        )
        input_index_type = type(data.index.levels[0][0])

        with warning_filter(), gluonts.core.settings.let(gluonts.env.env, use_tqdm=False):
            quantiles = quantile_levels or self.quantile_levels
            if not all(0 < q < 1 for q in quantiles):
                raise ValueError("Invalid quantile value specified. Quantiles must be between 0 and 1 (exclusive).")

            predicted_targets = self._predict_gluonts_forecasts(data, **kwargs)

            df = self._gluonts_forecasts_to_data_frame(
                predicted_targets,
                quantile_levels=quantile_levels or self.quantile_levels,
            )

            # if index type is different than the input data, cast it back
            if len(df.index.levels[0]) > 0:
                prediction_index_type = type(df.index.levels[0][0])
                if prediction_index_type is not input_index_type:
                    df.set_index(
                        df.index.set_levels([input_index_type(i) for i in df.index.levels[0]], level=0),
                        inplace=True,
                    )

        # Make sure the item_ids are sorted in the same order as in data
        return df.loc[data.item_ids]

    def _predict_gluonts_forecasts(self, data: TimeSeriesDataFrame, **kwargs) -> List[Forecast]:
        gts_data = self._to_gluonts_dataset(data)

        predictor_kwargs = dict(dataset=gts_data)
        if "num_samples" in kwargs:
            predictor_kwargs["num_samples"] = kwargs["num_samples"]

        return list(self.gts_predictor.predict(**predictor_kwargs))

    @staticmethod
    def _sample_to_quantile_forecast(forecast: SampleForecast, quantile_levels: List[float]) -> QuantileForecast:
        forecast_arrays = []

        quantile_keys = [str(q) for q in quantile_levels]
        for q in quantile_keys:
            forecast_arrays.append(forecast.quantile(q))

        forecast_init_args = dict(
            forecast_arrays=np.array(forecast_arrays),
            start_date=forecast.start_date,
            forecast_keys=quantile_keys,
            item_id=forecast.item_id,
        )
        if isinstance(forecast.start_date, pd.Timestamp):  # GluonTS version is <0.10
            forecast_init_args.update({"freq": forecast.freq})
        return QuantileForecast(**forecast_init_args)

    def _gluonts_forecasts_to_data_frame(
        self, forecasts: List[Forecast], quantile_levels: List[float]
    ) -> TimeSeriesDataFrame:
        if not isinstance(forecasts[0], (QuantileForecast, SampleForecast)):
            raise TypeError("DistributionForecast is not supported.")

        forecast_means = [f.mean for f in forecasts]

        # if predictions are gluonts SampleForecasts, convert to quantile forecasts
        if isinstance(forecasts[0], SampleForecast):
            forecasts = [self._sample_to_quantile_forecast(f, quantile_levels) for f in forecasts]

        # sanity check to ensure all quantiles are accounted for
        assert all(str(q) in forecasts[0].forecast_keys for q in quantile_levels), (
            "Some forecast quantiles are missing from GluonTS forecast outputs. Was"
            " the model trained to forecast all quantiles?"
        )

        result_dfs = []
        for i, forecast in enumerate(forecasts):
            item_forecast_dict = dict(mean=forecast_means[i])
            for quantile in quantile_levels:
                item_forecast_dict[str(quantile)] = forecast.quantile(str(quantile))

            df = pd.DataFrame(item_forecast_dict)
            df[ITEMID] = forecast.item_id
            df[TIMESTAMP] = pd.date_range(
                start=forecasts[i].start_date.to_timestamp(how="S"),
                periods=self.prediction_length,
                freq=self.freq,
            )
            result_dfs.append(df)

        return TimeSeriesDataFrame.from_data_frame(pd.concat(result_dfs))

    def _get_hpo_backend(self):
        return RAY_BACKEND

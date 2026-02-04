import logging
import os
import shutil
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Type, cast, overload

import gluonts
import gluonts.core.settings
import numpy as np
import pandas as pd
from gluonts.core.component import from_hyperparameters
from gluonts.dataset.common import Dataset as GluonTSDataset
from gluonts.env import env as gluonts_env
from gluonts.model.estimator import Estimator as GluonTSEstimator
from gluonts.model.forecast import Forecast, QuantileForecast, SampleForecast
from gluonts.model.predictor import Predictor as GluonTSPredictor

from autogluon.common.loaders import load_pkl
from autogluon.core.hpo.constants import RAY_BACKEND
from autogluon.tabular.models.tabular_nn.utils.categorical_encoders import (
    OneHotMergeRaresHandleUnknownEncoder as OneHotEncoder,
)
from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.warning_filters import disable_root_logger, warning_filter

if TYPE_CHECKING:
    from gluonts.torch.model.forecast import DistributionForecast

from .dataset import SimpleGluonTSDataset

# NOTE: We avoid imports for torch and lightning.pytorch at the top level and hide them inside class methods.
# This is done to skip these imports during multiprocessing (which may cause bugs)

logger = logging.getLogger(__name__)
gts_logger = logging.getLogger(gluonts.__name__)


class AbstractGluonTSModel(AbstractTimeSeriesModel):
    """Abstract class wrapping GluonTS estimators for use in autogluon.timeseries.

    Parameters
    ----------
    path
        directory to store model artifacts.
    freq
        string representation (compatible with GluonTS frequency strings) for the data provided.
        For example, "1D" for daily data, "1H" for hourly data, etc.
    prediction_length
        Number of time steps ahead (length of the forecast horizon) the model will be optimized
        to predict. At inference time, this will be the number of time steps the model will
        predict.
    name
        Name of the model. Also, name of subdirectory inside path where model will be saved.
    eval_metric
        objective function the model intends to optimize, will use WQL by default.
    hyperparameters
        various hyperparameters that will be used by model (can be search spaces instead of
        fixed values). See *Other Parameters* in each inheriting model's documentation for
        possible values.
    """

    gluonts_model_path = "gluon_ts"
    # we pass dummy freq compatible with pandas 2.1 & 2.2 to GluonTS models
    _dummy_gluonts_freq = "D"
    # default number of samples for prediction
    default_num_samples: int = 250

    #: whether the GluonTS model supports categorical variables as covariates
    _supports_cat_covariates: bool = False

    def __init__(
        self,
        freq: str | None = None,
        prediction_length: int = 1,
        path: str | None = None,
        name: str | None = None,
        eval_metric: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        **kwargs,  # noqa
    ):
        super().__init__(
            path=path,
            freq=freq,
            prediction_length=prediction_length,
            name=name,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            **kwargs,
        )
        self.gts_predictor: GluonTSPredictor | None = None
        self._ohe_generator_known: OneHotEncoder | None = None
        self._ohe_generator_past: OneHotEncoder | None = None
        self.callbacks = []
        # Following attributes may be overridden during fit() based on train_data & model parameters
        self.num_feat_static_cat = 0
        self.num_feat_static_real = 0
        self.num_feat_dynamic_cat = 0
        self.num_feat_dynamic_real = 0
        self.num_past_feat_dynamic_cat = 0
        self.num_past_feat_dynamic_real = 0
        self.feat_static_cat_cardinality: list[int] = []
        self.feat_dynamic_cat_cardinality: list[int] = []
        self.past_feat_dynamic_cat_cardinality: list[int] = []
        self.negative_data = True

    def save(self, path: str | None = None, verbose: bool = True) -> str:
        # we flush callbacks instance variable if it has been set. it can keep weak references which breaks training
        self.callbacks = []
        # The GluonTS predictor is serialized using custom logic
        predictor = self.gts_predictor
        self.gts_predictor = None
        saved_path = Path(super().save(path=path, verbose=verbose))

        with disable_root_logger():
            if predictor:
                Path.mkdir(saved_path / self.gluonts_model_path, exist_ok=True)
                predictor.serialize(saved_path / self.gluonts_model_path)

        self.gts_predictor = predictor

        return str(saved_path)

    @classmethod
    def load(
        cls, path: str, reset_paths: bool = True, load_oof: bool = False, verbose: bool = True
    ) -> "AbstractGluonTSModel":
        from gluonts.torch.model.predictor import PyTorchPredictor

        with warning_filter():
            model = load_pkl.load(path=os.path.join(path, cls.model_file_name), verbose=verbose)
            if reset_paths:
                model.set_contexts(path)
            model.gts_predictor = PyTorchPredictor.deserialize(Path(path) / cls.gluonts_model_path, device="auto")
        return model

    @property
    def supports_cat_covariates(self) -> bool:
        return self.__class__._supports_cat_covariates

    def _get_hpo_backend(self):
        return RAY_BACKEND

    def _deferred_init_hyperparameters(self, dataset: TimeSeriesDataFrame) -> None:
        """Update GluonTS specific hyperparameters with information available only at training time."""
        model_params = self.get_hyperparameters()
        disable_static_features = model_params.get("disable_static_features", False)
        if not disable_static_features:
            self.num_feat_static_cat = len(self.covariate_metadata.static_features_cat)
            self.num_feat_static_real = len(self.covariate_metadata.static_features_real)
            if self.num_feat_static_cat > 0:
                assert dataset.static_features is not None, (
                    "Static features must be provided if num_feat_static_cat > 0"
                )
                self.feat_static_cat_cardinality = list(self.covariate_metadata.static_cat_cardinality.values())

        disable_known_covariates = model_params.get("disable_known_covariates", False)
        if not disable_known_covariates and self.supports_known_covariates:
            self.num_feat_dynamic_cat = len(self.covariate_metadata.known_covariates_cat)
            self.num_feat_dynamic_real = len(self.covariate_metadata.known_covariates_real)
            if self.num_feat_dynamic_cat > 0:
                if self.supports_cat_covariates:
                    self.feat_dynamic_cat_cardinality = list(self.covariate_metadata.known_cat_cardinality.values())
                else:
                    feat_dynamic_cat = dataset[self.covariate_metadata.known_covariates_cat]
                    # If model doesn't support categorical covariates, convert them to real via one hot encoding
                    self._ohe_generator_known = OneHotEncoder(
                        max_levels=model_params.get("max_cat_cardinality", 100),
                        sparse=False,
                        dtype="float32",  # type: ignore
                    )
                    feat_dynamic_cat_ohe = self._ohe_generator_known.fit_transform(pd.DataFrame(feat_dynamic_cat))
                    self.num_feat_dynamic_cat = 0
                    self.num_feat_dynamic_real += feat_dynamic_cat_ohe.shape[1]

        disable_past_covariates = model_params.get("disable_past_covariates", False)
        if not disable_past_covariates and self.supports_past_covariates:
            self.num_past_feat_dynamic_cat = len(self.covariate_metadata.past_covariates_cat)
            self.num_past_feat_dynamic_real = len(self.covariate_metadata.past_covariates_real)
            if self.num_past_feat_dynamic_cat > 0:
                if self.supports_cat_covariates:
                    self.past_feat_dynamic_cat_cardinality = list(
                        self.covariate_metadata.past_cat_cardinality.values()
                    )
                else:
                    past_feat_dynamic_cat = dataset[self.covariate_metadata.past_covariates_cat]
                    # If model doesn't support categorical covariates, convert them to real via one hot encoding
                    self._ohe_generator_past = OneHotEncoder(
                        max_levels=model_params.get("max_cat_cardinality", 100),
                        sparse=False,
                        dtype="float32",  # type: ignore
                    )
                    past_feat_dynamic_cat_ohe = self._ohe_generator_past.fit_transform(
                        pd.DataFrame(past_feat_dynamic_cat)
                    )
                    self.num_past_feat_dynamic_cat = 0
                    self.num_past_feat_dynamic_real += past_feat_dynamic_cat_ohe.shape[1]

        self.negative_data = (dataset[self.target] < 0).any()

    def _get_default_hyperparameters(self):
        """Gets default parameters for GluonTS estimator initialization that are available after
        AbstractTimeSeriesModel initialization (i.e., before deferred initialization). Models may
        override this method to update default parameters.
        """
        return {
            "batch_size": 64,
            "context_length": min(512, max(10, 2 * self.prediction_length)),
            "predict_batch_size": 500,
            "early_stopping_patience": 20,
            "max_epochs": 100,
            "lr": 1e-3,
            "freq": self._dummy_gluonts_freq,
            "prediction_length": self.prediction_length,
            "quantiles": self.quantile_levels,
            "covariate_scaler": "global",
        }

    def get_hyperparameters(self) -> dict:
        """Gets params that are passed to the inner model."""
        # for backward compatibility with the old GluonTS MXNet API
        parameter_name_aliases = {
            "epochs": "max_epochs",
            "learning_rate": "lr",
        }

        init_args = super().get_hyperparameters()
        for alias, actual in parameter_name_aliases.items():
            if alias in init_args:
                if actual in init_args:
                    raise ValueError(f"Parameter '{alias}' cannot be specified when '{actual}' is also specified.")
                else:
                    init_args[actual] = init_args.pop(alias)

        return self._get_default_hyperparameters() | init_args

    def _get_estimator_init_args(self) -> dict[str, Any]:
        """Get GluonTS specific constructor arguments for estimator objects, an alias to `self.get_hyperparameters`
        for better readability."""
        return self.get_hyperparameters()

    def _get_estimator_class(self) -> Type[GluonTSEstimator]:
        raise NotImplementedError

    def _get_estimator(self) -> GluonTSEstimator:
        """Return the GluonTS Estimator object for the model"""
        # As GluonTSPyTorchLightningEstimator objects do not implement `from_hyperparameters` convenience
        # constructors, we re-implement the logic here.
        # we translate the "epochs" parameter to "max_epochs" for consistency in the AbstractGluonTSModel interface
        init_args = self._get_estimator_init_args()

        default_trainer_kwargs = {
            "limit_val_batches": 3,
            "max_epochs": init_args["max_epochs"],
            "callbacks": self.callbacks,
            "enable_progress_bar": False,
            "default_root_dir": self.path,
        }

        if self._is_gpu_available():
            default_trainer_kwargs["accelerator"] = "gpu"
            default_trainer_kwargs["devices"] = 1
        else:
            default_trainer_kwargs["accelerator"] = "cpu"

        default_trainer_kwargs.update(init_args.pop("trainer_kwargs", {}))
        logger.debug(f"\tTraining on device '{default_trainer_kwargs['accelerator']}'")

        return from_hyperparameters(
            self._get_estimator_class(),
            trainer_kwargs=default_trainer_kwargs,
            **init_args,
        )

    def _is_gpu_available(self) -> bool:
        import torch.cuda

        return torch.cuda.is_available()

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int | float]:
        minimum_resources: dict[str, int | float] = {"num_cpus": 1}
        # if GPU is available, we train with 1 GPU per trial
        if is_gpu_available:
            minimum_resources["num_gpus"] = 1
        return minimum_resources

    @overload
    def _to_gluonts_dataset(self, time_series_df: None, known_covariates=None) -> None: ...
    @overload
    def _to_gluonts_dataset(self, time_series_df: TimeSeriesDataFrame, known_covariates=None) -> GluonTSDataset: ...
    def _to_gluonts_dataset(
        self, time_series_df: TimeSeriesDataFrame | None, known_covariates: TimeSeriesDataFrame | None = None
    ) -> GluonTSDataset | None:
        if time_series_df is not None:
            # TODO: Preprocess real-valued features with StdScaler?
            if self.num_feat_static_cat > 0:
                assert time_series_df.static_features is not None, (
                    "Static features must be provided if num_feat_static_cat > 0"
                )
                feat_static_cat = time_series_df.static_features[
                    self.covariate_metadata.static_features_cat
                ].to_numpy()
            else:
                feat_static_cat = None

            if self.num_feat_static_real > 0:
                assert time_series_df.static_features is not None, (
                    "Static features must be provided if num_feat_static_real > 0"
                )
                feat_static_real = time_series_df.static_features[
                    self.covariate_metadata.static_features_real
                ].to_numpy()
            else:
                feat_static_real = None

            expected_known_covariates_len = len(time_series_df) + self.prediction_length * time_series_df.num_items
            # Convert TSDF -> DF to avoid overhead / input validation
            df = pd.DataFrame(time_series_df)
            if known_covariates is not None:
                known_covariates = pd.DataFrame(known_covariates)  # type: ignore
            if self.num_feat_dynamic_cat > 0:
                feat_dynamic_cat = df[self.covariate_metadata.known_covariates_cat].to_numpy()
                if known_covariates is not None:
                    feat_dynamic_cat = np.concatenate(
                        [feat_dynamic_cat, known_covariates[self.covariate_metadata.known_covariates_cat].to_numpy()]
                    )
                    assert len(feat_dynamic_cat) == expected_known_covariates_len
            else:
                feat_dynamic_cat = None

            if self.num_feat_dynamic_real > 0:
                feat_dynamic_real = df[self.covariate_metadata.known_covariates_real].to_numpy()
                # Append future values of known covariates
                if known_covariates is not None:
                    feat_dynamic_real = np.concatenate(
                        [feat_dynamic_real, known_covariates[self.covariate_metadata.known_covariates_real].to_numpy()]
                    )
                    assert len(feat_dynamic_real) == expected_known_covariates_len
                # Categorical covariates are one-hot-encoded as real
                if self._ohe_generator_known is not None:
                    feat_dynamic_cat_ohe: np.ndarray = self._ohe_generator_known.transform(
                        df[self.covariate_metadata.known_covariates_cat]
                    )  # type: ignore
                    if known_covariates is not None:
                        future_dynamic_cat_ohe: np.ndarray = self._ohe_generator_known.transform(  # type: ignore
                            known_covariates[self.covariate_metadata.known_covariates_cat]
                        )
                        feat_dynamic_cat_ohe = np.concatenate([feat_dynamic_cat_ohe, future_dynamic_cat_ohe])
                        assert len(feat_dynamic_cat_ohe) == expected_known_covariates_len
                    feat_dynamic_real = np.concatenate([feat_dynamic_real, feat_dynamic_cat_ohe], axis=1)
            else:
                feat_dynamic_real = None

            if self.num_past_feat_dynamic_cat > 0:
                past_feat_dynamic_cat = df[self.covariate_metadata.past_covariates_cat].to_numpy()
            else:
                past_feat_dynamic_cat = None

            if self.num_past_feat_dynamic_real > 0:
                past_feat_dynamic_real = df[self.covariate_metadata.past_covariates_real].to_numpy()
                if self._ohe_generator_past is not None:
                    past_feat_dynamic_cat_ohe: np.ndarray = self._ohe_generator_past.transform(  # type: ignore
                        df[self.covariate_metadata.past_covariates_cat]
                    )
                    past_feat_dynamic_real = np.concatenate(
                        [past_feat_dynamic_real, past_feat_dynamic_cat_ohe], axis=1
                    )
            else:
                past_feat_dynamic_real = None

            assert self.freq is not None
            return SimpleGluonTSDataset(
                target_df=time_series_df[[self.target]],  # type: ignore
                freq=self.freq,
                target_column=self.target,
                feat_static_cat=feat_static_cat,
                feat_static_real=feat_static_real,
                feat_dynamic_cat=feat_dynamic_cat,
                feat_dynamic_real=feat_dynamic_real,
                past_feat_dynamic_cat=past_feat_dynamic_cat,
                past_feat_dynamic_real=past_feat_dynamic_real,
                includes_future=known_covariates is not None,
                prediction_length=self.prediction_length,
            )
        else:
            return None

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: TimeSeriesDataFrame | None = None,
        time_limit: float | None = None,
        num_cpus: int | None = None,
        num_gpus: int | None = None,
        verbosity: int = 2,
        **kwargs,
    ) -> None:
        # necessary to initialize the loggers
        import lightning.pytorch  # noqa

        for logger_name in logging.root.manager.loggerDict:
            if "lightning" in logger_name:
                pl_logger = logging.getLogger(logger_name)
                pl_logger.setLevel(logging.ERROR if verbosity <= 3 else logging.INFO)
        gts_logger.setLevel(logging.ERROR if verbosity <= 3 else logging.INFO)

        if verbosity > 3:
            logger.warning(
                "GluonTS logging is turned on during training. Note that losses reported by GluonTS "
                "may not correspond to those specified via `eval_metric`."
            )

        self._check_fit_params()
        # update auxiliary parameters
        init_args = self._get_estimator_init_args()
        keep_lightning_logs = init_args.pop("keep_lightning_logs", False)
        self.callbacks = self._get_callbacks(
            time_limit=time_limit,
            early_stopping_patience=None if val_data is None else init_args["early_stopping_patience"],
        )
        self._deferred_init_hyperparameters(train_data)

        estimator = self._get_estimator()
        with warning_filter(), disable_root_logger(), gluonts.core.settings.let(gluonts_env, use_tqdm=False):
            self.gts_predictor = estimator.train(
                self._to_gluonts_dataset(train_data),
                validation_data=self._to_gluonts_dataset(val_data),
                cache_data=True,  # type: ignore
            )
            # Increase batch size during prediction to speed up inference
            if init_args["predict_batch_size"] is not None:
                self.gts_predictor.batch_size = init_args["predict_batch_size"]  # type: ignore

        lightning_logs_dir = Path(self.path) / "lightning_logs"
        if not keep_lightning_logs and lightning_logs_dir.exists() and lightning_logs_dir.is_dir():
            logger.debug(f"Removing lightning_logs directory {lightning_logs_dir}")
            shutil.rmtree(lightning_logs_dir)

    def _get_callbacks(
        self,
        time_limit: float | None,
        early_stopping_patience: int | None = None,
    ) -> list[Callable]:
        """Retrieve a list of callback objects for the GluonTS trainer"""
        from lightning.pytorch.callbacks import EarlyStopping, Timer

        callbacks = []
        if time_limit is not None:
            callbacks.append(Timer(timedelta(seconds=time_limit)))
        if early_stopping_patience is not None:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=early_stopping_patience))
        return callbacks

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: TimeSeriesDataFrame | None = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        if self.gts_predictor is None:
            raise ValueError("Please fit the model before predicting.")

        with warning_filter(), gluonts.core.settings.let(gluonts_env, use_tqdm=False):
            predicted_targets = self._predict_gluonts_forecasts(data, known_covariates=known_covariates)
            df = self._gluonts_forecasts_to_data_frame(
                predicted_targets,
                forecast_index=self.get_forecast_horizon_index(data),
            )
        return df

    def _predict_gluonts_forecasts(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: TimeSeriesDataFrame | None = None,
        num_samples: int | None = None,
    ) -> list[Forecast]:
        assert self.gts_predictor is not None, "GluonTS models must be fit before predicting."
        gts_data = self._to_gluonts_dataset(data, known_covariates=known_covariates)
        return list(
            self.gts_predictor.predict(
                dataset=gts_data,
                num_samples=num_samples or self.default_num_samples,
            )
        )

    def _stack_quantile_forecasts(self, forecasts: list[QuantileForecast], item_ids: pd.Index) -> pd.DataFrame:
        # GluonTS always saves item_id as a string
        item_id_to_forecast = {str(f.item_id): f for f in forecasts}
        result_dfs = []
        for item_id in item_ids:
            forecast = item_id_to_forecast[str(item_id)]
            result_dfs.append(pd.DataFrame(forecast.forecast_array.T, columns=forecast.forecast_keys))
        forecast_df = pd.concat(result_dfs)
        if "mean" not in forecast_df.columns:
            forecast_df["mean"] = forecast_df["0.5"]
        columns_order = ["mean"] + [str(q) for q in self.quantile_levels]
        return forecast_df[columns_order]

    def _stack_sample_forecasts(self, forecasts: list[SampleForecast], item_ids: pd.Index) -> pd.DataFrame:
        item_id_to_forecast = {str(f.item_id): f for f in forecasts}
        samples_per_item = []
        for item_id in item_ids:
            forecast = item_id_to_forecast[str(item_id)]
            samples_per_item.append(forecast.samples.T)
        samples = np.concatenate(samples_per_item, axis=0)
        quantiles = np.quantile(samples, self.quantile_levels, axis=1).T
        mean = samples.mean(axis=1, keepdims=True)
        forecast_array = np.concatenate([mean, quantiles], axis=1)
        return pd.DataFrame(forecast_array, columns=["mean"] + [str(q) for q in self.quantile_levels])

    def _stack_distribution_forecasts(
        self, forecasts: list["DistributionForecast"], item_ids: pd.Index
    ) -> pd.DataFrame:
        import torch
        from gluonts.torch.distributions import AffineTransformed
        from torch.distributions import Distribution

        # Sort forecasts in the same order as in the dataset
        item_id_to_forecast = {str(f.item_id): f for f in forecasts}
        dist_forecasts = [item_id_to_forecast[str(item_id)] for item_id in item_ids]

        assert all(isinstance(f.distribution, AffineTransformed) for f in dist_forecasts), (
            "Expected forecast.distribution to be an instance of AffineTransformed"
        )

        def stack_distributions(distributions: list[Distribution]) -> Distribution:
            """Stack multiple torch.Distribution objects into a single distribution"""
            last_dist: Distribution = distributions[-1]

            params_per_dist = []
            for dist in distributions:
                params = {name: getattr(dist, name) for name in dist.arg_constraints.keys()}
                params_per_dist.append(params)
            # Make sure that all distributions have same keys
            assert len(set(tuple(p.keys()) for p in params_per_dist)) == 1

            stacked_params = {}
            for key in last_dist.arg_constraints.keys():
                stacked_params[key] = torch.cat([p[key] for p in params_per_dist])
            return last_dist.__class__(**stacked_params)

        # We stack all forecast distribution into a single Distribution object.
        # This dramatically speeds up the quantiles calculation.
        stacked_base_dist = stack_distributions([f.distribution.base_dist for f in dist_forecasts])  # type: ignore

        stacked_loc = torch.cat([f.distribution.loc for f in dist_forecasts])  # type: ignore
        if stacked_loc.shape != stacked_base_dist.batch_shape:
            stacked_loc = stacked_loc.repeat_interleave(self.prediction_length)

        stacked_scale = torch.cat([f.distribution.scale for f in dist_forecasts])  # type: ignore
        if stacked_scale.shape != stacked_base_dist.batch_shape:
            stacked_scale = stacked_scale.repeat_interleave(self.prediction_length)

        stacked_dist = AffineTransformed(stacked_base_dist, loc=stacked_loc, scale=stacked_scale)

        mean_prediction = stacked_dist.mean.cpu().detach().numpy()
        quantiles = torch.tensor(self.quantile_levels, device=stacked_dist.mean.device).reshape(-1, 1)
        quantile_predictions = stacked_dist.icdf(quantiles).cpu().detach().numpy()  # type: ignore
        forecast_array = np.vstack([mean_prediction, quantile_predictions]).T
        return pd.DataFrame(forecast_array, columns=["mean"] + [str(q) for q in self.quantile_levels])

    def _gluonts_forecasts_to_data_frame(
        self,
        forecasts: list[Forecast],
        forecast_index: pd.MultiIndex,
    ) -> TimeSeriesDataFrame:
        from gluonts.torch.model.forecast import DistributionForecast

        item_ids = forecast_index.unique(level=TimeSeriesDataFrame.ITEMID)
        if isinstance(forecasts[0], SampleForecast):
            forecast_df = self._stack_sample_forecasts(cast(list[SampleForecast], forecasts), item_ids)
        elif isinstance(forecasts[0], QuantileForecast):
            forecast_df = self._stack_quantile_forecasts(cast(list[QuantileForecast], forecasts), item_ids)
        elif isinstance(forecasts[0], DistributionForecast):
            forecast_df = self._stack_distribution_forecasts(cast(list[DistributionForecast], forecasts), item_ids)
        else:
            raise ValueError(f"Unrecognized forecast type {type(forecasts[0])}")

        forecast_df.index = forecast_index
        return TimeSeriesDataFrame(forecast_df)

    def _more_tags(self) -> dict:
        return {"allow_nan": True, "can_use_val_data": True}

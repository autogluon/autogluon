import logging
import os
from typing import Any, Sequence

import numpy as np
import pandas as pd
from typing_extensions import Self

from autogluon.common.loaders import load_pkl
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.features import CovariateMetadata

try:
    from t0 import T0Forecaster
except ImportError:
    # T0 is optional; importing AutoGluon's model registry must still work without it.
    T0Forecaster = None

logger = logging.getLogger(__name__)


class T0Model(AbstractTimeSeriesModel):
    """t0-gamma pretrained time series forecasting model.

    t0-gamma is a 256M-parameter decoder-style patch transformer developed by The Forecasting Company. It produces
    probabilistic multi-horizon forecasts and was trained to predict 21 quantile levels from 0.01 to 0.99. Other
    quantile levels are interpolated by the model at inference time. More details are available on
    `Hugging Face <https://huggingface.co/theforecastingcompany/t0-gamma>`_ and
    `GitHub <https://github.com/theforecastingcompany/tfc-t0>`_.

    AutoGluon supports t0 for **inference only**, so the model is not trained or fine-tuned on the supplied
    training data. Numeric past covariates are supplied over the context, while numeric known covariates are supplied
    over both the context and prediction horizon. Categorical covariates and static features are ignored.
    Install the optional runtime with ``pip install "autogluon.timeseries[t0]"``.

    Other Parameters
    ----------------
    model_path : str, default = "theforecastingcompany/t0-gamma"
        Hugging Face model identifier or path to a local directory containing a compatible checkpoint.
        The alias ``"t0-gamma"`` resolves to the default checkpoint.
    batch_size : int, default = 64
        Number of time series processed together during inference. If a batch runs out of accelerator memory, the
        batch size is halved and the batch retried, as in tfc-t0's ``T0Predictor``. With ``cross_learning``, the
        batch is not retried, since a smaller batch would change which series are forecast together.
    device : str or None, default = None
        Device used for inference. If None, CUDA is used when available and CPU otherwise.
    context_length : int, default = 8192
        Maximum number of recent observations supplied to the model for each series.
    disable_known_covariates : bool, default = False
        If True, known covariates are not provided to the model.
    disable_past_covariates : bool, default = False
        If True, past covariates are not provided to the model.
    cross_learning : bool, default = False
        If True, all series in a batch are forecast together as one group, so the model reads across them. The group
        size is therefore ``batch_size``, and results depend on it. If False, each series is forecast on its own.
        This matches ``cross_learning`` in Chronos-2 and corresponds to ``cross_inference`` in tfc-t0's
        ``T0Predictor``.
    trim_leading_missing : bool, default = True
        If True, missing values before the first observation of each series are treated as padding. With
        ``cross_learning``, tfc-t0 trims each group as a whole, up to the first observation of any series in it.
    symmetric_averaging : bool, default = True
        If True, the forecast is averaged with the mirrored forecast of the negated series. This runs the model twice.
    clip_nonnegative : bool, default = True
        If True, forecasts are clipped at zero for series whose history is non-negative.
    snap_integers : bool, default = True
        If True, forecasts are rounded for series whose history is integer-valued.
    """

    ag_priority = 50
    default_model_path = "theforecastingcompany/t0-gamma"
    model_path_aliases = {
        "t0-gamma": "theforecastingcompany/t0-gamma",
    }
    _supports_known_covariates = True
    _supports_past_covariates = True

    def __init__(
        self,
        path: str | None = None,
        name: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        freq: str | None = None,
        prediction_length: int = 1,
        covariate_metadata: CovariateMetadata | None = None,
        target: str = "target",
        quantile_levels: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        eval_metric: Any = None,
    ):
        hyperparameters = hyperparameters if hyperparameters is not None else {}
        model_path = hyperparameters.get("model_path", self.default_model_path)
        self.model_path = self.model_path_aliases.get(model_path, model_path)

        super().__init__(
            path=path,
            name=name,
            hyperparameters=hyperparameters,
            freq=freq,
            prediction_length=prediction_length,
            covariate_metadata=covariate_metadata,
            target=target,
            quantile_levels=quantile_levels,
            eval_metric=eval_metric,
        )
        self._model: T0Forecaster | None = None

    def save(self, path: str | None = None, verbose: bool = True) -> str:
        model = self._model
        self._model = None
        path = super().save(path=path, verbose=verbose)
        self._model = model
        return str(path)

    @classmethod
    def load(cls, path: str, reset_paths: bool = True, load_oof: bool = False, verbose: bool = True) -> Self:
        model = load_pkl.load(path=os.path.join(path, cls.model_file_name), verbose=verbose)
        if reset_paths:
            model.set_contexts(path)
        return model

    def _is_gpu_available(self) -> bool:
        import torch.cuda

        return torch.cuda.is_available()

    @staticmethod
    def _empty_accelerator_cache() -> None:
        """Release cached accelerator memory left by a failed batch before retrying a smaller one."""
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int | float]:
        resources: dict[str, int | float] = {"num_cpus": 1}
        if is_gpu_available:
            resources["num_gpus"] = 1
        return resources

    def _get_device(self) -> str:
        device = self.get_hyperparameter("device")
        if device is None:
            device = "cuda" if self._is_gpu_available() else "cpu"
        return device

    def load_model(self) -> None:
        if T0Forecaster is None:
            raise ImportError(
                "T0Model requires the optional tfc-t0 runtime. Install it with "
                '`pip install "autogluon.timeseries[t0]"`.'
            )

        self._model = T0Forecaster.from_pretrained(self.model_path).eval().to(self._get_device())

    def persist(self) -> Self:
        if self._model is None:
            self.load_model()
        return self

    def _get_default_hyperparameters(self) -> dict:
        return {
            "batch_size": 64,
            "device": None,
            "context_length": 8192,
            "disable_known_covariates": False,
            "disable_past_covariates": False,
            "cross_learning": False,
            # Recommended inference settings of tfc-t0; each uses only the series' own history.
            "trim_leading_missing": True,
            "symmetric_averaging": True,
            "clip_nonnegative": True,
            "snap_integers": True,
        }

    @property
    def allowed_hyperparameters(self) -> list[str]:
        return super().allowed_hyperparameters + [
            "model_path",
            "batch_size",
            "device",
            "context_length",
            "disable_known_covariates",
            "disable_past_covariates",
            "cross_learning",
            "trim_leading_missing",
            "symmetric_averaging",
            "clip_nonnegative",
            "snap_integers",
        ]

    def _more_tags(self) -> dict:
        return {
            "allow_nan": True,
            "can_use_train_data": False,
            "can_use_val_data": False,
        }

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
        self._check_fit_params()
        self.load_model()

    def _predict(
        self, data: TimeSeriesDataFrame, known_covariates: TimeSeriesDataFrame | None = None, **kwargs
    ) -> TimeSeriesDataFrame:
        from .dataloader import T0DataLoader, T0InferenceDataset

        hyperparameters = self.get_hyperparameters()
        if self._model is None:
            self.load_model()
        assert self._model is not None, "t0 model failed to load"

        dataset = T0InferenceDataset(
            data=data,
            max_context_length=hyperparameters["context_length"],
            target_column=self.target,
            past_covariate_columns=(
                [] if hyperparameters["disable_past_covariates"] else self.covariate_metadata.past_covariates_real
            ),
            known_covariate_columns=(
                [] if hyperparameters["disable_known_covariates"] else self.covariate_metadata.known_covariates_real
            ),
            # AutoGluon keeps known covariates split: their historical values are in `data`, while
            # `known_covariates` contains only the prediction horizon.
            known_covariates=known_covariates,
            prediction_length=self.prediction_length,
        )
        loader = T0DataLoader(
            dataset,
            time_limit=kwargs.get("time_limit"),
            device=self._get_device(),
            cross_learning=hyperparameters["cross_learning"],
        )

        requested_quantiles = sorted({0.5, *self.quantile_levels})
        quantile_forecast = np.empty(
            (len(dataset), self.prediction_length, len(requested_quantiles)), dtype=np.float32
        )
        batch_size = hyperparameters["batch_size"]
        start = 0
        while start < len(dataset):
            stop = min(start + batch_size, len(dataset))
            try:
                indices, model_input, context_length = loader.make_batch(start, stop)
                forecast = self._model.predict(
                    model_input,
                    horizon=self.prediction_length,
                    quantile_levels=requested_quantiles,
                    context_length=context_length,
                    trim_leading_missing=hyperparameters["trim_leading_missing"],
                    symmetric_averaging=hyperparameters["symmetric_averaging"],
                    clip_nonnegative=hyperparameters["clip_nonnegative"],
                    snap_integers=hyperparameters["snap_integers"],
                )
            except RuntimeError as err:
                # torch.cuda.OutOfMemoryError subclasses RuntimeError; MPS raises a plain RuntimeError.
                if "out of memory" not in str(err).lower() or batch_size <= 1:
                    raise
                if hyperparameters["cross_learning"]:
                    # The batch is the cross-learning group, so a smaller retry would change the forecasts.
                    raise RuntimeError(
                        f"T0 ran out of memory at batch_size {batch_size} with cross_learning=True. Lower batch_size, "
                        "which also sets the size of the group of series forecast together."
                    ) from err
                self._empty_accelerator_cache()
                batch_size //= 2
                logger.warning(f"\tT0 ran out of memory at batch_size {batch_size * 2}, retrying with {batch_size}")
                loader.on_batch()
                continue
            # T0 returns one forecast per TARGET row in input order, and every item has exactly one: [items, H, Q].
            quantile_forecast[indices] = forecast.quantiles.cpu().numpy()
            start = stop
            loader.on_batch()

        quantile_forecast = quantile_forecast.reshape(-1, len(requested_quantiles))
        median_idx = requested_quantiles.index(0.5)
        predictions = {"mean": quantile_forecast[:, median_idx]}
        predictions |= {str(q): quantile_forecast[:, requested_quantiles.index(q)] for q in self.quantile_levels}

        df = pd.DataFrame(predictions, index=self.get_forecast_horizon_index(data))
        return TimeSeriesDataFrame(df)

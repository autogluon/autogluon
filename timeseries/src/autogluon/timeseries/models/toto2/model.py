import logging
import os
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd
from typing_extensions import Self

from autogluon.common.loaders import load_pkl
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.features import CovariateMetadata

if TYPE_CHECKING:
    from ._internal import Toto2Model as _Toto2Model

logger = logging.getLogger(__name__)


class Toto2Model(AbstractTimeSeriesModel):
    """Toto 2.0 [Khwaja2026]_ pretrained time series forecasting model.

    Toto 2.0 is a family of decoder-only foundation models for time series forecasting built by Datadog. It features a
    u-μP-scaled transformer with alternating time/variate attention and quantile-based probabilistic forecasting, with
    model sizes ranging from 4M to 2.5B parameters. The full collection of Toto 2.0 models is available on
    `Hugging Face <https://huggingface.co/collections/Datadog/toto-20>`_.

    The AutoGluon implementation of Toto 2.0 is a port of the original implementation. AutoGluon supports Toto 2.0 for
    **inference only**, i.e., the model will not be trained or fine-tuned on the provided training data. This wrapper
    currently uses the model in univariate mode and does not use covariates.

    References
    ----------
    .. [Khwaja2026] Khwaja, Emaad, Lettieri, Chris et al.
        "Toto 2.0: Time Series Forecasting Enters the Scaling Era." (2026).
        https://arxiv.org/abs/2605.20119


    Other Parameters
    ----------------
    model_path : str, default = "Datadog/Toto-2.0-22m"
        Model path used for the model, i.e., a HuggingFace ``name_or_path``. Can be a compatible model name on
        HuggingFace Hub or a local path to a model directory. Available checkpoints include ``Datadog/Toto-2.0-4m``,
        ``Datadog/Toto-2.0-22m``, ``Datadog/Toto-2.0-313m``, ``Datadog/Toto-2.0-1B``, and ``Datadog/Toto-2.0-2.5B``,
        which can also be referenced by the shorthands ``"Toto-2.0-4m"``, ``"Toto-2.0-22m"``, ``"Toto-2.0-313m"``,
        ``"Toto-2.0-1B"``, and ``"Toto-2.0-2.5B"``.
    batch_size : int, default = 256
        Size of batches used during inference.
    device : str, default = None
        Device to use for inference. If None, model will use the GPU if available, and the CPU otherwise.
    context_length : int, default = 4096
        The context length to use in the model. Shorter context lengths will decrease model accuracy, but result
        in faster inference.
    decode_block_size : int or None, default = None
        Block size used for autoregressive block decoding. If None, forecasts are produced in a single forward pass,
        which is faster and typically better for shorter horizons. Setting this to a positive multiple of the model's
        patch size enables block decoding, which can improve long-term stability for very long horizons.
    scaler_fallback_min_obs : int, default = 8
        Stabilizes the scaler on leading patches with fewer than this many observed values by backfilling their
        location/scale using statistics from the first ``scaler_fallback_min_obs`` observed points.
    quantile_real_cap_k : float, default = 1e4
        Clips each predicted quantile to ``[ctx_min - k * scale, ctx_max + k * scale]``, where ``ctx_min``/``ctx_max``
        are the observed context bounds. Guards against runaway predictions on near-degenerate inputs.
    """

    ag_priority = 50

    default_model_path: str = "Datadog/Toto-2.0-22m"
    model_path_aliases: dict[str, str] = {
        "Toto-2.0-4m": "Datadog/Toto-2.0-4m",
        "Toto-2.0-22m": "Datadog/Toto-2.0-22m",
        "Toto-2.0-313m": "Datadog/Toto-2.0-313m",
        "Toto-2.0-1B": "Datadog/Toto-2.0-1B",
        "Toto-2.0-2.5B": "Datadog/Toto-2.0-2.5B",
    }

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

        self._model: "_Toto2Model | None" = None

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

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int | float]:
        minimum_resources: dict[str, int | float] = {"num_cpus": 1}
        if is_gpu_available:
            minimum_resources["num_gpus"] = 1
        return minimum_resources

    def _get_device(self) -> str:
        device = self.get_hyperparameter("device")
        if device is None:
            device = "cuda" if self._is_gpu_available() else "cpu"
        return device

    def load_model(self):
        from ._internal import Toto2Model as _Toto2Model

        self._model = _Toto2Model.from_pretrained(self.model_path, device=self._get_device())

    def persist(self) -> Self:
        if self._model is None:
            self.load_model()
        return self

    def _get_default_hyperparameters(self) -> dict:
        return {
            "batch_size": 256,
            "device": None,
            "context_length": 4096,
            "decode_block_size": None,
            "scaler_fallback_min_obs": 8,
            "quantile_real_cap_k": 1e4,
        }

    @property
    def allowed_hyperparameters(self) -> list[str]:
        return super().allowed_hyperparameters + [
            "model_path",
            "batch_size",
            "device",
            "context_length",
            "decode_block_size",
            "scaler_fallback_min_obs",
            "quantile_real_cap_k",
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

    @staticmethod
    def _interpolate_quantiles(
        quantile_forecast: np.ndarray, knots: np.ndarray, quantile_levels: np.ndarray
    ) -> np.ndarray:
        """Linearly interpolate the quantiles at the requested levels from the quantiles at the model's native knots.

        Levels outside the range spanned by ``knots`` are clipped to the extreme knots.

        Parameters
        ----------
        quantile_forecast
            Shape ``(num_rows, len(knots))`` array with the quantiles predicted at the native ``knots``.
        knots
            Native quantile levels produced by the model, sorted in ascending order.
        quantile_levels
            Quantile levels to interpolate.

        Returns
        -------
        Shape ``(num_rows, len(quantile_levels))`` array with the quantiles at the requested ``quantile_levels``.
        """
        idx_above = np.clip(np.searchsorted(knots, quantile_levels), 1, len(knots) - 1)
        knot_below, knot_above = knots[idx_above - 1], knots[idx_above]
        weight = np.clip((quantile_levels - knot_below) / (knot_above - knot_below), 0.0, 1.0)
        return quantile_forecast[:, idx_above - 1] * (1 - weight) + quantile_forecast[:, idx_above] * weight

    def _predict(
        self, data: TimeSeriesDataFrame, known_covariates: TimeSeriesDataFrame | None = None, **kwargs
    ) -> TimeSeriesDataFrame:
        import torch

        from .dataloader import Toto2DataLoader, Toto2InferenceDataset

        hyperparameters = self.get_hyperparameters()

        if self._model is None:
            self.load_model()
        assert self._model is not None, "Toto 2.0 model failed to load"
        device = self._get_device()

        dataset = Toto2InferenceDataset(
            target_df=data,
            max_context_length=hyperparameters["context_length"],
            target_column=self.target,
        )
        loader = Toto2DataLoader(
            dataset,
            batch_size=hyperparameters["batch_size"],
            pad_to_multiple=self._model.config.patch_size,
            time_limit=kwargs.get("time_limit"),
            device=device,
        )

        # Quantile levels natively produced by Toto 2.0
        knots = np.array(self._model.knots, dtype=np.float64)

        forecast_per_batch = []
        with torch.inference_mode():
            for batch in loader:
                # (len(knots), batch, n_var=1, horizon)
                forecast = self._model.forecast(
                    batch,
                    horizon=self.prediction_length,
                    decode_block_size=hyperparameters["decode_block_size"],
                    has_missing_values=bool((~batch["target_mask"]).any().item()),
                    scaler_fallback_min_obs=hyperparameters["scaler_fallback_min_obs"],
                    quantile_real_cap_k=hyperparameters["quantile_real_cap_k"],
                )
                # -> (batch, horizon, len(knots))
                forecast_per_batch.append(forecast.squeeze(2).permute(1, 2, 0).cpu().numpy().astype(np.float64))

        # The median is requested first and used as the mean forecast
        interpolated = self._interpolate_quantiles(
            quantile_forecast=np.concatenate(forecast_per_batch, axis=0).reshape(-1, len(knots)),
            knots=knots,
            quantile_levels=np.array([0.5, *self.quantile_levels], dtype=np.float64),
        )

        predictions = {"mean": interpolated[:, 0]}
        predictions |= {str(q): interpolated[:, i + 1] for i, q in enumerate(self.quantile_levels)}

        df = pd.DataFrame(predictions, index=self.get_forecast_horizon_index(data))
        return TimeSeriesDataFrame(df)

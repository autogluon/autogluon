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
    from toto2 import Toto2Model as _Toto2Model

logger = logging.getLogger(__name__)


class Toto2Model(AbstractTimeSeriesModel):
    """Toto 2.0 [Khwaja2026]_ pretrained time series forecasting model.

    Toto 2.0 is a family of decoder-only foundation models for time series forecasting built by Datadog. It features a
    u-μP-scaled transformer with alternating time/variate attention and quantile-based probabilistic forecasting, with
    model sizes ranging from 4M to 2.5B parameters. The full collection of Toto 2.0 models is available on
    `Hugging Face <https://huggingface.co/collections/Datadog/toto-20>`_.

    AutoGluon supports Toto 2.0 for **inference only**, i.e., the model will not be trained or fine-tuned on the provided
    training data. This wrapper currently uses the model in univariate mode and does not use covariates. Unlike Toto 1.0,
    Toto 2.0 can run on both CPU and GPU (a CUDA-compatible GPU is recommended for faster inference).

    Toto 2.0 is provided by the optional ``toto-2`` package (which requires Python 3.12+ and PyTorch 2.5+) that must be
    installed separately with ``pip install toto-2``.

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
        ``Datadog/Toto-2.0-22m``, ``Datadog/Toto-2.0-313m``, ``Datadog/Toto-2.0-1B``, and ``Datadog/Toto-2.0-2.5B``.
    batch_size : int, default = 64
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
    """

    ag_priority = 50

    default_model_path: str = "Datadog/Toto-2.0-22m"

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

        self.model_path = hyperparameters.get("model_path", self.default_model_path)

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
        try:
            from toto2 import Toto2Model as _Toto2Model
        except ImportError as err:
            raise ImportError(
                f"{self.name} requires the `toto-2` package to be installed. "
                "Please install it with `pip install toto-2` (requires Python 3.12+ and PyTorch 2.5+)."
            ) from err

        model = _Toto2Model.from_pretrained(self.model_path)
        self._model = model.to(self._get_device()).eval()

    def persist(self) -> Self:
        if self._model is None:
            self.load_model()
        return self

    def _get_default_hyperparameters(self) -> dict:
        return {
            "batch_size": 64,
            "device": None,
            "context_length": 4096,
            "decode_block_size": None,
        }

    @property
    def allowed_hyperparameters(self) -> list[str]:
        return super().allowed_hyperparameters + [
            "model_path",
            "batch_size",
            "device",
            "context_length",
            "decode_block_size",
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
        model_quantiles = np.array(self._model.output_head.knots, dtype=np.float64)

        batch_quantiles = []
        with torch.inference_mode():
            for batch in loader:
                # (num_model_quantiles, batch, n_var=1, horizon)
                forecast = self._model.forecast(
                    {
                        "target": batch.target,
                        "target_mask": batch.target_mask,
                        "series_ids": batch.series_ids,
                    },
                    horizon=self.prediction_length,
                    decode_block_size=hyperparameters["decode_block_size"],
                    has_missing_values=bool((~batch.target_mask).any().item()),
                )
                # -> (batch, horizon, num_model_quantiles)
                qs = forecast.squeeze(2).permute(1, 2, 0).cpu().numpy().astype(np.float64)
                batch_quantiles.append(qs)

        # (num_items * horizon, num_model_quantiles)
        all_quantiles = np.concatenate(batch_quantiles, axis=0).reshape(-1, len(model_quantiles))

        # Linearly interpolate the requested quantile levels from the model's native quantiles.
        # Requested levels outside [min, max] of the native quantiles are clipped to the extreme quantiles.
        predicted_quantiles = np.stack(
            [np.interp(q, model_quantiles, row) for row in all_quantiles for q in self.quantile_levels]
        ).reshape(-1, len(self.quantile_levels))

        # Use the median (0.5 quantile) as the point forecast, since forecast() does not return a sample mean.
        median_idx = self.output_head_median_index(model_quantiles)
        mean = all_quantiles[:, median_idx].reshape(-1, 1)

        df = pd.DataFrame(
            np.concatenate([mean, predicted_quantiles], axis=1),
            columns=["mean"] + [str(q) for q in self.quantile_levels],
            index=self.get_forecast_horizon_index(data),
        )

        return TimeSeriesDataFrame(df)

    @staticmethod
    def output_head_median_index(model_quantiles: np.ndarray) -> int:
        return int(np.argmin(np.abs(model_quantiles - 0.5)))

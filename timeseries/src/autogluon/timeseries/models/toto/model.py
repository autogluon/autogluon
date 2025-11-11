import logging
import os
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from typing_extensions import Self

from autogluon.common.loaders import load_pkl
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.features import CovariateMetadata

if TYPE_CHECKING:
    from ._internal import TotoForecaster

logger = logging.getLogger(__name__)


class TotoModel(AbstractTimeSeriesModel):
    """Toto (Time-Series-Optimized Transformer for Observability) [CohenKhwajaetal2025]_ pretrained time series forecasting model.

    Toto is a 151M parameter model trained on over 1T data points from DataDog's internal observability systems, as well as
    the GIFT-eval pretrain, Chronos pretraining, and synthetically generated time series corpora. It is a decoder-only
    architecture that autoregressively outputs parametric distribution forecasts. More details can be found on
    `Hugging Face <https://huggingface.co/Datadog/Toto-Open-Base-1.0>`_ and `GitHub <https://github.com/DataDog/toto>`_.

    The AutoGluon implementation of Toto is on a port of the original implementation. AutoGluon supports Toto for
    **inference only**, i.e., the model will not be trained or fine-tuned on the provided training data. Toto is optimized
    for easy maintenance with the rest of the AutoGluon model zoo, and does not feature some important optimizations such
    as xformers and flash-attention available in the original model repository. The AutoGluon implementation of Toto
    requires a CUDA-compatible GPU.

    References
    ----------
    .. [CohenKhwajaetal2025] Cohen, Ben, Khwaja, Emaad et al.
        "This Time is Different: An Observability Perspective on Time Series Foundation Models."
        https://arxiv.org/abs/2505.14766


    Other Parameters
    ----------------
    model_path : str, default = "Datadog/Toto-Open-Base-1.0"
        Model path used for the model, i.e., a HuggingFace transformers ``name_or_path``. Can be a
        compatible model name on HuggingFace Hub or a local path to a model directory.
    batch_size : int, default = 24
        Size of batches used during inference.
    num_samples : int, default = 256
        Number of samples used during inference.
    device : str, default = "cuda"
        Device to use for inference. Toto requires a CUDA-compatible GPU to run.
    context_length : int or None, default = 4096
        The context length to use in the model. Shorter context lengths will decrease model accuracy, but result
        in faster inference.
    compile_model : bool, default = True
        Whether to compile the model using torch.compile() for faster inference. May increase initial loading time
        but can provide speedups during inference.
    """

    default_model_path: str = "Datadog/Toto-Open-Base-1.0"

    def __init__(
        self,
        path: Optional[str] = None,
        name: Optional[str] = None,
        hyperparameters: Optional[dict[str, Any]] = None,
        freq: Optional[str] = None,
        prediction_length: int = 1,
        covariate_metadata: Optional[CovariateMetadata] = None,
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

        self._forecaster: Optional[TotoForecaster] = None

    def save(self, path: Optional[str] = None, verbose: bool = True) -> str:
        forecaster = self._forecaster
        self._forecaster = None
        path = super().save(path=path, verbose=verbose)
        self._forecaster = forecaster

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

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, Union[int, float]]:
        return {"num_cpus": 1, "num_gpus": 1}

    def load_forecaster(self):
        from ._internal import TotoForecaster
        from .hf_pretrained_model import TotoConfig, TotoPretrainedModel

        if not self._is_gpu_available():
            raise RuntimeError(
                f"{self.name} requires a GPU to run, but no GPU was detected. "
                "Please make sure that you are using a computer with a CUDA-compatible GPU and "
                "`import torch; torch.cuda.is_available()` returns `True`."
            )

        hyperparameters = self.get_hyperparameters()
        pretrained_model = TotoPretrainedModel.from_pretrained(
            self.model_path,
            config=TotoConfig.from_pretrained(self.model_path),
            device_map=hyperparameters["device"],
        )

        if hyperparameters["compile_model"]:
            pretrained_model.model.compile()

        self._forecaster = TotoForecaster(model=pretrained_model.model)

    def persist(self) -> Self:
        if self._forecaster is None:
            self.load_forecaster()
        return self

    def _get_default_hyperparameters(self) -> dict:
        return {
            "batch_size": 24,
            "num_samples": 256,
            "device": "cuda",
            "context_length": 4096,
            "compile_model": True,
        }

    @property
    def allowed_hyperparameters(self) -> list[str]:
        return super().allowed_hyperparameters + [
            "model_path",
            "batch_size",
            "num_samples",
            "device",
            "context_length",
            "compile_model",
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
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[float] = None,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
        verbosity: int = 2,
        **kwargs,
    ) -> None:
        self._check_fit_params()
        self.load_forecaster()

    def _predict(
        self, data: TimeSeriesDataFrame, known_covariates: Optional[TimeSeriesDataFrame] = None, **kwargs
    ) -> TimeSeriesDataFrame:
        import torch

        from .dataloader import TotoDataLoader, TotoInferenceDataset

        hyperparameters = self.get_hyperparameters()

        if self._forecaster is None:
            self.load_forecaster()
        assert self._forecaster, "Toto model failed to load"
        device = self._forecaster.model.device

        dataset = TotoInferenceDataset(
            target_df=data.fill_missing_values("auto"),
            max_context_length=hyperparameters["context_length"],
        )
        loader = TotoDataLoader(
            dataset,
            freq=self.freq,
            batch_size=hyperparameters["batch_size"],
            time_limit=kwargs.get("time_limit"),
            device=device,
        )

        batch_means, batch_quantiles = [], []
        with torch.inference_mode():
            for masked_timeseries in loader:
                forecast = self._forecaster.forecast(
                    masked_timeseries,
                    prediction_length=self.prediction_length,
                    num_samples=hyperparameters["num_samples"],
                    samples_per_batch=32,
                )

                batch_means.append(forecast.mean.cpu().numpy())
                qs = np.array([forecast.quantile(q).cpu().numpy() for q in self.quantile_levels])
                batch_quantiles.append(qs.squeeze(2).transpose(1, 2, 0))

        df = pd.DataFrame(
            np.concatenate(
                [
                    np.concatenate(batch_means, axis=0).reshape(-1, 1),
                    np.concatenate(batch_quantiles, axis=0).reshape(-1, len(self.quantile_levels)),
                ],
                axis=1,
            ),
            columns=["mean"] + [str(q) for q in self.quantile_levels],
            index=self.get_forecast_horizon_index(data),
        )

        return TimeSeriesDataFrame(df)

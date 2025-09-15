import logging
import os
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from typing_extensions import Self

from autogluon.common.loaders import load_pkl
from autogluon.common.space import Space
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.features import CovariateMetadata

from .dataloader import TotoDataLoader, TotoInferenceDataset

if TYPE_CHECKING:
    from ._internal import TotoForecaster

logger = logging.getLogger(__name__)


class TotoModel(AbstractTimeSeriesModel):
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

        name = name or "Toto"

        if not isinstance(self.model_path, Space):
            # we truncate the name to avoid long path errors on Windows
            model_path_suffix = "[" + str(self.model_path).replace("/", "__").replace(os.path.sep, "__")[-50:] + "]"
            if model_path_suffix not in name:
                name += model_path_suffix

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
        from ._internal import TotoConfig, TotoForecaster, TotoPretrainedModel

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
            torch_dtype=hyperparameters["torch_dtype"],
            device_map=hyperparameters["device"],
        )

        self._forecaster = TotoForecaster(model=pretrained_model.model)

    def persist(self) -> Self:
        if self._forecaster is None:
            self.load_forecaster()
        return self

    def _get_default_hyperparameters(self) -> dict:
        return {
            "batch_size": 16,
            "num_samples": 256,
            "device": "cuda",
            "torch_dtype": "bfloat16",
            "context_length": 2048,
        }

    @property
    def allowed_hyperparameters(self) -> list[str]:
        return super().allowed_hyperparameters + [
            "model_path",
            "batch_size",
            "num_samples",
            "device",
            "context_length",
            "torch_dtype",
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

        hyperparameters = self.get_hyperparameters()

        if self._forecaster is None:
            self.load_forecaster()
        assert self._forecaster, "Toto model failed to load"
        device = self._forecaster.model.device

        dataset = TotoInferenceDataset(data, context_length=hyperparameters["context_length"])
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

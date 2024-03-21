import logging
import os
from itertools import chain
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import pandas as pd

from autogluon.common.loaders import load_pkl
from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.forecast import get_forecast_horizon_index_ts_dataframe
from autogluon.timeseries.utils.warning_filters import warning_filter

logger = logging.getLogger(__name__)


# allowed HuggingFace model paths with custom parameter definitions
MODEL_CONFIGS = {
    "amazon/chronos-t5-tiny": {
        "num_gpus": 0,  # minimum number of required GPUs
    },
    "amazon/chronos-t5-mini": {"num_gpus": 0},
    "amazon/chronos-t5-small": {"num_gpus": 1},
    "amazon/chronos-t5-base": {"num_gpus": 1},
    "amazon/chronos-t5-large": {"num_gpus": 1},
}


MODEL_ALIASES = {
    "tiny": "amazon/chronos-t5-tiny",
    "mini": "amazon/chronos-t5-mini",
    "small": "amazon/chronos-t5-small",
    "base": "amazon/chronos-t5-base",
    "large": "amazon/chronos-t5-large",
}


class ChronosInferenceDataset:
    """A container for time series datasets that implements the ``torch.utils.data.Dataset`` interface"""

    def __init__(
        self,
        target_df: TimeSeriesDataFrame,
        context_length: int,
        target_column: str = "target",
    ):
        assert context_length > 0
        self.context_length = context_length
        self.target_array = target_df[target_column].to_numpy(dtype=np.float32)
        self.freq = target_df.freq

        # store pointer to start:end of each time series
        cum_sizes = target_df.num_timesteps_per_item().values.cumsum()
        self.indptr = np.append(0, cum_sizes).astype(np.int32)

    def __len__(self):
        return len(self.indptr) - 1  # noqa

    def _get_context(self, a: np.ndarray, pad_value=np.nan):
        a = a[-self.context_length :]
        pad_size = self.context_length - len(a)
        if pad_size > 0:
            pad = np.full(shape=(pad_size,), fill_value=pad_value)
            a = np.concatenate((pad, a))
        return a

    def __getitem__(self, idx) -> np.ndarray:
        start_idx = self.indptr[idx]
        end_idx = self.indptr[idx + 1]

        return self._get_context(self.target_array[start_idx:end_idx])


class ChronosModel(AbstractTimeSeriesModel):
    """Chronos pretrained time series forecasting models, based on the original
    `ChronosModel <https://github.com/amazon-science/chronos-forecasting>`_ implementation.

    Chronos is family of pretrained models, based on the T5 family, with number of parameters ranging between 8M and 710M.
    The full collection of Chronos models is available on
    `Hugging Face <https://huggingface.co/collections/amazon/chronos-models-65f1791d630a8d57cb718444>`_. For Chronos small,
    base, and large variants a GPU is required to perform inference efficiently.

    Chronos takes a minimalistic approach to pretraining time series models, by discretizing time series data directly into bins
    which are treated as tokens, effectively performing regression by classification. This results in a simple and flexible framework
    for using any language model in the context of time series forecasting. See [Ansari2024]_ for more information.

    References
    ----------
    .. [Ansari2024] Ansari, Abdul Fatir, Stella, Lorenzo et al.
        "Chronos: Learning the Language of Time Series."
        http://arxiv.org/abs/2403.07815


    Other Parameters
    ----------------
    model_path: str, default = "amazon/chronos-t5-small"
        Model path used for the model, i.e., a HuggingFace transformers ``name_or_path``. Can be a
        compatible model name on HuggingFace Hub or a local path to a model directory. Original
        Chronos models (i.e., ``amazon/chronos-t5-{model_size}``) can be specified with aliases
        ``tiny``, ``mini`` , ``small``, ``base``, and ``large``.
    batch_size : int, default = 16
        Size of batches used during inference
    num_samples : int, default = 20
        Number of samples used during inference
    device : str, default = None
        Device to use for inference. If None, model will use the GPU if available. For larger model sizes
        `small`, `base`, and `large`; inference will fail if no GPU is available.
    context_length : int or None, default = None
        The context length to use in the model. Shorter context lengths will decrease model accuracy, but result
        in faster inference. If None, the model will infer context length from the data set length at inference
        time, but set it to a maximum of 512.
    optimization_strategy : {None, "onnx", "openvino"}, default = None
        Optimization strategy to use for inference on CPUs. If None, the model will use the default implementation.
        If `onnx`, the model will be converted to ONNX and the inference will be performed using ONNX. If ``openvino``,
        inference will be performed with the model compiled to OpenVINO.
    torch_dtype : torch.dtype or str, default = "auto"
        Torch data type for model weights, provided to ``from_pretrained`` method of Hugging Face AutoModels.
    data_loader_num_workers : int, default = 1
        Number of worker processes to be used in the data loader. See documentation on ``torch.utils.data.DataLoader``
        for more information.
    """

    # default number of samples for prediction
    default_num_samples: int = 20
    default_batch_size: int = 16
    default_model_path = "amazon/chronos-t5-small"
    maximum_context_length = 512

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
        hyperparameters = hyperparameters if hyperparameters is not None else {}

        model_path_input = hyperparameters.get("model_path", self.default_model_path)
        self.model_path = MODEL_ALIASES.get(model_path_input, model_path_input)

        # TODO: automatically determine batch size based on GPU / memory availability
        self.batch_size = hyperparameters.get("batch_size", self.default_batch_size)
        self.num_samples = hyperparameters.get("num_samples", self.default_num_samples)
        self.device = hyperparameters.get("device")
        self.torch_dtype = hyperparameters.get("torch_dtype", "auto")
        self.data_loader_num_workers = hyperparameters.get("data_loader_num_workers", 1)
        self.optimization_strategy: Optional[Literal["onnx", "openvino"]] = hyperparameters.get(
            "optimization_strategy", None
        )
        self.context_length = hyperparameters.get("context_length")

        if self.context_length is not None and self.context_length > self.maximum_context_length:
            logger.warning(
                f"\tContext length {self.context_length} exceeds maximum context length {self.maximum_context_length}."
                f"Context length will be set to {self.maximum_context_length}."
            )
            self.context_length = self.maximum_context_length

        model_path_safe = str.replace(model_path_input, "/", "__")
        name = (name if name is not None else "Chronos") + f"[{model_path_safe}]"

        super().__init__(
            path=path,
            freq=freq,
            prediction_length=prediction_length,
            name=name,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            **kwargs,
        )

        self.model_pipeline: Optional[Any] = None  # of type OptimizedChronosPipeline

    def save(self, path: str = None, verbose: bool = True) -> str:
        pipeline = self.model_pipeline
        self.model_pipeline = None
        path = super().save(path=path, verbose=verbose)
        self.model_pipeline = pipeline

        return str(path)

    @classmethod
    def load(cls, path: str, reset_paths: bool = True, verbose: bool = True) -> "ChronosModel":
        model = load_pkl.load(path=os.path.join(path, cls.model_file_name), verbose=verbose)
        if reset_paths:
            model.set_contexts(path)
        return model

    def _is_gpu_available(self) -> bool:
        import torch.cuda

        return torch.cuda.is_available()

    def get_minimum_resources(self, is_gpu_available: bool = False) -> Dict[str, Union[int, float]]:
        minimum_resources = {"num_cpus": 1}
        # if GPU is available, we train with 1 GPU per trial
        if is_gpu_available:
            minimum_resources["num_gpus"] = MODEL_CONFIGS.get(self.model_path, {}).get("num_gpus", 0)
        return minimum_resources

    def load_model_pipeline(self, context_length: Optional[int] = None):
        from .chronos import OptimizedChronosPipeline

        gpu_available = self._is_gpu_available()

        if not gpu_available and MODEL_CONFIGS.get(self.model_path, {}).get("num_gpus", 0) > 0:
            raise RuntimeError(
                f"{self.name} requires a GPU to run, but no GPU was detected. "
                "Please make sure that you are using a computer with a CUDA-compatible GPU and "
                "`import torch; torch.cuda.is_available()` returns `True`."
            )

        device = self.device or ("cuda" if gpu_available else "auto")

        pipeline = OptimizedChronosPipeline.from_pretrained(
            self.model_path,
            device_map=device,
            optimization_strategy=self.optimization_strategy,
            torch_dtype=self.torch_dtype,
            context_length=context_length or self.context_length,
        )

        self.model_pipeline = pipeline

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: int = None,
        **kwargs,
    ) -> None:
        self._check_fit_params()

    def _get_inference_data_loader(
        self,
        data: TimeSeriesDataFrame,
        context_length: int,
        num_workers: int = 1,
    ):
        import torch

        chronos_dataset = ChronosInferenceDataset(
            target_df=data,
            target_column=self.target,
            context_length=context_length,
        )

        return torch.utils.data.DataLoader(
            chronos_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        # We defer initialization of the model pipeline. i.e., the model is only loaded to device memory
        # during inference. We also infer the maximum length of the time series in the inference data set
        # and use that to determine the context length of the model. If the context length is specified
        # during initialization, this is always used. If not, the context length is set to the longest
        # item length. The context length is always capped by self.maximum_context_length.
        context_length = self.context_length or min(
            data.num_timesteps_per_item().max(),
            self.maximum_context_length,
        )

        with warning_filter(all_warnings=True):
            import torch

            # load model pipeline to device memory
            self.load_model_pipeline(context_length=context_length)

            self.model_pipeline.model.eval()
            with torch.inference_mode():
                prediction_samples = (
                    self.model_pipeline.predict(
                        batch,
                        prediction_length=self.prediction_length,
                        num_samples=self.num_samples,
                        limit_prediction_length=False,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    for batch in self._get_inference_data_loader(
                        data=data,
                        num_workers=self.data_loader_num_workers,
                        context_length=context_length,
                    )
                )

        samples = np.concatenate([c.T for c in chain.from_iterable(prediction_samples)], axis=0)

        mean = samples.mean(axis=-1, keepdims=True)
        quantiles = np.quantile(samples, self.quantile_levels, axis=-1).T

        df = pd.DataFrame(
            np.concatenate([mean, quantiles], axis=1),
            columns=["mean"] + [str(q) for q in self.quantile_levels],
            index=get_forecast_horizon_index_ts_dataframe(data, self.prediction_length),
        )

        return TimeSeriesDataFrame(df)
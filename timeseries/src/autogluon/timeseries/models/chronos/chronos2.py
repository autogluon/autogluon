import logging
import os
from typing import Any, Optional

import pandas as pd
from typing_extensions import Self

from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel

logger = logging.getLogger(__name__)


class Chronos2Model(AbstractTimeSeriesModel):
    """Chronos-2 forecasting model [Ansari2025]_, which provides strong zero-shot forecasting capability
    natively taking advantage of covariates. The model can also be fine-tuned in a task specific manner.

    This implementation wraps the original implementation in the `chronos-forecasting`
    `library <https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py>`_ .
    Chronos-2 has 120M parameters and requires a GPU for efficient training and inference.

    References
    ----------
    .. [Ansari2025] Ansari, Abdul Fatir, Shchur, Oleksandr, Kuken, Jaris et al.
        "Chronos-2: From Univariate to Universal Forecasting." (2025).
        https://arxiv.org/abs/2510.15821

    Other Parameters
    ----------------
    model_path : str, default = "autogluon/chronos-2"
        Model path used for the model, i.e., a HuggingFace transformers ``name_or_path``. Can be a
        compatible model name on HuggingFace Hub or a local path to a model directory.
    batch_size : int, default = 256
        Size of batches used during inference.
    device : str, default = None
        Device to use for inference (and fine-tuning, if enabled). If None, model will use the GPU if
        available.
    context_length : int or None, default = None
        The context length to use in the model. If None, the model will use its default context length
        of 2048. Shorter context lengths will decrease model accuracy, but result in faster inference.
    fine_tune : bool, default = False
        If True, the pretrained model will be fine-tuned.
    fine_tune_mode : str, default = "full"
        Fine-tuning mode, either "full" for full fine-tuning or "lora" for Low Rank Adaptation (LoRA).
        LoRA is faster and uses less memory.
    fine_tune_lr : float, default = 1e-6
        The learning rate used for fine-tuning. When using LoRA, a higher learning rate such as 1e-5
        is recommended.
    fine_tune_steps : int, default = 1000
        The number of gradient update steps to fine-tune for.
    fine_tune_batch_size : int, default = 256
        The batch size to use for fine-tuning.
    fine_tune_lora_config : dict, optional
        Configuration for LoRA fine-tuning when ``fine_tune_mode="lora"``. If None and LoRA is enabled,
        a default configuration will be used. Example: ``{"r": 8, "lora_alpha": 16}``.
    """

    _supports_known_covariates = True
    _supports_past_covariates = True

    def __init__(
        self,
        freq: Optional[str] = None,
        prediction_length: int = 1,
        path: Optional[str] = None,
        name: Optional[str] = None,
        eval_metric: Optional[str] = None,
        hyperparameters: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        hyperparameters = hyperparameters if hyperparameters is not None else {}
        name = name if name is not None else "Chronos2"
        super().__init__(
            path=path,
            freq=freq,
            prediction_length=prediction_length,
            name=name,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            **kwargs,
        )
        self._is_fine_tuned: bool = False
        self._model_pipeline = None

    @property
    def model_path(self) -> str:
        default_model_path = self.get_hyperparameters()["model_path"]

        if self._is_fine_tuned:
            model_path = os.path.join(self.path, "finetuned-ckpt")
            if not os.path.exists(model_path):
                raise ValueError("Cannot find finetuned checkpoint for Chronos2.")
            else:
                logger.info(f"Loading model from finetuned checkpoint at {model_path}")
                return model_path

        return default_model_path

    def save(self, path: Optional[str] = None, verbose: bool = True) -> str:
        pipeline = self._model_pipeline
        self._model_pipeline = None
        path = super().save(path=path, verbose=verbose)
        self._model_pipeline = pipeline

        return str(path)

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
        self.load_model_pipeline()

        if self.get_hyperparameters()["fine_tune"]:
            self._fine_tune(train_data, val_data)

    def _get_default_hyperparameters(self) -> dict:
        return {
            "model_path": "autogluon/chronos-2",
            "batch_size": 256,
            "device": None,
            "context_length": None,
            "torch_dtype": "auto",
            "fine_tune": False,
            "fine_tune_mode": "full",
            "fine_tune_lr": 1e-6,
            "fine_tune_steps": 1000,
            "fine_tune_batch_size": 64,
            "fine_tune_lora_config": None,
        }

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        if self._model_pipeline is None:
            self.load_model_pipeline()
        assert self._model_pipeline is not None

        if min(data.num_timesteps_per_item()) < 3:
            raise RuntimeError("Chronos2 requires at least 3 observations per time series.")

        batch_size = self.get_hyperparameters()["batch_size"]
        future_df = pd.DataFrame(known_covariates.reset_index()) if known_covariates is not None else None

        forecast_df = self._model_pipeline.predict_df(
            df=pd.DataFrame(data.reset_index()),
            future_df=future_df,
            target=self.target,
            prediction_length=self.prediction_length,
            quantile_levels=self.quantile_levels,
            batch_size=batch_size,
        )

        forecast_df = forecast_df.rename(columns={"predictions": "mean"}).drop(
            columns=["target_name"], errors="ignore"
        )

        return TimeSeriesDataFrame(forecast_df)

    def load_model_pipeline(self):
        from chronos.chronos2.pipeline import Chronos2Pipeline

        hyperparameters = self.get_hyperparameters()
        device = hyperparameters["device"]
        torch_dtype = hyperparameters["torch_dtype"]

        assert self.model_path is not None
        pipeline = Chronos2Pipeline.from_pretrained(
            self.model_path,
            device_map=device,
            torch_dtype=torch_dtype,
        )

        self._model_pipeline = pipeline

    def persist(self) -> Self:
        self.load_model_pipeline()
        return self

    def _fine_tune(self, train_data: TimeSeriesDataFrame, val_data: Optional[TimeSeriesDataFrame]):
        from chronos.df_utils import convert_df_input_to_list_of_dicts_input

        def convert_data(df: pd.DataFrame):
            inputs, _, _ = convert_df_input_to_list_of_dicts_input(
                df=pd.DataFrame(df.reset_index()),
                future_df=None,
                target_columns=[self.target],
                prediction_length=self.prediction_length,
            )
            return inputs

        assert self._model_pipeline is not None
        hyperparameters = self.get_hyperparameters()

        val_inputs = convert_data(val_data) if val_data is not None else None
        self._model_pipeline = self._model_pipeline.fit(
            inputs=convert_data(train_data),
            prediction_length=self.prediction_length,
            validation_inputs=val_inputs,
            # finetune_mode=hyperparameters["fine_tune_mode"],
            # lora_config=hyperparameters["fine_tune_lora_config"],
            context_length=hyperparameters["context_length"],
            learning_rate=hyperparameters["fine_tune_lr"],
            num_steps=hyperparameters["fine_tune_steps"],
            batch_size=hyperparameters["fine_tune_batch_size"],
            output_dir=self.path,
            finetuned_ckpt_name="finetuned-ckpt",
        )
        self._is_fine_tuned = True

    def _more_tags(self) -> dict[str, Any]:
        do_fine_tune = self.get_hyperparameters()["fine_tune"]
        return {
            "allow_nan": True,
            "can_use_train_data": do_fine_tune,
            "can_use_val_data": do_fine_tune,
        }

    def _is_gpu_available(self) -> bool:
        import torch

        return torch.cuda.is_available()

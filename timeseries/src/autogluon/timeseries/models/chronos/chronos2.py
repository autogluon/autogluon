import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from typing_extensions import Self

from autogluon.timeseries.dataset import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel

logger = logging.getLogger(__name__)


class Chronos2Model(AbstractTimeSeriesModel):
    """Chronos-2 pretrained time series forecasting model [Ansari2025]_, which provides strong zero-shot forecasting
    capability natively taking advantage of covariates. The model can also be fine-tuned in a task specific manner.

    This implementation wraps the original implementation in the `chronos-forecasting`
    `library <https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py>`_ .

    Chronos-2 can be used both on GPU and CPU. However, we recommend using a GPU for faster inference and fine-tuning.

    Chronos-2 variants can be fine-tuned by setting ``fine_tune=True`` and selecting appropriate fine-tuning parameters
    such as the learning rate (``fine_tune_lr``) and max steps (``fine_tune_steps``). By default, a low-rank adapter (LoRA)
    will be used for fine-tuning.

    References
    ----------
    .. [Ansari2025] Ansari, Abdul Fatir, Shchur, Oleksandr, Kuken, Jaris et al.
        "Chronos-2: From Univariate to Universal Forecasting." (2025).
        https://arxiv.org/abs/2510.15821

    Other Parameters
    ----------------
    model_path : str, default = "autogluon/chronos-2"
        Model path used for the model, i.e., a Hugging Face transformers ``name_or_path``. Can be a
        compatible model name on Hugging Face Hub or a local path to a model directory.
    batch_size : int, default = 256
        Size of batches used during inference.
    device : str, default = None
        Device to use for inference (and fine-tuning, if enabled). If None, model will use the GPU if
        available.
    cross_learning : bool, default = True
        If True, the cross-learning mode of Chronos-2 is enabled. This means that the model will make joint
        predictions across time series in a batch, by default True
        Note: Enabling this mode makes the results sensitive to the ``batch_size`` used.
    context_length : int or None, default = None
        The context length to use for inference. If None, the model will use its default context length
        of 8192. Shorter context lengths may reduce accuracy, but result in faster inference.
    fine_tune : bool, default = False
        If True, the pretrained model will be fine-tuned.
    fine_tune_mode : str, default = "lora"
        Fine-tuning mode, either "full" for full fine-tuning or "lora" for Low Rank Adaptation (LoRA).
        LoRA is faster and uses less memory.
    fine_tune_lr : float, default = 1e-5
        The learning rate used for fine-tuning. When using full fine-tuning, a lower learning rate such as 1e-6
        is recommended.
    fine_tune_steps : int, default = 1000
        The number of gradient update steps to fine-tune for.
    fine_tune_batch_size : int, default = 32
        The batch size to use for fine-tuning.
    fine_tune_context_length : int, default = 2048
        The maximum context_length to use for fine-tuning
    eval_during_fine_tune : bool, default = False
        If True, validation will be performed during fine-tuning to select the best checkpoint. Setting this
        argument to True may result in slower fine-tuning. This parameter is ignored if ``skip_model_selection=True``
        in ``TimeSeriesPredictor.fit``.
    fine_tune_eval_max_items : int, default = 256
        The maximum number of randomly-sampled time series to use from the validation set for evaluation
        during fine-tuning. If None, the entire validation dataset will be used.
    fine_tune_lora_config : dict, optional
        Configuration for LoRA fine-tuning when ``fine_tune_mode="lora"``. If None and LoRA is enabled,
        a default configuration will be used. Example: ``{"r": 8, "lora_alpha": 16}``.
    fine_tune_trainer_kwargs : dict, optional
        Extra keyword arguments passed to ``transformers.TrainingArguments``
    """

    ag_model_aliases = ["Chronos-2"]
    fine_tuned_ckpt_name: str = "fine-tuned-ckpt"

    _supports_known_covariates = True
    _supports_past_covariates = True

    def __init__(
        self,
        freq: str | None = None,
        prediction_length: int = 1,
        path: str | None = None,
        name: str | None = None,
        eval_metric: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        **kwargs,
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
        self._is_fine_tuned: bool = False
        self._model_pipeline = None

    @property
    def model_path(self) -> str:
        default_model_path = self.get_hyperparameter("model_path")

        if self._is_fine_tuned:
            model_path = os.path.join(self.path, self.fine_tuned_ckpt_name)
            if not os.path.exists(model_path):
                raise ValueError("Cannot find finetuned checkpoint for Chronos-2.")
            else:
                return model_path

        return default_model_path

    def save(self, path: str | None = None, verbose: bool = True) -> str:
        pipeline = self._model_pipeline
        self._model_pipeline = None
        path = super().save(path=path, verbose=verbose)
        self._model_pipeline = pipeline

        return str(path)

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
        self._log_unused_hyperparameters()
        self.load_model_pipeline()

        # NOTE: This must be placed after load_model_pipeline to ensure that the loggers are available in loggerDict
        self._update_transformers_loggers(logging.ERROR if verbosity <= 3 else logging.INFO)

        if self.get_hyperparameter("fine_tune"):
            self._fine_tune(train_data, val_data, time_limit=time_limit, verbosity=verbosity)

    def get_hyperparameters(self) -> dict:
        """Gets params that are passed to the inner model."""
        init_args = super().get_hyperparameters()

        fine_tune_trainer_kwargs = dict(disable_tqdm=True)
        user_fine_tune_trainer_kwargs = init_args.get("fine_tune_trainer_kwargs", {})
        fine_tune_trainer_kwargs.update(user_fine_tune_trainer_kwargs)
        init_args["fine_tune_trainer_kwargs"] = fine_tune_trainer_kwargs

        return init_args.copy()

    def _get_default_hyperparameters(self) -> dict:
        return {
            "model_path": "autogluon/chronos-2",
            "batch_size": 256,
            "device": None,
            "cross_learning": True,
            "context_length": None,
            "fine_tune": False,
            "fine_tune_mode": "lora",
            "fine_tune_lr": 1e-5,
            "fine_tune_steps": 1000,
            "fine_tune_batch_size": 32,
            "fine_tune_context_length": 2048,
            "eval_during_fine_tune": False,
            "fine_tune_eval_max_items": 256,
            "fine_tune_lora_config": None,
        }

    @property
    def allowed_hyperparameters(self) -> list[str]:
        return super().allowed_hyperparameters + [
            "model_path",
            "batch_size",
            "device",
            "cross_learning",
            "context_length",
            "fine_tune",
            "fine_tune_mode",
            "fine_tune_lr",
            "fine_tune_steps",
            "fine_tune_batch_size",
            "fine_tune_context_length",
            "eval_during_fine_tune",
            "fine_tune_eval_max_items",
            "fine_tune_lora_config",
            "fine_tune_trainer_kwargs",
        ]

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: TimeSeriesDataFrame | None = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        if self._model_pipeline is None:
            self.load_model_pipeline()
        assert self._model_pipeline is not None

        if max(data.num_timesteps_per_item()) < 3:
            # If all time series have length 2 or less, we prepend 2 dummy timesteps to the first series
            first_item_id = data.index.get_level_values(0)[0]
            dummy_timestamps = pd.date_range(end=data.loc[first_item_id].index[0], periods=3, freq=self.freq)[:-1]
            full_time_index_first_item = data.loc[first_item_id].index.union(dummy_timestamps)
            new_index = (
                pd.MultiIndex.from_product([[first_item_id], full_time_index_first_item], names=data.index.names)
            ).union(data.index)
            context_df = data.reindex(new_index).reset_index()
        else:
            context_df = data.reset_index().to_data_frame()

        batch_size = self.get_hyperparameter("batch_size")
        cross_learning = self.get_hyperparameter("cross_learning")
        context_length = self.get_hyperparameter("context_length")
        future_df = known_covariates.reset_index().to_data_frame() if known_covariates is not None else None

        forecast_df = self._model_pipeline.predict_df(
            df=context_df,
            future_df=future_df,
            target=self.target,
            prediction_length=self.prediction_length,
            quantile_levels=self.quantile_levels,
            context_length=context_length,
            batch_size=batch_size,
            validate_inputs=False,
            cross_learning=cross_learning,
        )

        forecast_df = forecast_df.rename(columns={"predictions": "mean"}).drop(columns="target_name")

        return TimeSeriesDataFrame(forecast_df)

    def load_model_pipeline(self):
        from chronos.chronos2.pipeline import Chronos2Pipeline

        device = (self.get_hyperparameter("device") or "cuda") if self._is_gpu_available() else "cpu"

        assert self.model_path is not None
        pipeline = Chronos2Pipeline.from_pretrained(self.model_path, device_map=device)

        self._model_pipeline = pipeline

    def persist(self) -> Self:
        self.load_model_pipeline()
        return self

    def _update_transformers_loggers(self, log_level: int):
        for logger_name in logging.root.manager.loggerDict:
            if "transformers" in logger_name:
                transformers_logger = logging.getLogger(logger_name)
                transformers_logger.setLevel(log_level)

    def _fine_tune(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: TimeSeriesDataFrame | None,
        time_limit: float | None = None,
        verbosity: int = 2,
    ):
        from chronos.df_utils import convert_df_input_to_list_of_dicts_input

        from .utils import LoggerCallback, TimeLimitCallback

        def convert_data(df: TimeSeriesDataFrame):
            inputs, _, _ = convert_df_input_to_list_of_dicts_input(
                df=df.reset_index().to_data_frame(),
                future_df=None,
                target_columns=[self.target],
                prediction_length=self.prediction_length,
                validate_inputs=False,
            )

            # The above utility will only split the dataframe into target and past_covariates, where past_covariates contains
            # past values of both past-only and known-future covariates. We need to add future_covariates to enable fine-tuning
            # with known covariates by indicating which covariates are known in the future.
            known_covariates = self.covariate_metadata.known_covariates

            if len(known_covariates) > 0:
                for input_dict in inputs:
                    # NOTE: the covariates are empty because the actual values are not used
                    # This only indicates which covariates are known in the future
                    input_dict["future_covariates"] = {name: np.array([]) for name in known_covariates}

            return inputs

        assert self._model_pipeline is not None
        hyperparameters = self.get_hyperparameters()

        callbacks = []
        if time_limit is not None:
            callbacks.append(TimeLimitCallback(time_limit=time_limit))

        val_inputs = None
        if val_data is not None and hyperparameters["eval_during_fine_tune"]:
            # evaluate on a randomly-sampled subset
            fine_tune_eval_max_items = (
                min(val_data.num_items, hyperparameters["fine_tune_eval_max_items"])
                if hyperparameters["fine_tune_eval_max_items"] is not None
                else val_data.num_items
            )

            if fine_tune_eval_max_items < val_data.num_items:
                eval_items = np.random.choice(val_data.item_ids.values, size=fine_tune_eval_max_items, replace=False)  # noqa: F841
                val_data = val_data.query("item_id in @eval_items")

            assert isinstance(val_data, TimeSeriesDataFrame)
            val_inputs = convert_data(val_data)

        if verbosity >= 3:
            logger.warning(
                "Transformers logging is turned on during fine-tuning. Note that losses reported by transformers "
                "do not correspond to those specified via `eval_metric`."
            )
            callbacks.append(LoggerCallback())

        self._model_pipeline = self._model_pipeline.fit(
            inputs=convert_data(train_data),
            prediction_length=self.prediction_length,
            validation_inputs=val_inputs,
            finetune_mode=hyperparameters["fine_tune_mode"],
            lora_config=hyperparameters["fine_tune_lora_config"],
            context_length=hyperparameters["fine_tune_context_length"],
            learning_rate=hyperparameters["fine_tune_lr"],
            num_steps=hyperparameters["fine_tune_steps"],
            batch_size=hyperparameters["fine_tune_batch_size"],
            output_dir=self.path,
            finetuned_ckpt_name=self.fine_tuned_ckpt_name,
            callbacks=callbacks,
            remove_printer_callback=True,
            min_past=1,
            **hyperparameters["fine_tune_trainer_kwargs"],
        )
        self._is_fine_tuned = True

    def _more_tags(self) -> dict[str, Any]:
        do_fine_tune = self.get_hyperparameter("fine_tune")
        return {
            "allow_nan": True,
            "can_use_train_data": do_fine_tune,
            "can_use_val_data": do_fine_tune,
        }

    def _is_gpu_available(self) -> bool:
        import torch.cuda

        return torch.cuda.is_available()

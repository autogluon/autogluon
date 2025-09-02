import logging
import os
import shutil
import warnings
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from autogluon.common.loaders import load_pkl
from autogluon.common.space import Space
from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.warning_filters import disable_duplicate_logs, warning_filter

logger = logging.getLogger("autogluon.timeseries.models.chronos")

# TODO: Replace `evaluation_strategy` with `eval_strategy` when upgrading to `transformers>=4.41` + remove warning filter
warnings.filterwarnings("ignore", category=FutureWarning, message="`evaluation_strategy` is deprecated")
# TODO: Remove warning filter when upgrading to `transformers>=4.40`
warnings.filterwarnings("ignore", category=FutureWarning, message="Passing the following arguments to ")


# allowed HuggingFace model paths with custom parameter definitions
MODEL_CONFIGS = {
    "chronos-t5-tiny": {
        "num_gpus": 0,  # minimum number of required GPUs
        "default_torch_dtype": "auto",
        "default_batch_size": 16,
    },
    "chronos-t5-mini": {
        "num_gpus": 0,
        "default_torch_dtype": "auto",
        "default_batch_size": 16,
    },
    "chronos-t5-small": {
        "num_gpus": 1,
        "default_torch_dtype": "bfloat16",
        "default_batch_size": 16,
    },
    "chronos-t5-base": {
        "num_gpus": 1,
        "default_torch_dtype": "bfloat16",
        "default_batch_size": 16,
    },
    "chronos-t5-large": {
        "num_gpus": 1,
        "default_torch_dtype": "bfloat16",
        "default_batch_size": 8,
    },
    "chronos-bolt-mini": {
        "num_gpus": 0,
        "default_torch_dtype": "auto",
        "default_batch_size": 256,
    },
    "chronos-bolt-small": {
        "num_gpus": 0,
        "default_torch_dtype": "auto",
        "default_batch_size": 256,
    },
    "chronos-bolt-base": {
        "num_gpus": 0,
        "default_torch_dtype": "auto",
        "default_batch_size": 256,
    },
}


MODEL_ALIASES = {
    "tiny": "autogluon/chronos-t5-tiny",
    "mini": "autogluon/chronos-t5-mini",
    "small": "autogluon/chronos-t5-small",
    "base": "autogluon/chronos-t5-base",
    "large": "autogluon/chronos-t5-large",
    "bolt_tiny": "autogluon/chronos-bolt-tiny",
    "bolt_mini": "autogluon/chronos-bolt-mini",
    "bolt_small": "autogluon/chronos-bolt-small",
    "bolt_base": "autogluon/chronos-bolt-base",
}


class ChronosModel(AbstractTimeSeriesModel):
    """Chronos [Ansari2024]_ pretrained time series forecasting models which can be used for zero-shot forecasting or fine-tuned
    in a task-specific manner. Models can be based on the original
    `ChronosModel <https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos.py>`_ implementation,
    as well as a newer family of Chronos-Bolt models capable of much faster inference.

    The original Chronos is a family of pretrained models, based on the T5 family, with number of parameters ranging between
    8M and 710M. The full collection of Chronos models is available on
    `Hugging Face <https://huggingface.co/collections/amazon/chronos-models-65f1791d630a8d57cb718444>`_. For Chronos small,
    base, and large variants a GPU is required to perform inference efficiently. Chronos takes a minimalistic approach to
    pretraining time series models, by discretizing time series data directly into bins which are treated as tokens,
    effectively performing regression by classification. This results in a simple and flexible framework
    for using any language model in the context of time series forecasting. See [Ansari2024]_ for more information.

    The newer Chronos-Bolt variants enable much faster inference by first "patching" the time series. The resulting
    time series is then fed into a T5 model for forecasting. The Chronos-Bolt variants are capable of much faster inference,
    and can all run on CPUs. Chronos-Bolt models are also available on Hugging Face <https://huggingface.co/autogluon/>`_.

    Both Chronos and Chronos-Bolt variants can be fine-tuned by setting ``fine_tune=True`` and selecting appropriate
    fine-tuning parameters such as the learning rate (``fine_tune_lr``) and max steps (``fine_tune_steps``).

    References
    ----------
    .. [Ansari2024] Ansari, Abdul Fatir, Stella, Lorenzo et al.
        "Chronos: Learning the Language of Time Series."
        http://arxiv.org/abs/2403.07815


    Other Parameters
    ----------------
    model_path : str, default = "autogluon/chronos-bolt-small"
        Model path used for the model, i.e., a HuggingFace transformers ``name_or_path``. Can be a
        compatible model name on HuggingFace Hub or a local path to a model directory. Original
        Chronos models (i.e., ``autogluon/chronos-t5-{model_size}``) can be specified with aliases
        ``tiny``, ``mini`` , ``small``, ``base``, and ``large``. Chronos-Bolt models can be specified
        with ``bolt_tiny``, ``bolt_mini``, ``bolt_small``, and ``bolt_base``.
    batch_size : int, default = 256
        Size of batches used during inference. The default ``batch_size`` is selected based on the model type. For Chronos-Bolt
        models the ``batch_size`` is set to 256 whereas Chronos models used a ``batch_size`` of 16, except Chronos (Large) which
        uses 8. For the Chronos-Bolt models, the ``batch_size`` is reduced by a factor of 4 when the prediction horizon is greater
        than the model's default prediction length.
    num_samples : int, default = 20
        Number of samples used during inference, only used for the original Chronos models
    device : str, default = None
        Device to use for inference (and fine-tuning, if enabled). If None, model will use the GPU if available.
        For larger Chronos model sizes ``small``, ``base``, and ``large``; inference will fail if no GPU is available.
        For Chronos-Bolt models, inference can be done on the CPU. Although fine-tuning the smaller Chronos models
        (``tiny`` and ``mini``) and all Chronos-Bolt is allowed on the CPU, we recommend using a GPU for faster fine-tuning.
    context_length : int or None, default = None
        The context length to use in the model. Shorter context lengths will decrease model accuracy, but result
        in faster inference. If None, the model will infer context length from the data set length at inference
        time, but set it to a maximum of 2048. Note that this is only the context length used to pass data into
        the model. Individual model implementations may have different context lengths specified in their configuration,
        and may truncate the context further. For example, original Chronos models have a context length of 512, but
        Chronos-Bolt models handle contexts up to 2048.
    torch_dtype : torch.dtype or {"auto", "bfloat16", "float32", "float64"}, default = "auto"
        Torch data type for model weights, provided to ``from_pretrained`` method of Hugging Face AutoModels. If
        original Chronos models are specified and the model size is ``small``, ``base``, or ``large``, the
        ``torch_dtype`` will be set to ``bfloat16`` to enable inference on GPUs.
    data_loader_num_workers : int, default = 0
        Number of worker processes to be used in the data loader. See documentation on ``torch.utils.data.DataLoader``
        for more information.
    fine_tune : bool, default = False
        If True, the pretrained model will be fine-tuned
    fine_tune_lr : float, default = 1e-5
        The learning rate used for fine-tuning. This default is suitable for Chronos-Bolt models; for the original
        Chronos models, we recommend using a higher learning rate such as ``1e-4``
    fine_tune_steps : int, default = 1000
        The number of gradient update steps to fine-tune for
    fine_tune_batch_size : int, default = 32
        The batch size to use for fine-tuning
    fine_tune_shuffle_buffer_size : int, default = 10000
        The size of the shuffle buffer to shuffle the data during fine-tuning. If None, shuffling will
        be turned off.
    eval_during_fine_tune : bool, default = False
        If True, validation will be performed during fine-tuning to select the best checkpoint.
        Setting this argument to True may result in slower fine-tuning.
    fine_tune_eval_max_items : int, default = 256
        The maximum number of randomly-sampled time series to use from the validation set for evaluation
        during fine-tuning. If None, the entire validation dataset will be used.
    fine_tune_trainer_kwargs : dict, optional
        Extra keyword arguments passed to ``transformers.TrainingArguments``
    fine_tune_expand_quantiles : bool, default False
        If True, expands the output layer of Chronos-Bolt model to produce all ``quantile_levels`` specified when
        creating the predictor during fine-tuning. Requires ``fine_tune=True``.
    keep_transformers_logs : bool, default = False
        If True, the logs generated by transformers will NOT be removed after fine-tuning
    """

    ag_priority = 55
    # default number of samples for prediction
    default_num_samples: int = 20
    default_model_path = "autogluon/chronos-bolt-small"
    default_max_time_limit_ratio = 0.8
    maximum_context_length = 2048
    fine_tuned_ckpt_name: str = "fine-tuned-ckpt"

    def __init__(
        self,
        freq: Optional[str] = None,
        prediction_length: int = 1,
        path: Optional[str] = None,
        name: Optional[str] = None,
        eval_metric: Optional[str] = None,
        hyperparameters: Optional[dict[str, Any]] = None,
        **kwargs,  # noqa
    ):
        hyperparameters = hyperparameters if hyperparameters is not None else {}

        model_path_input = hyperparameters.get("model_path", self.default_model_path)
        self.model_path = MODEL_ALIASES.get(model_path_input, model_path_input)

        name = name if name is not None else "Chronos"
        if not isinstance(model_path_input, Space):
            # we truncate the name to avoid long path errors on Windows
            model_path_suffix = "[" + str(model_path_input).replace("/", "__").replace(os.path.sep, "__")[-50:] + "]"
            if model_path_suffix not in name:
                name += model_path_suffix

        super().__init__(
            path=path,
            freq=freq,
            prediction_length=prediction_length,
            name=name,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            **kwargs,
        )

        self._model_pipeline: Optional[Any] = None  # of type BaseChronosPipeline

    def save(self, path: Optional[str] = None, verbose: bool = True) -> str:
        pipeline = self._model_pipeline
        self._model_pipeline = None
        path = super().save(path=path, verbose=verbose)
        self._model_pipeline = pipeline

        return str(path)

    @classmethod
    def load(cls, path: str, reset_paths: bool = True, verbose: bool = True) -> "ChronosModel":
        model = load_pkl.load(path=os.path.join(path, cls.model_file_name), verbose=verbose)
        if reset_paths:
            model.set_contexts(path)

        fine_tune_ckpt_path = Path(model.path) / cls.fine_tuned_ckpt_name
        if fine_tune_ckpt_path.exists():
            logger.debug(f"\tFine-tuned checkpoint exists, setting model_path to {fine_tune_ckpt_path}")
            model.model_path = str(fine_tune_ckpt_path)

        return model

    def _is_gpu_available(self) -> bool:
        import torch.cuda

        return torch.cuda.is_available()

    @property
    def model_pipeline(self) -> Any:  # of type BaseChronosPipeline
        """The model pipeline used for inference. If the model is not loaded, this will be None."""
        if self._model_pipeline is None:
            self.load_model_pipeline()  # load model pipeline to device memory
        return self._model_pipeline

    @property
    def ag_default_config(self) -> dict[str, Any]:
        """The default configuration of the model used by AutoGluon if the model is one of those
        defined in MODEL_CONFIGS. For now, these are ``autogluon/chronos-t5-*`` family of models.
        """
        for k in MODEL_CONFIGS:
            if k in self.model_path:
                return MODEL_CONFIGS[k]
        return {}

    @property
    def min_num_gpus(self) -> int:
        """Minimum number of GPUs required for the model. For models not defined in AutoGluon,
        this value defaults to 0.
        """
        return self.ag_default_config.get("num_gpus", 0)

    @property
    def default_batch_size(self) -> int:
        """Default batch size used for the model. For models not defined in AutoGluon, this value
        defaults to 8.
        """
        return self.ag_default_config.get("default_batch_size", 8)

    @property
    def default_torch_dtype(self) -> Any:
        """Default torch data type used for the model. For models not defined in AutoGluon, this value
        defaults to "auto".
        """
        return self.ag_default_config.get("default_torch_dtype", "auto")

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, Union[int, float]]:
        minimum_resources: dict[str, Union[int, float]] = {"num_cpus": 1}
        # if GPU is available, we train with 1 GPU per trial
        if is_gpu_available:
            minimum_resources["num_gpus"] = self.min_num_gpus
        return minimum_resources

    def load_model_pipeline(self, is_training: bool = False):
        from .pipeline import BaseChronosPipeline

        gpu_available = self._is_gpu_available()

        if not gpu_available and self.min_num_gpus > 0:
            raise RuntimeError(
                f"{self.name} requires a GPU to run, but no GPU was detected. "
                "Please make sure that you are using a computer with a CUDA-compatible GPU and "
                "`import torch; torch.cuda.is_available()` returns `True`."
            )

        device = self.device or ("cuda" if gpu_available else "cpu")

        pipeline = BaseChronosPipeline.from_pretrained(
            self.model_path,
            device_map=device,
            torch_dtype=self.torch_dtype,
        )

        self._model_pipeline = pipeline

    def persist(self) -> "ChronosModel":
        # TODO: Check the model has been fit before persist
        self.load_model_pipeline()
        return self

    def _has_tf32(self):
        import torch.cuda

        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

    def get_hyperparameters(self) -> dict:
        """Gets params that are passed to the inner model."""
        init_args = super().get_hyperparameters()

        eval_during_fine_tune = init_args["eval_during_fine_tune"]
        fine_tune_trainer_kwargs = self._get_fine_tune_trainer_kwargs(init_args, eval_during_fine_tune)
        user_fine_tune_trainer_kwargs = init_args.get("fine_tune_trainer_kwargs", {})
        fine_tune_trainer_kwargs.update(user_fine_tune_trainer_kwargs)
        init_args["fine_tune_trainer_kwargs"] = fine_tune_trainer_kwargs

        return init_args.copy()

    def _get_default_hyperparameters(self) -> dict:
        return {
            "batch_size": self.default_batch_size,
            "num_samples": self.default_num_samples,
            "device": None,
            "torch_dtype": self.default_torch_dtype,
            "data_loader_num_workers": 0,
            "context_length": None,
            "fine_tune": False,
            "keep_transformers_logs": False,
            "fine_tune_lr": 1e-5,
            "fine_tune_steps": 1000,
            "fine_tune_batch_size": 32,
            "eval_during_fine_tune": False,
            "fine_tune_eval_max_items": 256,
            "fine_tune_shuffle_buffer_size": 10_000,
            "fine_tune_expand_quantiles": False,
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
            "data_loader_num_workers",
            "fine_tune",
            "fine_tune_lr",
            "fine_tune_steps",
            "fine_tune_batch_size",
            "fine_tune_shuffle_buffer_size",
            "eval_during_fine_tune",
            "fine_tune_eval_max_items",
            "fine_tune_trainer_kwargs",
            "fine_tune_expand_quantiles",
            "keep_transformers_logs",
        ]

    def _get_fine_tune_trainer_kwargs(self, init_args, eval_during_fine_tune: bool):
        output_dir = Path(self.path) / "transformers_logs"
        fine_tune_trainer_kwargs = dict(
            output_dir=str(output_dir),
            per_device_train_batch_size=init_args["fine_tune_batch_size"],
            per_device_eval_batch_size=init_args["fine_tune_batch_size"],
            learning_rate=init_args["fine_tune_lr"],
            lr_scheduler_type="linear",
            warmup_ratio=0.0,
            optim="adamw_torch_fused",
            logging_dir=str(output_dir),
            logging_strategy="steps",
            logging_steps=100,
            disable_tqdm=True,
            report_to="none",
            max_steps=init_args["fine_tune_steps"],
            gradient_accumulation_steps=1,
            dataloader_num_workers=init_args["data_loader_num_workers"],
            tf32=self._has_tf32(),
            save_only_model=True,
            prediction_loss_only=True,
            save_total_limit=1,
            save_strategy="steps" if eval_during_fine_tune else "no",
            save_steps=100 if eval_during_fine_tune else None,
            evaluation_strategy="steps" if eval_during_fine_tune else "no",
            eval_steps=100 if eval_during_fine_tune else None,
            load_best_model_at_end=True if eval_during_fine_tune else False,
            metric_for_best_model="eval_loss" if eval_during_fine_tune else None,
        )

        return fine_tune_trainer_kwargs

    def _validate_and_assign_attributes(self, model_params: dict):
        # we validate the params here because their values are concrete,
        # unlike in the constructor where they may be a search space

        # TODO: automatically determine batch size based on GPU / memory availability
        self.batch_size = model_params["batch_size"]
        self.num_samples = model_params["num_samples"]
        self.device = model_params["device"]
        self.torch_dtype = model_params["torch_dtype"]
        self.data_loader_num_workers = model_params["data_loader_num_workers"]
        self.context_length = model_params["context_length"]

        if self.context_length is not None and self.context_length > self.maximum_context_length:
            logger.info(
                f"\tContext length {self.context_length} exceeds maximum context length {self.maximum_context_length}."
                f"Context length will be set to {self.maximum_context_length}."
            )
            self.context_length = self.maximum_context_length

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[int] = None,
        **kwargs,
    ) -> None:
        import transformers
        from packaging import version
        from transformers.trainer import PrinterCallback, Trainer, TrainingArguments

        from .pipeline import ChronosBoltPipeline, ChronosPipeline, patch_chronos_bolt_output_quantiles
        from .pipeline.utils import (
            ChronosFineTuningDataset,
            EvaluateAndSaveFinalStepCallback,
            LoggerCallback,
            TimeLimitCallback,
        )

        # TODO: Add support for fine-tuning models with context_length longer than the pretrained model

        # verbosity < 3: all logs and warnings from transformers will be suppressed
        # verbosity >= 3: progress bar and loss logs will be logged
        # verbosity 4: everything will be logged
        verbosity = kwargs.get("verbosity", 2)
        for logger_name in logging.root.manager.loggerDict:
            if "transformers" in logger_name:
                transformers_logger = logging.getLogger(logger_name)
                transformers_logger.setLevel(logging.ERROR if verbosity <= 3 else logging.INFO)

        self._check_fit_params()
        self._log_unused_hyperparameters()
        model_params = self.get_hyperparameters()
        self._validate_and_assign_attributes(model_params)
        do_fine_tune = model_params["fine_tune"]

        if do_fine_tune:
            assert train_data is not None, "train_data cannot be None when fine_tune=True"

        eval_during_fine_tune = val_data is not None and model_params["eval_during_fine_tune"]

        if do_fine_tune:
            context_length = self._get_context_length(train_data)
            # load model pipeline to device memory
            self.load_model_pipeline(is_training=True)

            fine_tune_prediction_length = self.prediction_length
            model_prediction_length = self.model_pipeline.inner_model.config.chronos_config["prediction_length"]

            if isinstance(self.model_pipeline, ChronosPipeline):
                pipeline_specific_trainer_kwargs = {}

                # Update prediction_length of the model
                # NOTE: We only do this for ChronosPipeline because the prediction length of ChronosBolt models
                # is fixed due to direct multistep forecasting setup
                self.model_pipeline.model.config.prediction_length = fine_tune_prediction_length
                self.model_pipeline.inner_model.config.chronos_config["prediction_length"] = (
                    fine_tune_prediction_length
                )

            elif isinstance(self.model_pipeline, ChronosBoltPipeline):
                # custom label_names is needed for validation to work with ChronosBolt models
                pipeline_specific_trainer_kwargs = dict(label_names=["target"])

                # truncate prediction_length if it goes beyond ChronosBolt's prediction_length
                fine_tune_prediction_length = min(model_prediction_length, self.prediction_length)

                if self.prediction_length != fine_tune_prediction_length:
                    logger.debug(
                        f"\tChronosBolt models can only be fine-tuned with a maximum prediction_length of {model_prediction_length}. "
                        f"Fine-tuning prediction_length has been changed to {fine_tune_prediction_length}."
                    )
                if model_params["fine_tune_expand_quantiles"]:
                    patch_chronos_bolt_output_quantiles(self.model_pipeline.model, self.quantile_levels)
            else:
                raise ValueError(f"Unsupported model pipeline: {type(self.model_pipeline)}")

            fine_tune_trainer_kwargs = model_params["fine_tune_trainer_kwargs"]
            fine_tune_trainer_kwargs["use_cpu"] = str(self.model_pipeline.inner_model.device) == "cpu"

            if fine_tune_trainer_kwargs["use_cpu"]:
                logger.info(
                    "\tFine-tuning on the CPU detected. We recommend using a GPU for faster fine-tuning of Chronos."
                )

                # TODO: adamw_torch_fused is not supported on CPU in torch <= 2.3. When torch 2.4 becomes the lower bound
                # this if block can be removed because torch >= 2.4 supports AdamW optimizer with fused=True on CPU
                if fine_tune_trainer_kwargs["optim"] == "adamw_torch_fused":
                    fine_tune_trainer_kwargs["optim"] = "adamw_torch"

            output_dir = Path(fine_tune_trainer_kwargs["output_dir"])

            if not eval_during_fine_tune:
                # turn off eval-related trainer args
                fine_tune_trainer_kwargs["evaluation_strategy"] = "no"
                fine_tune_trainer_kwargs["eval_steps"] = None
                fine_tune_trainer_kwargs["load_best_model_at_end"] = False
                fine_tune_trainer_kwargs["metric_for_best_model"] = None

            if version.parse(transformers.__version__) >= version.parse("4.46"):
                # transformers changed the argument name from `evaluation_strategy` to `eval_strategy`
                fine_tune_trainer_kwargs["eval_strategy"] = fine_tune_trainer_kwargs.pop("evaluation_strategy")

            training_args = TrainingArguments(**fine_tune_trainer_kwargs, **pipeline_specific_trainer_kwargs)
            tokenizer_train_dataset = ChronosFineTuningDataset(
                target_df=train_data,
                target_column=self.target,
                context_length=context_length,
                prediction_length=fine_tune_prediction_length,
                # if tokenizer exists, then the data is returned in the HF-style format accepted by
                # the original Chronos models otherwise the data is returned in ChronosBolt's format
                tokenizer=getattr(self.model_pipeline, "tokenizer", None),
                mode="training",
            ).shuffle(model_params["fine_tune_shuffle_buffer_size"])

            callbacks = []
            if time_limit is not None:
                callbacks.append(TimeLimitCallback(time_limit=time_limit))

            if val_data is not None:
                callbacks.append(EvaluateAndSaveFinalStepCallback())
                # evaluate on a randomly-sampled subset
                fine_tune_eval_max_items = (
                    min(val_data.num_items, model_params["fine_tune_eval_max_items"])
                    if model_params["fine_tune_eval_max_items"] is not None
                    else val_data.num_items
                )

                if fine_tune_eval_max_items < val_data.num_items:
                    eval_items = np.random.choice(
                        val_data.item_ids.values, size=fine_tune_eval_max_items, replace=False
                    )
                    val_data = val_data.loc[eval_items]

                tokenizer_val_dataset = ChronosFineTuningDataset(
                    target_df=val_data,
                    target_column=self.target,
                    context_length=context_length,
                    prediction_length=fine_tune_prediction_length,
                    tokenizer=getattr(self.model_pipeline, "tokenizer", None),
                    mode="validation",
                )

            trainer = Trainer(
                model=self.model_pipeline.inner_model,
                args=training_args,
                train_dataset=tokenizer_train_dataset,
                eval_dataset=tokenizer_val_dataset if val_data is not None else None,
                callbacks=callbacks,
            )

            # remove PrinterCallback from callbacks which logs to the console via a print() call,
            # so it cannot be handled by setting the log level
            trainer.pop_callback(PrinterCallback)

            if verbosity >= 3:
                logger.warning(
                    "Transformers logging is turned on during fine-tuning. Note that losses reported by transformers "
                    "may not correspond to those specified via `eval_metric`."
                )
                trainer.add_callback(LoggerCallback())

            trainer.train()

            fine_tuned_ckpt_path = Path(self.path) / self.fine_tuned_ckpt_name
            logger.info(f"\tSaving fine-tuned model to {fine_tuned_ckpt_path}")

            self.model_pipeline.inner_model.save_pretrained(Path(self.path) / self.fine_tuned_ckpt_name)

            if not model_params["keep_transformers_logs"]:
                logger.debug(f"Removing transformers_logs directory {output_dir}")
                shutil.rmtree(output_dir)

    def _get_inference_data_loader(
        self,
        data: TimeSeriesDataFrame,
        context_length: int,
        batch_size: int,
        num_workers: int = 0,
        time_limit: Optional[float] = None,
    ):
        from .pipeline.utils import ChronosInferenceDataLoader, ChronosInferenceDataset, timeout_callback

        chronos_dataset = ChronosInferenceDataset(
            target_df=data,
            target_column=self.target,
            context_length=context_length,
        )

        return ChronosInferenceDataLoader(
            chronos_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            on_batch=timeout_callback(seconds=time_limit),
        )

    def _get_context_length(self, data: TimeSeriesDataFrame) -> int:
        context_length = self.context_length or min(
            data.num_timesteps_per_item().max(),
            self.maximum_context_length,
        )
        return context_length

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        from .pipeline import ChronosBoltPipeline

        # We defer initialization of the model pipeline. i.e., the model is only loaded to device memory
        # during inference. We also infer the maximum length of the time series in the inference data set
        # and use that to determine the context length of the model. If the context length is specified
        # during initialization, this is always used. If not, the context length is set to the longest
        # item length. The context length is always capped by self.maximum_context_length.
        # Note that this is independent of the model's own context length set in the model's config file.
        # For example, if the context_length is set to 2048 here but the model expects context length
        # (according to its config.json file) of 512, it will further truncate the series during inference.
        context_length = self._get_context_length(data)

        with warning_filter(all_warnings=True):
            import torch

            self.model_pipeline.model.eval()
            batch_size = self.batch_size
            if (
                isinstance(self.model_pipeline, ChronosBoltPipeline)
                and self.prediction_length > self.model_pipeline.model_prediction_length
            ):
                batch_size = max(1, batch_size // 4)
                logger.debug(
                    f"\tThe prediction_length {self.prediction_length} exceeds model's prediction_length {self.model_pipeline.model_prediction_length}. "
                    f"The inference batch_size has been reduced from {self.batch_size} to {batch_size} to avoid OOM errors."
                )

            inference_data_loader = self._get_inference_data_loader(
                data=data,
                batch_size=batch_size,
                num_workers=self.data_loader_num_workers,
                context_length=context_length,
                time_limit=kwargs.get("time_limit"),
            )

            with torch.inference_mode(), disable_duplicate_logs(logger):
                batch_quantiles, batch_means = [], []
                for batch in inference_data_loader:
                    try:
                        qs, mn = self.model_pipeline.predict_quantiles(
                            batch,
                            prediction_length=self.prediction_length,
                            quantile_levels=self.quantile_levels,
                            num_samples=self.num_samples,
                        )
                    except torch.OutOfMemoryError as ex:
                        logger.error(
                            "The call to predict() resulted in an out of memory error. Try reducing the batch_size by setting:"
                            f" predictor.fit(..., hyperparameters={{'Chronos': {{'batch_size': {batch_size // 2}, ...}}}})"
                        )
                        raise ex
                    batch_quantiles.append(qs.numpy())
                    batch_means.append(mn.numpy())

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

    def _more_tags(self) -> dict:
        do_fine_tune = self.get_hyperparameters()["fine_tune"]
        return {
            "allow_nan": True,
            "can_use_train_data": do_fine_tune,
            "can_use_val_data": do_fine_tune,
        }

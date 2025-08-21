# Authors: Lorenzo Stella <stellalo@amazon.com>, Caner Turkmen <atturkm@amazon.com>

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import torch

from .utils import left_pad_and_stack_1D

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class ForecastType(Enum):
    SAMPLES = "samples"
    QUANTILES = "quantiles"


class PipelineRegistry(type):
    REGISTRY: dict[str, "PipelineRegistry"] = {}

    def __new__(cls, name, bases, attrs):
        """See, https://github.com/faif/python-patterns."""
        new_cls = type.__new__(cls, name, bases, attrs)
        if name is not None:
            cls.REGISTRY[name] = new_cls
        if aliases := attrs.get("_aliases"):
            for alias in aliases:
                cls.REGISTRY[alias] = new_cls
        return new_cls


class BaseChronosPipeline(metaclass=PipelineRegistry):
    forecast_type: ForecastType
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }

    def __init__(self, inner_model: "PreTrainedModel"):
        """
        Parameters
        ----------
        inner_model
            A hugging-face transformers PreTrainedModel, e.g., T5ForConditionalGeneration
        """
        # for easy access to the inner HF-style model
        self.inner_model = inner_model

    def _prepare_and_validate_context(self, context: Union[torch.Tensor, list[torch.Tensor]]):
        if isinstance(context, list):
            context = left_pad_and_stack_1D(context)
        assert isinstance(context, torch.Tensor)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        assert context.ndim == 2

        return context

    def predict(
        self,
        context: Union[torch.Tensor, list[torch.Tensor]],
        prediction_length: Optional[int] = None,
        **kwargs,
    ):
        """
        Get forecasts for the given time series.

        Parameters
        ----------
        context
            Input series. This is either a 1D tensor, or a list
            of 1D tensors, or a 2D tensor whose first dimension
            is batch. In the latter case, use left-padding with
            ``torch.nan`` to align series of different lengths.
        prediction_length
            Time steps to predict. Defaults to a model-dependent
            value if not given.

        Returns
        -------
        forecasts
            Tensor containing forecasts. The layout and meaning
            of the forecasts values depends on ``self.forecast_type``.
        """
        raise NotImplementedError()

    def predict_quantiles(
        self, context: torch.Tensor, prediction_length: int, quantile_levels: list[float], **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get quantile and mean forecasts for given time series. All
        predictions are returned on the CPU.

        Parameters
        ----------
        context
            Input series. This is either a 1D tensor, or a list
            of 1D tensors, or a 2D tensor whose first dimension
            is batch. In the latter case, use left-padding with
            ``torch.nan`` to align series of different lengths.
        prediction_length
            Time steps to predict. Defaults to a model-dependent
            value if not given.
        quantile_levels
            Quantile levels to compute

        Returns
        -------
        quantiles
            Tensor containing quantile forecasts. Shape
            (batch_size, prediction_length, num_quantiles)
        mean
            Tensor containing mean (point) forecasts. Shape
            (batch_size, prediction_length)
        """
        raise NotImplementedError()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        *model_args,
        force=False,
        **kwargs,
    ):
        """
        Load the model, either from a local path or from the HuggingFace Hub.
        Supports the same arguments as ``AutoConfig`` and ``AutoModel``
        from ``transformers``.

        When a local path is provided, supports both a folder or a .tar.gz archive.
        """
        from transformers import AutoConfig

        kwargs.setdefault("resume_download", None)  # silence huggingface_hub warning
        if str(pretrained_model_name_or_path).startswith("s3://"):
            from .utils import cache_model_from_s3

            local_model_path = cache_model_from_s3(str(pretrained_model_name_or_path), force=force)
            return cls.from_pretrained(local_model_path, *model_args, **kwargs)

        torch_dtype = kwargs.get("torch_dtype", "auto")
        if torch_dtype != "auto" and isinstance(torch_dtype, str):
            kwargs["torch_dtype"] = cls.dtypes[torch_dtype]

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        is_valid_config = hasattr(config, "chronos_pipeline_class") or hasattr(config, "chronos_config")

        if not is_valid_config:
            raise ValueError("Not a Chronos config file")

        pipeline_class_name = getattr(config, "chronos_pipeline_class", "ChronosPipeline")
        class_: Optional[BaseChronosPipeline] = PipelineRegistry.REGISTRY.get(pipeline_class_name)  # type: ignore
        if class_ is None:
            raise ValueError(f"Trying to load unknown pipeline class: {pipeline_class_name}")

        return class_.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

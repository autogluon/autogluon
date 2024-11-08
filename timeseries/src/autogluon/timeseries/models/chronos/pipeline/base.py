# Authors: Lorenzo Stella <stellalo@amazon.com>, Caner Turkmen <atturkm@amazon.com>

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch


class ForecastType(Enum):
    SAMPLES = "samples"
    QUANTILES = "quantiles"


class PipelineRegistry(type):
    REGISTRY: Dict[str, "PipelineRegistry"] = {}

    def __new__(cls, name, bases, attrs):
        """See, https://github.com/faif/python-patterns."""
        new_cls = type.__new__(cls, name, bases, attrs)
        if name is not None:
            cls.REGISTRY[name] = new_cls
        return new_cls


class BaseChronosPipeline(metaclass=PipelineRegistry):
    forecast_type: ForecastType

    def predict(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
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
        self, context: torch.Tensor, prediction_length: int, quantile_levels: List[float], **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        quantile_levels: List[float]
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

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        is_valid_config = hasattr(config, "chronos_pipeline_class") or hasattr(config, "chronos_config")

        if not is_valid_config:
            raise ValueError("Not a Chronos config file")

        pipeline_class_name = getattr(config, "chronos_pipeline_class", "ChronosPipeline")
        class_ = PipelineRegistry.REGISTRY.get(pipeline_class_name)

        return class_.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

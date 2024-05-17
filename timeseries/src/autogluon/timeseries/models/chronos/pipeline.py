# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Original Source: https://github.com/amazon-science/chronos-forecasting
# Authors: Lorenzo Stella <stellalo@amazon.com>, Abdul Fatir Ansari <ansarnd@amazon.com>

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, GenerationConfig, PreTrainedModel

from autogluon.timeseries.utils.warning_filters import set_loggers_level

logger = logging.getLogger(__name__)


__all__ = ["ChronosConfig", "ChronosPipeline", "OptimizedChronosPipeline"]


@dataclass
class ChronosConfig:
    """
    This class holds all the configuration parameters to be used
    by ``ChronosTokenizer`` and ``ChronosPretrainedModel``.
    """

    tokenizer_class: str
    tokenizer_kwargs: Dict[str, Any]
    n_tokens: int
    n_special_tokens: int
    pad_token_id: int
    eos_token_id: int
    use_eos_token: bool
    model_type: Literal["causal", "seq2seq"]
    context_length: int
    prediction_length: int
    num_samples: int
    temperature: float
    top_k: int
    top_p: float

    def __post_init__(self):
        assert (
            self.pad_token_id < self.n_special_tokens and self.eos_token_id < self.n_special_tokens
        ), f"Special token id's must be smaller than {self.n_special_tokens=}"

    def create_tokenizer(self) -> "ChronosTokenizer":
        if self.tokenizer_class == "MeanScaleUniformBins":
            return MeanScaleUniformBins(**self.tokenizer_kwargs, config=self)
        raise ValueError


class ChronosTokenizer:
    """
    A ``ChronosTokenizer`` defines how time series are mapped into token IDs
    and back.

    For details, see the ``input_transform`` and ``output_transform`` methods,
    which concrete classes must implement.
    """

    def input_transform(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Turn a batch of time series into token IDs, attention map, and scale.

        Parameters
        ----------
        context
            A tensor shaped (batch_size, time_length), containing the
            timeseries to forecast. Use left-padding with ``torch.nan``
            to align time series of different lengths.

        Returns
        -------
        token_ids
            A tensor of integers, shaped (batch_size, time_length + 1)
            if ``config.use_eos_token`` and (batch_size, time_length)
            otherwise, containing token IDs for the input series.
        attention_mask
            A boolean tensor, same shape as ``token_ids``, indicating
            which input observations are not ``torch.nan`` (i.e. not
            missing nor padding).
        tokenizer_state
            An object that will be passed to ``output_transform``.
            Contains the relevant context to decode output samples into
            real values, such as location and scale parameters.
        """
        raise NotImplementedError()

    def output_transform(self, samples: torch.Tensor, tokenizer_state: Any) -> torch.Tensor:
        """
        Turn a batch of sample token IDs into real values.

        Parameters
        ----------
        samples
            A tensor of integers, shaped (batch_size, num_samples, time_length),
            containing token IDs of sample trajectories.
        tokenizer_state
            An object returned by ``input_transform`` containing
            relevant context to decode samples, such as location and scale.
            The nature of this depends on the specific tokenizer.

        Returns
        -------
        forecasts
            A real tensor, shaped (batch_size, num_samples, time_length),
            containing forecasted sample paths.
        """
        raise NotImplementedError()


class MeanScaleUniformBins(ChronosTokenizer):
    def __init__(self, low_limit: float, high_limit: float, config: ChronosConfig) -> None:
        self.config = config
        self.centers = torch.linspace(
            low_limit,
            high_limit,
            config.n_tokens - config.n_special_tokens - 1,
        )
        self.boundaries = torch.concat(
            (
                torch.tensor([-1e20], device=self.centers.device),
                (self.centers[1:] + self.centers[:-1]) / 2,
                torch.tensor([1e20], device=self.centers.device),
            )
        )

    def input_transform(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, length = context.shape

        if length > self.config.context_length:
            context = context[..., -self.config.context_length :]

        attention_mask = ~torch.isnan(context)
        scale = torch.nansum(torch.abs(context) * attention_mask, dim=-1) / torch.nansum(attention_mask, dim=-1)
        scale[~(scale > 0)] = 1.0
        scaled_context = context / scale.unsqueeze(dim=-1)
        token_ids = (
            torch.bucketize(
                input=scaled_context,
                boundaries=self.boundaries,
                # buckets are open to the right, see:
                # https://pytorch.org/docs/2.1/generated/torch.bucketize.html#torch-bucketize
                right=True,
            )
            + self.config.n_special_tokens
        )
        token_ids[~attention_mask] = self.config.pad_token_id

        if self.config.use_eos_token:
            eos_tokens = torch.full((batch_size, 1), fill_value=self.config.eos_token_id)
            token_ids = torch.concat((token_ids, eos_tokens), dim=1)
            eos_mask = torch.full((batch_size, 1), fill_value=True)
            attention_mask = torch.concat((attention_mask, eos_mask), dim=1)

        return token_ids, attention_mask, scale

    def output_transform(self, samples: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        scale_unsqueezed = scale.unsqueeze(-1).unsqueeze(-1)
        indices = torch.clamp(
            samples - self.config.n_special_tokens - 1,
            min=0,
            max=len(self.centers) - 1,
        )
        return self.centers[indices] * scale_unsqueezed


class ChronosPretrainedModel(nn.Module):
    """
    A ``ChronosPretrainedModel`` wraps a ``PreTrainedModel`` object from ``transformers``
    and uses it to predict sample paths for time series tokens.

    Parameters
    ----------
    config
        The configuration to use.
    model
        The pre-trained model to use.
    """

    def __init__(self, config: ChronosConfig, model: PreTrainedModel) -> None:
        super().__init__()
        self.config = config
        self.model = model

    @property
    def device(self):
        return self.model.device

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """
        Extract the encoder embedding for the given token sequences.

        Parameters
        ----------
        input_ids
            Tensor of indices of input sequence tokens in the vocabulary
            with shape (batch_size, sequence_length).
        attention_mask
            A mask tensor of the same shape as input_ids to avoid attending
            on padding or missing tokens.

        Returns
        -------
        embedding
            A tensor of encoder embeddings with shape
            (batch_size, sequence_length, d_model).
        """
        assert self.config.model_type == "seq2seq", "Encoder embeddings are only supported for encoder-decoder models"
        return self.model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prediction_length: Optional[int] = None,
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Predict future sample tokens for the given token sequences.

        Arguments ``prediction_length``, ``num_samples``, ``temperature``,
        ``top_k``, ``top_p`` can be used to customize the model inference,
        and default to the corresponding attributes in ``self.config`` if
        not provided.

        Returns
        -------
        samples
            A tensor of integers, shaped (batch_size, num_samples, time_length),
            containing forecasted sample paths.
        """
        if prediction_length is None:
            prediction_length = self.config.prediction_length
        if num_samples is None:
            num_samples = self.config.num_samples
        if temperature is None:
            temperature = self.config.temperature
        if top_k is None:
            top_k = self.config.top_k
        if top_p is None:
            top_p = self.config.top_p

        preds = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask.long(),  # int64 (long) type conversion needed for ONNX
            generation_config=GenerationConfig(
                min_new_tokens=prediction_length,
                max_new_tokens=prediction_length,
                do_sample=True,
                num_return_sequences=num_samples,
                eos_token_id=self.config.eos_token_id,
                pad_token_id=self.config.pad_token_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            ),
        )

        if self.config.model_type == "seq2seq":
            preds = preds[..., 1:]  # remove the decoder start token
        else:
            assert self.config.model_type == "causal"
            assert preds.size(-1) == input_ids.size(-1) + prediction_length
            preds = preds[..., -prediction_length:]

        return preds.reshape(input_ids.size(0), num_samples, -1)


def left_pad_and_stack_1D(tensors: List[torch.Tensor]):
    max_len = max(len(c) for c in tensors)
    padded = []
    for c in tensors:
        assert isinstance(c, torch.Tensor)
        assert c.ndim == 1
        padding = torch.full(size=(max_len - len(c),), fill_value=torch.nan, device=c.device)
        padded.append(torch.concat((padding, c), dim=-1))
    return torch.stack(padded)


class ChronosPipeline:
    """
    A ``ChronosPipeline`` uses the given tokenizer and model to forecast
    input time series.

    Use the ``from_pretrained`` class method to load serialized models.
    Use the ``predict`` method to get forecasts.

    Parameters
    ----------
    tokenizer
        The tokenizer object to use.
    model
        The model to use.
    """

    tokenizer: ChronosTokenizer
    model: ChronosPretrainedModel

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def _prepare_and_validate_context(self, context: Union[torch.Tensor, List[torch.Tensor]]):
        if isinstance(context, list):
            context = left_pad_and_stack_1D(context)
        assert isinstance(context, torch.Tensor)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        assert context.ndim == 2

        return context

    @torch.no_grad()
    def embed(self, context: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[torch.Tensor, Any]:
        """
        Get encoder embeddings for the given time series.

        Parameters
        ----------
        context
            Input series. This is either a 1D tensor, or a list
            of 1D tensors, or a 2D tensor whose first dimension
            is batch. In the latter case, use left-padding with
            ``torch.nan`` to align series of different lengths.

        Returns
        -------
        embeddings, tokenizer_state
            A tuple of two tensors: the encoder embeddings and the tokenizer_state,
            e.g., the scale of the time series in the case of mean scaling.
            The encoder embeddings are shaped (batch_size, context_length, d_model)
            or (batch_size, context_length + 1, d_model), where context_length
            is the size of the context along the time axis if a 2D tensor was provided
            or the length of the longest time series, if a list of 1D tensors was
            provided, and the extra 1 is for EOS.
        """
        context = self._prepare_and_validate_context(context=context)
        token_ids, attention_mask, tokenizer_state = self.tokenizer.input_transform(context)
        embeddings = self.model.encode(
            input_ids=token_ids.to(self.model.device),
            attention_mask=attention_mask.to(self.model.device),
        ).cpu()
        return embeddings, tokenizer_state

    def predict(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        limit_prediction_length: bool = True,
    ) -> torch.Tensor:
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
            Time steps to predict. Defaults to what specified
            in ``self.model.config``.
        num_samples
            Number of sample paths to predict. Defaults to what
            specified in ``self.model.config``.
        temperature
            Temperature to use for generating sample tokens.
            Defaults to what specified in ``self.model.config``.
        top_k
            Top-k parameter to use for generating sample tokens.
            Defaults to what specified in ``self.model.config``.
        top_p
            Top-p parameter to use for generating sample tokens.
            Defaults to what specified in ``self.model.config``.
        limit_prediction_length
            Force prediction length smaller or equal than the
            built-in prediction length from the model. True by
            default. When true, fail loudly if longer predictions
            are requested, otherwise longer predictions are allowed.

        Returns
        -------
        samples
            Tensor of sample forecasts, of shape
            (batch_size, num_samples, prediction_length).
        """
        context = self._prepare_and_validate_context(context=context)
        if prediction_length is None:
            prediction_length = self.model.config.prediction_length

        if prediction_length > self.model.config.prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {self.model.config.prediction_length}. "
                f"The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)

        predictions = []
        remaining = prediction_length

        while remaining > 0:
            token_ids, attention_mask, scale = self.tokenizer.input_transform(context)
            samples = self.model(
                token_ids.to(self.model.device),
                attention_mask.to(self.model.device),
                min(remaining, self.model.config.prediction_length),
                num_samples,
                temperature,
                top_k,
                top_p,
            )
            prediction = self.tokenizer.output_transform(samples.to(scale.device), scale)

            predictions.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            context = torch.cat([context, prediction.median(dim=1).values], dim=-1)

        return torch.cat(predictions, dim=-1)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Load the model, either from a local path or from the HuggingFace Hub.
        Supports the same arguments as ``AutoConfig`` and ``AutoModel``
        from ``transformers``.
        """

        config = AutoConfig.from_pretrained(*args, **kwargs)

        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        chronos_config = ChronosConfig(**config.chronos_config)

        if chronos_config.model_type == "seq2seq":
            inner_model = AutoModelForSeq2SeqLM.from_pretrained(*args, **kwargs)
        else:
            assert config.model_type == "causal"
            inner_model = AutoModelForCausalLM.from_pretrained(*args, **kwargs)

        return cls(
            tokenizer=chronos_config.create_tokenizer(),
            model=ChronosPretrainedModel(config=chronos_config, model=inner_model),
        )


class OptimizedChronosPipeline(ChronosPipeline):
    """A wrapper around the ChronosPipeline object for CPU-optimized model classes from
    HuggingFace optimum.
    """

    dtypes = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Load the model, either from a local path or from the HuggingFace Hub.
        Supports the same arguments as ``AutoConfig`` and ``AutoModel``
        from ``transformers``.
        """
        kwargs = kwargs.copy()

        optimization_strategy = kwargs.pop("optimization_strategy", None)
        context_length = kwargs.pop("context_length", None)

        config = AutoConfig.from_pretrained(*args, **kwargs)
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        if context_length is not None:
            config.chronos_config["context_length"] = context_length
        chronos_config = ChronosConfig(**config.chronos_config)

        torch_dtype = kwargs.get("torch_dtype", "auto")
        if torch_dtype != "auto" and isinstance(torch_dtype, str):
            kwargs["torch_dtype"] = cls.dtypes[torch_dtype]

        if chronos_config.model_type == "seq2seq":
            if optimization_strategy is None:
                inner_model = AutoModelForSeq2SeqLM.from_pretrained(*args, **kwargs)
            else:
                assert optimization_strategy in [
                    "onnx",
                    "openvino",
                ], "optimization_strategy not recognized. Please provide one of `onnx` or `openvino`"
                torch_dtype = kwargs.pop("torch_dtype", "auto")
                if torch_dtype != "auto":
                    logger.warning(
                        f"\t`torch_dtype` will be ignored for optimization_strategy {optimization_strategy}"
                    )

                if optimization_strategy == "onnx":
                    try:
                        from optimum.onnxruntime import ORTModelForSeq2SeqLM
                    except ImportError:
                        raise ImportError(
                            "Huggingface Optimum library must be installed with ONNX for using the `onnx` strategy"
                        )

                    assert kwargs.pop("device_map", "cpu") in ["cpu", "auto"], "ONNX mode only available on the CPU"
                    with set_loggers_level(regex=r"^optimum.*", level=logging.ERROR):
                        inner_model = ORTModelForSeq2SeqLM.from_pretrained(*args, **{**kwargs, "export": True})
                elif optimization_strategy == "openvino":
                    try:
                        from optimum.intel import OVModelForSeq2SeqLM
                    except ImportError:
                        raise ImportError(
                            "Huggingface Optimum library must be installed with OpenVINO for using the `openvino` strategy"
                        )
                    with set_loggers_level(regex=r"^optimum.*", level=logging.ERROR):
                        inner_model = OVModelForSeq2SeqLM.from_pretrained(
                            *args, **{**kwargs, "device_map": "cpu", "export": True}
                        )
        else:
            assert config.model_type == "causal"
            inner_model = AutoModelForCausalLM.from_pretrained(*args, **kwargs)

        return cls(
            tokenizer=chronos_config.create_tokenizer(),
            model=ChronosPretrainedModel(config=chronos_config, model=inner_model),
        )

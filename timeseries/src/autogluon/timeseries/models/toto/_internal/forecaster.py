# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

from dataclasses import dataclass
from typing import Optional, Union, cast

import numpy as np
import torch
from einops import rearrange, repeat
from gluonts.torch.distributions import AffineTransformed
from torch.distributions import Distribution

from .backbone import TotoBackbone
from .dataset import (
    MaskedTimeseries,
    pad_array,
    pad_id_mask,
    replace_extreme_values,
)


@dataclass(frozen=True)
class Forecast:
    mean: torch.Tensor
    samples: Optional[torch.Tensor]

    def quantile(self, q: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Compute the quantile of the forecast samples.
        """
        assert self.samples is not None, "samples must be provided to compute quantiles"
        assert isinstance(q, float) or isinstance(q, torch.Tensor), "q must be a float or a tensor"
        if isinstance(q, float):
            q = torch.tensor(q, device=self.samples.device, dtype=self.samples.dtype)
        return self.samples.quantile(q, dim=-1)

    @property
    def median(self) -> torch.Tensor:
        """
        The median of the forecast samples.
        """
        return self.quantile(0.5)

    @property
    def std(self) -> torch.Tensor:
        """
        Compute the standard deviation of the forecast samples.
        """
        assert self.samples is not None, "samples must be provided to compute standard deviation"
        return self.samples.std(dim=-1)


class TotoForecaster:
    """
    A forecaster class for the Toto model that handles autoregressive decoding for time series forecasting.

    This class wraps a TotoBackbone model and provides methods to generate forecasts for time series data.
    The forecasting process uses an autoregressive decoding algorithm:

    1. The model first processes the entire input context (historical data)
    2. For each future time step:
       - The model generates a distribution over possible values
       - Either the mean or random samples are drawn from this distribution
       - The generated value(s) are appended to the input sequence
       - The process repeats with this extended sequence

    When generating multiple samples (num_samples > 1), the model creates separate trajectories for each sample:
    - Each trajectory starts with the same historical context
    - As sampling progresses, each trajectory evolves independently
    - This results in num_samples different possible future paths
    - Samples can be processed in batches (samples_per_batch) to manage memory usage

    The forecaster efficiently reuses computation from the context processing phase using a key-value cache,
    which stores intermediate transformer attention states to avoid redundant computation.

    The forecaster handles data preprocessing, including padding to match the model's patch size,
    and postprocessing to format the outputs as a Forecast object containing means and optional samples.
    """

    model: TotoBackbone

    def __init__(self, model: TotoBackbone):
        self.model = model

    def forecast(
        self,
        inputs: MaskedTimeseries,
        prediction_length: int,
        num_samples: Optional[int] = None,
        samples_per_batch: int = 10,
        use_kv_cache: bool = True,
    ) -> Forecast:
        """
        Generate a forecast for a batch of time series. This method works autoregressively,
        i.e. it feeds the model's predictions back into itself. The decoding process is as follows:

        1. The model first processes the entire input context (historical data)
        2. For each future time step:
            - The model generates a distribution over possible values
            - Either the mean or random samples are drawn from this distribution
            - The generated value(s) are appended to the input sequence
            - The process repeats with this extended sequence

        There are two modes of operation:
        1. num_samples is None: generate a single mean prediction
        2. num_samples is not None: generate num_samples random samples

        When num_samples is not None, the model creates num_samples separate trajectories for each sample:
        - Each trajectory starts with the same historical context
        - As sampling progresses, each trajectory evolves independently
        - This results in num_samples different possible future paths
        - Samples can be processed in batches (samples_per_batch) to manage memory usage

        When using samples_per_batch, this batch size compounds with the optional batch dimension of the input.
        For example, if you have a batch of 10 time series, and you set samples_per_batch to 10,
        the effective batch size is 100. For the best performance, set samples_per_batch
        as high as possible, subject to memory constraints.

        Args:
            inputs: A MaskedTimeseries object containing the input time series.
            prediction_length: The number of future time steps to predict.
            num_samples:
                The number of samples to generate.
                If None, a single mean prediction is generated. However,
                the mean point forecast tends to be less accurate than the
                median or mean of the samples (provided enough samples are generated).
                It's recommended to use at least 128 samples for reliable forecasts.
            samples_per_batch:
                The number of samples to generate per batch.
                In most cases, this should be as high as possible, subject to memory constraints.
                When the inputs have a batch dimension, the effective batch size is samples_per_batch * batch_size.
            use_kv_cache:
                Whether to use a key-value cache for the model. In most cases, this should be True,
                as it significantly speeds up inference.
        """
        if len(inputs.series.shape) == 2:
            # unbatched input, variates x time_steps
            batch = cast(MaskedTimeseries, torch.utils.data.default_collate([inputs]))
        else:
            # input is already batched
            batch = inputs

        # pad the input to the nearest multiple of the patch size
        series = pad_array(batch.series, self.model.patch_embed.stride)
        padding_mask = pad_array(batch.padding_mask, self.model.patch_embed.stride)
        id_mask = batch.id_mask
        if id_mask is not None:
            id_mask = pad_id_mask(batch.id_mask, self.model.patch_embed.stride)
        timestamp_seconds = pad_array(batch.timestamp_seconds, self.model.patch_embed.stride)
        time_interval_seconds: torch.Tensor = torch.as_tensor(
            batch.time_interval_seconds, device=series.device, dtype=torch.int
        )

        if num_samples is not None:
            samples = self.generate_samples(
                inputs=series,
                prediction_length=prediction_length,
                num_samples=num_samples,
                timestamp_seconds=timestamp_seconds,
                time_interval_seconds=time_interval_seconds,
                input_padding_mask=padding_mask,
                id_mask=id_mask,
                sampling_batch_size=samples_per_batch,
                use_kv_cache=use_kv_cache,
            )
            mean = samples.mean(dim=-1)
        else:
            mean = self.generate_mean(
                inputs=series,
                prediction_length=prediction_length,
                timestamp_seconds=timestamp_seconds,
                time_interval_seconds=time_interval_seconds,
                input_padding_mask=padding_mask,
                id_mask=id_mask,
                use_kv_cache=use_kv_cache,
            )
            samples = None

        return Forecast(mean=mean, samples=samples)

    @torch.no_grad()
    def generate_mean(
        self,
        inputs: torch.Tensor,
        prediction_length: int,
        timestamp_seconds: torch.Tensor,
        time_interval_seconds: torch.Tensor,
        input_padding_mask: Optional[torch.Tensor] = None,
        id_mask: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
    ) -> torch.Tensor:
        """
        Generate a point prediction by taking the mean of the output distribution at each step.
        This method works autoregressively, i.e. it feeds the model's predictions back into itself
        to generate the next prediction.
        """
        if input_padding_mask is None:
            input_padding_mask = torch.ones_like(inputs, dtype=torch.bool, device=inputs.device)
        if id_mask is None:
            id_mask = torch.zeros_like(inputs, dtype=torch.int, device=inputs.device)

        ## round up the prediction length to the nearest multiple of the patch size
        patch_size = self.model.patch_embed.stride
        rounded_steps = int(np.ceil(prediction_length / patch_size) * patch_size)
        start_index = inputs.shape[-1]
        end_index = start_index + prediction_length

        # TODO: maybe pass in future masks, rather than making assumptions here?
        dummy_padding = torch.ones(
            (input_padding_mask.shape[0], input_padding_mask.shape[1], patch_size),
            device=inputs.device,
            dtype=torch.bool,
        )
        dummy_id_mask = repeat(
            id_mask[:, :, -1:],
            "batch variates 1 -> batch variates patch_size",
            patch_size=patch_size,
        )
        if use_kv_cache:
            kv_cache = self.model.allocate_kv_cache(
                batch_size=inputs.shape[0],
                num_variates=inputs.shape[1],
                max_time_steps=inputs.shape[2] + rounded_steps,
                device=inputs.device,
                dtype=inputs.dtype,
            )
        else:
            kv_cache = None

        scaling_prefix_length = inputs.shape[-1]

        for _ in range(rounded_steps // patch_size):
            base_distr, loc, scale = self.model(
                inputs=inputs,
                input_padding_mask=input_padding_mask,
                id_mask=id_mask,
                kv_cache=kv_cache,
                scaling_prefix_length=scaling_prefix_length,
            )
            distr = self.create_affine_transformed(base_distr, loc, scale)

            # We remove extreme values that can occur early in training
            # and cause validation metrics to be NaN
            samples = replace_extreme_values(distr.mean[:, :, -patch_size:])

            inputs = torch.cat([inputs, samples], dim=-1)
            id_mask = torch.cat([id_mask, dummy_id_mask], dim=-1)
            input_padding_mask = torch.cat([input_padding_mask, dummy_padding], dim=-1)
            for _ in range(patch_size):
                next_timestamp: torch.Tensor = timestamp_seconds[:, :, -1] + time_interval_seconds
                timestamp_seconds = torch.cat([timestamp_seconds, next_timestamp.unsqueeze(-1)], dim=-1)

        return inputs.detach()[:, :, start_index:end_index]

    @torch.no_grad()
    def generate_samples(
        self,
        inputs: torch.Tensor,
        prediction_length: int,
        num_samples: int,
        timestamp_seconds: torch.Tensor,
        time_interval_seconds: torch.Tensor,
        input_padding_mask: Optional[torch.Tensor] = None,
        id_mask: Optional[torch.Tensor] = None,
        sampling_batch_size: int = 10,
        use_kv_cache: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples from the output distribution.
        This method works autorregressively, i.e. it feeds the model's predictions back into itself.
        It works by creating num_samples chains. Each chain is a separate sequence of predictions.
        At each time step, for each chain we take a single sample from the output distribution and append
        it to the end of the sequence.
        """
        if input_padding_mask is None:
            input_padding_mask = torch.ones_like(inputs, dtype=torch.bool, device=inputs.device)
        if id_mask is None:
            id_mask = torch.zeros_like(inputs, dtype=torch.int, device=inputs.device)

        assert num_samples % sampling_batch_size == 0, "num_samples must be divisible by sampling_batch_size"
        num_batches = num_samples // sampling_batch_size

        # round up the prediction length to the nearest multiple of the patch size
        patch_size = self.model.patch_embed.patch_size
        rounded_steps = int(np.ceil(prediction_length / patch_size) * patch_size)
        start_index = inputs.shape[-1]
        end_index = start_index + prediction_length

        dummy_padding = torch.ones(
            (
                input_padding_mask.shape[0] * sampling_batch_size,
                input_padding_mask.shape[1],
                patch_size,
            ),
            dtype=torch.bool,
            device=inputs.device,
        )
        dummy_id_mask = repeat(
            id_mask[:, :, -1:],
            "batch variates 1 -> (sampling_batch_size batch) variates patch_size",
            sampling_batch_size=sampling_batch_size,
            patch_size=patch_size,
        )
        inputs = repeat(
            inputs,
            "batch variates seq_len -> (sampling_batch_size batch) variates seq_len",
            sampling_batch_size=sampling_batch_size,
        )
        input_padding_mask = repeat(
            input_padding_mask,
            "batch variates seq_len -> (sampling_batch_size batch) variates seq_len",
            sampling_batch_size=sampling_batch_size,
        )
        id_mask = repeat(
            id_mask,
            "batch variates seq_len -> (sampling_batch_size batch) variates seq_len",
            sampling_batch_size=sampling_batch_size,
        )
        timestamp_seconds = repeat(
            timestamp_seconds,
            "batch variates seq_len -> (sampling_batch_size batch) variates seq_len",
            sampling_batch_size=sampling_batch_size,
        )
        time_interval_seconds = repeat(
            time_interval_seconds,
            "batch variates -> (sampling_batch_size batch) variates",
            sampling_batch_size=sampling_batch_size,
        )

        all_samples = []
        if use_kv_cache:
            kv_cache = self.model.allocate_kv_cache(
                batch_size=inputs.shape[0],
                num_variates=inputs.shape[1],
                max_time_steps=inputs.shape[2] + rounded_steps,
                device=inputs.device,
                dtype=inputs.dtype,
            )
        else:
            kv_cache = None

        scaling_prefix_length = inputs.shape[-1]

        for _ in range(num_batches):
            batch_inputs = torch.clone(inputs)
            batch_input_padding_mask = torch.clone(input_padding_mask)
            batch_id_mask = torch.clone(id_mask)
            batch_timestamp_seconds = torch.clone(timestamp_seconds)

            for _ in range(rounded_steps // patch_size):
                base_distr, loc, scale = self.model(
                    inputs=batch_inputs,
                    input_padding_mask=batch_input_padding_mask,
                    id_mask=batch_id_mask,
                    kv_cache=kv_cache,
                    scaling_prefix_length=scaling_prefix_length,
                )
                distr = self.create_affine_transformed(base_distr, loc, scale)

                sample = distr.sample()
                assert sample is not None

                # We remove extreme values that can occur early in training
                # and cause validation metrics to be NaN
                samples = replace_extreme_values(sample[:, :, -patch_size:])
                batch_inputs = torch.cat([batch_inputs, samples], dim=-1)
                batch_id_mask = torch.cat([batch_id_mask, dummy_id_mask], dim=-1)
                batch_input_padding_mask = torch.cat([batch_input_padding_mask, dummy_padding], dim=-1)
                for _ in range(patch_size):
                    next_timestamp = batch_timestamp_seconds[:, :, -1] + time_interval_seconds
                    batch_timestamp_seconds = torch.cat(
                        [batch_timestamp_seconds, next_timestamp.unsqueeze(-1)], dim=-1
                    )
            all_samples.append(batch_inputs)
            if kv_cache is not None:
                kv_cache.reset()

        outputs = torch.cat(all_samples, dim=0)
        unfolded_outputs = rearrange(
            outputs,
            "(samples batch) variates seq_len -> batch variates seq_len samples",
            samples=num_samples,
        ).detach()

        trimmed_predictions = unfolded_outputs[:, :, start_index:end_index, :]
        return trimmed_predictions

    @staticmethod
    def create_affine_transformed(base_distr: Distribution, loc: torch.Tensor, scale: torch.Tensor) -> Distribution:
        """
        Creates an AffineTransformed distribution with correctly matched shapes.

        Handles three cases:
        1. When loc/scale are per-timestep (from CausalStdMeanScaler)
        2. When base_distr only contains the distribution for the latest patch
           while loc/scale contain values for the entire sequence
        3. When loc/scale have a single time step (from StdMeanScaler/StdMinScaler)
           and need to be broadcast to match a multi-step base distribution

        Args:
            base_distr: The base distribution to transform
            loc: Location parameter
            scale: Scale parameter

        Returns:
            An AffineTransformed distribution with properly handled shapes
        """
        # Get the shape of the base distribution
        # We'll use this to match the time dimension of loc/scale
        base_shape = base_distr.mean.shape

        base_time_dim = base_shape[-1]  # Time dimension of base distribution
        loc_time_dim = loc.shape[-1]  # Time dimension of loc

        if loc_time_dim == 1:
            # Case 1: If loc/scale have time dimension 1 (standard scalers), PyTorch broadcasting will handle it
            return AffineTransformed(base_distr, loc=loc, scale=scale)

        # Case 2: If loc/scale have time dimension > 1 (causal scaler with history)
        # We need to extract only the suffix that matches the base distribution
        return AffineTransformed(base_distr, loc=loc[:, :, -base_time_dim:], scale=scale[:, :, -base_time_dim:])

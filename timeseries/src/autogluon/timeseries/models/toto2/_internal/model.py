# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

"""Inference-only implementation of the Toto 2.0 forecasting model."""

import dataclasses
import json
import math
import os
import warnings
from typing import Optional, TypedDict

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat

from .backbone import KVCache, QuantileKnotsOutputHead, ResidualMLP, VariateTimeTransformerDecoder
from .configuration import Toto2ModelConfig

# Quantile levels natively predicted by all published Toto 2.0 checkpoints.
QUANTILE_KNOTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class Toto2ForecastInputs(TypedDict):
    """A batch of series to forecast.

    ``target`` and ``target_mask`` have shape ``(batch, num_variates, context_length)``, where ``target_mask`` marks
    the observed entries. ``series_ids`` has shape ``(batch, num_variates)`` and identifies which variates belong to
    the same multivariate series; variates with different ids do not attend to each other.
    """

    target: torch.Tensor
    target_mask: torch.Tensor
    series_ids: torch.Tensor


class PatchedCausalStdScaler(nn.Module):
    """Causal standardization using, for every position in a patch, the statistics up to the end of that patch."""

    def __init__(self, patch_size: int, correction: int = 1, minimum_scale: float = 1e-6):
        super().__init__()
        self.patch_size = patch_size
        self.correction = correction
        self.minimum_scale = minimum_scale

    def forward(self, data: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the standardized ``data`` together with the location and the scale used, all shaped like ``data``."""
        try:
            high_precision = data.to(torch.float64)
        except TypeError:
            warnings.warn(f"Float64 is not supported on {data.device}, using float32 for the scaler.", RuntimeWarning)
            high_precision = data.to(torch.float32)

        loc, scale = self._compute_loc_scale(high_precision, mask)
        loc, scale = loc.to(data.dtype), scale.to(data.dtype)
        return torch.where(mask, (data - loc) / scale, 0), loc, scale

    def _compute_loc_scale(self, data: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        denominator = mask.cumsum(dim=-1).clamp_min(1)
        causal_loc = (data * mask).cumsum(dim=-1) / denominator

        # Welford's online algorithm for the causal variance.
        prev_loc = torch.cat([torch.zeros_like(causal_loc[..., :1]), causal_loc[..., :-1]], dim=-1)
        m_2 = torch.cumsum((data - prev_loc) * (data - causal_loc) * mask, dim=-1)
        causal_var = m_2 / (denominator - self.correction).clamp(min=1)
        causal_scale = causal_var.sqrt().clamp(min=self.minimum_scale)

        # Broadcast the statistics at the last position of each patch across the whole patch.
        return tuple(
            repeat(
                rearrange(stat, "... (seq patch) -> ... seq patch", patch=self.patch_size)[..., -1],
                "... seq -> ... (seq patch)",
                patch=self.patch_size,
            )
            for stat in (causal_loc, causal_scale)
        )


def backfill_short_patches(
    target: torch.Tensor,
    loc: torch.Tensor,
    scale: torch.Tensor,
    obs_mask: torch.Tensor,
    patch_size: int,
    min_obs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stabilize ``(loc, scale)`` on leading patches that contain too few observations.

    Every leading patch whose cumulative observation count is below ``min_obs`` has its location and scale replaced by
    the statistics of the first ``min_obs`` observed points. The donor statistics are local to that leading region, so
    no later observation leaks into it. A ``min_obs <= 0`` is a no-op, and series with fewer than ``min_obs``
    observations in total fall back to the statistics of whatever is observed.
    """
    if min_obs <= 0:
        return loc, scale

    in_first_n = (obs_mask.cumsum(dim=-1) <= min_obs) & obs_mask
    n = in_first_n.sum(dim=-1, keepdim=True).clamp(min=1)
    donor_loc = (target * in_first_n).sum(dim=-1, keepdim=True) / n
    donor_var = (((target - donor_loc) * in_first_n) ** 2).sum(dim=-1, keepdim=True) / (n - 1).clamp(min=1)
    donor_scale = donor_var.sqrt().clamp(min=1e-6)

    below_min_obs = obs_mask.unflatten(-1, (-1, patch_size)).sum(dim=-1).cumsum(dim=-1) < min_obs
    below_min_obs = below_min_obs.repeat_interleave(patch_size, dim=-1)
    return (
        torch.where(below_min_obs, donor_loc.expand_as(loc), loc),
        torch.where(below_min_obs, donor_scale.expand_as(scale), scale),
    )


class Toto2Model(nn.Module):
    """Decoder-only foundation model that predicts, for each patch, the quantiles of the next patch."""

    def __init__(self, config: Toto2ModelConfig):
        super().__init__()
        self.config = config
        self.scaler = PatchedCausalStdScaler(patch_size=config.patch_size)
        # The input to the patch embedding is the standardized values of a patch concatenated with its missingness mask.
        self.patch_proj = ResidualMLP(
            in_dim=2 * config.patch_size, hidden_dim=4 * config.d_model, out_dim=config.d_model
        )
        self.transformer = VariateTimeTransformerDecoder(config)
        self.output_head = QuantileKnotsOutputHead(
            knots=list(QUANTILE_KNOTS),
            embeds_dim=config.d_model,
            hidden_dim=4 * config.d_model,
            patch_size=config.patch_size,
        )
        self._kv_cache: Optional[KVCache] = None
        self._kv_cache_key: Optional[tuple] = None

    @property
    def knots(self) -> list[float]:
        """Quantile levels natively predicted by the model, in ascending order."""
        return self.output_head.knots

    def _embed_patches(self, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Embed standardized values and their missingness mask into one token per patch."""
        patch = self.config.patch_size
        return self.patch_proj(
            torch.cat(
                [
                    rearrange(data, "... (seq patch) -> ... seq patch", patch=patch),
                    rearrange((~mask).to(data.dtype), "... (seq patch) -> ... seq patch", patch=patch),
                ],
                dim=-1,
            )
        )

    @staticmethod
    def _clamp_nonfinite(values: torch.Tensor) -> torch.Tensor:
        """Replace ``+/-inf`` quantiles with the largest/smallest finite quantile of the same patch."""
        return torch.where(
            values == float("inf"),
            torch.where(values.isfinite(), values, -float("inf")).amax(dim=-1, keepdim=True),
            torch.where(
                values == -float("inf"),
                torch.where(values.isfinite(), values, float("inf")).amin(dim=-1, keepdim=True),
                values,
            ),
        )

    def _get_kv_cache(self, max_cache_size: int, batch_shape: torch.Size, device: torch.device) -> KVCache:
        """Return a KV cache for the given shapes, reusing the previously allocated one when it still fits."""
        cache_key = (max_cache_size, tuple(batch_shape))
        if self._kv_cache is not None and self._kv_cache_key == cache_key:
            self._kv_cache.reset()
        else:
            self._kv_cache = KVCache(self.config.num_time_layers, max_size=max_cache_size).to(device)
            self._kv_cache_key = cache_key
        return self._kv_cache

    @torch.no_grad()
    def forecast(
        self,
        inputs: Toto2ForecastInputs,
        horizon: int,
        decode_block_size: Optional[int] = None,
        has_missing_values: bool = True,
        scaler_fallback_min_obs: int = 0,
        quantile_real_cap_k: float = 0.0,
    ) -> torch.Tensor:
        """Predict the quantiles of the next ``horizon`` steps of every series in ``inputs``.

        The model is a next-patch predictor: the output at patch position ``i`` predicts the values of patch ``i + 1``.
        The last context patch therefore acts as the anchor producing the first forecast patch.

        By default the whole horizon is produced in a single forward pass. When ``decode_block_size`` (a multiple of the
        model's patch size) is set and the horizon spans more than one block, the horizon is decoded block by block: the
        predicted medians of one block are fed back as observed inputs for the next, using a KV cache to avoid
        recomputing the context. This can improve stability over very long horizons at the cost of extra forward passes.

        Parameters
        ----------
        inputs
            Batch of series to forecast. The context length must be divisible by the model's patch size.
        horizon
            Number of steps to predict.
        decode_block_size
            Block size for autoregressive block decoding, or None to predict the whole horizon in one pass.
        has_missing_values
            Whether ``inputs["target_mask"]`` may contain unobserved entries. When False, attention uses a faster path
            that skips the construction of an explicit mask.
        scaler_fallback_min_obs
            Stabilizes the scaler on leading patches with fewer than this many observations; see
            :func:`backfill_short_patches`. Disabled when ``<= 0``.
        quantile_real_cap_k
            Clips each predicted quantile to ``[ctx_min - k * scale, ctx_max + k * scale]``, where ``ctx_min`` and
            ``ctx_max`` are the observed context bounds and ``scale`` is the scale at the last context position.
            Guards against runaway predictions on near-degenerate inputs. Disabled when ``<= 0``.

        Returns
        -------
        Quantile forecasts of shape ``(len(self.knots), batch, num_variates, horizon)``, sorted along the first axis.
        """
        patch_size = self.config.patch_size
        num_patches = math.ceil(horizon / patch_size)
        median_idx = self.knots.index(0.5)

        initial_len = inputs["target"].shape[-1]
        if initial_len % patch_size != 0:
            raise ValueError(f"Context length ({initial_len}) must be divisible by patch_size ({patch_size})")
        initial_patches = initial_len // patch_size
        device = inputs["target"].device

        if decode_block_size:
            if decode_block_size % patch_size != 0:
                raise ValueError(
                    f"decode_block_size ({decode_block_size}) must be divisible by patch_size ({patch_size})"
                )
            block_size_patches = min(decode_block_size // patch_size, num_patches)
        else:
            block_size_patches = num_patches

        full_target, full_mask = self._prepare_forecast_inputs(inputs, num_patches)
        base_group_ids = self._prepare_group_ids(
            inputs["series_ids"],
            full_mask[..., :initial_len],
            num_positions=max(initial_patches + num_patches, 2 * block_size_patches),
            initial_patches=initial_patches,
        )

        use_cache = block_size_patches < num_patches
        kv_cache: Optional[KVCache] = None
        all_time_ids: Optional[torch.Tensor] = None
        cache_valid: Optional[torch.Tensor] = None
        if use_cache:
            # Each iteration writes one block of context tokens plus one ephemeral block of prediction tokens.
            kv_cache = self._get_kv_cache(initial_patches + 2 * num_patches, full_target.shape[:-1], device)
            all_time_ids = torch.arange(initial_patches, initial_patches + 2 * num_patches, device=device)

        quantiles = torch.zeros(
            len(self.knots),
            *full_target.shape[:-1],
            num_patches,
            patch_size,
            device=device,
            dtype=full_target.dtype,
        )
        patches_predicted = 0
        cache_len = 0
        context_x: Optional[torch.Tensor] = None
        scaled_context: Optional[torch.Tensor] = None
        cap_min = cap_max = None

        while patches_predicted < num_patches:
            block = min(block_size_patches, num_patches - patches_predicted)
            pred_start = initial_len + patches_predicted * patch_size
            pred_end = pred_start + block * patch_size

            # The scaler is causal, so the statistics of the context never change; those of the prediction region do,
            # as previously predicted medians get filled in, and are therefore recomputed on every iteration.
            _, loc, scale = self.scaler(full_target, full_mask)
            if scaler_fallback_min_obs > 0:
                loc, scale = backfill_short_patches(
                    full_target, loc, scale, full_mask, patch_size, scaler_fallback_min_obs
                )

            if quantile_real_cap_k > 0 and cap_min is None:
                cap_min, cap_max = self._quantile_caps(
                    full_target[..., :initial_len],
                    full_mask[..., :initial_len],
                    anchor_scale=scale[..., initial_len - 1 : initial_len],
                    cap_k=quantile_real_cap_k,
                )

            if scaled_context is None:
                scaled_context = self._standardize(full_target, full_mask, loc, scale, 0, initial_len)
                context_x = self._embed_patches(scaled_context, full_mask[..., :initial_len])

            pred_x = self._embed_patches(
                self._standardize(full_target, full_mask, loc, scale, pred_start, pred_end),
                full_mask[..., pred_start:pred_end],
            )

            if patches_predicted == 0:
                combined_x = torch.cat([context_x, pred_x], dim=-2)
                combined_group_ids = base_group_ids[..., : initial_patches + block]
                time_ids = None
            else:
                # Re-embed the previous block, whose missing values have been replaced by the predicted medians.
                assert all_time_ids is not None
                prev_start = pred_start - block_size_patches * patch_size
                median_x = self._embed_patches(
                    self._standardize(full_target, full_mask, loc, scale, prev_start, pred_start),
                    torch.ones_like(full_mask[..., prev_start:pred_start]),
                )
                combined_x = torch.cat([median_x, pred_x], dim=-2)
                combined_group_ids = base_group_ids[..., : block_size_patches + block]
                tid_start = patches_predicted - block_size_patches
                time_ids = all_time_ids[tid_start : tid_start + block_size_patches + block]

            if kv_cache is not None:
                kv_cache.ephemeral_len = block
                kv_read_len = cache_len + combined_x.shape[-2]
            else:
                kv_read_len = None

            x_out = self.transformer(
                combined_x,
                time_ids=time_ids,
                group_ids=combined_group_ids,
                kv_cache=kv_cache,
                kv_read_len=kv_read_len,
                cache_valid=cache_valid,
                has_missing_values=has_missing_values,
            )

            if kv_cache is not None:
                if cache_valid is None:
                    # Record which prefilled positions hold observed data, so that later decode steps can keep masking
                    # out the fully unobserved context patches. Positions written after the prefill are always valid.
                    cache_valid = torch.ones(
                        math.prod(combined_x.shape[:-2]), kv_cache.max_size, dtype=torch.bool, device=device
                    )
                    flat_group_ids = combined_group_ids.expand(combined_x.shape[:-1]).reshape(-1, combined_x.shape[-2])
                    cache_valid[:, : combined_x.shape[-2]] = flat_group_ids != -1
                cache_len += combined_x.shape[-2] - block

            # Drop the final position: the output at patch i predicts patch i + 1, so the block of `block` forecast
            # patches is produced by the `block` positions ending just before the last one.
            block_quantiles = self.output_head(x_out[..., -(block + 1) : -1, :])

            block_loc = rearrange(loc[..., pred_start:pred_end], "... (s p) -> ... s p", p=patch_size)
            block_scale = rearrange(scale[..., pred_start:pred_end], "... (s p) -> ... s p", p=patch_size)
            block_real = self._clamp_nonfinite(block_quantiles.sinh() * block_scale + block_loc)
            if cap_min is not None:
                block_real.clamp_(cap_min, cap_max)
            # Quantile crossing is possible, since the knots are predicted independently.
            block_real = block_real.sort(dim=0).values
            quantiles[..., patches_predicted : patches_predicted + block, :] = block_real

            patches_predicted += block

            if patches_predicted < num_patches:
                # Feed the predicted medians back in as observed values for the next block.
                full_target[..., pred_start:pred_end] = rearrange(block_real[median_idx], "... s p -> ... (s p)")
                full_mask[..., pred_start:pred_end] = True

        return rearrange(quantiles, "... seq patch -> ... (seq patch)")[..., :horizon]

    def _prepare_forecast_inputs(
        self, inputs: Toto2ForecastInputs, num_patches: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extend the target and its mask with a zero-filled, unobserved prediction region.

        The final context patch is marked as fully observed: a short series whose tail was padded with unobserved
        positions would otherwise be out of distribution, since training never systematically leaves the end of the
        context unobserved. Block decoding later marks prediction patches as observed as their medians are filled in.
        """
        patch_size = self.config.patch_size
        target, target_mask = inputs["target"], inputs["target_mask"]
        pred_len = num_patches * patch_size

        pred_shape = target.shape[:-1] + (pred_len,)
        full_target = torch.cat([target, torch.zeros(pred_shape, device=target.device, dtype=target.dtype)], dim=-1)
        full_mask = torch.cat(
            [
                target_mask[..., :-patch_size],
                torch.ones_like(target_mask[..., -patch_size:]),
                torch.zeros(pred_shape, device=target_mask.device, dtype=torch.bool),
            ],
            dim=-1,
        )
        return full_target, full_mask

    def _prepare_group_ids(
        self, series_ids: torch.Tensor, context_mask: torch.Tensor, num_positions: int, initial_patches: int
    ) -> torch.Tensor:
        """Expand ``series_ids`` to one id per patch, marking fully unobserved context patches with ``-1``."""
        group_ids = repeat(series_ids, "... n_var -> ... n_var seq", seq=num_positions).clone()
        patch_obs = reduce(context_mask, "... (seq patch) -> ... seq", "sum", patch=self.config.patch_size)
        group_ids[..., :initial_patches][patch_obs == 0] = -1
        return group_ids

    @staticmethod
    def _standardize(
        target: torch.Tensor, mask: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor, start: int, end: int
    ) -> torch.Tensor:
        """Standardize ``target[..., start:end]`` and compress it with ``asinh``, zeroing out unobserved entries."""
        standardized = (target[..., start:end] - loc[..., start:end]) / scale[..., start:end]
        return torch.where(mask[..., start:end], standardized, 0.0).asinh()

    @staticmethod
    def _quantile_caps(
        context: torch.Tensor, context_mask: torch.Tensor, anchor_scale: torch.Tensor, cap_k: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(min, max)`` bounds that the predicted quantiles are clipped to.

        The bounds extend the observed range of the context by ``cap_k`` times the scale at the last context position.
        """
        unobserved = ~context_mask
        # A single buffer is reused for both reductions to avoid allocating a second copy of the context.
        buffer = context.masked_fill(unobserved, float("-inf"))
        margin = cap_k * anchor_scale
        cap_max = (torch.nan_to_num(buffer.amax(-1, keepdim=True), neginf=0.0) + margin).unsqueeze(-1)
        buffer.masked_fill_(unobserved, float("inf"))
        cap_min = (torch.nan_to_num(buffer.amin(-1, keepdim=True), posinf=0.0) - margin).unsqueeze(-1)
        return cap_min, cap_max

    @classmethod
    def from_pretrained(cls, model_id: str, device: str | torch.device = "cpu") -> "Toto2Model":
        """Load a pretrained checkpoint from a local directory or from the Hugging Face Hub.

        Parameters
        ----------
        model_id
            Either a path to a directory containing ``config.json`` and ``model.safetensors``, or the id of such a
            repository on the Hugging Face Hub (e.g. ``"Datadog/Toto-2.0-22m"``).
        device
            Device to load the weights onto.
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        if os.path.isdir(model_id):
            config_path = os.path.join(model_id, "config.json")
            weights_path = os.path.join(model_id, "model.safetensors")
        else:
            config_path = hf_hub_download(repo_id=model_id, filename="config.json")
            weights_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")

        with open(config_path) as f:
            raw_config = json.load(f)
        known_fields = {field.name for field in dataclasses.fields(Toto2ModelConfig)}
        config = Toto2ModelConfig(**{k: v for k, v in raw_config.items() if k in known_fields})

        model = cls(config)
        model.load_state_dict(load_file(weights_path, device=str(device)))
        return model.to(device).eval()

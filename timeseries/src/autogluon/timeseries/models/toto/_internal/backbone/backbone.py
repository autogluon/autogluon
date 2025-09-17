# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

import math
from typing import NamedTuple, Optional

import torch

from .distribution import MixtureOfStudentTsOutput
from .kvcache import KVCache
from .scaler import CausalPatchStdMeanScaler
from .transformer import Transformer


class TotoOutput(NamedTuple):
    """
    Output of the Toto model. Contains the output distribution, the location parameters,
    and the scale parameters.
    """

    distribution: torch.distributions.Distribution
    loc: torch.Tensor
    scale: torch.Tensor


def patchify_id_mask(id_mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    patched_id_mask = id_mask.unfold(dimension=-1, size=patch_size, step=patch_size)
    patched_id_mask_min = patched_id_mask.min(-1).values
    patched_id_mask_max = patched_id_mask.max(-1).values
    assert torch.eq(patched_id_mask_min, patched_id_mask_max).all(), "Patches cannot span multiple datasets"
    return patched_id_mask_min


class PatchEmbedding(torch.nn.Module):
    """
    Multivariate time series patch embedding.
    Patchifies each variate separately.
    """

    def __init__(self, patch_size: int, stride: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.stride = stride
        self.projection = torch.nn.Linear(self.patch_size, self.embed_dim)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        return x.unfold(dimension=-1, size=self.patch_size, step=self.stride)

    def forward(
        self,
        x: torch.Tensor,
        id_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[-1] % self.patch_size == 0, (
            f"Series length ({x.shape=}) must be divisible by ({self.patch_size=})"
        )
        x_patched: torch.Tensor = self._patchify(x)
        id_mask_patched: torch.Tensor = self._patchify(id_mask)

        assert torch.eq(id_mask_patched.min(-1).values, id_mask_patched.max(-1).values).all(), (
            "Patches cannot span multiple datasets"
        )

        return (
            self.projection(x_patched),
            id_mask_patched.min(-1).values,
        )


class TotoBackbone(torch.nn.Module):
    """
    Toto (Timeseries-Optimized Transformer for Observability) is a transformer-based model for multivariate
    time series forecasting. It applies a patch embedding to the input data, followed by a transformer
    that alternates between time-wise and space-wise attention. The transformer is followed by a linear projection
    that maps the transformer output to the output distribution.

    The output distribution can be a single distribution (e.g. Gaussian) or a mixture of distributions.
    If a mixture of distributions is used, the model will learn to predict the mixture weights
    as well as the parameters of the individual distributions.

    Parameters
    ----------
    patch_size
        Size of the patch to use for the patch embedding.
    stride
        Stride to use for the patch embedding.
    embed_dim
        Dimension of the model's latent space.
    num_layers
        Number of transformer layers to use.
    num_heads
        Number of attention heads to use in each self-attention layer.
    mlp_hidden_dim
        Dimension of the hidden layer in the feedforward network.
    dropout
        Dropout rate to use in the model.
    spacewise_every_n_layers
        How many time-wise transformer layers to apply between each space-wise transformer layer.
    spacewise_first
        Whether to apply space-wise attention before time-wise attention.
    scaler_cls
        Class to use for scaling the input data.
    output_distribution_classes
        List of classes to use for the output distribution. If a single class is provided, the model
        will output a single distribution. If multiple classes are provided, the model will output a
        learned mixture of distributions.
    output_distribution_kwargs
        Keyword arguments to pass to the output distribution class. Note: this currently only works
        with a single output distribution class.
    use_memory_efficient_attention:
        Whether to use memory-efficient attention. If True, the model will use the memory-efficient from xFormers.
    stabilize_with_global:
        Whether to use global statistics to stabilize causal statistics by clamping extreme values. Only applies to causal scalers.
    scale_factor_exponent:
        Exponent that controls the allowed range of deviation from global scale for causal scalers.
    """

    def __init__(
        self,
        patch_size: int,
        stride: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float,
        spacewise_every_n_layers: int,
        scaler_cls: str,
        output_distribution_classes: list[str],
        spacewise_first: bool = True,
        output_distribution_kwargs: Optional[dict] = None,
        use_memory_efficient_attention: bool = True,
        stabilize_with_global: bool = True,
        scale_factor_exponent: float = 10.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # strings are used when loading a safetensors checkpoint
        # Initialize patch-based scalers with the correct patch_size

        self.scaler = CausalPatchStdMeanScaler(
            patch_size=patch_size,
            stabilize_with_global=stabilize_with_global,
            scale_factor_exponent=scale_factor_exponent,
        )
        self.patch_embed = PatchEmbedding(patch_size, stride, embed_dim)
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_memory_efficient_attention = use_memory_efficient_attention
        self.transformer = Transformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=self.num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout,
            spacewise_every_n_layers=spacewise_every_n_layers,
            spacewise_first=spacewise_first,
            use_memory_efficient_attention=self.use_memory_efficient_attention,
        )
        self.unembed = torch.nn.Linear(embed_dim, embed_dim * patch_size)

        # TODO[BEN] this doesn't need to be a list
        output_distribution_classes_ = [MixtureOfStudentTsOutput]
        self.output_distribution = output_distribution_classes_[0](embed_dim, **(output_distribution_kwargs or {}))

    def allocate_kv_cache(
        self,
        batch_size: int,
        num_variates: int,
        max_time_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> KVCache:
        return KVCache(
            batch_size=batch_size,
            num_variates=num_variates,
            transformer_layers=list(self.transformer.layers),
            num_layers=self.num_layers,
            embed_dim=self.embed_dim,
            num_heads=self.transformer.layers[0].num_heads,  # type: ignore
            max_seq_len=math.ceil(max_time_steps / self.patch_embed.stride),
            device=device,
            dtype=dtype,
            use_memory_efficient_attention=self.use_memory_efficient_attention,
        )

    def backbone(
        self,
        inputs: torch.Tensor,
        input_padding_mask: torch.Tensor,
        id_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        scaling_prefix_length: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scaled_inputs: torch.Tensor
        loc: torch.Tensor
        scale: torch.Tensor

        # Standard scaling operation, same API but without ID mask.
        scaled_inputs, loc, scale = self.scaler(
            inputs,
            weights=torch.ones_like(inputs, device=inputs.device),
            padding_mask=input_padding_mask,
            prefix_length=scaling_prefix_length,
        )

        if kv_cache is not None:
            prefix_len = self.patch_embed.stride * kv_cache.current_len(0)

            # Truncate inputs so that the transformer only processes
            # the last patch in the sequence. We'll use the KVCache
            # for the earlier patches.
            scaled_inputs = scaled_inputs[:, :, prefix_len:]

            # As a simplification, when using kv cache we only allow decoding
            # one step at a time after the initial forward pass.
            assert (prefix_len == 0) or (scaled_inputs.shape[-1] == self.patch_embed.stride), (
                "Must decode one step at a time."
            )

            input_padding_mask = input_padding_mask[:, :, prefix_len:]
            id_mask = id_mask[:, :, prefix_len:]

        embeddings: torch.Tensor
        reduced_id_mask: torch.Tensor

        embeddings, reduced_id_mask = self.patch_embed(scaled_inputs, id_mask)

        # Apply the transformer on the embeddings
        transformed: torch.Tensor = self.transformer(embeddings, reduced_id_mask, kv_cache)

        # Unembed and flatten the sequence
        unembedded = self.unembed(transformed)
        batch_size, num_variates, seq_len = unembedded.shape[:3]
        patch_size = unembedded.shape[-1] // self.embed_dim
        flattened = unembedded.view(batch_size, num_variates, seq_len * patch_size, self.embed_dim)
        return flattened, loc, scale

    def forward(
        self,
        inputs: torch.Tensor,
        input_padding_mask: torch.Tensor,
        id_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        scaling_prefix_length: Optional[int] = None,
    ) -> TotoOutput:
        flattened, loc, scale = self.backbone(
            inputs,
            input_padding_mask,
            id_mask,
            kv_cache,
            scaling_prefix_length,
        )

        return TotoOutput(self.output_distribution(flattened), loc, scale)

    @property
    def device(self):
        return next(self.parameters()).device
